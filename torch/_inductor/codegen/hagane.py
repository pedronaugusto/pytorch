# Sprint IX MVP — Inductor codegen scaffold for the Hagane platform fork.
#
# Architectural note: unlike the MPS / Triton / Pallas backends which emit
# kernel-source strings (Metal Shading Language, Triton IR, JAX/Pallas
# Python), Hagane has no source-level kernel emission path. Every op
# already has an MLX-backed implementation that dispatches via
# ``aten/src/ATen/hip/HaganeOps.cpp``. The MVP codegen target is therefore
# *Python that calls torch.* ops on cuda-device tensors*, which the
# existing ``aten`` dispatcher routes through ``HaganeOps.cpp`` →
# ``haganeOps*`` C-ABI → MLX. Inductor still drives scheduling, buffer
# assignment, and FX-graph capture; only the kernel emission is
# effectively a flat sequence of ATen fallbacks. Loop fusion, autotune,
# and full op breadth defer to Sprint X+.
#
# Reference scaffolding: ``mps.py`` (1217 LOC, Metal target),
# ``pallas.py`` (4938 LOC, JAX/Pallas target).
from __future__ import annotations

import functools
import logging
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import ExprPrinter as ExprPrinter_

from ..virtualized import NullHandler, V
from .common import (
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from ..ops_handler import ReductionType, StoreMode
    from ..scheduler import Scheduler, SchedulerNode

log = logging.getLogger(__name__)

DTYPE_TO_TORCH_NAME: dict[torch.dtype, str] = {
    torch.bool: "torch.bool",
    torch.int8: "torch.int8",
    torch.int16: "torch.int16",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.uint8: "torch.uint8",
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.bfloat16: "torch.bfloat16",
}


def value_to_python(val: float | int | bool | str | CSEVariable) -> str:
    if isinstance(val, float):
        if val == torch.inf:
            return "float('inf')"
        elif val == -torch.inf:
            return "float('-inf')"
        elif val != val:
            return "float('nan')"
        return repr(val)
    elif isinstance(val, bool):
        return "True" if val else "False"
    return str(val)


class HaganeExprPrinter(ExprPrinter_):
    """Sympy → Python expression printer for Hagane kernel bodies.

    Hagane kernels emit Python source; this printer converts indexing
    sympy expressions to Python integer arithmetic. No fancy
    transformations — sympy's default Python printer suffices for the
    MVP since indexing is computed eagerly per-tensor by the wrapper,
    not per-element inside a kernel body.
    """

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        return f"({self.doprint(x)} // {self.doprint(div)})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.doprint(x)
        if div != 1:
            x = f"({x} // {self.doprint(div)})"
        return f"({x} % {self.doprint(mod)})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return "min(" + ", ".join(self.doprint(a) for a in expr.args) + ")"

    def _print_Max(self, expr: sympy.Expr) -> str:
        return "max(" + ", ".join(self.doprint(a) for a in expr.args) + ")"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        return f"abs({self.doprint(expr.args[0])})"

    def _print_Where(self, expr: sympy.Expr) -> str:
        cond, true_v, false_v = expr.args
        return (
            f"({self.doprint(true_v)} if {self.doprint(cond)} "
            f"else {self.doprint(false_v)})"
        )


# ---------------------------------------------------------------------------
# OpOverrides — element-wise op codegen.
#
# Each method returns a Python *expression* string. The expression is
# composed by SIMDKernel into kernel-body statements like
# ``auto var_N = <expr>``. For Hagane, a, b, c are scalar Python values
# (from a per-element kernel body OR — when the scaffold runs in
# whole-tensor fallback mode — torch.Tensor object names). The MVP
# emission below assumes scalar-Python semantics so that SIMDKernel's
# normal CSE pipeline composes correctly. Real tensor-granularity codegen
# (where each op is a single ``torch.X(...)`` call on whole tensors) lives
# in HaganeKernel.codegen_kernel.
# ---------------------------------------------------------------------------


class HaganeOverrides(OpOverrides):
    """Lower Inductor IR ops to Python scalar expressions.

    Hagane MVP emits Python kernel bodies that operate on tensors as
    whole-array units. Per-element semantics are preserved through
    NumPy-style broadcasting; the kernel function signature receives
    torch.Tensor inputs and returns torch.Tensor outputs.
    """

    @staticmethod
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        return f"({x}).to({DTYPE_TO_TORCH_NAME[dtype]})"

    @staticmethod
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        return f"({x}).view({DTYPE_TO_TORCH_NAME[dtype]})"

    @staticmethod
    def constant(val: bool | float | int, dtype: torch.dtype) -> str:
        return value_to_python(val)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        # X+63: route through kernel.prepare_indexing to trigger axis-variable
        # emission via codegen_iteration_ranges_entry per free symbol.
        # Without this, sympy symbols like `x2` leak into kernel body as
        # undefined names (X+62 kernel_3 NameError surface).
        # X+69: rename_indexing registers free sizevars (dynamic-shape
        # symbols like s1) as kernel args; without it they leak into the
        # body as undefined names under automatic_dynamic_shapes.
        kernel = V.kernel
        expr = kernel.prepare_indexing(expr)
        return kernel.sexpr(kernel.rename_indexing(expr))

    # ----- Binary arithmetic --------------------------------------------------
    @staticmethod
    def add(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.add({a}, {b})"

    @staticmethod
    def sub(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.sub({a}, {b})"

    @staticmethod
    def mul(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.mul({a}, {b})"

    @staticmethod
    def div(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.div({a}, {b})"

    @staticmethod
    def truediv(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.true_divide({a}, {b})"

    @staticmethod
    def floordiv(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.floor_divide({a}, {b})"

    @staticmethod
    def mod(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.remainder({a}, {b})"

    @staticmethod
    def remainder(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.remainder({a}, {b})"

    @staticmethod
    def fmod(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.pow({a}, {b})"

    # ----- Comparison ---------------------------------------------------------
    # X+67: as_tensor on the lhs — torch.le(int, Tensor) rejects scalar-first
    # operands (surfaced by HF sdpa_mask arange comparisons on the math path).
    @staticmethod
    def lt(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.lt(torch.as_tensor({a}), {b})"

    @staticmethod
    def le(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.le(torch.as_tensor({a}), {b})"

    @staticmethod
    def gt(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.gt(torch.as_tensor({a}), {b})"

    @staticmethod
    def ge(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.ge(torch.as_tensor({a}), {b})"

    @staticmethod
    def eq(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.eq(torch.as_tensor({a}), {b})"

    @staticmethod
    def ne(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.ne(torch.as_tensor({a}), {b})"

    @staticmethod
    def minimum(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.maximum({a}, {b})"

    # ----- Bitwise / logical --------------------------------------------------
    @staticmethod
    def bitwise_and(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.bitwise_and({a}, {b})"

    @staticmethod
    def bitwise_or(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.bitwise_or({a}, {b})"

    @staticmethod
    def bitwise_xor(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.bitwise_xor({a}, {b})"

    @staticmethod
    def bitwise_not(a):  # type: ignore[no-untyped-def, override]
        return f"torch.bitwise_not({a})"

    @staticmethod
    def bitwise_left_shift(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.bitwise_left_shift({a}, {b})"

    @staticmethod
    def bitwise_right_shift(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.bitwise_right_shift({a}, {b})"

    @staticmethod
    def logical_and(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.logical_and(torch.as_tensor({a}), torch.as_tensor({b}))"

    @staticmethod
    def logical_or(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.logical_or(torch.as_tensor({a}), torch.as_tensor({b}))"

    @staticmethod
    def logical_xor(a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.logical_xor(torch.as_tensor({a}), torch.as_tensor({b}))"

    @staticmethod
    def logical_not(a):  # type: ignore[no-untyped-def, override]
        return f"torch.logical_not({a})"

    # ----- Unary math ---------------------------------------------------------
    @staticmethod
    def sqrt(x):  # type: ignore[no-untyped-def, override]
        return f"torch.sqrt({x})"

    @staticmethod
    def rsqrt(x):  # type: ignore[no-untyped-def, override]
        return f"torch.rsqrt({x})"

    @staticmethod
    def exp(x):  # type: ignore[no-untyped-def, override]
        return f"torch.exp({x})"

    @staticmethod
    def expm1(x):  # type: ignore[no-untyped-def, override]
        return f"torch.expm1({x})"

    @staticmethod
    def log(x):  # type: ignore[no-untyped-def, override]
        return f"torch.log({x})"

    @staticmethod
    def log1p(x):  # type: ignore[no-untyped-def, override]
        return f"torch.log1p({x})"

    @staticmethod
    def log2(x):  # type: ignore[no-untyped-def, override]
        return f"torch.log2({x})"

    @staticmethod
    def log10(x):  # type: ignore[no-untyped-def, override]
        return f"torch.log10({x})"

    @staticmethod
    def sin(x):  # type: ignore[no-untyped-def, override]
        return f"torch.sin({x})"

    @staticmethod
    def cos(x):  # type: ignore[no-untyped-def, override]
        return f"torch.cos({x})"

    @staticmethod
    def tan(x):  # type: ignore[no-untyped-def, override]
        return f"torch.tan({x})"

    @staticmethod
    def sinh(x):  # type: ignore[no-untyped-def, override]
        return f"torch.sinh({x})"

    @staticmethod
    def cosh(x):  # type: ignore[no-untyped-def, override]
        return f"torch.cosh({x})"

    @staticmethod
    def tanh(x):  # type: ignore[no-untyped-def, override]
        return f"torch.tanh({x})"

    @staticmethod
    def asin(x):  # type: ignore[no-untyped-def, override]
        return f"torch.asin({x})"

    @staticmethod
    def acos(x):  # type: ignore[no-untyped-def, override]
        return f"torch.acos({x})"

    @staticmethod
    def atan(x):  # type: ignore[no-untyped-def, override]
        return f"torch.atan({x})"

    @staticmethod
    def atan2(x, y):  # type: ignore[no-untyped-def, override]
        return f"torch.atan2({x}, {y})"

    @staticmethod
    def atanh(x):  # type: ignore[no-untyped-def, override]
        return f"torch.atanh({x})"

    # ----- Unary other --------------------------------------------------------
    @staticmethod
    def abs(x):  # type: ignore[no-untyped-def, override]
        return f"torch.abs({x})"

    @staticmethod
    def sign(x):  # type: ignore[no-untyped-def, override]
        return f"torch.sign({x})"

    @staticmethod
    def signbit(x):  # type: ignore[no-untyped-def, override]
        return f"torch.signbit({x})"

    @staticmethod
    def neg(x):  # type: ignore[no-untyped-def, override]
        return f"torch.neg({x})"

    @staticmethod
    def floor(x):  # type: ignore[no-untyped-def, override]
        return f"torch.floor({x})"

    @staticmethod
    def ceil(x):  # type: ignore[no-untyped-def, override]
        return f"torch.ceil({x})"

    @staticmethod
    def round(x):  # type: ignore[no-untyped-def, override]
        return f"torch.round({x})"

    @staticmethod
    def trunc(x):  # type: ignore[no-untyped-def, override]
        return f"torch.trunc({x})"

    @staticmethod
    def reciprocal(x):  # type: ignore[no-untyped-def, override]
        return f"torch.reciprocal({x})"

    @staticmethod
    def square(x):  # type: ignore[no-untyped-def, override]
        return f"torch.square({x})"

    @staticmethod
    def isnan(x):  # type: ignore[no-untyped-def, override]
        return f"torch.isnan({x})"

    @staticmethod
    def isinf(x):  # type: ignore[no-untyped-def, override]
        return f"torch.isinf({x})"

    # ----- Ternary ------------------------------------------------------------
    @staticmethod
    def where(cond, a, b):  # type: ignore[no-untyped-def, override]
        return f"torch.where({cond}, {a}, {b})"

    @staticmethod
    def clamp(x, lo, hi):  # type: ignore[no-untyped-def, override]
        return f"torch.clamp({x}, {lo}, {hi})"


HaganeOverrides._initialize_pointwise_overrides("hagane")


# ---------------------------------------------------------------------------
# Kernel — emits Python kernel bodies.
#
# A HaganeKernel encapsulates a fused group of Inductor IR ops. The MVP
# disables fusion (HaganeScheduling.can_fuse_* return False), so each
# kernel typically corresponds to a single FX-graph node. The kernel body
# is a sequence of Python statements assembled from HaganeOverrides
# expressions; loads and stores convert tensor names to local Python
# variables that the launcher binds at call time.
# ---------------------------------------------------------------------------


class HaganeKernel(SIMDKernel):
    """Inductor SIMDKernel that emits Python source rather than MSL/HIP.

    Sprint IX MVP scope: each kernel is a flat sequence of ATen fallback
    calls operating on whole tensors. The ``loads`` / ``stores`` /
    ``compute`` IndentedBuffers accumulate Python statements.
    """

    overrides = HaganeOverrides  # type: ignore[assignment]
    suffix = ""
    newvar_prefix = ""
    pexpr = PythonPrinter().doprint
    sexpr = HaganeExprPrinter().doprint
    kexpr = sexpr
    headers: OrderedSet[str] = OrderedSet()

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs: Any,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self._emitted_indexers: set[str] = set()

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_TORCH_NAME[dtype]

    def codegen_iteration_ranges_entry(self, entry: Any) -> None:
        # X+63: emit axis-var definitions into self.indexing_code. Each
        # range tree's root index symbol (xindex, r0index, ...) is emitted
        # once as a 1-D torch.arange of the tree's full numel; per-entry
        # lines then reference that root via FloorDiv/ModularIndexing or
        # alias it directly when expr collapses to the root symbol.
        # Whole-tensor broadcasting handles per-element semantics during
        # downstream torch.* ops. Mirrors MPS pattern but uses tensors
        # instead of scalar declarations.
        root = entry.root
        root_sym = str(root.index_sym())
        if root_sym not in self._emitted_indexers:
            root_size = self.pexpr(self.rename_indexing(root.numel))
            arange = (
                f"torch.arange({root_size}, dtype=torch.int64, device='cuda')"
            )
            # X+68: tiled kernels (>1 pointwise tree, e.g. GPT-2 Conv1D
            # transposes) view each root arange on its own broadcast dim so
            # cross-tree index arithmetic forms the outer product. Single-tree
            # kernels keep the flat 1-D form.
            prefixes = [t.prefix for t in self.range_trees if not t.is_reduction]
            if len(prefixes) > 1 and root.prefix in prefixes:
                view = ["1"] * len(prefixes)
                view[prefixes.index(root.prefix)] = "-1"
                arange = f"{arange}.view({', '.join(view)})"
            self.indexing_code.writeline(f"{root_sym} = {arange}")
            self._emitted_indexers.add(root_sym)
        expr_str = self.sexpr(self.rename_indexing(entry.expr))
        self.indexing_code.writeline(f"{entry.name} = {expr_str}")

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # X+67: no emitted bounds asserts; indirect-index correctness is
        # gated by the eager parity sweeps. Surfaced by HF sdpa_mask's
        # arange-indexed mask construction on the SDPA math path.
        pass

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        # X+64: Pointwise kernels emit indexed gather
        # `var.reshape(-1)[idx_str]` honoring Inductor's per-element
        # index. Composes with X+63 axis-var emission (xindex/x0/x1
        # defined in indexing_code) to produce a 1-D (kernel_numel,)
        # tensor that broadcasts correctly with axis-var arithmetic.
        # Reduction kernels keep X+55 whole-tensor semantics because
        # reduction codegen below uses value.shape.index(red_size) to
        # locate the reduction axis — needs multi-dim native shape.
        if self.inside_reduction:
            line = f"{var}"
        else:
            # X+68: memory-linear alias — index exprs compute MEMORY
            # offsets; reshape(-1) copies transpose-strided inputs into
            # logical order, silently decoupling the two (GPT-2 Conv1D).
            line = (
                f"torch.as_strided({var}, ({var}.numel(),), (1,))"
                f"[{self.sexpr(self.rename_indexing(index))}]"
            )
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: StoreMode = None,
    ) -> None:
        var = self.args.output(name)
        if mode is None:
            if self.inside_reduction:
                line = f"{var}.copy_({value}.reshape({var}.shape))"
            else:
                # X+68: honor the store index via a memory-linear alias.
                # The copy_(value.reshape(shape)) form assumed iteration
                # order == logical row-major order, which breaks on
                # transpose-strided outputs (GPT-2 Conv1D) and tiled
                # (multi-tree) kernels.
                idx = self.prepare_indexing(index)
                val = f"{value}"
                ptrees = [t for t in self.range_trees if not t.is_reduction]
                if self.features.reduction_numel != 1 and len(ptrees) == 1:
                    # X+69: reduction results stored under disable_reduction
                    # may keep collapsed dims (e.g. (32,1) vs flat (32,));
                    # only flatten in true reduction kernels with a flat
                    # pointwise side — tiled pointwise kernels carry a
                    # numel-1 reduction tree and must keep their 2-D value.
                    val = f"{value}.reshape(-1)"
                line = (
                    f"torch.as_strided({var}, ({var}.numel(),), (1,))"
                    f"[{self.sexpr(self.rename_indexing(idx))}] = {val}"
                )
        elif mode == "atomic_add":
            line = f"{var}.add_({value})"
        else:
            raise RuntimeError(f"Unimplemented store mode {mode}")
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # Scalar output (features.numel == 1) → whole-tensor emission
        # (X+56 invariant). Dim-aware reductions look up the reduction
        # axis at runtime via value.shape.index(reduction_numel); load()
        # returns the whole input tensor (see HaganeKernel.load above), so
        # the kept output dim equals value.dim() - reduction_axes_count.
        assert self.inside_reduction
        REDUCE_OPS = {
            "sum": "torch.sum",
            "prod": "torch.prod",
            "max": "torch.amax",
            "min": "torch.amin",
            "argmin": "torch.argmin",
            "argmax": "torch.argmax",
            "any": "torch.any",
            "all": "torch.all",
        }
        if reduction_type not in REDUCE_OPS:
            raise NotImplementedError(f"reduction_type {reduction_type}")
        op = REDUCE_OPS[reduction_type]
        if self.features.numel == 1:
            line = f"{op}({value})"
        else:
            red_size = self.pexpr(self.rename_indexing(self.features.reduction_numel))
            line = f"{op}({value}, dim=list({value}.shape).index({red_size}))"
        return self.cse.generate(self.compute, line, dtype=dtype)

    def call_kernel(
        self, name: str, node: Any = None, deallocate_ws: bool = True
    ) -> None:
        wrapper = V.graph.wrapper_code
        for v in self.args.sizevars:
            wrapper.ensure_size_computed(v)
        # Bypass wrapper.generate_kernel_call, which forces CUDA stream
        # handling (get_raw_stream + c_void_p(stream)) incompatible with the
        # Hagane MVP's whole-tensor torch.* dispatch. python_argdefs() yields
        # call args in canonical order matching codegen_kernel's def signature.
        _, call_args, _, _ = self.args.python_argdefs()
        args = [str(a) for a in call_args]
        wrapper.writeline(wrapper.wrap_kernel_call(name, args))

    def codegen_kernel(self, name: str | None = None) -> str:
        """Assemble the kernel function source as Python."""
        if name is None:
            name = "<KERNEL_NAME>"
        code = IndentedBuffer()
        code.writeline("import torch")
        code.writeline("")
        # Argument list — derived from SIMDKernel.args.python_argdefs().
        argdefs, _, _, _ = self.args.python_argdefs()
        arg_list = ", ".join(a.name for a in argdefs)
        code.writeline(f"def {name}({arg_list}):")
        with code.indent():
            code.splice(self.indexing_code)  # X+63: axis variable defs
            for buf in (self.loads, self.compute, self.stores):
                code.splice(buf)
            code.writeline("return")
        return code.getvalue()


# ---------------------------------------------------------------------------
# Scheduling — registers HaganeKernel as the kernel type and disables
# fusion in the MVP. SIMDScheduling drives the per-node lowering pipeline;
# fusion gates control whether multiple FX nodes coalesce into one
# HaganeKernel emission.
# ---------------------------------------------------------------------------


class HaganeScheduling(SIMDScheduling):
    """Inductor SIMDScheduling that emits HaganeKernel (Python source).

    MVP: ``can_fuse_*`` return False so each FX node lowers to its own
    kernel. Sprint X+ enables horizontal/vertical fusion once the MLX
    runtime exposes a fused-op path.
    """

    kernel_type = HaganeKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Scheduler | None) -> None:
        super().__init__(scheduler)
        if isinstance(V.graph, NullHandler):
            return
        wrapper = V.graph.wrapper_code
        if wrapper is not None and not V.graph.cpp_wrapper:
            wrapper.header.splice(
                "# Hagane Inductor MVP — kernels are flat sequences of "
                "torch.* fallbacks dispatching through HaganeOps.cpp."
            )

    def can_fuse_horizontal(self, node1, node2) -> bool:  # type: ignore[no-untyped-def, override]
        return False

    def can_fuse_vertical(self, node1, node2) -> bool:  # type: ignore[no-untyped-def, override]
        return False

    def define_kernel(
        self,
        src_code: str,
        node_schedule: list[SchedulerNode],
        kernel: HaganeKernel,
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        kernel_name = f"hagane_kernel_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[src_code] = kernel_name
        src_code = src_code.replace("<KERNEL_NAME>", kernel_name)
        # PythonWrapperCodegen.define_kernel formats as ``<name> = <body>`` which
        # cannot wrap a multi-statement ``def`` block. Splice the function
        # definition directly into the module header instead.
        wrapper.header.splice("\n\n" + src_code)
        return kernel_name


# ---------------------------------------------------------------------------
# Backend registration helper — invoked from
# ``init_backend_registration`` only when the Hagane runtime DSO is
# loadable (gated by hagane.inductor_backend._target_marker).
# ---------------------------------------------------------------------------


@functools.cache
def _hagane_runtime_available() -> bool:
    try:
        import hagane.inductor_backend._target_marker  # noqa: F401
    except Exception:
        return False
    return True


def maybe_register_hagane_scheduling() -> bool:
    """Replace the "cuda" device scheduling with HaganeScheduling on Hagane builds.

    No-op on vanilla PyTorch where the Hagane runtime DSO is unavailable.
    Returns True if registration succeeded.
    """
    if not _hagane_runtime_available():
        return False
    from .common import register_backend_for_device
    from .hagane_wrapper import HaganeWrapperCodegen

    # HaganeWrapperCodegen brackets the generated call(args) with
    # set_compile_active(1/0) so the runtime's unified-queue deferred drain
    # eager-drains during compiled execution (A4 S0/S1) — a bare
    # torch.compile(model) can't defer-then-NaN (the X+74 failure).
    register_backend_for_device(
        "cuda",
        HaganeScheduling,
        HaganeWrapperCodegen,
    )
    # X+66: Override Inductor's sdpa_constraint with a passthrough on Hagane.
    # sdpa_constraint at lowering.py:3113 calls ExternKernel.require_stride_order
    # which emits assert_size_stride that mismatches Hagane's stride emission
    # for the natural BSHD-memory-transposed-to-BHSD layout (X+64 surfaced;
    # X+65 verified Dynamo-layer disallow_in_graph inert because SDPA
    # composite->flash decomposition happens at AOT autograd, deeper than
    # Dynamo's hook). _maybe_layout_constraints is looked up via late binding
    # at graph.py:1397; dict-update via add_layout_constraint overwrites.
    from torch._inductor.lowering import add_layout_constraint

    def _hagane_sdpa_no_constraint(fx_node, *args, **kwargs):
        return args, kwargs

    add_layout_constraint(
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        _hagane_sdpa_no_constraint,
    )
    # X+67: SDPA-composite-to-math routing lives in _inductor/decomposition.py
    # (module scope) — registering here is too late: this function runs at
    # backend init during lowering, after AOT capture already decomposed SDPA.
    return True
