"""Hagane Python wrapper codegen — brackets compiled execution (ADR-042 A4).

The Hagane runtime's unified queue (``HAGANE_METALLIB_UNIFIED_QUEUE``) defers the
cross-queue drain for routed metallib outputs and orders them GPU-side via a
shared event. That is correct in plain eager, but under ``torch.compile``
Inductor's extern/generated kernels read those outputs directly — bypassing the
runtime's read-boundary hooks AND the event chain — so a deferred drain NaNs (the
X+74 failure). ``unified_defer_safe()`` must therefore eager-drain while compiled
code is executing.

``torch.compiler.is_compiling()`` is True only during tracing, not during
compiled-graph execution, so it can't gate the runtime at op-invocation time.
This wrapper emits a runtime signal instead: it brackets the generated
``call(args)`` body with ``set_compile_active(1)`` … ``finally:
set_compile_active(0)`` (a re-entrancy-safe counter), giving the runtime a direct
"compiled execution in flight" flag. With it, a bare ``torch.compile(model)`` is
correct under the unified queue without the eager-eval-barrier harness knob.

Coverage: this is the default Python-wrapper path. ``cpp_wrapper`` / AOTInductor
generate C++ and never instantiate this class — they fall back to the
``HAGANE_COMPILE_ACTIVE`` env + eager-eval-barrier defense in
``unified_defer_safe()``.
"""

from __future__ import annotations

from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)


class HaganeWrapperCodegen(PythonWrapperCodegen):
    # Imported once into the generated module header; referenced inside the
    # bracket calls below.
    _SIGNAL_IMPORT = (
        "from hagane.inductor_backend.runtime import "
        "set_compile_active as _hagane_set_compile_active"
    )

    @staticmethod
    def create(
        is_subgraph,
        subgraph_name,
        parent_wrapper,
        partition_signatures=None,
    ):
        # The base create() is a staticmethod hardcoded to PythonWrapperCodegen,
        # so registering the subclass is not enough — we must return our class
        # for the top-level wrapper. Subgraphs keep the stock subgraph wrapper
        # (their kernels run inside the bracketed top-level call() anyway).
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return HaganeWrapperCodegen()

    def get_wrapper_call_indent(self) -> int:
        # Indent the whole call() body one level deeper so it sits inside the
        # try: opened in write_prefix.
        return super().get_wrapper_call_indent() + 1

    def write_prefix(self) -> None:
        super().write_prefix()
        self.header.writeline(self._SIGNAL_IMPORT)
        # try: sits at the call() body indent (one shallower than the body).
        body_indent = super().get_wrapper_call_indent()
        with self.prefix.indent(body_indent):
            self.prefix.writeline("_hagane_set_compile_active(1)")
            self.prefix.writeline("try:")

    def generate_before_suffix(self, result) -> None:
        super().generate_before_suffix(result)
        # Close the try: opened in write_prefix. finally: at the call() body
        # indent; the decrement one level deeper. A finally (not a plain call
        # before return) so an exception in the body can't leak the counter
        # (a stuck counter would force every later eager op onto the slow
        # eager-drain — correct, but a perf regression).
        body_indent = super().get_wrapper_call_indent()
        with result.indent(body_indent):
            result.writeline("finally:")
            with result.indent():
                result.writeline("_hagane_set_compile_active(0)")
