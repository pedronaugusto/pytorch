// Copyright (c) 2026 Pedro Augusto
// SPDX-License-Identifier: MIT
//
// hagane_dispatch.h — Sprint X+21 extraction (X+29 TU split: helpers +
// trampolines now live in hagane_dispatch_detail.h; this file keeps
// OpConfig structs + rows + bridge templates).
//
// What this is
// ============
// The dispatch logic for HaganeMetallibBridge.cpp lifted into a header so the
// bridge .cpp shrinks to a REGISTER_DISPATCH-only TU. All `OpConfig` tables
// + bridge templates live here; their `cpu_dispatch_*` + helper symbols come
// from `hagane_dispatch_detail.h` via the `#include` below.
// HaganeMetallibBridge.cpp now contains only:
//
//   #include "hagane_dispatch.h"
//   namespace at::native {
//       using namespace at::native::hagane_dispatch::detail;
//       REGISTER_DISPATCH(abs_stub, &hagane_kernel_bridge<kAbsCfg>)
//       ...
//   } // at::native
//
// Why extracted in X+21 / re-split in X+29
// ========================================
// X+18 → X+20 grew the bridge from ~250 LOC to 759 LOC. Adding the X+21 row
// batch (1 lgamma + 1 frac + 11 binary) plus X+22+ variant templates (e.g.
// hardswish `TensorIterator&`, reductions) would cross the 800 LOC threshold
// in a single TU. Extracting now means subsequent sprints add a `cpu_dispatch_X`
// + `OpConfig` row + `REGISTER_DISPATCH` line without touching extraction
// logistics again.
//
// X+28 grew the header to 1139 LOC (past the 1100 soft cap). X+29 split the
// upper half (helpers + per-op cpu-dispatch + fp64 trampolines + scalar
// wrappers) into `hagane_dispatch_detail.h`, leaving this file at ~790 LOC
// with the OpConfig + bridge template halves intact.
//
// Linkage rules
// =============
// - All free functions `inline`.
// - All config rows `inline constexpr` at namespace scope (ODR-safe across TUs).
// - All templates header-defined (instantiated at the REGISTER_DISPATCH call
//   site in the bridge .cpp).
// - `MetallibState<Cfg>` uses `static inline` members for per-instantiation
//   `once_flag` + `available` storage (single linker symbol per Cfg).
// - The whole API lives in `at::native::hagane_dispatch::detail` so the .cpp
//   can `using namespace` it into `at::native` for REGISTER_DISPATCH.

#pragma once

#include <ATen/Functions.h>
#include <ATen/WrapDimUtils.h>      // E2E-1 — at::maybe_wrap_dim for cross_stub composition.
#include <ATen/native/Cross.h>     // E2E-1 — cross_fn / cross_stub signature.
#include <ATen/native/Activation.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Pow.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <hip/hip_runtime.h>
#include <hagane_ops.h>
#include <hip/hagane_detail/hagane_kernel_registry.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>

// X+29 TU split: helpers + per-op cpu-dispatch + fp64 trampolines + scalar
// wrappers live in detail header; OpConfig + bridge templates stay here.
#include "hagane_dispatch_detail.h"

namespace at::native::hagane_dispatch::detail {

using at::Tensor;
using c10::Scalar;

// ---- Per-op configuration -------------------------------------------------

struct OpConfig {
    const char* op_name;
    const char* metallib_kernel;     // nullptr → no metallib coverage.
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*);  // nullptr → CPU-only.
    void (*cpu_fallback)(TensorIteratorBase&);
    Tensor (*at_fp64_fn)(const Tensor&);  // nullptr → no fp64 round-trip.
};

// X+73 M3b: metallib_kernel now holds the kernel BASE name (the bridge appends
// _<dtypeTag>[_shape]_unrolled_contig per dispatch). abs ships float-only —
// the int/bool/bfloat AbsFunctor arms hit an MSL `abs`-overload gap, so only
// abs.metallib's float arm exists; other dtypes fall back to the MLX c_abi.
inline constexpr OpConfig kAbsCfg = {
    "abs",
    "hagane_AbsFunctor",
    &haganeOpsAbs,
    &cpu_dispatch_abs,
    &fp64_abs,
};

inline constexpr OpConfig kNegCfg  = {"neg",  nullptr, &haganeOpsNeg,  &cpu_dispatch_neg,  &fp64_neg};
inline constexpr OpConfig kSignCfg = {"sign", nullptr, &haganeOpsSign, &cpu_dispatch_sign, nullptr };
inline constexpr OpConfig kSgnCfg  = {"sgn",  nullptr, nullptr,        &cpu_dispatch_sgn,  nullptr };

// X+73 M3d: unary math wired onto the native metallib path (base name; the
// bridge appends _<dtypeTag>_unrolled_contig). All emit float/half/bfloat16
// arms; other dtypes fall back to the MLX c_abi_fn.
inline constexpr OpConfig kExpCfg     = {"exp",     "hagane_exp_kernel_cuda",   &haganeOpsExp,     &cpu_dispatch_exp,     nullptr};
inline constexpr OpConfig kLogCfg     = {"log",     "hagane_log_kernel_cuda",   &haganeOpsLog,     &cpu_dispatch_log,     nullptr};
inline constexpr OpConfig kSqrtCfg    = {"sqrt",    "hagane_sqrt_kernel_cuda",  &haganeOpsSqrt,    &cpu_dispatch_sqrt,    nullptr};
inline constexpr OpConfig kSinCfg     = {"sin",     nullptr, &haganeOpsSin,     &cpu_dispatch_sin,     nullptr};
inline constexpr OpConfig kCosCfg     = {"cos",     nullptr, &haganeOpsCos,     &cpu_dispatch_cos,     nullptr};
inline constexpr OpConfig kCeilCfg    = {"ceil",    nullptr, &haganeOpsCeil,    &cpu_dispatch_ceil,    nullptr};
inline constexpr OpConfig kFloorCfg   = {"floor",   nullptr, &haganeOpsFloor,   &cpu_dispatch_floor,   nullptr};
inline constexpr OpConfig kTanhCfg    = {"tanh",    "hagane_tanh_kernel_cuda",  &haganeOpsTanh,    &cpu_dispatch_tanh,    nullptr};
inline constexpr OpConfig kSigmoidCfg = {"sigmoid", nullptr, &haganeOpsSigmoid, &cpu_dispatch_sigmoid, nullptr};

inline constexpr OpConfig kReciprocalCfg = {"reciprocal", nullptr, &haganeOpsReciprocal, &cpu_dispatch_reciprocal, nullptr};
inline constexpr OpConfig kRsqrtCfg      = {"rsqrt",      nullptr, &haganeOpsRsqrt,      &cpu_dispatch_rsqrt,      nullptr};
inline constexpr OpConfig kRoundCfg      = {"round",      nullptr, &haganeOpsRound,      &cpu_dispatch_round,      nullptr};
inline constexpr OpConfig kTruncCfg      = {"trunc",      nullptr, &haganeOpsTrunc,      &cpu_dispatch_trunc,      nullptr};
inline constexpr OpConfig kErfCfg        = {"erf",        nullptr, &haganeOpsErf,        &cpu_dispatch_erf,        nullptr};
inline constexpr OpConfig kLog2Cfg       = {"log2",       "hagane_log2_kernel_cuda",  &haganeOpsLog2,       &cpu_dispatch_log2,       nullptr};
inline constexpr OpConfig kLog10Cfg      = {"log10",      "hagane_log10_kernel_cuda", &haganeOpsLog10,      &cpu_dispatch_log10,      nullptr};
inline constexpr OpConfig kLog1pCfg      = {"log1p",      nullptr, &haganeOpsLog1p,      &cpu_dispatch_log1p,      nullptr};
inline constexpr OpConfig kExp2Cfg       = {"exp2",       nullptr, &haganeOpsExp2,       &cpu_dispatch_exp2,       nullptr};

inline constexpr OpConfig kExpm1Cfg      = {"expm1",       nullptr, &haganeOpsExpm1,      &cpu_dispatch_expm1,       nullptr};
inline constexpr OpConfig kBitwiseNotCfg = {"bitwise_not", nullptr, &haganeOpsBitwiseNot, &cpu_dispatch_bitwise_not, nullptr};
inline constexpr OpConfig kLogicalNotCfg = {"logical_not", nullptr, &haganeOpsLogicalNot, &cpu_dispatch_logical_not, nullptr};

inline constexpr OpConfig kTanCfg    = {"tan",    nullptr, &haganeOpsTan,    &cpu_dispatch_tan,    nullptr};
inline constexpr OpConfig kAcosCfg   = {"acos",   nullptr, &haganeOpsAcos,   &cpu_dispatch_acos,   nullptr};
inline constexpr OpConfig kAsinCfg   = {"asin",   nullptr, &haganeOpsAsin,   &cpu_dispatch_asin,   nullptr};
inline constexpr OpConfig kAtanCfg   = {"atan",   nullptr, &haganeOpsAtan,   &cpu_dispatch_atan,   nullptr};
inline constexpr OpConfig kCoshCfg   = {"cosh",   nullptr, &haganeOpsCosh,   &cpu_dispatch_cosh,   nullptr};
inline constexpr OpConfig kSinhCfg   = {"sinh",   nullptr, &haganeOpsSinh,   &cpu_dispatch_sinh,   nullptr};
inline constexpr OpConfig kErfcCfg   = {"erfc",   nullptr, &haganeOpsErfc,   &cpu_dispatch_erfc,   nullptr};

// X+21 — lgamma re-migration after `haganeOpsLgamma` flush fix.
inline constexpr OpConfig kLgammaCfg = {"lgamma", nullptr, &haganeOpsLgamma, &cpu_dispatch_lgamma, nullptr};

// X+21 — frac brace-defect bug-fix-by-migration.
inline constexpr OpConfig kFracCfg   = {"frac",   nullptr, &haganeOpsFrac,   &cpu_dispatch_frac,   nullptr};

// X+72: silu + hardsigmoid moved onto the native transpiler-harvested metallib
// path (float-contiguous fast path in hagane_kernel_bridge; MLX c_abi_fn still
// covers half/bfloat/non-contiguous). Kernels harvested by the AST corpus
// engine from the GPU_LAMBDA activations; numerics verified vs CPU at atol=1e-5.
inline constexpr OpConfig kHardsigmoidCfg = {"hardsigmoid", "hagane_hardsigmoid_kernel", &haganeOpsHardsigmoid, &cpu_dispatch_hardsigmoid, nullptr};
inline constexpr OpConfig kMishCfg        = {"mish",        nullptr, &haganeOpsMish,        &cpu_dispatch_mish,        nullptr};
inline constexpr OpConfig kSiluCfg        = {"silu",        "hagane_silu_kernel", &haganeOpsSilu,        &cpu_dispatch_silu,        nullptr};

// X+24 Lane B — sinc / signbit via new pure-MLX runtime C-ABIs.
inline constexpr OpConfig kSincCfg    = {"sinc",    nullptr, &haganeOpsSinc,    &cpu_dispatch_sinc,    nullptr};
inline constexpr OpConfig kSignbitCfg = {"signbit", nullptr, &haganeOpsSignbit, &cpu_dispatch_signbit, nullptr};

// ---- Binary-op configuration ----------------------------------------------
// Parallel to OpConfig for `structured_binary_fn = void(*)(TensorIteratorBase&)`
// stubs with 3-tensor `int(in_a, in_b, out)` C-ABI.

struct BinaryOpConfig {
    const char* op_name;
    const char* metallib_kernel;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, const haganeOpsTensor_t*);
    void (*cpu_fallback)(TensorIteratorBase&);
    Tensor (*at_fp64_fn)(const Tensor&, const Tensor&);
};

inline constexpr BinaryOpConfig kEqCfg  = {"eq",  nullptr, &haganeOpsEq,  &cpu_dispatch_eq,  &fp64_eq };
inline constexpr BinaryOpConfig kNeCfg  = {"ne",  nullptr, &haganeOpsNe,  &cpu_dispatch_ne,  &fp64_ne };
inline constexpr BinaryOpConfig kLtCfg  = {"lt",  nullptr, &haganeOpsLt,  &cpu_dispatch_lt,  &fp64_lt };
inline constexpr BinaryOpConfig kGtCfg  = {"gt",  nullptr, &haganeOpsGt,  &cpu_dispatch_gt,  &fp64_gt };
inline constexpr BinaryOpConfig kLeCfg  = {"le",  nullptr, &haganeOpsLe,  &cpu_dispatch_le,  &fp64_le };
inline constexpr BinaryOpConfig kGeCfg  = {"ge",  nullptr, &haganeOpsGe,  &cpu_dispatch_ge,  &fp64_ge };
inline constexpr BinaryOpConfig kMulCfg = {"mul", "hagane_MulFunctor", &haganeOpsMul, &cpu_dispatch_mul, &fp64_mul};

inline constexpr BinaryOpConfig kPowTtCfg     = {"pow_tt",    nullptr, &haganeOpsPow,       &cpu_dispatch_pow_tt,    nullptr  };
inline constexpr BinaryOpConfig kAtan2Cfg     = {"atan2",     nullptr, &haganeOpsAtan2,     &cpu_dispatch_atan2,     nullptr  };
inline constexpr BinaryOpConfig kRemainderCfg = {"remainder", nullptr, &haganeOpsRemainder, &cpu_dispatch_remainder, nullptr  };
inline constexpr BinaryOpConfig kFmodCfg      = {"fmod",      nullptr, &haganeOpsFmod,      &cpu_dispatch_fmod,      nullptr  };
inline constexpr BinaryOpConfig kDivTrueCfg   = {"div_true",  "hagane_DivFunctor", &haganeOpsDiv,       &cpu_dispatch_div_true,  &fp64_div};
inline constexpr BinaryOpConfig kDivFloorCfg  = {"div_floor", nullptr, &haganeOpsDivFloor,  &cpu_dispatch_div_floor, nullptr  };
inline constexpr BinaryOpConfig kDivTruncCfg  = {"div_trunc", nullptr, &haganeOpsDivTrunc,  &cpu_dispatch_div_trunc, nullptr  };

// X+21 Lane D — 6 binary ops on `structured_binary_fn` (TensorIteratorBase&).
inline constexpr BinaryOpConfig kBitwiseAndCfg = {"bitwise_and", nullptr, &haganeOpsBitwiseAnd, &cpu_dispatch_bitwise_and, nullptr};
inline constexpr BinaryOpConfig kBitwiseOrCfg  = {"bitwise_or",  nullptr, &haganeOpsBitwiseOr,  &cpu_dispatch_bitwise_or,  nullptr};
inline constexpr BinaryOpConfig kBitwiseXorCfg = {"bitwise_xor", nullptr, &haganeOpsBitwiseXor, &cpu_dispatch_bitwise_xor, nullptr};
inline constexpr BinaryOpConfig kMaximumCfg    = {"maximum",     nullptr, &haganeOpsMaximum,    &cpu_dispatch_maximum,     nullptr};
inline constexpr BinaryOpConfig kMinimumCfg    = {"minimum",     nullptr, &haganeOpsMinimum,    &cpu_dispatch_minimum,     nullptr};
inline constexpr BinaryOpConfig kCopysignCfg   = {"copysign",    nullptr, &haganeOpsCopysign,   &cpu_dispatch_copysign,    nullptr};

// T2.1 — integer gcd/lcm run NATIVE via the transpiler-owned metallib
// (GcdLcmKernel.hip → hagane_{gcd,lcm}_kernel_cuda_<int-dtype>_a2_wptN_...).
// No MLX C-ABI (c_abi_fn=nullptr) and no fp64 (integral-only) → the bridge's
// nullptr-c_abi branch falls straight to cpu_dispatch_{gcd,lcm} when the
// metallib is unavailable / dtype/shape unsupported.
inline constexpr BinaryOpConfig kGcdCfg = {"gcd", "hagane_gcd_kernel_cuda", nullptr, &cpu_dispatch_gcd, nullptr};
inline constexpr BinaryOpConfig kLcmCfg = {"lcm", "hagane_lcm_kernel_cuda", nullptr, &cpu_dispatch_lcm, nullptr};

// X+23 Lane B — fmax/fmin via NaN-aware MLX composition C-ABIs added in
// Sprint X+23 (hagane/src/runtime/hagane_ops.cpp). Same BinaryOpConfig
// shape as maximum/minimum but with NaN propagation semantics.
inline constexpr BinaryOpConfig kFmaxCfg       = {"fmax",        nullptr, &haganeOpsFmax,       &cpu_dispatch_fmax,        nullptr};
inline constexpr BinaryOpConfig kFminCfg       = {"fmin",        nullptr, &haganeOpsFmin,       &cpu_dispatch_fmin,        nullptr};

// X+26 Lane B — hardsigmoid_backward via BinaryOpConfig (TensorIteratorBase&).
inline constexpr BinaryOpConfig kHardsigmoidBackwardCfg = {"hardsigmoid_backward", nullptr, &haganeOpsHardsigmoidBackward, &cpu_dispatch_hardsigmoid_backward, nullptr};

// X+28 Lane A — sigmoid/tanh backwards via BinaryOpConfig (structured_binary_fn
// = void(*)(TensorIteratorBase&)). TORCH_META_FUNC at BinaryOps.cpp:310/314
// builds iter as (out, grad_output, output) — bridge passes a=grad_output,
// b=output directly to the C-ABI's (grad_out, output, grad_in) signature.
inline constexpr BinaryOpConfig kSigmoidBackwardCfg = {"sigmoid_backward", nullptr, &haganeOpsSigmoidBackward, &cpu_dispatch_sigmoid_backward, nullptr};
inline constexpr BinaryOpConfig kTanhBackwardCfg    = {"tanh_backward",    nullptr, &haganeOpsTanhBackward,    &cpu_dispatch_tanh_backward,    nullptr};


// ---- Binary-iter-op configuration (X+21) ----------------------------------
// Parallel to BinaryOpConfig for `binary_fn = void(*)(TensorIterator&)`
// stubs (logical_*, max/min_elementwise). TensorIterator&-typed signature
// because PyTorch declares these stubs with the derived type. Helpers all
// take TensorIteratorBase& (upcast inside the bridge body).

struct BinaryIterOpConfig {
    const char* op_name;
    const char* metallib_kernel;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, const haganeOpsTensor_t*);
    void (*cpu_fallback)(TensorIterator&);
    Tensor (*at_fp64_fn)(const Tensor&, const Tensor&);
};

inline constexpr BinaryIterOpConfig kLogicalAndCfg = {"logical_and", nullptr, &haganeOpsLogicalAnd, &cpu_dispatch_logical_and, nullptr};
inline constexpr BinaryIterOpConfig kLogicalOrCfg  = {"logical_or",  nullptr, &haganeOpsLogicalOr,  &cpu_dispatch_logical_or,  nullptr};
inline constexpr BinaryIterOpConfig kLogicalXorCfg = {"logical_xor", nullptr, &haganeOpsLogicalXor, &cpu_dispatch_logical_xor, nullptr};

// X+26 Lane B — hardswish_backward / mish_backward via BinaryIterOpConfig.
inline constexpr BinaryIterOpConfig kHardswishBackwardCfg = {"hardswish_backward", nullptr, &haganeOpsHardswishBackward, &cpu_dispatch_hardswish_backward, nullptr};
inline constexpr BinaryIterOpConfig kMishBackwardCfg      = {"mish_backward",      nullptr, &haganeOpsMishBackward,      &cpu_dispatch_mish_backward,      nullptr};


// ---- Unary-iter-op configuration (X+22) -----------------------------------
// Parallel to OpConfig for `reduce_fn` / `hardswish_fn` stubs — both alias
// to `void(*)(TensorIterator&)`. C-ABI is identical to OpConfig
// (`int(in, out)`); only the cpu_fallback signature changes
// (TensorIterator& vs TensorIteratorBase&).

struct UnaryIterOpConfig {
    const char* op_name;
    const char* metallib_kernel;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*);
    void (*cpu_fallback)(TensorIterator&);
    Tensor (*at_fp64_fn)(const Tensor&);
};

// A9: kSumCfg/kMeanCfg carry a non-null metallib_kernel sentinel ("reduce") so
// the bridge attempts try_launch_reduction_metallib (the owned Tier-a host
// 2-pass) before the MLX c_abi_fn. The sentinel only gates the attempt; the
// actual kernel names are picked by dtype inside the helper. All other reduce
// configs keep nullptr → MLX (unchanged).
inline constexpr UnaryIterOpConfig kSumCfg       = {"sum",        "reduce", &haganeOpsSum,       &cpu_dispatch_sum,        nullptr};
inline constexpr UnaryIterOpConfig kMeanCfg      = {"mean",       "reduce", &haganeOpsMean,      &cpu_dispatch_mean,       nullptr};
inline constexpr UnaryIterOpConfig kProdCfg      = {"prod",       nullptr, &haganeOpsProd,      &cpu_dispatch_prod,       nullptr};
inline constexpr UnaryIterOpConfig kArgmaxCfg    = {"argmax",     nullptr, &haganeOpsArgmax,    &cpu_dispatch_argmax,     nullptr};
inline constexpr UnaryIterOpConfig kArgminCfg    = {"argmin",     nullptr, &haganeOpsArgmin,    &cpu_dispatch_argmin,     nullptr};
inline constexpr UnaryIterOpConfig kMaxValuesCfg = {"max_values", nullptr, &haganeOpsMaxValues, &cpu_dispatch_max_values, nullptr};
inline constexpr UnaryIterOpConfig kMinValuesCfg = {"min_values", nullptr, &haganeOpsMinValues, &cpu_dispatch_min_values, nullptr};
// and_stub is the reduction kernel behind `torch.all` (logical AND across
// elements), or_stub backs `torch.any` (logical OR). The C-ABI names match
// the semantic, not the stub name.
inline constexpr UnaryIterOpConfig kAndCfg       = {"and",        nullptr, &haganeOpsAll,       &cpu_dispatch_and,        nullptr};
inline constexpr UnaryIterOpConfig kOrCfg        = {"or",         nullptr, &haganeOpsAny,       &cpu_dispatch_or,         nullptr};
inline constexpr UnaryIterOpConfig kHardswishCfg = {"hardswish",  nullptr, &haganeOpsHardswish, &cpu_dispatch_hardswish,  nullptr};

// ---- Reduce-with-flag op configuration (X+23) -----------------------------
// `reduce_fn_flag = void(*)(TensorIterator&, const Scalar&)` stubs
// (norm, powsum). The bridge derives the reduction dim from input/output
// shape comparison (mirror of the retired hagane_norm_kernel body) and
// passes (p, dim) through to the C-ABI's existing `(double p, int32_t dim)`
// signature. For full-tensor reductions (numel(out)==1), the C-ABI's single
// `dim` parameter can't express multi-axis; route via `at_full_fn` instead.

inline Tensor at_full_norm   (const Tensor& t, const Scalar& p) { return at::norm(t, p); }
inline Tensor at_full_powsum (const Tensor& t, const Scalar& p) {
    return at::sum(at::pow(at::abs(t), p.toDouble()));
}

struct ReduceFlagOpConfig {
    const char* op_name;
    const char* metallib_kernel;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, double, int32_t);
    void (*cpu_fallback)(TensorIterator&, const Scalar& p);
    Tensor (*at_fp64_fn)(const Tensor&, const Scalar& p);
    Tensor (*at_full_fn)(const Tensor&, const Scalar& p);
};

inline constexpr ReduceFlagOpConfig kNormCfg   = {"norm",   nullptr, &haganeOpsNormVal, &cpu_dispatch_norm,   nullptr, &at_full_norm  };
inline constexpr ReduceFlagOpConfig kPowsumCfg = {"powsum", nullptr, &haganeOpsPowsum,  &cpu_dispatch_powsum, nullptr, &at_full_powsum};

// ---- Cumulative-op configuration (X+24) -----------------------------------
// cumsum/cumprod use structured_cum_fn (const Tensor&, const Tensor&, int64_t);
// logcumsumexp uses cum_fn (Tensor&, const Tensor&, int64_t). All three
// C-ABIs share (in, out, int32_t dim); the bridge normalizes negative dim
// before the cast (haganeOpsCumsum/Cumprod/Logcumsumexp don't normalize
// internally — they pass directly to mx::cumsum/cumprod which require
// positive axes).

struct CumOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int32_t dim);
    // Exactly one of the two cpu fallbacks is non-null per row, matching the
    // upstream stub signature. The dispatch templates select by which is set.
    void (*cpu_structured_fn)(const Tensor& self, const Tensor& result, int64_t dim);
    void (*cpu_mutable_fn)   (Tensor& result, const Tensor& self, int64_t dim);
};

inline constexpr CumOpConfig kCumsumCfg       = {"cumsum",       &haganeOpsCumsum,       &cpu_dispatch_cumsum,     nullptr};
inline constexpr CumOpConfig kCumprodCfg      = {"cumprod",      &haganeOpsCumprod,      &cpu_dispatch_cumprod,    nullptr};
inline constexpr CumOpConfig kLogcumsumexpCfg = {"logcumsumexp", &haganeOpsLogcumsumexp, nullptr,                  &cpu_dispatch_logcumsumexp};

// ---- Reduce-std/var op configuration (X+24) -------------------------------
// std_var_stub takes (TensorIterator&, double correction, bool take_sqrt).
// haganeOpsVar already exists at hagane_ops.cpp:971 with full-axis branch
// (out_n == 1 ⇒ mx::var(x, /*keepdims=*/false)). Wrap shim casts bool→int.

inline int std_var_wrap(const haganeOpsTensor_t* in, const haganeOpsTensor_t* out,
                        double correction, bool take_sqrt) {
    return haganeOpsVar(in, out, correction, take_sqrt ? 1 : 0);
}

struct ReduceStdVarOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    double correction, bool take_sqrt);
    void (*cpu_fallback)(TensorIterator&, double, bool);
};

inline constexpr ReduceStdVarOpConfig kStdVarCfg = {"std_var", &std_var_wrap, &cpu_dispatch_std_var};

// ---- Binary-alpha-op configuration ----------------------------------------

struct BinaryAlphaOpConfig {
    const char* op_name;
    const char* metallib_kernel;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    const haganeOpsTensor_t*, float alpha);
    void (*cpu_fallback)(TensorIteratorBase&, const Scalar& alpha);
    Tensor (*at_fp64_fn)(const Tensor&, const Tensor&, const Scalar& alpha);
};

inline constexpr BinaryAlphaOpConfig kAddCfg = {"add", nullptr, &haganeOpsAdd, &cpu_fallback_add, &fp64_add_alpha};
inline constexpr BinaryAlphaOpConfig kSubCfg = {"sub", nullptr, &haganeOpsSub, &cpu_fallback_sub, &fp64_sub_alpha};

// X+26 Lane B — leaky_relu_backward via BinaryAlphaOpConfig + arg-swap shim.
// TORCH_META_FUNC builds the iter with pos 1 = self_or_result, pos 2 = grad_out
// (Activation.cpp:190) — leaky_relu_backward_wrap inverts to match C-ABI's
// (grad_out, input, grad_in, negval) signature.
inline constexpr BinaryAlphaOpConfig kLeakyReluBackwardCfg = {"leaky_relu_backward", nullptr, &leaky_relu_backward_wrap, &cpu_dispatch_leaky_relu_backward, nullptr};

// X+28 Lane A — logit_backward via BinaryAlphaOpConfig (binary_fn_alpha =
// void(*)(TensorIteratorBase&, const Scalar&)). TORCH_META_FUNC at
// BinaryOps.cpp:306 builds iter as (out, grad_output, input); eps threaded
// through alpha. C-ABI signature is (grad_out, input, grad_in, eps) — direct
// match, no swap shim.
inline constexpr BinaryAlphaOpConfig kLogitBackwardCfg = {"logit_backward", nullptr, &haganeOpsLogitBackward, &cpu_dispatch_logit_backward, nullptr};


// ---- Unary-scalar-op configuration ----------------------------------------

struct UnaryScalarOpConfig {
    const char* op_name;
    const char* metallib_kernel;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, float scalar);
    void (*cpu_fallback)(TensorIteratorBase&, const Scalar& scalar);
    Tensor (*at_fp64_fn)(const Tensor&, const Scalar& scalar);
};

inline constexpr UnaryScalarOpConfig kPowTsCfg      = {"pow_ts",     nullptr, &pow_scalar_wrap,     &cpu_dispatch_pow_ts,     nullptr};
inline constexpr UnaryScalarOpConfig kLeakyReluCfg  = {"leaky_relu", "hagane_leaky_relu_kernel", &haganeOpsLeakyRelu,  &cpu_dispatch_leaky_relu, nullptr};
inline constexpr UnaryScalarOpConfig kHardshrinkCfg = {"hardshrink", nullptr, &haganeOpsHardshrink, &cpu_dispatch_hardshrink, nullptr};
inline constexpr UnaryScalarOpConfig kSoftshrinkCfg = {"softshrink", nullptr, &haganeOpsSoftshrink, &cpu_dispatch_softshrink, nullptr};
inline constexpr UnaryScalarOpConfig kLogitCfg      = {"logit",      nullptr, &haganeOpsLogit,      &cpu_dispatch_logit,      nullptr};

// Drains both the MLX lazy graph and the Hagane Metal command queue. (M3c moved
// the post-launch drain to the MLX read boundary — see haganeOpsMarkMetallibWrite
// — so this is no longer called per metallib launch; kept for the contract.)
extern "C" hipError_t hipDeviceSynchronize();

// X+74 — a raw metallib launch (hagane_launch_kernel_mixed) has no capture
// hook, so it is invisible to the hagane tape and would be dropped on replay.
// While a tape is recording (haganeOpsTapeRecording), the launch helpers fall
// back to the MLX c_abi path, which IS recorded (HAGANE_CAPTURE_OP) and replays
// correctly. Distinct from haganeOpsCaptureActive() (hipStreamBeginCapture).
// Declared here (hagane_capture.h is not included by this header).
extern "C" int haganeOpsTapeRecording(void);


// ---- Native metallib dispatch (X+73 M3b) -----------------------------------
// The corpus engine names harvested kernels
// `hagane_<base>_<dtypeTag><shape>_unrolled_contig`. <Config>.metallib_kernel
// holds the <base> (e.g. "hagane_silu_kernel"); the bridge computes the
// per-dtype name at dispatch, so one metallib (every dtype arm of the op,
// loaded at once via haganeRegisterMetallibAll) serves float/half/bfloat16/…
// The <shape> is implicit in the bridge arity ("" unary, "_a2" binary,
// "_a1_s1" captured-scalar) — matching the transpiler's disambiguation.

// Torch dtype → transpiler dtype tag (see ast_transpiler msl_type_for). nullptr
// for dtypes with no Metal kernel (float64/complex) → handled host-side.
inline const char* metallib_dtype_tag(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Float:    return "float";
        case c10::ScalarType::Half:     return "half";
        case c10::ScalarType::BFloat16: return "bfloat16";
        case c10::ScalarType::Byte:     return "uint8";
        case c10::ScalarType::Char:     return "int8";
        case c10::ScalarType::Short:    return "int16";
        case c10::ScalarType::Int:      return "int32";
        case c10::ScalarType::Long:     return "int64";
        case c10::ScalarType::Bool:     return "bool";
        default:                        return nullptr;
    }
}

// A8: work-per-thread vectorization factor. MUST equal the transpiler's
// emit_functor_specialization (wpt = 8/sizeof(elem)): bf16/half->4, float/int32
// ->2, int8/bool->8, int64->1. Encoded in the kernel name (_wptN) AND used to
// size the launch grid (ceil(N/wpt) threads), so the kernel and its launch
// agree by construction; any drift -> kernel-not-found -> MLX c_abi fallback
// (correct-but-slow, never wrong).
inline int metallib_work_per_thread(c10::ScalarType st) {
    int bytes = static_cast<int>(c10::elementSize(st));
    return (bytes > 0 && bytes <= 8) ? 8 / bytes : 1;
}

inline std::string metallib_kernel_name(const char* base, c10::ScalarType st,
                                        const char* shape) {
    const char* tag = metallib_dtype_tag(st);
    if (!tag) return {};
    return std::string(base) + "_" + tag + shape
         + "_wpt" + std::to_string(metallib_work_per_thread(st))
         + "_unrolled_contig";
}

// ---- Lazy metallib registration --------------------------------------------
// Per-(config) state. `auto&` so one template serves OpConfig / BinaryOpConfig
// / UnaryScalarOpConfig — all expose op_name + metallib_kernel. Each Cfg gets
// its own MetallibState<Cfg> (once_flag + availability); only instantiated for
// Cfgs whose metallib_kernel != nullptr (the bridge `if constexpr` gates it).

template <auto& Cfg>
struct MetallibState {
    static inline bool available = false;
    static inline std::once_flag once;
};

template <auto& Cfg>
inline void maybe_register_metallib() {
    // A4 (route default-on): the native metallib route is ON by default;
    // HAGANE_USE_METALLIB_ROUTE=0 is the escape back to the bit-identical
    // MLX-only path (the working spine). Any other value (or unset) routes.
    const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
    if (route && route[0] == '0') {
        std::fprintf(stderr,
                     "[hagane-path-alpha] %s_stub owned by bridge "
                     "(metallib=off)\n",
                     Cfg.op_name);
        return;
    }
    // Load EVERY dtype arm in one library open (fatbin equivalent). The bridge
    // computes the exact per-dtype name at dispatch; an absent arm fails the
    // launch cleanly and falls back to the MLX c_abi path.
    std::string path = metallib_dir() + "/" + Cfg.op_name + ".metallib";
    hipError_t rc = haganeRegisterMetallibAll(path.c_str());
    if (rc != hipSuccess) {
        std::fprintf(stderr,
                     "[hagane-path-alpha] failed to register %s (%d); "
                     "falling back to direct path\n",
                     path.c_str(), static_cast<int>(rc));
        return;
    }
    MetallibState<Cfg>::available = true;
    std::fprintf(stderr,
                 "[hagane-path-alpha] %s metallib registered (all dtype arms); "
                 "%s_stub owned by bridge (metallib=on)\n",
                 Cfg.op_name, Cfg.op_name);
}

// Log the first native dispatch of each distinct kernel name (one line per
// dtype arm). Confirms from the bridge log which dtypes actually route native
// vs fall back to MLX — e.g. silu_kernel_half_/bfloat16_ appearing here proves
// half/bf16 run on the harvested kernel, not the MLX c_abi path.
inline void note_native_launch(const std::string& kname) {
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> g(mu);
    if (seen.insert(kname).second)
        std::fprintf(stderr, "[hagane-path-alpha] native dispatch: %s\n", kname.c_str());
}

// X+77: settle a metallib kernel's MLX-produced input. Under the unified queue
// (HAGANE_METALLIB_UNIFIED_QUEUE) this event-orders the input GPU-side — commit
// the producer async (no host wait) + blit it into the torch block on the same
// queue — so a native→MLX→native chain pays no blocking host eval. Flag-off
// keeps the byte-identical X+75 region flush (drain_submitted + mx::eval).
inline void flush_or_commit_metallib_input(void* d_in, int64_t nbytes) {
    if (haganeMetallibUnifiedQueueEnabled())
        ::haganeOpsCommitInputProducer(d_in, nbytes);
    else
        ::haganeOpsFlushRegion(d_in, nbytes);
}

// ---- Launch helpers (one per arity) ----------------------------------------
// Each flushes pending MLX writes to the input buffer(s), dispatches the named
// kernel, then drains the Hagane queue so the shared MTLBuffer is coherent
// before the cross-queue MLX read (X+72 fix). Return false on launch failure
// so the caller falls back to the MLX c_abi path.
//
// M3b: the per-launch drain costs the X+71 async win for routed ops; M3c
// replaces it with a deferred drain fired only at the actual MLX/host read
// boundary, so native→native chains run drain-free.

inline bool try_launch_unary_metallib(const std::string& kname,
                                      TensorIteratorBase& iter) {
    if (haganeOpsTapeRecording()) return false;  // record via MLX so replay is correct
    void* d_out = iter.data_ptr(0);
    void* d_in  = iter.data_ptr(1);
    int N = static_cast<int>(iter.numel());
    if (N <= 0) return true;
    // X+75: minimal per-input region flush (NOT blanket haganeOpsFlush, which
    // bumps generation_ and purges wrap_deps_ — corrupting the decode KV-cache
    // lazy chain). Only the input bytes must be materialized; mirrors
    // hagane_copy_kernel's haganeOpsFlushRegion(src, nbytes) (HaganeOps.cpp).
    // X+77: under the unified queue this event-orders the input GPU-side instead.
    flush_or_commit_metallib_input(d_in, static_cast<int64_t>(N) * iter.element_size(1));
    // A8: each thread handles wpt elements (matches the kernel's _wptN); the
    // grid covers ceil(N/wpt) threads.
    const int wpt = metallib_work_per_thread(iter.dtype());
    const int nthreads = (N + wpt - 1) / wpt;
    dim3 block(256, 1, 1), grid((nthreads + 255) / 256, 1, 1);
    void*  args[]      = {d_in, d_out, &N};
    int    arg_types[] = {0, 0, 1};
    size_t arg_sizes[] = {0, 0, sizeof(int)};
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                   args, arg_types, arg_sizes, 3) != hipSuccess)
        return false;
    note_native_launch(kname);
    // M3c / X+76: mark the output dirty-on-Hagane-queue instead of draining now.
    // The drain fires lazily at a read boundary (haganeOpsMarkMetallibWrite). On
    // the unified queue (HAGANE_METALLIB_UNIFIED_QUEUE) the MLX-input boundary
    // skips that drain — downstream MLX readers commit onto Hagane's queue and
    // order against this write via the shared MTLSharedEvent chain — while the
    // host-readback / FREE boundaries still drain precisely (correctness). All of
    // that queue policy lives in the runtime now, so this path is op-agnostic and
    // flag-off stays bit-identical to X+75.
    haganeOpsMarkMetallibWrite(d_out, static_cast<size_t>(N) * iter.element_size(0));
    return true;
}

// ---- A6 (Phase 2): native norm route (rms_norm + layer_norm) ----------------
// Unlike the elementwise ops (DispatchStub + OpConfig + bridge), the fused
// rms_norm / layer_norm entries (_fused_rms_norm_cuda, layer_norm_cuda) are
// direct C10_EXPORTs, so the shared norm.metallib is registered here via a
// standalone thread-safe once-init rather than MetallibState<Cfg>. It bundles
// all 12 transpiler-owned kernels ({float,half,bfloat16} × {ln,rms} ×
// {RowwiseMoments,LayerNormForward}); haganeRegisterMetallibAll is idempotent so
// one registration serves both norms. Two kernels per dtype run the per-row
// geometry grid=(M) block=256 (one threadgroup per row): the real PyTorch
// RowwiseMoments (Welford → mean,rstd) then LayerNormForward (apply). Route-off
// (HAGANE_USE_METALLIB_ROUTE=0) skips registration → unavailable → MLX fallback
// (the bit-identical spine).
inline bool norm_metallib_available() {
    static const bool available = [] {
        const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
        if (route && route[0] == '0') return false;
        std::string path = metallib_dir() + "/norm.metallib";
        if (haganeRegisterMetallibAll(path.c_str()) != hipSuccess) {
            std::fprintf(stderr,
                "[hagane-path-alpha] failed to register %s; norm stays on MLX\n",
                path.c_str());
            return false;
        }
        std::fprintf(stderr,
            "[hagane-path-alpha] norm metallib registered "
            "(rms+layer_norm moments+apply, all dtype arms)\n");
        return true;
    }();
    return available;
}

// Per-thread persistent scratch for the moments outputs, reused across calls so
// the hot decode path never frees a buffer with an in-flight metallib write —
// which would trip the unconditional FREE-boundary drain and serialize the
// deferred queue (a ~15% decode regression vs reusing). Grows monotonically;
// leaked (never destructed) to dodge static-teardown ordering vs the allocator.
// Safe to reuse on the in-order Hagane queue: call N's apply consumes these
// before call N+1's moments overwrites them.
inline void* rms_moments_scratch(const at::TensorOptions& opts, int64_t M, int slot) {
    static thread_local at::Tensor* scratch[2] = {nullptr, nullptr};
    at::Tensor*& t = scratch[slot];
    if (t == nullptr) t = new at::Tensor();
    if (!t->defined() || t->numel() < M || t->device() != opts.device())
        *t = at::empty({M}, opts.dtype(at::kFloat));
    return t->data_ptr();
}

// A10: the FAST norm path — PyTorch's own FUSED vectorized_layer_norm_kernel
// (single moments+apply pass, aligned_vector<T,4> vec loads, dynamic threadgroup
// scratch). Eligible only for N % 4 == 0 + 16-byte-aligned X/gamma/beta/Y, dtype
// {float,bfloat16} (no half fused arm). Geometry mirrors
// launch_vectorized_layer_norm_kernel: block(warp_size, num_threads/warp_size),
// grid(M), shared = threads.y>1 ? threads.y*3/2*sizeof(float) : 0. Returns false →
// the caller falls back to the byte-unchanged scalar 2-kernel path. The kernel
// name carries the variant, so any codegen/launch drift → name-not-registered →
// hagane_launch_kernel_mixed fails → false → MLX (never silent-wrong).
inline bool fused_norm_eligible(const char* tag, int64_t N,
                                void* dX, void* dgamma, void* dbeta, void* dY) {
    if (std::strcmp(tag, "half") == 0) return false;   // no fused half arm
    if (N % 4 != 0) return false;
    auto a16 = [](void* p) {
        return (reinterpret_cast<size_t>(p) % 16) == 0;  // arm64: size_t==ptr width
    };
    return a16(dX) && a16(dgamma) && a16(dY) && (dbeta == nullptr || a16(dbeta));
}

inline bool launch_fused_norm(const char* tag, bool rms, void* dX, void* dgamma,
                              void* dbeta, void* dmean, void* drstd, void* dY,
                              int64_t M, int64_t N, double eps) {
    const std::string k = std::string("hagane_vectorized_layer_norm_kernel_") +
                          tag + (rms ? "_rms" : "_ln");
    const int warp_size = 32, num_threads = 256;
    const int threads_y = num_threads / warp_size;  // 8 (matches the launcher)
    const unsigned shared = threads_y > 1
        ? static_cast<unsigned>(threads_y * 3 / 2 * sizeof(float)) : 0u;
    dim3 grid(static_cast<unsigned>(M), 1, 1), block(warp_size, threads_y, 1);
    int   c_N   = static_cast<int>(N);
    float c_eps = static_cast<float>(eps);
    void*  args[] = {&c_N, &c_eps, dX, dgamma, dbeta, dmean, drstd, dY};
    int    at[]   = {1, 1, 0, 0, 0, 0, 0, 0};
    size_t as[]   = {sizeof(int), sizeof(float), 0, 0, 0, 0, 0, 0};
    if (hagane_launch_kernel_mixed_tracked(k.c_str(), grid, block, shared, nullptr,
                                   args, at, as, 8) != hipSuccess)
        return false;
    note_native_launch(k);
    return true;
}

// Run rms_norm natively (moments→apply). On success writes `output` and, when
// `need_rstd` (backward needed), hands back the per-row rstd (== rrms =
// rsqrt(E[x^2]+eps)) in `rstd_out` so the caller returns it without recomputing
// (consistent with the exact forward normalization); that fresh tensor is held
// by autograd, so it isn't freed during the forward either. Returns false →
// caller falls back to the MLX path: route-off, an unsupported/non-float dtype,
// tape recording (so torch.compile capture/replay stays on MLX), or any launch
// failure.
inline bool try_launch_rms_norm_metallib(const at::Tensor& input_c,
                                         const std::optional<at::Tensor>& weight,
                                         at::Tensor& output,
                                         int64_t M, int64_t N, double eps,
                                         bool need_rstd, at::Tensor& rstd_out) {
    if (haganeOpsTapeRecording()) return false;
    if (M <= 0 || N <= 0) return false;
    if (!norm_metallib_available()) return false;
    const char* tag = nullptr;
    switch (input_c.scalar_type()) {
        case at::kFloat:    tag = "float";    break;
        case at::kHalf:     tag = "half";     break;
        case at::kBFloat16: tag = "bfloat16"; break;
        default: return false;
    }

    // T_ACC = float for every arm. mean is never read out (rms uses only rstd),
    // so it's always reusable scratch; rstd is a fresh result only when backward
    // needs it (returned), else reusable scratch too.
    const auto opts = input_c.options();
    void* dmean = rms_moments_scratch(opts, M, 0);
    at::Tensor rstd_fresh;
    void* drstd;
    if (need_rstd) {
        rstd_fresh = at::empty({M}, opts.dtype(at::kFloat));
        drstd = rstd_fresh.data_ptr();
    } else {
        drstd = rms_moments_scratch(opts, M, 1);
    }

    // gamma must be a valid (non-null) buffer — the mixed launcher rejects null
    // buffer args. With a weight, gamma = weight; without, a ones-vector
    // reproduces the kernel's `gamma == nullptr ? 1 : gamma[j]` identity branch.
    // beta is never read in the rms arm (if constexpr DCE) but is still a bound
    // param, so reuse gamma as a harmless non-null dummy.
    at::Tensor gamma = (weight.has_value() && weight->defined())
        ? weight->contiguous()
        : at::ones({N}, input_c.options());

    void* dX     = input_c.data_ptr();
    void* dgamma = gamma.data_ptr();
    void* dY     = output.data_ptr();
    const int64_t elt = input_c.element_size();

    flush_or_commit_metallib_input(dX, M * N * elt);
    flush_or_commit_metallib_input(dgamma, N * gamma.element_size());

    // A10: the fused vec4 kernel (one pass) when aligned + N%4==0. beta is a bound
    // param the rms arm never dereferences (if constexpr DCE), so the gamma dummy
    // is safe — identical to the scalar path's beta=gamma.
    if (fused_norm_eligible(tag, N, dX, dgamma, dgamma, dY) &&
        launch_fused_norm(tag, /*rms=*/true, dX, dgamma, dgamma, dmean, drstd, dY,
                          M, N, eps)) {
        haganeOpsMarkMetallibWrite(dY, static_cast<size_t>(M) * N * elt);
        if (need_rstd) rstd_out = rstd_fresh;
        return true;
    }

    std::string moments = std::string("hagane_RowwiseMomentsCUDAKernel_") + tag + "_rms";
    std::string apply   = std::string("hagane_LayerNormForwardCUDAKernel_") + tag + "_rms";

    dim3 grid(static_cast<unsigned>(M), 1, 1), block(256, 1, 1);
    int64_t c_N = N;
    float c_eps = static_cast<float>(eps);

    void*  margs[] = {&c_N, &c_eps, dX, dmean, drstd};
    int    mat[]   = {1, 1, 0, 0, 0};
    size_t mas[]   = {sizeof(int64_t), sizeof(float), 0, 0, 0};
    if (hagane_launch_kernel_mixed_tracked(moments.c_str(), grid, block, 0, nullptr,
                                   margs, mat, mas, 5) != hipSuccess)
        return false;

    // gamma at [4], beta=gamma (dummy) at [5]; the two kernels run in submission
    // order on the in-order Hagane queue, so apply reads the rstd moments wrote.
    void*  aargs[] = {&c_N, dX, dmean, drstd, dgamma, dgamma, dY};
    int    aat[]   = {1, 0, 0, 0, 0, 0, 0};
    size_t aas[]   = {sizeof(int64_t), 0, 0, 0, 0, 0, 0};
    if (hagane_launch_kernel_mixed_tracked(apply.c_str(), grid, block, 0, nullptr,
                                   aargs, aat, aas, 7) != hipSuccess)
        return false;

    note_native_launch(moments);
    note_native_launch(apply);
    haganeOpsMarkMetallibWrite(dY, static_cast<size_t>(M) * N * elt);
    if (need_rstd) rstd_out = rstd_fresh;
    return true;
}

// Run layer_norm natively (moments→apply), mirroring the rms path but with the
// `_ln` kernel arms: both mean AND rstd are live (RowwiseMoments writes both;
// LayerNormForward applies Y = (X-mean)*rstd*gamma + beta). When `need_stats`
// (backward needed) the caller gets fresh mean+rstd back (held by autograd, so
// never freed during the forward) feeding layer_norm_backward without an ATen
// recompute; otherwise both are reusable per-thread scratch. The mixed launcher
// rejects null buffers, so no-weight binds a ones-vector gamma and no-bias binds
// a zeros-vector beta (reproducing the kernel's nullptr-guard identities).
// Returns false → MLX fallback: route-off, unsupported/non-float dtype, tape
// recording, or any launch failure.
inline bool try_launch_layer_norm_metallib(const at::Tensor& input_c,
                                           const std::optional<at::Tensor>& weight,
                                           const std::optional<at::Tensor>& bias,
                                           at::Tensor& output,
                                           int64_t M, int64_t N, double eps,
                                           bool need_stats,
                                           at::Tensor& mean_out, at::Tensor& rstd_out) {
    if (haganeOpsTapeRecording()) return false;
    if (M <= 0 || N <= 0) return false;
    if (!norm_metallib_available()) return false;
    const char* tag = nullptr;
    switch (input_c.scalar_type()) {
        case at::kFloat:    tag = "float";    break;
        case at::kHalf:     tag = "half";     break;
        case at::kBFloat16: tag = "bfloat16"; break;
        default: return false;
    }

    // T_ACC = float for every arm. mean and rstd are both live (apply reads
    // them). When backward needs the stats they're fresh tensors (returned);
    // otherwise reusable per-thread scratch.
    const auto opts = input_c.options();
    at::Tensor mean_fresh, rstd_fresh;
    void* dmean;
    void* drstd;
    if (need_stats) {
        mean_fresh = at::empty({M}, opts.dtype(at::kFloat));
        rstd_fresh = at::empty({M}, opts.dtype(at::kFloat));
        dmean = mean_fresh.data_ptr();
        drstd = rstd_fresh.data_ptr();
    } else {
        dmean = rms_moments_scratch(opts, M, 0);
        drstd = rms_moments_scratch(opts, M, 1);
    }

    at::Tensor gamma = (weight.has_value() && weight->defined())
        ? weight->contiguous()
        : at::ones({N}, input_c.options());
    at::Tensor beta = (bias.has_value() && bias->defined())
        ? bias->contiguous()
        : at::zeros({N}, input_c.options());

    void* dX     = input_c.data_ptr();
    void* dgamma = gamma.data_ptr();
    void* dbeta  = beta.data_ptr();
    void* dY     = output.data_ptr();
    const int64_t elt = input_c.element_size();

    flush_or_commit_metallib_input(dX, M * N * elt);
    flush_or_commit_metallib_input(dgamma, N * gamma.element_size());
    flush_or_commit_metallib_input(dbeta, N * beta.element_size());

    // A10: the fused vec4 kernel (one pass) when aligned + N%4==0.
    if (fused_norm_eligible(tag, N, dX, dgamma, dbeta, dY) &&
        launch_fused_norm(tag, /*rms=*/false, dX, dgamma, dbeta, dmean, drstd, dY,
                          M, N, eps)) {
        haganeOpsMarkMetallibWrite(dY, static_cast<size_t>(M) * N * elt);
        if (need_stats) { mean_out = mean_fresh; rstd_out = rstd_fresh; }
        return true;
    }

    std::string moments = std::string("hagane_RowwiseMomentsCUDAKernel_") + tag + "_ln";
    std::string apply   = std::string("hagane_LayerNormForwardCUDAKernel_") + tag + "_ln";

    dim3 grid(static_cast<unsigned>(M), 1, 1), block(256, 1, 1);
    int64_t c_N = N;
    float c_eps = static_cast<float>(eps);

    void*  margs[] = {&c_N, &c_eps, dX, dmean, drstd};
    int    mat[]   = {1, 1, 0, 0, 0};
    size_t mas[]   = {sizeof(int64_t), sizeof(float), 0, 0, 0};
    if (hagane_launch_kernel_mixed_tracked(moments.c_str(), grid, block, 0, nullptr,
                                   margs, mat, mas, 5) != hipSuccess)
        return false;

    void*  aargs[] = {&c_N, dX, dmean, drstd, dgamma, dbeta, dY};
    int    aat[]   = {1, 0, 0, 0, 0, 0, 0};
    size_t aas[]   = {sizeof(int64_t), 0, 0, 0, 0, 0, 0};
    if (hagane_launch_kernel_mixed_tracked(apply.c_str(), grid, block, 0, nullptr,
                                   aargs, aat, aas, 7) != hipSuccess)
        return false;

    note_native_launch(moments);
    note_native_launch(apply);
    haganeOpsMarkMetallibWrite(dY, static_cast<size_t>(M) * N * elt);
    if (need_stats) { mean_out = mean_fresh; rstd_out = rstd_fresh; }
    return true;
}

// ---- A9 (Phase 2): native reduction route (sum / mean, full contiguous) ------
// PyTorch's real reduce_kernel can't be transpiled (its ReduceOp carries an
// un-nameable GPU_LAMBDA ops type — see hagane/kernels/reduce.hip), so we own a
// generic Tier-a block reducer instead. reduce.metallib bundles 3 concrete
// kernels: f32→f32, bf16→f32 (pass 1), f32→bf16 (pass 2). Route-off
// (HAGANE_USE_METALLIB_ROUTE=0) skips registration → MLX (the bit-identical spine).
inline bool reduction_metallib_available() {
    static const bool available = [] {
        const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
        if (route && route[0] == '0') return false;
        std::string path = metallib_dir() + "/reduce.metallib";
        if (haganeRegisterMetallibAll(path.c_str()) != hipSuccess) {
            std::fprintf(stderr,
                "[hagane-path-alpha] failed to register %s; reductions stay on MLX\n",
                path.c_str());
            return false;
        }
        std::fprintf(stderr,
            "[hagane-path-alpha] reduce metallib registered (sum/mean Tier-a, f32+bf16)\n");
        return true;
    }();
    return available;
}

// Per-thread persistent float partials buffer for the reduction 2-pass — reused
// across calls (mirrors rms_moments_scratch: never freed mid-flight, grows
// monotonically, safe on the in-order queue since pass 2 consumes pass 1's
// partials before the next call overwrites them).
inline void* reduce_partials_scratch(const at::TensorOptions& opts, int64_t nblocks) {
    static thread_local at::Tensor* t = nullptr;
    if (t == nullptr) t = new at::Tensor();
    if (!t->defined() || t->numel() < nblocks || t->device() != opts.device())
        *t = at::empty({nblocks}, opts.dtype(at::kFloat));
    return t->data_ptr();
}

// Full contiguous sum/mean as a host 2-pass: pass 1 reduces the input → per-block
// float partials (grid=(nblocks) block=256, grid-strided); pass 2 reduces the
// partials → the single output (grid=(1) block=256). `scale` (1 for sum, 1/N for
// mean) is applied once at the pass-2 store. Returns false → MLX fallback:
// route-off, tape recording, non-contiguous, unsupported dtype, partial-dim
// reduction (out.numel()!=1, deferred to a Tier-b follow-on), or launch failure.
inline bool try_launch_reduction_metallib(TensorIterator& iter, bool is_mean) {
    if (haganeOpsTapeRecording()) return false;
    if (!reduction_metallib_available()) return false;
    const at::Tensor& in_t  = iter.tensor(1);
    const at::Tensor& out_t = iter.tensor(0);
    if (out_t.numel() != 1) return false;          // full reduction only (v1)
    if (!in_t.is_contiguous()) return false;
    const int64_t N = in_t.numel();
    if (N <= 0) return false;

    const char *k1, *k2;
    const auto inty = in_t.scalar_type(), outty = out_t.scalar_type();
    if (inty == at::kFloat && outty == at::kFloat) {
        k1 = "hagane_reduce_f32_f32"; k2 = "hagane_reduce_f32_f32";
    } else if (inty == at::kBFloat16 && outty == at::kBFloat16) {
        k1 = "hagane_reduce_bf16_f32"; k2 = "hagane_reduce_f32_bf16";
    } else {
        return false;  // half / mixed-acc dtypes stay on MLX
    }

    const int block = 256;
    int64_t nbtmp = (N + block - 1) / block;     // ~one block per 256 elements,
    if (nbtmp < 1) nbtmp = 1;                     // clamped to [1, 256] so pass 2's
    if (nbtmp > 256) nbtmp = 256;                 // single block reduces the partials
    const int nblocks = static_cast<int>(nbtmp);
    void* dpart = reduce_partials_scratch(in_t.options(), nblocks);
    void* dX    = in_t.data_ptr();
    void* dout  = out_t.data_ptr();
    flush_or_commit_metallib_input(dX, N * in_t.element_size());

    int64_t c_N = N, c_NB = nblocks;
    float s1 = 1.0f, s2 = is_mean ? static_cast<float>(1.0 / static_cast<double>(N)) : 1.0f;
    const int    at_[] = {1, 0, 0, 1};
    const size_t as_[] = {sizeof(int64_t), 0, 0, sizeof(float)};
    dim3 b(block, 1, 1), g1(static_cast<unsigned>(nblocks), 1, 1), g2(1, 1, 1);

    void* a1[] = {&c_N, dX, dpart, &s1};   // pass 1: X(N) → partials(nblocks), scale 1
    if (hagane_launch_kernel_mixed_tracked(k1, g1, b, 0, nullptr, a1, at_, as_, 4) != hipSuccess)
        return false;
    void* a2[] = {&c_NB, dpart, dout, &s2}; // pass 2: partials(nblocks) → out(1), scale
    if (hagane_launch_kernel_mixed_tracked(k2, g2, b, 0, nullptr, a2, at_, as_, 4) != hipSuccess)
        return false;

    note_native_launch(k1);
    note_native_launch(k2);
    haganeOpsMarkMetallibWrite(dout, static_cast<size_t>(out_t.element_size()));
    return true;
}

// A6 softmax: register the warp-per-row softmax metallib (PyTorch's REAL
// softmax_warp_forward, owned via explicit-instantiation injection — float +
// bfloat16, log2_elements 0..11, is_log {false,true}). Separate from norm.metallib;
// haganeRegisterMetallibAll is idempotent. Route-off skips registration → MLX.
inline bool softmax_metallib_available() {
    static const bool available = [] {
        const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
        if (route && route[0] == '0') return false;
        std::string path = metallib_dir() + "/softmax.metallib";
        if (haganeRegisterMetallibAll(path.c_str()) != hipSuccess) {
            std::fprintf(stderr,
                "[hagane-path-alpha] failed to register %s; softmax stays on MLX\n",
                path.c_str());
            return false;
        }
        std::fprintf(stderr,
            "[hagane-path-alpha] softmax metallib registered "
            "(warp-per-row, float+bfloat16, log2 0..11, fwd+log)\n");
        return true;
    }();
    return available;
}

// Persistent per-thread dummy mask buffer. is_masked=false → the kernel never
// dereferences mask, but it is still bound as buffer(5); sized to M*N bytes so a
// Metal bounds validation of the bound buffer (and the kernel's `mask += offset`
// pointer arithmetic) stays in range. Reused; leaked to dodge static-teardown.
inline void* softmax_mask_scratch(const at::TensorOptions& opts, int64_t bytes) {
    static thread_local at::Tensor* scratch = nullptr;
    if (scratch == nullptr) scratch = new at::Tensor();
    if (!scratch->defined() || scratch->numel() < bytes ||
        scratch->device() != opts.device())
        *scratch = at::zeros({bytes}, opts.dtype(at::kBool));
    return scratch->data_ptr();
}

// Run softmax / log_softmax natively via the warp-per-row kernel. Writes
// `output` and returns true; false → caller falls back to MLX. Covers only what
// the warp kernel owns: reduction over the LAST (innermost) dim, N ≤ 2048,
// contiguous, output dtype == input dtype (so half_to_float stays on MLX),
// dtype ∈ {float, bfloat16} (fp16 deferred). Backward stays on MLX (reads this
// dtype-preserved output). Geometry is N-derived, identical to
// dispatch_softmax_forward (PersistentSoftmax.cuh:311). is_masked=false.
inline bool try_launch_softmax_metallib(const at::Tensor& input_c,
                                        const at::Tensor& output,
                                        int64_t dim, bool is_log) {
    if (haganeOpsTapeRecording()) return false;
    if (!softmax_metallib_available()) return false;
    const int64_t ndim = input_c.dim();
    if (ndim == 0) return false;
    const int64_t d = dim < 0 ? dim + ndim : dim;
    if (d != ndim - 1) return false;                       // last-dim only
    if (output.scalar_type() != input_c.scalar_type()) return false;  // no half_to_float
    if (!input_c.is_contiguous() || !output.is_contiguous()) return false;
    const char* tag = nullptr;
    switch (input_c.scalar_type()) {
        case at::kFloat:    tag = "float";    break;
        case at::kBFloat16: tag = "bfloat16"; break;
        default: return false;                              // fp16 deferred
    }
    const int64_t N = input_c.size(-1);
    if (N <= 0 || N > 2048) return false;                   // warp kernel coverage
    const int64_t M = input_c.numel() / N;
    if (M <= 0) return false;

    // Geometry — identical to dispatch_softmax_forward.
    int log2e = 0; while ((1 << log2e) < static_cast<int>(N)) ++log2e;
    const int npot = 1 << log2e;
    const int warp_size = (npot < 32) ? npot : 32;
    const int batches_per_warp = (npot <= 128) ? 2 : 1;
    const int warps_per_block = 128 / warp_size;
    const int batches_per_block = warps_per_block * batches_per_warp;
    const unsigned blocks =
        static_cast<unsigned>((M + batches_per_block - 1) / batches_per_block);

    std::string kernel = std::string("hagane_softmax_warp_forward_") + tag +
                         "_l2e" + std::to_string(log2e) + "_ws32" +
                         (is_log ? "_log" : "");

    void* dst = output.data_ptr();
    void* src = input_c.data_ptr();
    const int64_t elt = input_c.element_size();
    void* dmask = softmax_mask_scratch(input_c.options(), M * N);

    flush_or_commit_metallib_input(src, M * N * elt);

    int batch_size = static_cast<int>(M), stride = static_cast<int>(N),
        element_count = static_cast<int>(N), head_chunk = -1;
    bool is_tmask = false;
    void*  args[] = {dst, src, &batch_size, &stride, &element_count,
                     dmask, &head_chunk, &is_tmask};
    int    at_[]  = {0, 0, 1, 1, 1, 0, 1, 1};
    size_t as_[]  = {0, 0, sizeof(int), sizeof(int), sizeof(int),
                     0, sizeof(int), sizeof(bool)};
    dim3 grid(blocks, 1, 1), block(warp_size, warps_per_block, 1);
    if (hagane_launch_kernel_mixed_tracked(kernel.c_str(), grid, block, 0, nullptr,
                                   args, at_, as_, 8) != hipSuccess)
        return false;
    note_native_launch(kernel);
    haganeOpsMarkMetallibWrite(dst, static_cast<size_t>(M) * N * elt);
    return true;
}

inline bool try_launch_binary_metallib(const std::string& kname,
                                       TensorIteratorBase& iter) {
    if (haganeOpsTapeRecording()) return false;  // record via MLX so replay is correct
    void* d_out = iter.data_ptr(0);
    void* d_a   = iter.data_ptr(1);
    void* d_b   = iter.data_ptr(2);
    int N = static_cast<int>(iter.numel());
    if (N <= 0) return true;
    // X+75: minimal per-input region flush for both operands (see unary above).
    // X+77: event-order both inputs GPU-side under the unified queue.
    flush_or_commit_metallib_input(d_a, static_cast<int64_t>(N) * iter.element_size(1));
    flush_or_commit_metallib_input(d_b, static_cast<int64_t>(N) * iter.element_size(2));
    // A8: wpt elements per thread (matches the kernel's _wptN).
    const int wpt = metallib_work_per_thread(iter.dtype());
    const int nthreads = (N + wpt - 1) / wpt;
    dim3 block(256, 1, 1), grid((nthreads + 255) / 256, 1, 1);
    void*  args[]      = {d_a, d_b, d_out, &N};
    int    arg_types[] = {0, 0, 0, 1};
    size_t arg_sizes[] = {0, 0, 0, sizeof(int)};
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                   args, arg_types, arg_sizes, 4) != hipSuccess)
        return false;
    note_native_launch(kname);
    // M3c: mark the output dirty-on-Hagane-queue instead of draining now; the
    // drain fires lazily at the MLX/host read boundary, so native chains don't
    // pay the per-op sync (the X+71 async win is preserved for routed ops).
    haganeOpsMarkMetallibWrite(d_out, static_cast<size_t>(N) * iter.element_size(0));
    return true;
}

// ---- #1010: MLX's elementwise kernels, writing into OUR output --------------
//
// Real ROCm hands rocBLAS the caller's pointer and the library writes into it.
// We were letting MLX allocate, which leaves the bytes with two homes and makes
// every subsequent free reconcile them — measured at ~173 us/op against 2.8 us
// for the identical MLX kernel when the output is already allocated. So the
// kernel stays MLX's; only the destination comes back to us.
//
// Routed as a FAMILY. A boundary between an op that owns its output and one
// that does not costs ~180 us/op (a 16-op chain measures 6.4 us/op all-native,
// 176.5 all-MLX, 273.1 alternating), so covering half of a chain is worse than
// covering none of it. Every op below that MLX's corpus carries is routed
// together; an op whose name does not map keeps its existing path untouched.

// torch's stub name -> MLX's operator name as it appears in the kernel symbol.
// nullptr means "not ours" — never a guess. MLX's corpus is checked at dispatch
// (kernel-not-found declines), so a wrong name here cannot produce a wrong
// answer, only a fallback.
inline const char* mlx_binary_op_name(const char* op) {
    struct Row { const char* torch_name; const char* mlx_name; };
    static constexpr Row kMap[] = {
        {"add",         "Add"},
        {"sub",         "Subtract"},
        {"mul",         "Multiply"},
        {"div_true",    "Divide"},
        {"maximum",     "Maximum"},
        {"minimum",     "Minimum"},
        {"pow_tt",      "Power"},
        {"remainder",   "Remainder"},
        {"eq",          "Equal"},
        {"ne",          "NotEqual"},
        {"lt",          "Less"},
        {"le",          "LessEqual"},
        {"gt",          "Greater"},
        {"ge",          "GreaterEqual"},
        {"bitwise_and", "BitwiseAnd"},
        {"bitwise_or",  "BitwiseOr"},
        {"bitwise_xor", "BitwiseXor"},
        {"logical_and", "LogicalAnd"},
        {"logical_or",  "LogicalOr"},
    };
    for (const auto& r : kMap)
        if (std::strcmp(op, r.torch_name) == 0) return r.mlx_name;
    return nullptr;
}

// torch dtype -> HAGANE_DTYPE_*. -1 for anything with no Metal representation
// (float64, complex) — declined, never narrowed behind the caller's back.
inline int hagane_vendor_dtype(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Float:    return HAGANE_DTYPE_FLOAT32;
        case c10::ScalarType::Half:     return HAGANE_DTYPE_FLOAT16;
        case c10::ScalarType::BFloat16: return HAGANE_DTYPE_BFLOAT16;
        case c10::ScalarType::Char:     return HAGANE_DTYPE_INT8;
        case c10::ScalarType::Short:    return HAGANE_DTYPE_INT16;
        case c10::ScalarType::Int:      return HAGANE_DTYPE_INT32;
        case c10::ScalarType::Long:     return HAGANE_DTYPE_INT64;
        case c10::ScalarType::Byte:     return HAGANE_DTYPE_UINT8;
        case c10::ScalarType::Bool:     return HAGANE_DTYPE_BOOL;
        default:                        return -1;
    }
}

// Raw bytes of a CPU-scalar operand, in the COMPUTE dtype (which is what the
// kernel's `device const T*` expects). Returns false for a dtype we do not
// spell, so the caller declines rather than passing garbage.
inline bool vendor_scalar_bytes(TensorIteratorBase& iter, int arg,
                                c10::ScalarType st, uint64_t* out) {
    *out = 0;
    switch (st) {
        case c10::ScalarType::Float:
            { float v = iter.scalar_value<float>(arg);   std::memcpy(out, &v, 4); return true; }
        case c10::ScalarType::Half:
            { auto v = iter.scalar_value<c10::Half>(arg);     std::memcpy(out, &v, 2); return true; }
        case c10::ScalarType::BFloat16:
            { auto v = iter.scalar_value<c10::BFloat16>(arg); std::memcpy(out, &v, 2); return true; }
        case c10::ScalarType::Char:
            { int8_t v = iter.scalar_value<int8_t>(arg);   std::memcpy(out, &v, 1); return true; }
        case c10::ScalarType::Short:
            { int16_t v = iter.scalar_value<int16_t>(arg); std::memcpy(out, &v, 2); return true; }
        case c10::ScalarType::Int:
            { int32_t v = iter.scalar_value<int32_t>(arg); std::memcpy(out, &v, 4); return true; }
        case c10::ScalarType::Long:
            { int64_t v = iter.scalar_value<int64_t>(arg); std::memcpy(out, &v, 8); return true; }
        case c10::ScalarType::Byte:
            { uint8_t v = iter.scalar_value<uint8_t>(arg); std::memcpy(out, &v, 1); return true; }
        case c10::ScalarType::Bool:
            { bool v = iter.scalar_value<bool>(arg);       std::memcpy(out, &v, 1); return true; }
        default: return false;
    }
}

// Dispatch one MLX elementwise kernel into iter's output. Returns false having
// encoded NOTHING when anything is outside what we can prove correct.
// ON by default -- writing into the caller's output IS the intended behaviour.
// HAGANE_VENDOR_ELEMENTWISE=0 is the escape back to the MLX-allocates path, and
// is what the route-off gate uses to prove nothing else changed. (#1011 tracks
// removing this knob once the whole family is settled.)
inline bool vendor_elementwise_route_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("HAGANE_VENDOR_ELEMENTWISE");
        return !(s && s[0] == '0');
    }();
    return on;
}

inline bool try_vendor_binary(const char* torch_op, TensorIteratorBase& iter) {
    if (!vendor_elementwise_route_enabled()) return false;
    const char* mlx_op = mlx_binary_op_name(torch_op);
    if (!mlx_op) return false;
    if (haganeOpsTapeRecording()) return false;  // capture records via MLX
    if (iter.ninputs() != 2) return false;
    if (!haganeOpsVendorElementwiseAvailable()) return false;

    // The vv_/sv_/vs_ kernels are flat and unbroadcast: one linear index into
    // each operand. So every operand must be contiguous and the same shape as
    // the output, and the compute dtype must be the storage dtype.
    if (!iter.is_contiguous()) return false;
    const auto compute = iter.common_dtype();
    const int dt = hagane_vendor_dtype(compute);
    if (dt < 0) return false;

    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to compute
    if (N > static_cast<int64_t>(UINT32_MAX)) return false;

    const void* ptrs[2] = {nullptr, nullptr};
    bool is_scalar[2] = {false, false};
    uint64_t scalars[2] = {0, 0};
    for (int i = 0; i < 2; i++) {
        const int arg = i + 1;
        if (iter.is_cpu_scalar(arg)) {
            if (!vendor_scalar_bytes(iter, arg, compute, &scalars[i])) return false;
            is_scalar[i] = true;
            continue;
        }
        // A device operand must already be in the compute dtype and full-sized:
        // a promoted or broadcast operand would need MLX's g*_ strided kernels,
        // which this path does not select.
        if (iter.tensor(arg).scalar_type() != compute) return false;
        if (iter.tensor(arg).sizes() != iter.tensor(0).sizes()) return false;
        ptrs[i] = iter.data_ptr(arg);
    }
    if (is_scalar[0] && is_scalar[1]) return false;
    // A comparison writes bool while computing in the input dtype; the kernel
    // handles that, but the output BUFFER must then be bool-sized. Only accept
    // out dtype == compute dtype, or bool out for a comparison.
    const auto out_st = iter.dtype(0);
    if (out_st != compute && out_st != c10::ScalarType::Bool) return false;
    if (out_st == c10::ScalarType::Bool && compute != c10::ScalarType::Bool) {
        // MLX names these by the INPUT dtype and writes bool — supported, but
        // only for the ops that actually produce bool.
        static constexpr const char* kBoolOut[] = {
            "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual"};
        bool ok = false;
        for (const char* b : kBoolOut) if (std::strcmp(mlx_op, b) == 0) ok = true;
        if (!ok) return false;
    }

    // The output span must come from the OUTPUT dtype: a comparison computes in
    // the input dtype and writes bool, and sizing the span from the input would
    // supersede/mark bytes past the end of the output — corrupting whatever
    // tensor's stash sits after it.
    const int dt_out = hagane_vendor_dtype(out_st);
    if (dt_out < 0) return false;
    if (haganeOpsVendorBinary(mlx_op, dt, dt_out, ptrs[0], ptrs[1], iter.data_ptr(0), N,
                              is_scalar[0] ? 1 : 0, is_scalar[1] ? 1 : 0,
                              scalars[0], scalars[1]) != HAGANE_OPS_SUCCESS)
        return false;
    note_native_launch(std::string("mlx:") + mlx_op);

    return true;
}

inline bool try_launch_unary_scalar_metallib(const std::string& kname,
                                             TensorIteratorBase& iter, float scalar) {
    if (haganeOpsTapeRecording()) return false;  // record via MLX so replay is correct
    void* d_out = iter.data_ptr(0);
    void* d_in  = iter.data_ptr(1);
    int N = static_cast<int>(iter.numel());
    if (N <= 0) return true;
    // X+75: minimal per-input region flush (see unary above).
    // X+77: event-order the input GPU-side under the unified queue.
    flush_or_commit_metallib_input(d_in, static_cast<int64_t>(N) * iter.element_size(1));
    float sc = scalar;
    // A8: wpt elements per thread (matches the kernel's _wptN).
    const int wpt = metallib_work_per_thread(iter.dtype());
    const int nthreads = (N + wpt - 1) / wpt;
    dim3 block(256, 1, 1), grid((nthreads + 255) / 256, 1, 1);
    void*  args[]      = {d_in, d_out, &sc, &N};
    int    arg_types[] = {0, 0, 1, 1};
    size_t arg_sizes[] = {0, 0, sizeof(float), sizeof(int)};
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                   args, arg_types, arg_sizes, 4) != hipSuccess)
        return false;
    note_native_launch(kname);
    // M3c: mark the output dirty-on-Hagane-queue instead of draining now; the
    // drain fires lazily at the MLX/host read boundary, so native chains don't
    // pay the per-op sync (the X+71 async win is preserved for routed ops).
    haganeOpsMarkMetallibWrite(d_out, static_cast<size_t>(N) * iter.element_size(0));
    return true;
}

// ---- Templated dispatch bridges -------------------------------------------
// One instantiation per config row. `if constexpr` branches eliminate dead
// paths per-instantiation. Each instantiation generates a unique symbol so
// REGISTER_DISPATCH in the .cpp can reference `&hagane_kernel_bridge<kXCfg>`.

template <const OpConfig& Cfg>
inline void hagane_kernel_bridge(TensorIteratorBase& iter) {
    if constexpr (Cfg.metallib_kernel != nullptr) {
        std::call_once(MetallibState<Cfg>::once, maybe_register_metallib<Cfg>);
    }

    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto in = iter.tensor(1).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(in).to(c10::ScalarType::Double));
            return;
        }
    }

    if constexpr (Cfg.c_abi_fn == nullptr) {
        iter.tensor(0).copy_(at::sign(iter.tensor(1)));
    } else {
        if constexpr (Cfg.metallib_kernel != nullptr) {
            if (MetallibState<Cfg>::available && iter.is_contiguous()) {
                std::string kname =
                    metallib_kernel_name(Cfg.metallib_kernel, iter.dtype(), "");
                if (!kname.empty() && try_launch_unary_metallib(kname, iter)) return;
            }
        }

        auto out = make_ops_tensor_local(iter, 0);
        auto in  = make_ops_tensor_local(iter, 1);
        if (Cfg.c_abi_fn(&in, &out) != HAGANE_OPS_SUCCESS) {
            ::haganeOpsFlush();
            Cfg.cpu_fallback(iter);
        }
    }
}

template <const BinaryOpConfig& Cfg>
inline void hagane_binary_bridge(TensorIteratorBase& iter) {
    if constexpr (Cfg.metallib_kernel != nullptr) {
        std::call_once(MetallibState<Cfg>::once, maybe_register_metallib<Cfg>);
    }

    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto a = iter.tensor(1).to(c10::ScalarType::Float);
            auto b = iter.tensor(2).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(a, b));
            return;
        }
    }

    // Native tensor-tensor (a2) path: both inputs must be real tensors of the
    // output dtype with identical shape (the unrolled_contig kernel is pure
    // element-wise — no broadcast, no scalar operand, no mixed dtype). Anything
    // else falls through to the MLX c_abi path below.
    if constexpr (Cfg.metallib_kernel != nullptr) {
        if (MetallibState<Cfg>::available && iter.ninputs() == 2
            && iter.is_contiguous()
            && iter.tensor(1).scalar_type() == iter.dtype()
            && iter.tensor(2).scalar_type() == iter.dtype()
            && iter.tensor(1).sizes() == iter.tensor(0).sizes()
            && iter.tensor(2).sizes() == iter.tensor(0).sizes()) {
            std::string kname =
                metallib_kernel_name(Cfg.metallib_kernel, iter.dtype(), "_a2");
            if (!kname.empty() && try_launch_binary_metallib(kname, iter)) return;
        }
    }

    // #1010: MLX's kernel, our output block. Tried after our own metallib (that
    // one is already ours end-to-end) and before the MLX C-ABI (which is the
    // path that makes MLX allocate).
    if (try_vendor_binary(Cfg.op_name, iter)) return;

    if constexpr (Cfg.c_abi_fn == nullptr) {
        // No MLX path (e.g. integer gcd/lcm): the metallib is the only device
        // route; fall to the CPU stub on fallthrough (route-off/unsupported).
        Cfg.cpu_fallback(iter);
    } else {
        auto out = make_ops_tensor_local(iter, 0);
        at::Tensor sa, sb;
        auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
        auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
        if (Cfg.c_abi_fn(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
            ::haganeOpsFlush();
            Cfg.cpu_fallback(iter);
        }
    }
}

// X+21 — TensorIterator&-typed variant for `binary_fn` stubs. Same body as
// hagane_binary_bridge with the signature swapped so REGISTER_DISPATCH for
// logical_*/max_elementwise/min_elementwise (`binary_fn`) accepts the
// function pointer.
template <const BinaryIterOpConfig& Cfg>
inline void hagane_binary_iter_bridge(TensorIterator& iter) {
    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto a = iter.tensor(1).to(c10::ScalarType::Float);
            auto b = iter.tensor(2).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(a, b));
            return;
        }
    }

    auto out = make_ops_tensor_local(iter, 0);
    at::Tensor sa, sb;
    auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
    auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
    if (Cfg.c_abi_fn(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter);
    }
}

// X+22 — TensorIterator&-typed variant for `reduce_fn` / `hardswish_fn` stubs.
// Same body as hagane_kernel_bridge minus the metallib fast-path (none of
// the 10 X+22 candidates have metallibs today; deferred to X+23+ if needed).
template <const UnaryIterOpConfig& Cfg>
inline void hagane_unary_iter_bridge(TensorIterator& iter) {
    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto in = iter.tensor(1).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(in).to(c10::ScalarType::Double));
            return;
        }
    }

    // A9: full contiguous sum/mean route through the owned Tier-a reducer
    // (metallib_kernel != nullptr enables it). Any miss (partial-dim, non-
    // contiguous, unsupported dtype, route-off) falls through to the MLX C-ABI —
    // the bit-identical spine.
    if (Cfg.metallib_kernel != nullptr &&
        try_launch_reduction_metallib(iter, std::string(Cfg.op_name) == "mean"))
        return;

    auto out = make_ops_tensor_local(iter, 0);
    auto in  = make_ops_tensor_local(iter, 1);
    if (Cfg.c_abi_fn(&in, &out) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter);
    }
}

// X+23 — reduce_fn_flag variant. norm_stub / powsum_stub take
// (TensorIterator&, const Scalar&). Bridge derives the reduction dim by
// shape comparison (mirrors the retired hagane_norm_kernel at
// HaganeOps.cpp:2147-2154) and threads (p.toDouble(), dim) to the C-ABI.
// For full-tensor reductions (out_t.dim() == 0), the C-ABI takes only one
// reduction axis — delegate to CPU stub instead.
template <const ReduceFlagOpConfig& Cfg>
inline void hagane_unary_iter_flag_bridge(TensorIterator& iter, const Scalar& p) {
    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto in = iter.tensor(1).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(in, p).to(c10::ScalarType::Double));
            return;
        }
    }

    const at::Tensor& in_t  = iter.tensor(1);
    const at::Tensor& out_t = iter.tensor(0);

    // Full-tensor reductions: out has numel==1 over a multi-axis input. The
    // single-`dim` C-ABI can't express multi-axis reduction; compute via
    // at_full_fn (at::norm / at::sum(at::pow(at::abs(...))) etc.) on a
    // CPU clone — calling at::* on the GPU tensor recurses through the
    // same bridge.
    if constexpr (Cfg.at_full_fn != nullptr) {
        if (out_t.numel() == 1 && in_t.dim() > 1) {
            iter.tensor(0).copy_(Cfg.at_full_fn(in_t.cpu(), p));
            return;
        }
    }

    auto out = make_ops_tensor_local(iter, 0);
    auto in  = make_ops_tensor_local(iter, 1);

    int32_t dim = -1;
    for (int64_t d = 0; d < in_t.dim(); d++) {
        if (in_t.size(d) > 1 && (d >= out_t.dim() || out_t.size(d) == 1)) {
            dim = static_cast<int32_t>(d);
            break;
        }
    }
    if (dim < 0) dim = static_cast<int32_t>(in_t.dim() - 1);

    if (Cfg.c_abi_fn(&in, &out, p.toDouble(), dim) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter, p);
    }
}

template <const BinaryAlphaOpConfig& Cfg>
inline void hagane_binary_alpha_bridge(TensorIteratorBase& iter, const Scalar& alpha) {
    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto a = iter.tensor(1).to(c10::ScalarType::Float);
            auto b = iter.tensor(2).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(a, b, alpha));
            return;
        }
    }

    // #1010: alpha != 1 means out = a + alpha*b, which is a DIFFERENT kernel
    // than MLX's Add. Route only the alpha == 1 case; anything else keeps the
    // MLX path. Dropping alpha here would be exactly the silent-wrong class the
    // project forbids.
    if (alpha.toDouble() == 1.0 && try_vendor_binary(Cfg.op_name, iter)) return;

    auto out = make_ops_tensor_local(iter, 0);
    at::Tensor sa, sb;
    auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
    auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
    if (Cfg.c_abi_fn(&a, &b, &out, alpha.toFloat()) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter, alpha);
    }
}

template <const UnaryScalarOpConfig& Cfg>
inline void hagane_unary_scalar_bridge(TensorIteratorBase& iter, const Scalar& scalar) {
    if constexpr (Cfg.metallib_kernel != nullptr) {
        std::call_once(MetallibState<Cfg>::once, maybe_register_metallib<Cfg>);
    }

    if constexpr (Cfg.at_fp64_fn != nullptr) {
        if (iter.common_dtype() == c10::ScalarType::Double) {
            auto in = iter.tensor(1).to(c10::ScalarType::Float);
            iter.tensor(0).copy_(Cfg.at_fp64_fn(in, scalar).to(c10::ScalarType::Double));
            return;
        }
    }

    // Native captured-scalar (a1_s1) path: the kernel takes the input tensor,
    // output, the opmath scalar (always float) and N.
    if constexpr (Cfg.metallib_kernel != nullptr) {
        if (MetallibState<Cfg>::available && iter.is_contiguous()) {
            std::string kname =
                metallib_kernel_name(Cfg.metallib_kernel, iter.dtype(), "_a1_s1");
            if (!kname.empty()
                && try_launch_unary_scalar_metallib(kname, iter, scalar.toFloat()))
                return;
        }
    }

    auto out = make_ops_tensor_local(iter, 0);
    auto in  = make_ops_tensor_local(iter, 1);
    if (Cfg.c_abi_fn(&in, &out, scalar.toFloat()) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter, scalar);
    }
}

// ---- Cumulative-op bridges (X+24) -----------------------------------------
// Distinct from the iterator-typed bridges above — cumsum/cumprod/logcumsumexp
// are dispatched against the raw Tensor (no TensorIterator). Two templates
// carry the const-ref vs mutable-ref signature asymmetry.

inline haganeOpsTensor_t make_ops_tensor_from_tensor(const Tensor& t) {
    haganeOpsTensor_t desc;
    desc.data    = const_cast<void*>(t.data_ptr());
    desc.shape   = t.sizes().data();
    desc.strides = t.strides().data();
    desc.ndim    = static_cast<int32_t>(t.dim());
    desc.dtype   = to_hagane_dtype_local(t.scalar_type());
    return desc;
}

template <const CumOpConfig& Cfg>
inline void hagane_cum_bridge_structured(const Tensor& self, const Tensor& result, int64_t dim) {
    haganeOpsTensor_t in  = make_ops_tensor_from_tensor(self);
    haganeOpsTensor_t out = make_ops_tensor_from_tensor(result);
    int32_t dim32 = static_cast<int32_t>(dim < 0 ? dim + self.dim() : dim);
    if (Cfg.c_abi_fn(&in, &out, dim32) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_structured_fn(self, result, dim);
    }
}

template <const CumOpConfig& Cfg>
inline void hagane_cum_bridge_mutable(Tensor& result, const Tensor& self, int64_t dim) {
    haganeOpsTensor_t in  = make_ops_tensor_from_tensor(self);
    haganeOpsTensor_t out = make_ops_tensor_from_tensor(result);
    int32_t dim32 = static_cast<int32_t>(dim < 0 ? dim + self.dim() : dim);
    if (Cfg.c_abi_fn(&in, &out, dim32) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_mutable_fn(result, self, dim);
    }
}

// ---- Reduce-std/var bridge (X+24) -----------------------------------------
// std_var_stub iter has noutputs() == 1 (var alone) or 2 (var_mean). When 2,
// output 0 is var, output 1 is mean, input is at index nout. The C-ABI only
// produces var; for nout == 2 we additionally call haganeOpsMean to populate
// output 1, matching the retired hagane_std_var_kernel:2542.

template <const ReduceStdVarOpConfig& Cfg>
inline void hagane_unary_iter_stdvar_bridge(TensorIterator& iter,
                                             double correction, bool take_sqrt) {
    int nout = iter.noutputs();
    auto var_out = make_ops_tensor_local(iter, 0);
    auto in      = make_ops_tensor_local(iter, nout);
    if (Cfg.c_abi_fn(&in, &var_out, correction, take_sqrt) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter, correction, take_sqrt);
        return;
    }
    if (nout == 2) {
        auto mean_out = make_ops_tensor_local(iter, 1);
        if (haganeOpsMean(&in, &mean_out) != HAGANE_OPS_SUCCESS) {
            ::haganeOpsFlush();
            Cfg.cpu_fallback(iter, correction, take_sqrt);
        }
    }
}

// ---- Unary-optional-triple bridge (X+25) ----------------------------------
// nan_to_num_stub signature is
// `void(*)(TensorIteratorBase&, optional<double>×3)`. The C-ABI takes plain
// `double`s; .value_or() chooses defaults matching the retired
// hagane_nan_to_num_kernel (HaganeOps.cpp:2675 — 0 / DBL_MAX / DBL_LOWEST).

struct UnaryOptionalTripleOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    double, double, double);
    void (*cpu_fallback)(TensorIteratorBase&,
                         std::optional<double>, std::optional<double>, std::optional<double>);
};

inline constexpr UnaryOptionalTripleOpConfig kNanToNumCfg = {
    "nan_to_num", &haganeOpsNanToNum, &cpu_dispatch_nan_to_num};

template <const UnaryOptionalTripleOpConfig& Cfg>
inline void hagane_unary_optional_triple_bridge(TensorIteratorBase& iter,
                                                 std::optional<double> nan_val,
                                                 std::optional<double> pos_inf_val,
                                                 std::optional<double> neg_inf_val) {
    auto out = make_ops_tensor_local(iter, 0);
    auto in  = make_ops_tensor_local(iter, 1);
    double n = nan_val.value_or(0.0);
    double p = pos_inf_val.value_or(std::numeric_limits<double>::max());
    double m = neg_inf_val.value_or(std::numeric_limits<double>::lowest());
    if (Cfg.c_abi_fn(&in, &out, n, p, m) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter, nan_val, pos_inf_val, neg_inf_val);
    }
}

// ---- X+28 Lane B variants — backward stubs with extra scalars/bool --------
//
// elu_backward_stub:      void(*)(TensorIteratorBase&, Scalar, Scalar, Scalar, bool)
// softplus_backward_stub: void(*)(TensorIteratorBase&, Scalar, Scalar)
// hardtanh_backward_stub: void(*)(TensorIterator&, Scalar, Scalar)
//
// All 3 use the standard binary-iter layout (out=pos 0, grad_output=pos 1,
// self/self_or_result=pos 2), so the bridges pass a=grad_output, b=self
// directly to the C-ABI's (grad_out, self, grad_in, ...) signature.

struct EluBackwardOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    const haganeOpsTensor_t*, float, float, float, bool);
    void (*cpu_fallback)(TensorIteratorBase&,
                         const Scalar&, const Scalar&, const Scalar&, bool);
};

inline constexpr EluBackwardOpConfig kEluBackwardCfg = {
    "elu_backward", &haganeOpsEluBackward, &cpu_dispatch_elu_backward};

template <const EluBackwardOpConfig& Cfg>
inline void hagane_elu_backward_bridge(TensorIteratorBase& iter,
                                        const Scalar& alpha, const Scalar& scale,
                                        const Scalar& input_scale, bool is_result) {
    auto out = make_ops_tensor_local(iter, 0);
    at::Tensor sa, sb;
    auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
    auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
    if (Cfg.c_abi_fn(&a, &b, &out,
                     alpha.toFloat(), scale.toFloat(), input_scale.toFloat(),
                     is_result) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(iter, alpha, scale, input_scale, is_result);
    }
}

struct BinaryDoubleScalarOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    const haganeOpsTensor_t*, float, float);
    // cpu_fallback's iter type varies per stub (Base& for softplus, derived
    // for hardtanh); store as TensorIteratorBase& and the bridge upcasts.
    void (*cpu_fallback_base)(TensorIteratorBase&, const Scalar&, const Scalar&);
    void (*cpu_fallback_iter)(TensorIterator&,    const Scalar&, const Scalar&);
};

inline constexpr BinaryDoubleScalarOpConfig kSoftplusBackwardCfg = {
    "softplus_backward", &haganeOpsSoftplusBackward, &cpu_dispatch_softplus_backward, nullptr};
inline constexpr BinaryDoubleScalarOpConfig kHardtanhBackwardCfg = {
    "hardtanh_backward", &haganeOpsHardtanhBackward, nullptr, &cpu_dispatch_hardtanh_backward};

template <const BinaryDoubleScalarOpConfig& Cfg>
inline void hagane_binary_double_scalar_bridge(TensorIteratorBase& iter,
                                                const Scalar& s1, const Scalar& s2) {
    auto out = make_ops_tensor_local(iter, 0);
    at::Tensor sa, sb;
    auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
    auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
    if (Cfg.c_abi_fn(&a, &b, &out, s1.toFloat(), s2.toFloat()) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback_base(iter, s1, s2);
    }
}

template <const BinaryDoubleScalarOpConfig& Cfg>
inline void hagane_binary_double_scalar_iter_bridge(TensorIterator& iter,
                                                     const Scalar& s1, const Scalar& s2) {
    auto out = make_ops_tensor_local(iter, 0);
    at::Tensor sa, sb;
    auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
    auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
    if (Cfg.c_abi_fn(&a, &b, &out, s1.toFloat(), s2.toFloat()) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback_iter(iter, s1, s2);
    }
}

// ---- X+30 Lane C/D: upsample-scale bridges --------------------------------
// Stub signatures (mirroring CPU TORCH_IMPL_FUNC at UpSampleNearest*d.cpp):
//   1d: (grad_input, grad_output, scales_w)
//   2d: (grad_input, grad_output, scales_h, scales_w)
//   3d: (grad_input, grad_output, scales_d, scales_h, scales_w)
// The C-ABI in all 3 cases reads (grad_output, grad_input) descriptors;
// scales are recoverable from the shape ratio in the kernel.

// X+31 Lane B — added `bool exact` field. The C-ABI now takes a 3rd `int exact`
// parameter (centered source index for exact-nearest variants). Non-exact ops
// set `exact = false`; exact variants set `exact = true`.
struct UpSampleScale1dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int);
    void (*cpu_fallback)(const Tensor&, const Tensor&, std::optional<double>);
    bool exact;
};
struct UpSampleScale2dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int);
    void (*cpu_fallback)(const Tensor&, const Tensor&,
                         std::optional<double>, std::optional<double>);
    bool exact;
};
struct UpSampleScale3dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int);
    void (*cpu_fallback)(const Tensor&, const Tensor&,
                         std::optional<double>, std::optional<double>, std::optional<double>);
    bool exact;
};

inline constexpr UpSampleScale1dOpConfig kUpsampleNearest1dBackwardCfg = {
    "upsample_nearest1d_backward", &haganeOpsUpsampleNearest1dBackward,
    &cpu_dispatch_upsample_nearest1d_backward, /*exact=*/false};
inline constexpr UpSampleScale2dOpConfig kUpsampleNearest2dBackwardCfg = {
    "upsample_nearest2d_backward", &haganeOpsUpsampleNearest2dBackward,
    &cpu_dispatch_upsample_nearest2d_backward, /*exact=*/false};
inline constexpr UpSampleScale3dOpConfig kUpsampleNearest3dBackwardCfg = {
    "upsample_nearest3d_backward", &haganeOpsUpsampleNearest3dBackward,
    &cpu_dispatch_upsample_nearest3d_backward, /*exact=*/false};

// X+31 Lane B — exact-nearest backward variants reuse the same C-ABIs with
// `exact = true`; CPU fallback routes to `_upsample_nearest_exact*d_backward_kernel`.
inline constexpr UpSampleScale1dOpConfig kUpsampleNearestExact1dBackwardCfg = {
    "_upsample_nearest_exact1d_backward", &haganeOpsUpsampleNearest1dBackward,
    &cpu_dispatch_upsample_nearest_exact1d_backward, /*exact=*/true};
inline constexpr UpSampleScale2dOpConfig kUpsampleNearestExact2dBackwardCfg = {
    "_upsample_nearest_exact2d_backward", &haganeOpsUpsampleNearest2dBackward,
    &cpu_dispatch_upsample_nearest_exact2d_backward, /*exact=*/true};
inline constexpr UpSampleScale3dOpConfig kUpsampleNearestExact3dBackwardCfg = {
    "_upsample_nearest_exact3d_backward", &haganeOpsUpsampleNearest3dBackward,
    &cpu_dispatch_upsample_nearest_exact3d_backward, /*exact=*/true};

template <const UpSampleScale1dOpConfig& Cfg>
inline void hagane_upsample_scale1d_bridge(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_w) {
    auto grad_in_d  = make_ops_tensor_local_t(grad_input);
    auto grad_out_d = make_ops_tensor_local_t(grad_output);
    if (Cfg.c_abi_fn(&grad_out_d, &grad_in_d, Cfg.exact ? 1 : 0) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(grad_input, grad_output, scales_w);
    }
}
template <const UpSampleScale2dOpConfig& Cfg>
inline void hagane_upsample_scale2d_bridge(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    auto grad_in_d  = make_ops_tensor_local_t(grad_input);
    auto grad_out_d = make_ops_tensor_local_t(grad_output);
    if (Cfg.c_abi_fn(&grad_out_d, &grad_in_d, Cfg.exact ? 1 : 0) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(grad_input, grad_output, scales_h, scales_w);
    }
}
template <const UpSampleScale3dOpConfig& Cfg>
inline void hagane_upsample_scale3d_bridge(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w) {
    auto grad_in_d  = make_ops_tensor_local_t(grad_input);
    auto grad_out_d = make_ops_tensor_local_t(grad_output);
    if (Cfg.c_abi_fn(&grad_out_d, &grad_in_d, Cfg.exact ? 1 : 0) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(grad_input, grad_output, scales_d, scales_h, scales_w);
    }
}

// X+31 Lane A — align-corners upsample backward family (linear1d / bilinear2d /
// trilinear3d). C-ABI signature mirrors the non-align-corners family but takes
// an additional `int align_corners` flag.
//
// Stub signatures (mirroring CPU TORCH_IMPL_FUNC at UpSample{Linear1d,Bilinear2d,Trilinear3d}.cpp):
//   1d: (grad_input, grad_output, align_corners, scales_w)
//   2d: (grad_input, grad_output, align_corners, scales_h, scales_w)
//   3d: (grad_input, grad_output, align_corners, scales_d, scales_h, scales_w)
// bicubic2d_backward is omitted — no DispatchStub (no DECLARE_DISPATCH at
// UpSample.h); retirement deferred until upstream provides one or .cu-patch
// mechanism unblocks (Lane E).

struct UpSampleScaleAlignCorners1dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int);
    void (*cpu_fallback)(const Tensor&, const Tensor&,
                         bool, std::optional<double>);
};
struct UpSampleScaleAlignCorners2dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int);
    void (*cpu_fallback)(const Tensor&, const Tensor&,
                         bool, std::optional<double>, std::optional<double>);
};
struct UpSampleScaleAlignCorners3dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*, int);
    void (*cpu_fallback)(const Tensor&, const Tensor&,
                         bool, std::optional<double>, std::optional<double>,
                         std::optional<double>);
};

inline constexpr UpSampleScaleAlignCorners1dOpConfig kUpsampleLinear1dBackwardCfg = {
    "upsample_linear1d_backward", &haganeOpsUpsampleLinear1dBackward,
    &cpu_dispatch_upsample_linear1d_backward};
inline constexpr UpSampleScaleAlignCorners2dOpConfig kUpsampleBilinear2dBackwardCfg = {
    "upsample_bilinear2d_backward", &haganeOpsUpsampleBilinear2dBackward,
    &cpu_dispatch_upsample_bilinear2d_backward};
inline constexpr UpSampleScaleAlignCorners3dOpConfig kUpsampleTrilinear3dBackwardCfg = {
    "upsample_trilinear3d_backward", &haganeOpsUpsampleTrilinear3dBackward,
    &cpu_dispatch_upsample_trilinear3d_backward};
// X+32 Lane D — bicubic2d_backward retirement (upstream-patched DispatchStub).
// Reuses UpSampleScaleAlignCorners2dOpConfig + existing haganeOpsUpsampleBicubic2dBackward
// C-ABI (X+30 Lane A parity fix already applied).
inline constexpr UpSampleScaleAlignCorners2dOpConfig kUpsampleBicubic2dBackwardCfg = {
    "upsample_bicubic2d_backward", &haganeOpsUpsampleBicubic2dBackward,
    &cpu_dispatch_upsample_bicubic2d_backward};
// X+32 Lane C — anti-aliased upsample backwards reuse UpSampleScaleAlignCorners2dOpConfig.
inline constexpr UpSampleScaleAlignCorners2dOpConfig kUpsampleBilinear2dAABackwardCfg = {
    "_upsample_bilinear2d_aa_backward", &haganeOpsUpsampleBilinear2dAABackward,
    &cpu_dispatch_upsample_bilinear2d_aa_backward};
inline constexpr UpSampleScaleAlignCorners2dOpConfig kUpsampleBicubic2dAABackwardCfg = {
    "_upsample_bicubic2d_aa_backward", &haganeOpsUpsampleBicubic2dAABackward,
    &cpu_dispatch_upsample_bicubic2d_aa_backward};

template <const UpSampleScaleAlignCorners1dOpConfig& Cfg>
inline void hagane_upsample_scale_alignc_1d_bridge(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners, std::optional<double> scales_w) {
    auto grad_in_d  = make_ops_tensor_local_t(grad_input);
    auto grad_out_d = make_ops_tensor_local_t(grad_output);
    if (Cfg.c_abi_fn(&grad_out_d, &grad_in_d, align_corners ? 1 : 0) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(grad_input, grad_output, align_corners, scales_w);
    }
}
template <const UpSampleScaleAlignCorners2dOpConfig& Cfg>
inline void hagane_upsample_scale_alignc_2d_bridge(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    auto grad_in_d  = make_ops_tensor_local_t(grad_input);
    auto grad_out_d = make_ops_tensor_local_t(grad_output);
    if (Cfg.c_abi_fn(&grad_out_d, &grad_in_d, align_corners ? 1 : 0) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(grad_input, grad_output, align_corners, scales_h, scales_w);
    }
}
template <const UpSampleScaleAlignCorners3dOpConfig& Cfg>
inline void hagane_upsample_scale_alignc_3d_bridge(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w) {
    auto grad_in_d  = make_ops_tensor_local_t(grad_input);
    auto grad_out_d = make_ops_tensor_local_t(grad_output);
    if (Cfg.c_abi_fn(&grad_out_d, &grad_in_d, align_corners ? 1 : 0) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(grad_input, grad_output, align_corners, scales_d, scales_h, scales_w);
    }
}

// X+31 Lane C — pool backward family. avg_pool2d/3d backward TORCH_IMPL_FUNCs
// retire via stub. max_pool3d_with_indices_backward is C10_EXPORT — stub is
// still REGISTER_DISPATCH'd via the bridge; the C10_EXPORT entry point in
// HaganeOps.cpp routes through it (the +1 RD movement preserves ADR-036
// trajectory without changing native_functions.yaml).

struct AvgPool2dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    int, int, int, int, int, int, int, int64_t);
    avg_pool2d_backward_fn cpu_fallback;
};
struct AvgPool3dOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    int, int, int, int, int, int, int, int, int, int, int64_t);
    avg_pool3d_backward_fn cpu_fallback;
};
struct MaxPool3dBackwardOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    const haganeOpsTensor_t*);
    max_pool3d_backward_fn cpu_fallback;
};
// X+31 Lane D / X+39 Lane B rename — pool-with-indices backward signature
// shared across max_pool2d_backward + adaptive_max_pool{2,3}d_backward:
// `void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices)`.
// All 3 DispatchStubs typedef to the identical function-pointer type, so a
// single struct + template serves all 3 ops. Originally named
// MaxPool2dBackwardOpConfig (X+31 Lane D) — renamed X+39 Lane B once it
// served >1 op (X+37 added the 2 adaptive ops). cpu_fallback uses
// `max_pool2d_backward_fn` typedef name out of historical preference; the
// type is identical to `adaptive_max_pooling{2,3}d_backward_fn`.
struct PoolBackwardWithIndicesOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    const haganeOpsTensor_t*);
    max_pool2d_backward_fn cpu_fallback;
};

inline constexpr AvgPool2dOpConfig kAvgPool2dBackwardCfg = {
    "avg_pool2d_backward", &haganeOpsAvgPool2dBackward,
    &cpu_dispatch_avg_pool2d_backward};
inline constexpr AvgPool3dOpConfig kAvgPool3dBackwardCfg = {
    "avg_pool3d_backward", &haganeOpsAvgPool3dBackward,
    &cpu_dispatch_avg_pool3d_backward};
inline constexpr MaxPool3dBackwardOpConfig kMaxPool3dBackwardCfg = {
    "max_pool3d_backward", &haganeOpsMaxPool3dBackward,
    &cpu_dispatch_max_pool3d_backward};
inline constexpr PoolBackwardWithIndicesOpConfig kMaxPool2dBackwardCfg = {
    "max_pool2d_backward", &haganeOpsMaxPool2dBackward,
    &cpu_dispatch_max_pool2d_backward};
// X+37 Lane B — adaptive_max_pool{2,3}d_backward Cfg constants. Reuse
// PoolBackwardWithIndicesOpConfig + hagane_pool_backward_with_indices_bridge template;
// the C-ABI + DispatchStub + cpu_fallback signatures all match exactly
// (Tensor& gradInput, gradOutput, indices). Bridge RDs land at
// HaganeMetallibBridge.cpp via REGISTER_DISPATCH(adaptive_max_pool{2,3}d_backward_kernel).
inline constexpr PoolBackwardWithIndicesOpConfig kAdaptiveMaxPool2dBackwardCfg = {
    "adaptive_max_pool2d_backward", &haganeOpsAdaptiveMaxPool2dBackward,
    &cpu_dispatch_adaptive_max_pool2d_backward};
inline constexpr PoolBackwardWithIndicesOpConfig kAdaptiveMaxPool3dBackwardCfg = {
    "adaptive_max_pool3d_backward", &haganeOpsAdaptiveMaxPool3dBackward,
    &cpu_dispatch_adaptive_max_pool3d_backward};

// X+40 Lane A — AdaptiveAvgPool backward shape (2-arg, no indices). Shared
// across 2D + 3D variants per AdaptivePooling.h:12,22 typedefs. C-ABIs:
// haganeOpsAdaptiveAvgPool{2,3}dBackward(grad_output, grad_input) — note
// arg order is swapped vs DispatchStub typedef but the bridge template
// re-orders them.
struct AdaptiveAvgPoolBackwardOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*);
    adaptive_avg_pooling2d_backward_fn cpu_fallback;  // identical to 3d_backward_fn
};
inline constexpr AdaptiveAvgPoolBackwardOpConfig kAdaptiveAvgPool2dBackwardCfg = {
    "adaptive_avg_pool2d_backward", &haganeOpsAdaptiveAvgPool2dBackward,
    &cpu_dispatch_adaptive_avg_pool2d_backward};
inline constexpr AdaptiveAvgPoolBackwardOpConfig kAdaptiveAvgPool3dBackwardCfg = {
    "adaptive_avg_pool3d_backward", &haganeOpsAdaptiveAvgPool3dBackward,
    &cpu_dispatch_adaptive_avg_pool3d_backward};

// X+42 Lane B — AdaptiveAvgPool forward shape (3-arg with output_size). Shared
// across 2D + 3D per AdaptivePooling.h:11,21 typedefs. C-ABIs:
// haganeOpsAdaptiveAvgPool{2,3}d(input, output) — bridge re-orders into the
// DispatchStub-typed (output, input) shape. output_size is implicit in
// `output`'s shape (set by the entry point before dispatch).
struct AdaptiveAvgPoolForwardOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*);
    adaptive_avg_pooling2d_fn cpu_fallback;  // identical to 3d_fn
};
inline constexpr AdaptiveAvgPoolForwardOpConfig kAdaptiveAvgPool2dForwardCfg = {
    "adaptive_avg_pool2d", &haganeOpsAdaptiveAvgPool2d,
    &cpu_dispatch_adaptive_avg_pool2d};
inline constexpr AdaptiveAvgPoolForwardOpConfig kAdaptiveAvgPool3dForwardCfg = {
    "adaptive_avg_pool3d", &haganeOpsAdaptiveAvgPool3d,
    &cpu_dispatch_adaptive_avg_pool3d};

// X+44 Lane B — AvgPool forward shapes. C-ABI signature is identical to the
// backward variants (8 ints + int64_t divisor for 2D; 11 ints + int64_t for
// 3D) so the existing AvgPool{2,3}dOpConfig c_abi_fn type fits. However
// `avg_pool{2,3}d_fn` (Pool.h:21,30) uses int64_t args while
// `avg_pool{2,3}d_backward_fn` (Pool.h:23,34) uses int args — so the
// cpu_fallback field type differs, requiring a distinct struct.
struct AvgPool2dForwardOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    int, int, int, int, int, int, int, int64_t);
    avg_pool2d_fn cpu_fallback;
};
struct AvgPool3dForwardOpConfig {
    const char* op_name;
    int (*c_abi_fn)(const haganeOpsTensor_t*, const haganeOpsTensor_t*,
                    int, int, int, int, int, int, int, int, int, int, int64_t);
    avg_pool3d_fn cpu_fallback;
};
inline constexpr AvgPool2dForwardOpConfig kAvgPool2dForwardCfg = {
    "avg_pool2d", &haganeOpsAvgPool2d, &cpu_dispatch_avg_pool2d};
inline constexpr AvgPool3dForwardOpConfig kAvgPool3dForwardCfg = {
    "avg_pool3d", &haganeOpsAvgPool3d, &cpu_dispatch_avg_pool3d};

template <const AvgPool2dOpConfig& Cfg>
inline void hagane_avg_pool2d_backward_bridge(
    const Tensor& gradInput, const Tensor& gradOutput,
    int kW, int kH, int dW, int dH, int padW, int padH,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
    auto gi = make_ops_tensor_local_t(gradInput);
    auto go = make_ops_tensor_local_t(gradOutput);
    if (Cfg.c_abi_fn(&go, &gi, kH, kW, dH, dW, padH, padW,
                     count_include_pad ? 1 : 0,
                     divisor_override.value_or(0)) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(gradInput, gradOutput, kW, kH, dW, dH, padW, padH,
                         count_include_pad, divisor_override);
    }
}
template <const AvgPool3dOpConfig& Cfg>
inline void hagane_avg_pool3d_backward_bridge(
    const Tensor& gradInput, const Tensor& gradOutput,
    int kW, int kH, int kD, int dW, int dH, int dD,
    int padW, int padH, int padD,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
    auto gi = make_ops_tensor_local_t(gradInput);
    auto go = make_ops_tensor_local_t(gradOutput);
    if (Cfg.c_abi_fn(&go, &gi, kD, kH, kW, dD, dH, dW, padD, padH, padW,
                     count_include_pad ? 1 : 0,
                     divisor_override.value_or(0)) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(gradInput, gradOutput, kW, kH, kD, dW, dH, dD,
                         padW, padH, padD, count_include_pad, divisor_override);
    }
}
template <const MaxPool3dBackwardOpConfig& Cfg>
inline void hagane_max_pool3d_backward_bridge(
    Tensor& gradInput, const Tensor& gradOutput, const Tensor& indices) {
    auto gi  = make_ops_tensor_local_t(gradInput);
    auto go  = make_ops_tensor_local_t(gradOutput);
    auto idx = make_ops_tensor_local_t(indices);
    if (Cfg.c_abi_fn(&go, &gi, &idx) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(gradInput, gradOutput, indices);
    }
}
template <const PoolBackwardWithIndicesOpConfig& Cfg>
inline void hagane_pool_backward_with_indices_bridge(
    const Tensor& gradInput, const Tensor& gradOutput, const Tensor& indices) {
    auto gi  = make_ops_tensor_local_t(gradInput);
    auto go  = make_ops_tensor_local_t(gradOutput);
    auto idx = make_ops_tensor_local_t(indices);
    if (Cfg.c_abi_fn(&go, &gi, &idx) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(gradInput, gradOutput, indices);
    }
}
// X+40 Lane A — adaptive avg-pool backward bridge (2-arg, no indices).
// Serves both 2D and 3D variants. DispatchStub signature passes Tensor&
// grad_input (non-const) since the backward kernel writes into it; the
// bridge template signature mirrors that.
template <const AdaptiveAvgPoolBackwardOpConfig& Cfg>
inline void hagane_adaptive_avg_pool_backward_bridge(
    Tensor& gradInput, const Tensor& gradOutput) {
    auto gi = make_ops_tensor_local_t(gradInput);
    auto go = make_ops_tensor_local_t(gradOutput);
    if (Cfg.c_abi_fn(&go, &gi) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(gradInput, gradOutput);
    }
}
// X+42 Lane B — adaptive avg-pool forward bridge (3-arg with output_size).
// DispatchStub signature passes Tensor& output + const Tensor& input +
// IntArrayRef output_size; the C-ABI's (input, output) ordering is swapped at
// the call site. output_size is implicit in output's shape (caller has already
// resized output before dispatch).
template <const AdaptiveAvgPoolForwardOpConfig& Cfg>
inline void hagane_adaptive_avg_pool_forward_bridge(
    Tensor& output, const Tensor& input, IntArrayRef output_size) {
    auto i = make_ops_tensor_local_t(input);
    auto o = make_ops_tensor_local_t(output);
    if (Cfg.c_abi_fn(&i, &o) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(output, input, output_size);
    }
}
// X+44 Lane B — avg_pool forward bridges. DispatchStub passes (output, input,
// kW, kH, ...) with int64_t args; C-ABI takes (input, output, kH, kW, ...) so
// the bridge re-orders args + downcasts to int at the call site (matches the
// pattern of hagane_avg_pool{2,3}d_backward_bridge at lines 1079+1094).
template <const AvgPool2dForwardOpConfig& Cfg>
inline void hagane_avg_pool2d_forward_bridge(
    const Tensor& output, const Tensor& input,
    int64_t kW, int64_t kH, int64_t dW, int64_t dH,
    int64_t padW, int64_t padH, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
    auto i = make_ops_tensor_local_t(input);
    auto o = make_ops_tensor_local_t(output);
    if (Cfg.c_abi_fn(&i, &o, (int)kH, (int)kW, (int)dH, (int)dW,
                     (int)padH, (int)padW,
                     count_include_pad ? 1 : 0,
                     divisor_override.value_or(0)) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(output, input, kW, kH, dW, dH, padW, padH,
                         count_include_pad, divisor_override);
    }
}
template <const AvgPool3dForwardOpConfig& Cfg>
inline void hagane_avg_pool3d_forward_bridge(
    const Tensor& output, const Tensor& input,
    int64_t kW, int64_t kH, int64_t kD, int64_t dW, int64_t dH, int64_t dD,
    int64_t padW, int64_t padH, int64_t padD, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
    auto i = make_ops_tensor_local_t(input);
    auto o = make_ops_tensor_local_t(output);
    if (Cfg.c_abi_fn(&i, &o, (int)kD, (int)kH, (int)kW,
                     (int)dD, (int)dH, (int)dW,
                     (int)padD, (int)padH, (int)padW,
                     count_include_pad ? 1 : 0,
                     divisor_override.value_or(0)) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        Cfg.cpu_fallback(output, input, kW, kH, kD, dW, dH, dD,
                         padW, padH, padD, count_include_pad, divisor_override);
    }
}

// E2E-1 (o-voxel flexible_dual_grid_to_mesh) surfaced torch.cross / linalg_cross
// crashing on Hagane: the stock CrossKernel.hip launches a raw cross_kernel<<<>>>
// whose OffsetCalculator/TensorIterator GPU path has no Metal backing on the Hagane
// fork -> segfault. Own cross_stub via REGISTER_DISPATCH (the bridge mechanism that
// reliably overrides the hipified artifact's stub, unlike an operator-level m.impl
// which loses the link-order race to the generated structured CUDA kernel). Compose
// from OWNED primitives (roll/mul/sub) — exact vs the CPU reference (float epsilon):
//   cross(x1,x2) along d = roll(x1,-1,d)*roll(x2,-2,d) - roll(x1,-2,d)*roll(x2,-1,d)
// result/x1/x2 are already broadcast + sized by the structured linalg_cross.out.
// ADR-036-clean (REGISTER_DISPATCH in the bridge TU); ADR-032-clean (a generic vector
// op composed from owned primitives, not model-shaped runtime code).
inline void hagane_cross_impl(const at::Tensor& result, const at::Tensor& x1,
                              const at::Tensor& x2, const int64_t dim) {
    int64_t d = at::maybe_wrap_dim(dim, x1.dim());
    result.copy_(at::sub(
        at::mul(at::roll(x1, {-1}, {d}), at::roll(x2, {-2}, {d})),
        at::mul(at::roll(x1, {-2}, {d}), at::roll(x2, {-1}, {d}))));
}

} // namespace at::native::hagane_dispatch::detail
