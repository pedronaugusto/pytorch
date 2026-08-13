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

#include <algorithm>
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

    // Does this op's CUDA kernel take a DIFFERENT code path when an operand is
    // a CPU scalar AND the dtype is floating?
    //
    // div_floor and div_trunc do: they trade the division for a reciprocal
    // multiply (`(a - fmod(a,b)) * inv_b`), and upstream's own comment says it
    // "may lose one bit of precision". So for those the harvested `_a1_sA` /
    // `_a1_sB` kernels are wrappers of the GENERAL-float functor, which is a
    // branch torch never dispatches for a float cpu scalar — and the two are
    // not interchangeable: over 300k adversarial cases they agree exactly at
    // float32 but differ on 2537/300k at fp16 and 2275/300k at bf16, by far
    // more than 1 ULP (e.g. -3776 vs -3760). MEASURED, not argued.
    //
    // So when this is true the metallib scalar arm is restricted to INTEGRAL
    // dtypes (where torch has no such branch and the harvest is exact) and the
    // float case is owned separately. fmod/remainder set it false: their
    // gpu_kernel_with_scalars presents ONE functor for every operand shape, so
    // the harvested arms are exactly what torch dispatches.
    bool float_cpu_scalar_differs = false;
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
inline constexpr BinaryOpConfig kDivTrueCfg   = {"div_true",  "hagane_DivFunctor", &haganeOpsDiv,       &cpu_dispatch_div_true,  &fp64_div};

// 1.11-A — div_floor / div_trunc / fmod run NATIVE from PyTorch's OWN harvested
// kernels, and `c_abi_fn = nullptr` is what does the work: it removes
// haganeOps{DivFloor,DivTrunc,Fmod} from the path entirely, so the stash AND
// the silent-wrong go in one edit, and a DECLINE lands on torch's own exact CPU
// kernel instead of a wrong device answer. kGcdCfg is the precedent.
//
// Why these three and not `remainder`: 1.11 step 0 classified all four against
// the shipped build. `mx::divide` round-trips integers through float32, so
// div_floor took 34 divergences and div_trunc 23 — `16777217 // 1` -> 16777216
// at BOTH int32 and int64 (the giveaway), `INT_MIN // -1` saturating to
// +INT_MAX, bf16 `4.0 // 0.4` = 10 against CUDA's 9. fmod took 4 (fp16 loses
// the sign of a zero result). `remainder` ALREADY routes through MLX's
// `Remainder` kernel, which is torch's exact formula for integers and floats
// alike, and stashes zero — it keeps the same 4 fp16 signed-zero rows, and its
// own kernel does NOT harvest (an `if` in the functor body, #1069), so it is
// filed rather than folded in here.
//
// div_floor/div_trunc set float_cpu_scalar_differs: upstream has a CUDA-only
// reciprocal branch for a floating cpu scalar and the harvest does not carry
// it. fmod has no such branch.
inline constexpr BinaryOpConfig kFmodCfg      = {"fmod",      "hagane_fmod_kernel_cuda",      nullptr, &cpu_dispatch_fmod,      nullptr, false};
inline constexpr BinaryOpConfig kDivFloorCfg  = {"div_floor", "hagane_div_floor_kernel_cuda", nullptr, &cpu_dispatch_div_floor, nullptr, true };
inline constexpr BinaryOpConfig kDivTruncCfg  = {"div_trunc", "hagane_div_trunc_kernel_cuda", nullptr, &cpu_dispatch_div_trunc, nullptr, true };

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
    // MLX's all_reduce operator name, for the ops whose FULL reduction can be
    // dispatched into the caller's own block (#1010 1.9). nullptr means it
    // cannot. sum/prod are nullptr deliberately: MLX's remap_reduce_types
    // WIDENS their output for integers and torch has its own accumulation
    // contract there, so routing them would be a numerics decision rather than
    // an ownership one — and sum/mean already have the owned Tier-a reducer.
    const char* mlx_reduce_op;
};

// A9: kSumCfg/kMeanCfg carry a non-null metallib_kernel sentinel ("reduce") so
// the bridge attempts try_launch_reduction_metallib (the owned Tier-a host
// 2-pass) before the MLX c_abi_fn. The sentinel only gates the ATTEMPT; the
// actual kernel names are picked by op and dtype inside the helper.
//
// #1135 extended the sentinel to prod / max_values / min_values / and / or, so
// their PARTIAL (dim=) reductions reach the owned outer/red/inner kernel. The
// helpers take the op NAME and decline by name anything they have no arm for
// (`op_unowned`, `dtype_pair_unowned`), which is what makes the sentinel a
// request to try rather than a claim to own: try_launch_reduction_metallib is
// still sum/mean-only, and now says so instead of assuming it.
//
// argmax/argmin keep nullptr: they reduce to an INDEX, not to a value, and
// share none of these kernels.
inline constexpr UnaryIterOpConfig kSumCfg       = {"sum",        "reduce", &haganeOpsSum,       &cpu_dispatch_sum,        nullptr, nullptr};
inline constexpr UnaryIterOpConfig kMeanCfg      = {"mean",       "reduce", &haganeOpsMean,      &cpu_dispatch_mean,       nullptr, nullptr};
inline constexpr UnaryIterOpConfig kProdCfg      = {"prod",       "reduce", &haganeOpsProd,      &cpu_dispatch_prod,       nullptr, nullptr};
inline constexpr UnaryIterOpConfig kArgmaxCfg    = {"argmax",     nullptr, &haganeOpsArgmax,    &cpu_dispatch_argmax,     nullptr, nullptr};
inline constexpr UnaryIterOpConfig kArgminCfg    = {"argmin",     nullptr, &haganeOpsArgmin,    &cpu_dispatch_argmin,     nullptr, nullptr};
inline constexpr UnaryIterOpConfig kMaxValuesCfg = {"max_values", "reduce", &haganeOpsMaxValues, &cpu_dispatch_max_values, nullptr, "max"};
inline constexpr UnaryIterOpConfig kMinValuesCfg = {"min_values", "reduce", &haganeOpsMinValues, &cpu_dispatch_min_values, nullptr, "min"};
// and_stub is the reduction kernel behind `torch.all` (logical AND across
// elements), or_stub backs `torch.any` (logical OR). The C-ABI names match
// the semantic, not the stub name.
inline constexpr UnaryIterOpConfig kAndCfg       = {"and",        "reduce", &haganeOpsAll,       &cpu_dispatch_and,        nullptr, "and"};
inline constexpr UnaryIterOpConfig kOrCfg        = {"or",         "reduce", &haganeOpsAny,       &cpu_dispatch_or,         nullptr, "or"};
inline constexpr UnaryIterOpConfig kHardswishCfg = {"hardswish",  nullptr, &haganeOpsHardswish, &cpu_dispatch_hardswish,  nullptr, nullptr};

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
    // MLX's scan operator name, when the C-ABI above lands on MLX's scan kernel
    // and this op can therefore be dispatched into the caller's own block
    // (#1010 1.9). nullptr means it cannot, which is a statement about the
    // FALLBACK and not about the corpus: logcumsumexp is nullptr because
    // haganeOpsLogcumsumexp composes max/subtract/cumsum(exp)/log/add rather
    // than calling the logaddexp scan the metallib does carry, so routing it
    // would change the arithmetic instead of only the ownership.
    const char* mlx_scan_op;
};

inline constexpr CumOpConfig kCumsumCfg       = {"cumsum",       &haganeOpsCumsum,       &cpu_dispatch_cumsum,     nullptr, "sum"};
inline constexpr CumOpConfig kCumprodCfg      = {"cumprod",      &haganeOpsCumprod,      &cpu_dispatch_cumprod,    nullptr, "prod"};
inline constexpr CumOpConfig kLogcumsumexpCfg = {"logcumsumexp", &haganeOpsLogcumsumexp, nullptr,                  &cpu_dispatch_logcumsumexp, nullptr};

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

// ---- 1.8: native gelu route -------------------------------------------------
// GeluCUDAKernelImpl is a direct C10_EXPORT that upstream's Activation.cpp
// calls, not a DispatchStub with an OpConfig row, so registration goes through a
// standalone once-init exactly as the norms above do rather than through
// MetallibState<Cfg>. MLX ships no gelu kernel at all (checked with `strings`
// over mlx.metallib), which is why this is transpiler-owned like silu.
inline bool gelu_metallib_available() {
    static const bool available = [] {
        const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
        if (route && route[0] == '0') return false;
        std::string path = metallib_dir() + "/gelu.metallib";
        if (haganeRegisterMetallibAll(path.c_str()) != hipSuccess) {
            std::fprintf(stderr,
                "[hagane-path-alpha] failed to register %s; gelu stays on MLX\n",
                path.c_str());
            return false;
        }
        std::fprintf(stderr,
            "[hagane-path-alpha] gelu metallib registered (erf + tanh arms, "
            "all dtype arms)\n");
        return true;
    }();
    return available;
}

// The two `approximate` arms are two GPU_LAMBDAs inside one function, so they
// collide on the transpiler's function-derived name and the second takes an
// ordinal. Which arm holds which name is POSITIONAL — see the gelu row in
// tools/metallib_manifest.txt, the collision diagnostic hagane-compile prints,
// and scripts/test_gelu_route.py, which separates the arms numerically so a
// swap fails loudly instead of silently changing the activation.
inline const char* gelu_metallib_base(at::native::GeluType approximate) {
    return approximate == at::native::GeluType::Tanh
        ? "hagane_GeluCUDAKernelImpl"       // arm 1 — tanh, ActivationGeluKernel.hip:24
        : "hagane_GeluCUDAKernelImpl__2";   // arm 2 — erf,  ActivationGeluKernel.hip:35
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

// The decline-trace helpers are defined further down (with the vendor routes)
// but the owned routes above them need to report too. Declared here rather than
// moved, so the definitions stay next to the family that documents them.
inline bool decline(const char* route, const char* op, const char* why);
inline bool decline_layout(const char* route, const char* why,
                           const at::Tensor& t);
inline bool decline_dtypes(const char* route, const char* op, const char* why,
                           c10::ScalarType from, c10::ScalarType to);

// ---- A9 (Phase 2): native reduction route (sum / mean, full contiguous) ------
// PyTorch's real reduce_kernel can't be transpiled (its ReduceOp carries an
// un-nameable GPU_LAMBDA ops type — see hagane/kernels/reduce.hip), so we own a
// generic Tier-a block reducer instead. reduce.metallib bundles 7 concrete
// kernels: the float ladder f32→f32, {bf16,f16}→f32 (pass 1), f32→{bf16,f16}
// (pass 2), and the two integer reducers i64→i64 / i32→i32 (#1124), which are
// their own pass 1 AND pass 2 because TIN == TOUT == TACC. Route-off
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
            "[hagane-path-alpha] reduce metallib registered "
            "(Tier-a: sum/mean f32+bf16+f16, integer sum i32+i64)\n");
        return true;
    }();
    return available;
}

// Per-thread persistent partials buffer for the reduction 2-pass — reused
// across calls (mirrors rms_moments_scratch: never freed mid-flight, grows
// monotonically, safe on the in-order queue since pass 2 consumes pass 1's
// partials before the next call overwrites them).
//
// acc_ty IS LOAD-BEARING, not a tidy-up. This buffer used to be hardcoded
// at::kFloat while the reuse predicate compared only numel and device. Once
// #1124 added an int64 arm, a buffer first created by a float reduction (4
// bytes/elt) and then handed to an int64 pass 1 (8 bytes/elt) would have been
// overrun by exactly 2x — a heap corruption, not a wrong number, and one that
// would have reproduced only when the two dtypes met on one thread in that
// order. The dtype is part of the reuse identity.
inline void* reduce_partials_scratch(const at::TensorOptions& opts,
                                     int64_t nblocks, at::ScalarType acc_ty) {
    static thread_local at::Tensor* t = nullptr;
    if (t == nullptr) t = new at::Tensor();
    if (!t->defined() || t->numel() < nblocks || t->device() != opts.device()
        || t->scalar_type() != acc_ty)
        *t = at::empty({nblocks}, opts.dtype(acc_ty));
    return t->data_ptr();
}

// Full contiguous sum/mean as a host 2-pass: pass 1 reduces the input → per-block
// float partials (grid=(nblocks) block=256, grid-strided); pass 2 reduces the
// partials → the single output (grid=(1) block=256). `scale` (1 for sum, 1/N for
// mean) is applied once at the pass-2 store. Returns false → MLX fallback:
// route-off, tape recording, non-contiguous, unsupported dtype, partial-dim
// reduction (out.numel()!=1, deferred to a Tier-b follow-on), or launch failure.
//
// #1135 step 1 — this takes the OP NAME, not a `bool is_mean`. The bool was
// sound only while sum and mean were the only two configs carrying a
// `metallib_kernel`: a third op reaching here with `false` would have been
// computed as a SUM and returned as itself. Declining by name is what makes
// giving `prod`/`max`/`min`/`all`/`any` that sentinel a routing change rather
// than a silent-wrong.
inline bool try_launch_reduction_metallib(TensorIterator& iter, const char* op) {
    const bool is_mean = std::strcmp(op, "mean") == 0;
    if (!is_mean && std::strcmp(op, "sum") != 0)
        return decline("reduce_owned", op, "op_unowned");
    if (haganeOpsTapeRecording())
        return decline("reduce_owned", op, "tape_recording");
    if (!reduction_metallib_available())
        return decline("reduce_owned", op, "metallib_unavailable");
    const at::Tensor& in_t  = iter.tensor(1);
    const at::Tensor& out_t = iter.tensor(0);
    // Full reduction only. NOTE this is dtype-independent: a sum(dim=) whose
    // output has >1 element stays on MLX for float32 too. #1124 widened the
    // dtype gate below, NOT this one — the partial-dim case is the row/col
    // reduce family (1.9b) and is a separate piece of work.
    if (out_t.numel() != 1)
        return decline("reduce_owned", op, "partial_dim_out_numel_gt_1");
    if (!in_t.is_contiguous())
        return decline_layout("reduce_owned", "input_not_contiguous", in_t);
    const int64_t N = in_t.numel();
    if (N <= 0) return decline("reduce_owned", op, "empty_input");

    // in→out dtype pairs this route owns. acc_ty is the PARTIALS dtype (what
    // pass 1 writes and pass 2 reads); `scaled` says whether the kernel takes
    // the trailing float `scale` argument at all.
    //
    // Integers accumulate in their own width, never float: a float32
    // accumulator is exact only to 2^24. They also carry no scale — mean is
    // unreachable for them (TORCH_META_FUNC2(mean,dim) rejects an integral
    // inferred dtype, and mean(int, dtype=float) casts first and lands on the
    // f32 arm), so one kernel serves both passes.
    //
    // The narrow integer types (bool/uint8/int8/int16) never appear here:
    // ReduceOpsUtils.h's make_reduction does self.to(in_dtype) with
    // in_dtype == out_dtype, so torch has already widened them to int64 before
    // the stub is reached. That is why there is no i8/i16 arm and why adding
    // one would be dead code.
    const char *k1, *k2;
    at::ScalarType acc_ty = at::kFloat;
    bool scaled = true;
    const auto inty = in_t.scalar_type(), outty = out_t.scalar_type();
    if (inty == at::kFloat && outty == at::kFloat) {
        k1 = "hagane_reduce_f32_f32";  k2 = "hagane_reduce_f32_f32";
    } else if (inty == at::kBFloat16 && outty == at::kBFloat16) {
        k1 = "hagane_reduce_bf16_f32"; k2 = "hagane_reduce_f32_bf16";
    } else if (inty == at::kBFloat16 && outty == at::kFloat) {
        k1 = "hagane_reduce_bf16_f32"; k2 = "hagane_reduce_f32_f32";
    } else if (inty == at::kHalf && outty == at::kHalf) {
        k1 = "hagane_reduce_f16_f32";  k2 = "hagane_reduce_f32_f16";
    } else if (inty == at::kHalf && outty == at::kFloat) {
        k1 = "hagane_reduce_f16_f32";  k2 = "hagane_reduce_f32_f32";
    } else if (inty == at::kLong && outty == at::kLong) {
        k1 = k2 = "hagane_reduce_i64_i64"; acc_ty = at::kLong; scaled = false;
    } else if (inty == at::kInt && outty == at::kInt) {
        k1 = k2 = "hagane_reduce_i32_i32"; acc_ty = at::kInt;  scaled = false;
    } else {
        return decline_dtypes("reduce_owned", op, "dtype_pair_unowned",
                              inty, outty);
    }
    // Belt and braces on the paragraph above: if an integral mean ever does
    // reach here, decline rather than silently drop the 1/N.
    if (is_mean && !scaled)
        return decline("reduce_owned", "mean", "integral_mean_has_no_scale");

    const int block = 256;
    int64_t nbtmp = (N + block - 1) / block;     // ~one block per 256 elements,
    if (nbtmp < 1) nbtmp = 1;                     // clamped to [1, 256] so pass 2's
    if (nbtmp > 256) nbtmp = 256;                 // single block reduces the partials
    const int nblocks = static_cast<int>(nbtmp);
    void* dpart = reduce_partials_scratch(in_t.options(), nblocks, acc_ty);
    void* dX    = in_t.data_ptr();
    void* dout  = out_t.data_ptr();
    flush_or_commit_metallib_input(dX, N * in_t.element_size());

    int64_t c_N = N, c_NB = nblocks;
    float s1 = 1.0f, s2 = is_mean ? static_cast<float>(1.0 / static_cast<double>(N)) : 1.0f;
    const int    at_[] = {1, 0, 0, 1};
    const size_t as_[] = {sizeof(int64_t), 0, 0, sizeof(float)};
    const int nargs = scaled ? 4 : 3;   // the integer arms take no `scale`
    dim3 b(block, 1, 1), g1(static_cast<unsigned>(nblocks), 1, 1), g2(1, 1, 1);

    void* a1[] = {&c_N, dX, dpart, &s1};   // pass 1: X(N) → partials(nblocks), scale 1
    if (hagane_launch_kernel_mixed_tracked(k1, g1, b, 0, nullptr, a1, at_, as_, nargs) != hipSuccess)
        return decline("reduce_owned", op, "pass1_launch_failed");
    void* a2[] = {&c_NB, dpart, dout, &s2}; // pass 2: partials(nblocks) → out(1), scale
    if (hagane_launch_kernel_mixed_tracked(k2, g2, b, 0, nullptr, a2, at_, as_, nargs) != hipSuccess)
        return decline("reduce_owned", op, "pass2_launch_failed");

    note_native_launch(k1);
    note_native_launch(k2);
    haganeOpsMarkMetallibWrite(dout, static_cast<size_t>(out_t.element_size()));
    return true;
}

// #1135 — the PARTIAL reduction owned: `sum` / `mean` / `prod` / `amax` /
// `amin` / `all` / `any`, each over a `dim=`.
//
// try_launch_reduction_metallib above declines every one of these by name
// (`partial_dim_out_numel_gt_1`) and has since A9, with a comment calling the
// partial-dim case "a separate piece of work". Measured, that left sum, mean
// and prod stashing AND blocking the host on every dim and every dtype — 158
// dirty op x dtype x dim x keepdim combinations. It survived because ARDY never
// calls a partial sum; a model-free gate (test_workload_stress) found it on the
// first run it was ever given.
//
// `result` carries the INPUT's rank with extent 1 at every reduced axis —
// review_reduce_result's doing (ReduceOpsUtils.h:170) — which is what lets the
// reduced axes be identified here without threading a dim list through every
// stub signature. Same trick try_vendor_reduce_dim uses, and this route is
// strictly more general than that one: it takes a reduced block ANYWHERE, not
// only trailing.
//
// Geometry is outer/red/inner (see kernels/reduce.hip). inner == 1 is the row
// reduce and inner > 1 the col reduce, from one kernel. A reduced axis set that
// is not adjacent (`sum(dim=(0,2))`) is not expressible this way and declines
// by name rather than being approximated.
//
// Takes the OP NAME for the reason spelled out over try_launch_reduction_metallib
// above: the ops this route owns are named here and nowhere else, so an op that
// arrives without an arm declines instead of being computed as the arm that
// happens to be first.
inline bool try_launch_reduction_dim_metallib(TensorIterator& iter, const char* op) {
    constexpr const char* R = "reduce_dim_owned";
    // The families this route owns, by name. `mean` shares every one of sum's
    // arms and differs from it only by `scale`, applied once at the store. An op
    // not listed declines right here, which is what lets a config carry the
    // `reduce` sentinel without that being a claim that an arm exists for it.
    enum Family { kFamSum, kFamProd, kFamMax, kFamMin, kFamAll, kFamAny };
    Family fam;
    bool is_mean = false;
    if      (std::strcmp(op, "sum")        == 0) fam = kFamSum;
    else if (std::strcmp(op, "mean")       == 0) { fam = kFamSum; is_mean = true; }
    else if (std::strcmp(op, "prod")       == 0) fam = kFamProd;
    else if (std::strcmp(op, "max_values") == 0) fam = kFamMax;
    else if (std::strcmp(op, "min_values") == 0) fam = kFamMin;
    else if (std::strcmp(op, "and")        == 0) fam = kFamAll;
    else if (std::strcmp(op, "or")         == 0) fam = kFamAny;
    else return decline(R, op, "op_unowned");
    if (haganeOpsTapeRecording()) return decline(R, op, "tape_recording");
    if (!reduction_metallib_available()) return decline(R, op, "metallib_unavailable");
    const at::Tensor& in_t  = iter.tensor(1);
    const at::Tensor& out_t = iter.tensor(0);

    // The full reduction belongs to try_launch_reduction_metallib, which runs
    // first. Declining here keeps the two routes' counts disjoint.
    if (out_t.numel() <= 1) return decline(R, op, "full_reduction");
    if (!in_t.is_contiguous())  return decline_layout(R, "in_not_contiguous", in_t);
    if (!out_t.is_contiguous()) return decline_layout(R, "out_not_contiguous", out_t);
    const int64_t nd = in_t.dim();
    if (nd < 1) return decline(R, op, "rank_0");
    if (out_t.dim() != nd) return decline(R, op, "rank_differs");
    if (in_t.numel() <= 0) return decline(R, op, "empty");

    // Find the reduced block. An axis is reduced when the output holds extent 1
    // where the input does not; an axis of extent 1 in the INPUT reduces to
    // itself and belongs to whichever side keeps the block contiguous, so it is
    // treated as neither.
    int64_t first = -1, last = -1;
    for (int64_t d = 0; d < nd; ++d) {
        const bool reduced = out_t.size(d) == 1 && in_t.size(d) != 1;
        if (!reduced) continue;
        if (first < 0) first = d;
        last = d;
    }
    if (first < 0) return decline(R, op, "no_reduced_axis");
    for (int64_t d = first; d <= last; ++d)
        if (out_t.size(d) != 1 && in_t.size(d) != 1)
            return decline(R, op, "reduced_axes_not_adjacent");

    int64_t outer = 1, red = 1, inner = 1;
    for (int64_t d = 0;        d < first; ++d) outer *= in_t.size(d);
    for (int64_t d = first;    d <= last; ++d) red   *= in_t.size(d);
    for (int64_t d = last + 1; d < nd;    ++d) inner *= in_t.size(d);
    if (red <= 1) return decline(R, op, "no_reduced_axis");
    if (outer * inner != out_t.numel()) return decline(R, op, "shape_mismatch");
    // One threadgroup per output element, so the grid is the output size. Bound
    // it rather than letting an unsigned narrowing decide: a truncated grid
    // writes part of the output and leaves the rest as whatever the block held,
    // which is finite, plausible and wrong — the failure mode this project does
    // not tolerate.
    if (outer * inner > 0x7fffffffLL) return decline(R, op, "grid_too_large");

    // Every kernel name is a literal here, so `grep hagane_reduce_dim_ ` over
    // this file and over kernels/reduce.hip must produce the same set. A name
    // assembled from pieces would break that, and a typo in one would surface as
    // a launch failure at runtime rather than as a diff.
    //
    // `scaled` says whether the kernel takes the trailing float `scale` at all.
    // sum/mean/prod's float arms do; the integer arms and the whole
    // max/min/all/any set do not, because a parameter that can only ever be 1 is
    // a lie in the signature (the same rule kernels/reduce.hip states).
    const char* k = nullptr;
    bool scaled = false;
    const auto inty = in_t.scalar_type(), outty = out_t.scalar_type();
    switch (fam) {
    case kFamSum:
        // The full reducer's table minus the two arms that exist there only to
        // serve its SECOND pass (f32->bf16, f32->f16 read a float partials
        // buffer). This route is single-pass, so its pairs are the ones torch
        // actually presents: same-dtype, or a float32 accumulate-out.
        scaled = true;
        if      (inty == at::kFloat     && outty == at::kFloat)    k = "hagane_reduce_dim_sum_f32_f32";
        else if (inty == at::kBFloat16  && outty == at::kBFloat16) k = "hagane_reduce_dim_sum_bf16_bf16";
        else if (inty == at::kBFloat16  && outty == at::kFloat)    k = "hagane_reduce_dim_sum_bf16_f32";
        else if (inty == at::kHalf      && outty == at::kHalf)     k = "hagane_reduce_dim_sum_f16_f16";
        else if (inty == at::kHalf      && outty == at::kFloat)    k = "hagane_reduce_dim_sum_f16_f32";
        else if (inty == at::kLong      && outty == at::kLong)   { k = "hagane_reduce_dim_sum_i64_i64"; scaled = false; }
        else if (inty == at::kInt       && outty == at::kInt)    { k = "hagane_reduce_dim_sum_i32_i32"; scaled = false; }
        break;
    case kFamProd:
        // prod has no `mean` partner, so its scale is always 1 — but its float
        // arms keep sum's 6-argument signature rather than gaining a shorter
        // one, because two float kernels of different arity is a launch-site
        // branch that buys nothing.
        scaled = true;
        if      (inty == at::kFloat     && outty == at::kFloat)    k = "hagane_reduce_dim_prod_f32_f32";
        else if (inty == at::kBFloat16  && outty == at::kBFloat16) k = "hagane_reduce_dim_prod_bf16_bf16";
        else if (inty == at::kHalf      && outty == at::kHalf)     k = "hagane_reduce_dim_prod_f16_f16";
        else if (inty == at::kLong      && outty == at::kLong)   { k = "hagane_reduce_dim_prod_i64_i64"; scaled = false; }
        else if (inty == at::kInt       && outty == at::kInt)    { k = "hagane_reduce_dim_prod_i32_i32"; scaled = false; }
        break;
    case kFamMax:
        // amax/amin PRESERVE the dtype (TORCH_META_FUNC(amax) builds its
        // iterator on self's dtype with no promotion), so a pair that differs is
        // not this family's and declines. That also means the narrow integers
        // reach here as themselves, where the sum path never sees them.
        if (inty != outty) break;
        if      (inty == at::kFloat)    k = "hagane_reduce_dim_max_f32_f32";
        else if (inty == at::kBFloat16) k = "hagane_reduce_dim_max_bf16_bf16";
        else if (inty == at::kHalf)     k = "hagane_reduce_dim_max_f16_f16";
        else if (inty == at::kLong)     k = "hagane_reduce_dim_max_i64_i64";
        else if (inty == at::kInt)      k = "hagane_reduce_dim_max_i32_i32";
        else if (inty == at::kShort)    k = "hagane_reduce_dim_max_i16_i16";
        else if (inty == at::kChar)     k = "hagane_reduce_dim_max_i8_i8";
        else if (inty == at::kByte)     k = "hagane_reduce_dim_max_u8_u8";
        else if (inty == at::kBool)     k = "hagane_reduce_dim_max_bool_bool";
        break;
    case kFamMin:
        if (inty != outty) break;
        if      (inty == at::kFloat)    k = "hagane_reduce_dim_min_f32_f32";
        else if (inty == at::kBFloat16) k = "hagane_reduce_dim_min_bf16_bf16";
        else if (inty == at::kHalf)     k = "hagane_reduce_dim_min_f16_f16";
        else if (inty == at::kLong)     k = "hagane_reduce_dim_min_i64_i64";
        else if (inty == at::kInt)      k = "hagane_reduce_dim_min_i32_i32";
        else if (inty == at::kShort)    k = "hagane_reduce_dim_min_i16_i16";
        else if (inty == at::kChar)     k = "hagane_reduce_dim_min_i8_i8";
        else if (inty == at::kByte)     k = "hagane_reduce_dim_min_u8_u8";
        else if (inty == at::kBool)     k = "hagane_reduce_dim_min_bool_bool";
        break;
    case kFamAll:
        // all/any run with a Bool OUTPUT and the INPUT's own dtype
        // (get_allany_iter, ReduceOps.cpp), so the in/out pair genuinely differs
        // and the kernel reduces the predicate `x != 0` rather than the values.
        // The MLX route declines exactly this pair by name
        // (try_vendor_reduce_dim, `in_out_dtype_differ`), so a non-bool input is
        // not a route MOVE here — it is a case that had no owned path at all.
        //
        // uint8 is the ONE dtype torch does not convert: uint8 in, uint8 out,
        // valued 0/1. #1136 — that arm is a wrong-VALUE fix, not a route move.
        if (inty == at::kByte && outty == at::kByte) {
            k = "hagane_reduce_dim_all_u8_u8"; break;
        }
        if (outty != at::kBool) break;
        if      (inty == at::kFloat)    k = "hagane_reduce_dim_all_f32";
        else if (inty == at::kBFloat16) k = "hagane_reduce_dim_all_bf16";
        else if (inty == at::kHalf)     k = "hagane_reduce_dim_all_f16";
        else if (inty == at::kLong)     k = "hagane_reduce_dim_all_i64";
        else if (inty == at::kInt)      k = "hagane_reduce_dim_all_i32";
        else if (inty == at::kShort)    k = "hagane_reduce_dim_all_i16";
        else if (inty == at::kChar)     k = "hagane_reduce_dim_all_i8";
        else if (inty == at::kByte)     k = "hagane_reduce_dim_all_u8";
        else if (inty == at::kBool)     k = "hagane_reduce_dim_all_bool";
        break;
    case kFamAny:
        if (inty == at::kByte && outty == at::kByte) {
            k = "hagane_reduce_dim_any_u8_u8"; break;
        }
        if (outty != at::kBool) break;
        if      (inty == at::kFloat)    k = "hagane_reduce_dim_any_f32";
        else if (inty == at::kBFloat16) k = "hagane_reduce_dim_any_bf16";
        else if (inty == at::kHalf)     k = "hagane_reduce_dim_any_f16";
        else if (inty == at::kLong)     k = "hagane_reduce_dim_any_i64";
        else if (inty == at::kInt)      k = "hagane_reduce_dim_any_i32";
        else if (inty == at::kShort)    k = "hagane_reduce_dim_any_i16";
        else if (inty == at::kChar)     k = "hagane_reduce_dim_any_i8";
        else if (inty == at::kByte)     k = "hagane_reduce_dim_any_u8";
        else if (inty == at::kBool)     k = "hagane_reduce_dim_any_bool";
        break;
    }
    if (k == nullptr) return decline_dtypes(R, op, "dtype_pair_unowned", inty, outty);
    // Belt and braces, exactly as the full reducer carries it: an integral mean
    // is unreachable by construction (TORCH_META_FUNC2(mean,dim) rejects an
    // integral inferred dtype), and if one ever does arrive, decline rather
    // than silently drop the 1/N.
    if (is_mean && !scaled) return decline(R, "mean", "integral_mean_has_no_scale");

    void* dX   = in_t.data_ptr();
    void* dout = out_t.data_ptr();
    flush_or_commit_metallib_input(dX, in_t.numel() * in_t.element_size());

    int64_t c_outer = outer, c_red = red, c_inner = inner;
    float scale = is_mean ? static_cast<float>(1.0 / static_cast<double>(red)) : 1.0f;
    const int    at_[] = {1, 1, 1, 0, 0, 1};
    const size_t as_[] = {sizeof(int64_t), sizeof(int64_t), sizeof(int64_t),
                          0, 0, sizeof(float)};
    const int nargs = scaled ? 6 : 5;   // the integer arms take no `scale`
    void* args[] = {&c_outer, &c_red, &c_inner, dX, dout, &scale};
    dim3 b(256, 1, 1), g(static_cast<unsigned>(outer * inner), 1, 1);
    if (hagane_launch_kernel_mixed_tracked(k, g, b, 0, nullptr, args, at_, as_,
                                           nargs) != hipSuccess)
        return decline(R, op, "launch_failed");

    note_native_launch(k);
    haganeOpsMarkMetallibWrite(dout, static_cast<size_t>(out_t.numel() *
                                                         out_t.element_size()));
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
        // Operand order matches: torch's atan2(input, other) and MLX's
        // ArcTan2(a, b) both take y first and x second. The gate pins that on
        // the four quadrants plus the axes, where a swapped pair is a different
        // angle rather than a small error. `arctan2` is torch's alias for the
        // same stub, so one row serves both. Float only in the corpus
        // (float16/float32/bfloat16); an integer compute dtype declines at the
        // kernel lookup, which is where torch would have promoted anyway.
        {"atan2",       "ArcTan2"},
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

// The same convention for a caller that has a Scalar rather than an iterator
// operand — try_vendor_fill/arange/clamp and the scalar-binary unary rows.
inline bool vendor_scalar_bytes_of(const c10::Scalar& v, c10::ScalarType st,
                                   uint64_t* out) {
    *out = 0;
    switch (st) {
        case c10::ScalarType::Float:
            { float x = v.to<float>();             std::memcpy(out, &x, 4); return true; }
        case c10::ScalarType::Half:
            { auto x = v.to<c10::Half>();          std::memcpy(out, &x, 2); return true; }
        case c10::ScalarType::BFloat16:
            { auto x = v.to<c10::BFloat16>();      std::memcpy(out, &x, 2); return true; }
        case c10::ScalarType::Char:
            { int8_t x = v.to<int8_t>();           std::memcpy(out, &x, 1); return true; }
        case c10::ScalarType::Short:
            { int16_t x = v.to<int16_t>();         std::memcpy(out, &x, 2); return true; }
        case c10::ScalarType::Int:
            { int32_t x = v.to<int32_t>();         std::memcpy(out, &x, 4); return true; }
        case c10::ScalarType::Long:
            { int64_t x = v.to<int64_t>();         std::memcpy(out, &x, 8); return true; }
        case c10::ScalarType::Byte:
            { uint8_t x = v.to<uint8_t>();         std::memcpy(out, &x, 1); return true; }
        case c10::ScalarType::Bool:
            { bool x = v.to<bool>();               std::memcpy(out, &x, 1); return true; }
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

// #1023 — a decline site names itself.
//
// A third of the stashes ARDY still pays sit in routes below, which means a
// precondition here returns false and nobody knows which. Reading them did not
// answer it, and inferring it put two ops in the wrong bucket. So every
// `return false` becomes `return decline(route, op, why)` — same control flow,
// and with HAGANE_DECLINE_TRACE set the reason is counted in the runtime, where
// a probe can read it without going through torch.
//
// route_enter() is the other half: a route with zero ENTERs is not declining,
// it is never reached, and those two need different fixes but read identically
// from the source.
inline bool decline_trace_on() {
    static const bool on = haganeOpsDeclineTraceEnabled() != 0;
    return on;
}
inline bool decline(const char* route, const char* op, const char* why) {
    if (decline_trace_on()) haganeOpsDeclineNote(route, op, why);
    return false;
}
inline void route_enter(const char* route, const char* op) {
    if (decline_trace_on()) haganeOpsDeclineNote(route, op, "ENTER");
}
// A layout decline is only actionable with the layout: "extend the route" and
// "this caller should not be making that copy" are different fixes.
inline bool decline_layout(const char* route, const char* why,
                           const at::Tensor& t) {
    if (!decline_trace_on()) return false;
    std::string w = std::string(why) + "[";
    for (int64_t i = 0; i < t.dim(); ++i)
        w += (i ? "," : "") + std::to_string(t.size(i));
    w += "|";
    for (int64_t i = 0; i < t.dim(); ++i)
        w += (i ? "," : "") + std::to_string(t.stride(i));
    w += "]";
    haganeOpsDeclineNote(route, nullptr, w.c_str());
    return false;
}
// A dtype-mismatch decline is only actionable with the pair that mismatched:
// "route the cast" and "pick a mixed-dtype kernel" are different fixes and the
// bare reason cannot tell them apart.
inline bool decline_dtypes(const char* route, const char* op, const char* why,
                           c10::ScalarType from, c10::ScalarType to) {
    if (!decline_trace_on()) return false;
    std::string w = std::string(why) + "[" + c10::toString(from) + "->"
                  + c10::toString(to) + "]";
    haganeOpsDeclineNote(route, op, w.c_str());
    return false;
}

inline bool try_vendor_binary_flat(const char* torch_op, TensorIteratorBase& iter) {
    constexpr const char* R = "bin_flat";
    if (!vendor_elementwise_route_enabled()) return decline(R, torch_op, "route_off");
    const char* mlx_op = mlx_binary_op_name(torch_op);
    if (!mlx_op) return decline(R, torch_op, "no_mlx_name");
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (iter.ninputs() != 2) return decline(R, torch_op, "ninputs");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, torch_op, "corpus_unavailable");

    // The vv_/sv_/vs_ kernels are flat and unbroadcast: one linear index into
    // each operand. So every operand must be contiguous and the same shape as
    // the output, and the compute dtype must be the storage dtype.
    if (!iter.is_contiguous()) return decline(R, torch_op, "iter_not_contiguous");
    const auto compute = iter.common_dtype();
    const int dt = hagane_vendor_dtype(compute);
    if (dt < 0) return decline(R, torch_op, "compute_dtype");

    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to compute
    if (N > static_cast<int64_t>(UINT32_MAX)) return decline(R, torch_op, "numel_too_big");

    const void* ptrs[2] = {nullptr, nullptr};
    bool is_scalar[2] = {false, false};
    uint64_t scalars[2] = {0, 0};
    for (int i = 0; i < 2; i++) {
        const int arg = i + 1;
        if (iter.is_cpu_scalar(arg)) {
            if (!vendor_scalar_bytes(iter, arg, compute, &scalars[i]))
                return decline(R, torch_op, "scalar_dtype");
            is_scalar[i] = true;
            continue;
        }
        // A device operand must already be in the compute dtype and full-sized:
        // a promoted or broadcast operand would need MLX's g*_ strided kernels,
        // which this path does not select.
        if (iter.tensor(arg).scalar_type() != compute)
            return decline_dtypes(R, torch_op, "operand_dtype_promoted",
                                  iter.tensor(arg).scalar_type(), compute);
        if (iter.tensor(arg).sizes() != iter.tensor(0).sizes())
            return decline(R, torch_op, "operand_broadcast");
        ptrs[i] = iter.data_ptr(arg);
    }
    if (is_scalar[0] && is_scalar[1]) return decline(R, torch_op, "both_scalar");
    // A comparison writes bool while computing in the input dtype; the kernel
    // handles that, but the output BUFFER must then be bool-sized. Only accept
    // out dtype == compute dtype, or bool out for a comparison.
    const auto out_st = iter.dtype(0);
    if (out_st != compute && out_st != c10::ScalarType::Bool)
        return decline(R, torch_op, "out_dtype");
    if (out_st == c10::ScalarType::Bool && compute != c10::ScalarType::Bool) {
        // MLX names these by the INPUT dtype and writes bool — supported, but
        // only for the ops that actually produce bool.
        static constexpr const char* kBoolOut[] = {
            "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual"};
        bool ok = false;
        for (const char* b : kBoolOut) if (std::strcmp(mlx_op, b) == 0) ok = true;
        if (!ok) return decline(R, torch_op, "bool_out_not_cmp");
    }

    // The output span must come from the OUTPUT dtype: a comparison computes in
    // the input dtype and writes bool, and sizing the span from the input would
    // supersede/mark bytes past the end of the output — corrupting whatever
    // tensor's stash sits after it.
    const int dt_out = hagane_vendor_dtype(out_st);
    if (dt_out < 0) return decline(R, torch_op, "out_dtype_unspellable");
    if (haganeOpsVendorBinary(mlx_op, dt, dt_out, ptrs[0], ptrs[1], iter.data_ptr(0), N,
                              is_scalar[0] ? 1 : 0, is_scalar[1] ? 1 : 0,
                              scalars[0], scalars[1]) != HAGANE_OPS_SUCCESS)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(std::string("mlx:") + mlx_op);

    return true;
}

// The same binary op when an operand is BROADCAST, EXPANDED or non-contiguous —
// which is most of what a model does with one, and all of which the flat path
// above declines because its kernels index every operand with one linear index.
//
// MLX's g1_/g2_/g3_ variants take a stride per operand instead. The operand's
// stride along each of the output's dimensions comes straight from the tensor
// (right-aligned, 0 where it broadcasts), so an expanded operand needs no
// materialisation and a transposed one needs no copy. Adjacent dimensions then
// collapse wherever BOTH operands stay linear across them, which keeps the
// common cases inside the three dimensions the corpus carries.
//
// The OUTPUT must be contiguous — the kernel writes it with a linear index.
inline bool try_vendor_binary_g(const char* torch_op, TensorIteratorBase& iter) {
    constexpr const char* R = "bin_g";
    if (!vendor_elementwise_route_enabled()) return decline(R, torch_op, "route_off");
    const char* mlx_op = mlx_binary_op_name(torch_op);
    if (!mlx_op) return decline(R, torch_op, "no_mlx_name");
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (iter.ninputs() != 2) return decline(R, torch_op, "ninputs");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, torch_op, "corpus_unavailable");

    const auto compute = iter.common_dtype();
    const int dt = hagane_vendor_dtype(compute);
    if (dt < 0) return decline(R, torch_op, "compute_dtype");
    const auto out_st = iter.dtype(0);
    if (out_st != compute && out_st != c10::ScalarType::Bool)
        return decline(R, torch_op, "out_dtype");
    if (out_st == c10::ScalarType::Bool && compute != c10::ScalarType::Bool) {
        static constexpr const char* kBoolOut[] = {
            "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual"};
        bool ok = false;
        for (const char* b : kBoolOut) if (std::strcmp(mlx_op, b) == 0) ok = true;
        if (!ok) return decline(R, torch_op, "bool_out_not_cmp");
    }
    const int dt_out = hagane_vendor_dtype(out_st);
    if (dt_out < 0) return decline(R, torch_op, "out_dtype_unspellable");

    const at::Tensor& out = iter.tensor(0);
    if (!out.is_contiguous()) return decline(R, torch_op, "out_not_contiguous");
    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to compute
    if (N > static_cast<int64_t>(INT32_MAX)) return decline(R, torch_op, "numel_too_big");

    const auto osz = out.sizes();
    const int64_t n = static_cast<int64_t>(osz.size());
    if (n > 16) return decline(R, torch_op, "rank_gt16");

    // Each operand's stride along every OUTPUT dimension, right-aligned.
    int64_t st[2][16];
    const void* ptrs[2] = {nullptr, nullptr};
    bool is_scalar[2] = {false, false};
    uint64_t scalars[2] = {0, 0};
    int src_dt[2] = {-1, -1};   // >=0: operand is stored in a dtype needing a cast
    for (int i = 0; i < 2; i++) {
        const int arg = i + 1;
        if (iter.is_cpu_scalar(arg)) {
            if (!vendor_scalar_bytes(iter, arg, compute, &scalars[i]))
                return decline(R, torch_op, "scalar_dtype");
            is_scalar[i] = true;
            for (int64_t k = 0; k < n; ++k) st[i][k] = 0;
            continue;
        }
        const at::Tensor& t = iter.tensor(arg);
        if (t.scalar_type() != compute) {
            // TensorIterator's own CUDA kernels cast this operand on load, so
            // the tensor stays in its own dtype and the op computes in another.
            // MLX has no mixed-dtype binary kernel, so the runtime casts the
            // operand with MLX's v_copy first; a pair the corpus does not carry
            // declines there and lands here as kernel_dispatch.
            src_dt[i] = hagane_vendor_dtype(t.scalar_type());
            if (src_dt[i] < 0)
                return decline_dtypes(R, torch_op, "operand_dtype_unspellable",
                                      t.scalar_type(), compute);
        }
        const int64_t tn = t.dim();
        if (tn > n) return decline(R, torch_op, "operand_rank_gt_out");
        for (int64_t k = 0; k < n; ++k) {
            const int64_t j = k - (n - tn);
            if (j < 0)                    { st[i][k] = 0; continue; }
            if (t.size(j) == osz[k])        st[i][k] = t.stride(j);
            else if (t.size(j) == 1)        st[i][k] = 0;
            else return decline(R, torch_op, "not_broadcastable");
            if (st[i][k] < 0) return decline(R, torch_op, "negative_stride");
        }
        // The tensor's own base, since the strides above are the tensor's own.
        ptrs[i] = t.const_data_ptr();
    }
    if (is_scalar[0] && is_scalar[1]) return decline(R, torch_op, "both_scalar");

    // Collapse: merge an inner dimension into the group outside it only when
    // EVERY operand's index stays linear across the pair.
    int64_t ext[16], sa[16], sb[16];
    int ng = 0;
    for (int64_t k = 0; k < n; ++k) {
        if (osz[k] == 1) continue;
        const int64_t e = osz[k], A = st[0][k], B = st[1][k];
        if (ng > 0 && sa[ng - 1] == A * e && sb[ng - 1] == B * e) {
            ext[ng - 1] *= e; sa[ng - 1] = A; sb[ng - 1] = B;
        } else {
            ext[ng] = e; sa[ng] = A; sb[ng] = B; ++ng;
        }
    }
    if (ng == 0) { ext[0] = 1; sa[0] = 0; sb[0] = 0; ng = 1; }   // all dims were 1
    if (ng > 3) return decline(R, torch_op, "groups_gt3");       // beyond g3_

    // A promoted operand is cast over its SPAN, and its strides are then reused
    // against the scratch — so a broadcast one costs its own size, not the
    // output's. The span can EXCEED the output when the operand is a narrow
    // slice of a wide tensor (extent 1 x k, stride M x 1); casting more elements
    // than the op computes is not a win, so decline under its own name and let
    // the count say whether that shape ever occurs.
    for (int i = 0; i < 2; i++) {
        if (src_dt[i] < 0) continue;
        const int64_t* s = i == 0 ? sa : sb;
        int64_t span = 1;
        for (int g = 0; g < ng; ++g) span += (ext[g] - 1) * s[g];
        if (span > N) return decline(R, torch_op, "cast_span_gt_numel");
    }

    if (haganeOpsVendorBinaryGCast(mlx_op, dt, dt_out, ptrs[0], ptrs[1],
                                   iter.data_ptr(0),
                                   is_scalar[0] ? 1 : 0, is_scalar[1] ? 1 : 0,
                                   scalars[0], scalars[1],
                                   ext, sa, sb, ng,
                                   src_dt[0], src_dt[1]) != HAGANE_OPS_SUCCESS)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(std::string("mlx:g_") + mlx_op);
    return true;
}

// The flat kernels first — they are cheaper to set up and cover the shape most
// ops actually have; the strided ones catch everything else.
//
// Only ENTER is recorded here: the flat path declining is NORMAL (it is the
// narrow one), so the decline that matters is bin_g's, and its reason counts
// already sum to this route's total declines.
inline bool try_vendor_binary(const char* torch_op, TensorIteratorBase& iter) {
    route_enter("bin", torch_op);
    return try_vendor_binary_flat(torch_op, iter)
        || try_vendor_binary_g(torch_op, iter);
}

// Which MLX op an alpha-carrying stub actually computes. nullptr means "no
// single kernel does this", and the caller keeps the MLX path.
inline const char* alpha_effective_op(const char* op, const c10::Scalar& alpha) {
    const double a = alpha.toDouble();
    if (a == 1.0) return op;
    if (a == -1.0) {
        // Exact for every dtype MLX carries: integer wraparound is the same
        // two's-complement subtraction on both sides, and there is no rounding
        // step to differ on.
        if (std::strcmp(op, "add") == 0) return "sub";
        if (std::strcmp(op, "sub") == 0) return "add";
    }
    return nullptr;
}

// #1026 — can `out = a + alpha*b` be evaluated in FLOAT32 without changing the
// answer? That is the question the MLX C-ABI actually asks, because
// `haganeOpsAdd` takes a `float alpha` and builds it as a float32 mx::array, so
// the alpha multiply happens in float32 whatever the tensors are.
//
//   float dtypes  YES, and it is what parity requires: torch's CUDA kernel
//                 accumulates this in at::acc_type<scalar_t, true>, which is
//                 float for Half, BFloat16 and Float alike. The only extra
//                 condition is that the alpha itself survive double -> float.
//   integer       NO. torch computes in int64; float32 carries 24 bits. Neither
//                 the alpha nor the operand values are safe, and the values
//                 cannot be checked from here without reading device memory.
//   complex       NO. toFloat() would drop the imaginary part outright.
//
// alpha == 1 and alpha == -1 never reach this test — alpha_effective_op maps
// those onto MLX's Add and Subtract exactly — so it only ever judges a general
// alpha. Measured on the shipped build: torch.add(int64, 3, alpha=2**24+1) came
// back 4 off, silently.
inline bool alpha_expressible_in_float(c10::ScalarType st,
                                       const c10::Scalar& alpha) {
    if (c10::isIntegralType(st, /*includeBool=*/true)) return false;
    if (alpha.isComplex()) return false;
    return static_cast<double>(alpha.toFloat()) == alpha.toDouble();
}

// A general alpha that reaches the MLX path records NOTHING today, so its
// stashes arrive with no named reason — 60/step on ARDY with an empty decline
// table. Name them. The three facts here are the ones that decide whether the
// caller could be folded onto a vs_ kernel instead:
//
//   b=cpu_scalar   alpha*b is ONE number and can be folded on the host.
//                  b=tensor means it cannot, whatever else is true.
//   dtype          float32 and the integers can fold; Half/BFloat16 must not,
//                  because CUDA's ufunc computes those in opmath_t = float
//                  (UfuncCUDA_add.cu: CUDAFunctorOnSelf_add holds other_ and
//                  alpha_ at opmath_t) while a fold has to materialise the
//                  product in the narrow dtype.
//   prod_exact     whether alpha*b is exactly representable in the compute
//                  dtype. Only then is `a + p` a single rounding, which is
//                  what CUDA's fma-contracted `self + alpha*other` computes.
//
// A count alone would not distinguish "51 foldable" from "51 that cannot be",
// which is the whole lesson of 1.7a.
inline void note_general_alpha(const char* op, TensorIteratorBase& iter,
                               const c10::Scalar& alpha) {
    if (!decline_trace_on()) return;
    const auto ct = iter.common_dtype();
    std::string w = std::string("general_alpha[") + c10::toString(ct);
    const bool b_scalar = iter.is_cpu_scalar(2);
    w += b_scalar ? "|b=cpu_scalar" : "|b=tensor";
    if (iter.is_cpu_scalar(1)) w += "|a=cpu_scalar";
    if (b_scalar) {
        bool exact = false;
        if (c10::isIntegralType(ct, /*includeBool=*/true)) {
            exact = true;                    // integer multiply, exact mod 2^N
        } else if (ct == c10::ScalarType::Float) {
            // double holds the product of two float32s exactly, so this is a
            // representability test and not an approximation of one
            const double p = static_cast<double>(alpha.toFloat())
                           * static_cast<double>(iter.scalar_value<float>(2));
            exact = (static_cast<double>(static_cast<float>(p)) == p);
        }
        w += exact ? "|prod_exact" : "|prod_inexact";
    }
    w += "]";
    haganeOpsDeclineNote("bin_alpha", op, w.c_str());
}

// torch's unary stub name -> MLX's operator name. nullptr means "not ours".
//
// Availability was checked against the shipped corpus, but availability is not
// the bar — SEMANTICS are, and two entries here were checked rather than
// assumed:
//
//   round  MLX uses metal::rint, i.e. half-to-even, which is what torch's
//          round does. Pinned on exact .5 inputs by the gate, because random
//          data never lands on the tie that would expose a wrong mode.
//   sign   MLX computes `(x > 0) - (x < 0)`, which yields 0 for NaN. torch does
//          the same — verified against stock CPU torch, which returns
//          [-1, 0, 1, 0, 1] for [-2, 0, 3, nan, inf]. (The intuition that torch
//          propagates NaN here is wrong; sgn only differs for complex, which
//          hagane_vendor_dtype declines outright.)
//
// bitwise_not on BOOL is remapped to LogicalNot below, and that is an EXACT
// mapping rather than a substitution: torch's aten schema defines `~` on a Bool
// tensor AS logical negation, and MLX spells logical negation LogicalNot. MLX
// instantiates BitwiseInvert for integers only, so v_BitwiseInvertbool_bool_ is
// not in the corpus while v_LogicalNotbool_bool_ is — checked against the
// shipped metallib, not inferred.
//
// This reverses an earlier note here which said the decline WAS the fix. That
// was written before the corpus was searched for the operation under MLX's name
// for it, and it cost 21 stashes a step. The general form of the mistake: an op
// missing under one name is not an op MLX lacks. The decline path stays and is
// still load-bearing — the registry builds a pipeline eagerly and threw on a
// missing function, so `~mask` used to abort mid-generate instead of falling
// back, and vendor_elementwise's lookup catching and memoising that miss is
// what makes "decline honestly" true at all.
//
// Not carried by MLX at all, so absent by nature: erfc, frac, lgamma, mish,
// sinc, signbit, trunc, silu, hardsigmoid, exp2.
//
// `reciprocal` was on that list and did not belong there. MLX ships no
// Reciprocal kernel, but 1/x IS sv_Divide, which it ships for every dtype — so
// the op was absent under torch's name for it, not absent from the corpus. It
// is handled by unary_as_scalar_binary below rather than by this table, because
// what differs is the dispatch and not just the name. Same correction the
// bitwise_not note above records, on the same table, one slice later.
inline const char* mlx_unary_op_name(const char* op) {
    struct Row { const char* torch_name; const char* mlx_name; };
    static constexpr Row kMap[] = {
        {"abs",         "Abs"},
        {"acos",        "ArcCos"},
        {"asin",        "ArcSin"},
        {"atan",        "ArcTan"},
        {"ceil",        "Ceil"},
        {"cos",         "Cos"},
        {"cosh",        "Cosh"},
        {"erf",         "Erf"},
        {"exp",         "Exp"},
        {"expm1",       "Expm1"},
        {"floor",       "Floor"},
        {"log",         "Log"},
        {"log10",       "Log10"},
        {"log1p",       "Log1p"},
        {"log2",        "Log2"},
        {"neg",         "Negative"},
        {"round",       "Round"},
        {"rsqrt",       "Rsqrt"},
        {"sigmoid",     "Sigmoid"},
        {"sign",        "Sign"},
        {"sgn",         "Sign"},
        {"sin",         "Sin"},
        {"sinh",        "Sinh"},
        {"sqrt",        "Sqrt"},
        {"tan",         "Tan"},
        {"tanh",        "Tanh"},
        {"bitwise_not", "BitwiseInvert"},
        {"logical_not", "LogicalNot"},
    };
    for (const auto& r : kMap)
        if (std::strcmp(op, r.torch_name) == 0) return r.mlx_name;
    return nullptr;
}

// A unary op MLX has no kernel for, but which IS one of MLX's BINARY kernels
// with a constant left operand. Separate from mlx_unary_op_name because the
// dispatch differs and not just the name: these run binary_sv, which is
// `Op()(a[0], b[index])`, so the tensor is the SECOND operand and the constant
// crosses as raw bits in the compute dtype. Spelling it as a name remap would
// put the tensor first and compute x/1.
//
// reciprocal is the only row. The shipped metallib carries no Reciprocal; it
// carries sv_Divide for all 13 dtypes.
//
// Measured before this was written, exhaustively over every fp16 and bf16 bit
// pattern and over 2^20 float32 ones: sv_Divide(1, x) is BIT-IDENTICAL to the
// mx::reciprocal fallback it replaces, on all three float dtypes. The row is a
// pure stash removal and changes no value anywhere.
//
// That is also why upstream's opmath question does not arise here. c10::Half's
// operator/ is `static_cast<float>(a) / static_cast<float>(b)`, so torch divides
// half in float32 and rounds TWICE while sv_Dividefloat16 divides natively and
// rounds once — but for a reciprocal those never disagree, because float32
// carries 2p+2 bits relative to both fp16 (24 >= 24) and bf16 (24 >= 18), which
// is exactly when double rounding is harmless. 0 disagreeing values out of
// 65536 for each, measured rather than left standing on the bound.
struct UnaryAsScalarBinary {
    const char* torch_name;
    const char* mlx_op;
    double lhs;                          // the constant LEFT operand
};

inline const UnaryAsScalarBinary* unary_as_scalar_binary(const char* op) {
    static constexpr UnaryAsScalarBinary kMap[] = {
        {"reciprocal", "Divide", 1.0},
    };
    for (const auto& r : kMap)
        if (std::strcmp(op, r.torch_name) == 0) return &r;
    return nullptr;
}

inline bool try_vendor_unary(const char* torch_op, TensorIteratorBase& iter) {
    constexpr const char* R = "unary";
    route_enter(R, torch_op);
    if (!vendor_elementwise_route_enabled()) return decline(R, torch_op, "route_off");
    const char* mlx_op = mlx_unary_op_name(torch_op);
    // A scalar-binary row survives the name check and then takes every guard
    // below unchanged: MLX's sv_ kernels are flat and unbroadcast in exactly the
    // way the unary ones are, and walk the same single operand.
    const UnaryAsScalarBinary* sb = mlx_op ? nullptr : unary_as_scalar_binary(torch_op);
    if (!mlx_op && !sb) return decline(R, torch_op, "no_mlx_name");
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (iter.ninputs() != 1) return decline(R, torch_op, "ninputs");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, torch_op, "corpus_unavailable");

    // unary_v is flat and unbroadcast: one linear index into each operand.
    if (!iter.is_contiguous()) return decline(R, torch_op, "iter_not_contiguous");

    // The COMPUTE dtype is the output's. MLX names a unary kernel by BOTH
    // dtypes and instantiates only matched pairs, so the kernel we name always
    // computes in one type — but torch does not hand us one type.
    //
    // #1031: torch promotes a unary op's OUTPUT and leaves the INPUT alone,
    // because upstream's gpu_kernel casts on the load. `reciprocal(int64)` and
    // `sqrt(int64)` therefore arrive Long-in / Float-out, and this used to
    // decline all of them (11 per ARDY step). The input is now CAST first with
    // MLX's own v_copy, in the runtime, which is 1.4's operand-promotion pass
    // one family over — and it guards the whole unary family, not one op.
    //
    // A BOOL output is still a decline and must stay one: `logical_not` over a
    // float input is a different kernel, not a cast of this one, and casting the
    // input would compute the wrong thing rather than fail to compute.
    const auto st = iter.dtype(0);
    const auto in_st = iter.dtype(1);
    if (st == c10::ScalarType::Bool && in_st != st)
        return decline_dtypes(R, torch_op, "bool_out_from_other_in", in_st, st);
    const int dt = hagane_vendor_dtype(st);
    if (dt < 0) return decline(R, torch_op, "dtype");
    int src_dt = -1;
    if (in_st != st) {
        src_dt = hagane_vendor_dtype(in_st);
        if (src_dt < 0)
            return decline_dtypes(R, torch_op, "in_dtype_unspellable", in_st, st);
    }

    // `~` on Bool IS logical negation in torch, and that is what MLX calls
    // LogicalNot — see the note on mlx_unary_op_name. Integer bitwise_not keeps
    // BitwiseInvert. Remapped here rather than in the table because the table
    // does not see a dtype, and moving the name lookup down to where it does
    // would change which decline name fires for every op that is not ours.
    if (st == c10::ScalarType::Bool && std::strcmp(torch_op, "bitwise_not") == 0)
        mlx_op = "LogicalNot";

    if (iter.tensor(1).sizes() != iter.tensor(0).sizes())
        return decline(R, torch_op, "operand_broadcast");

    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to compute
    if (N > static_cast<int64_t>(UINT32_MAX)) return decline(R, torch_op, "numel_too_big");

    if (sb) {
        // Divide on an integer COMPUTE dtype is INTEGER division, which is not
        // what reciprocal means. torch promotes the output above the kernel
        // (AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2), so `st` is float even
        // when the INPUT is Long — which is exactly the #1031 case handled
        // below. This guard stays for a compute dtype that is genuinely
        // integral: decline rather than truncate.
        if (!c10::isFloatingType(st))
            return decline(R, torch_op, "scalar_binary_needs_float");
        uint64_t lhs_bits = 0;
        if (!vendor_scalar_bytes_of(c10::Scalar(sb->lhs), st, &lhs_bits))
            return decline(R, torch_op, "scalar_binary_lhs");
        if (src_dt >= 0) {
            // Promoted input: the cast pass lives on the binary_g entry, which
            // already allocates its own scratch and runs v_copy before the op
            // (1.4). The flat sv_ path below stays byte-identical for the
            // un-promoted case, so 1.7c's bit-identity claim is untouched.
            const int64_t ext[1] = {N};
            const int64_t sa[1] = {0};      // the scalar rides at stride 0
            const int64_t sbst[1] = {1};    // the tensor is contiguous
            if (haganeOpsVendorBinaryGCast(sb->mlx_op, dt, dt, /*a=*/nullptr,
                                           iter.data_ptr(1), iter.data_ptr(0),
                                           /*a_is_scalar=*/1, /*b_is_scalar=*/0,
                                           lhs_bits, 0, ext, sa, sbst, 1,
                                           /*a_src_dtype=*/-1, src_dt)
                != HAGANE_OPS_SUCCESS)
                return decline(R, torch_op, "kernel_dispatch");
            note_native_launch(std::string("mlx:g_") + sb->mlx_op);
            return true;
        }
        if (haganeOpsVendorBinary(sb->mlx_op, dt, dt, /*a=*/nullptr,
                                  iter.data_ptr(1), iter.data_ptr(0), N,
                                  /*a_is_scalar=*/1, /*b_is_scalar=*/0,
                                  lhs_bits, 0) != HAGANE_OPS_SUCCESS)
            return decline(R, torch_op, "kernel_dispatch");
        note_native_launch(std::string("mlx:sv_") + sb->mlx_op);
        return true;
    }

    if (haganeOpsVendorUnaryCast(mlx_op, dt, dt, iter.data_ptr(1),
                                 iter.data_ptr(0), N, src_dt) != HAGANE_OPS_SUCCESS)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(std::string("mlx:") + mlx_op);
    return true;
}

// masked_fill_ through MLX's Select. self is contiguous and is both operands;
// the mask may be broadcast, which is the whole reason this uses MLX's strided
// g*_ variants instead of the flat ones the binary/unary routes use.
//
// The mask's stride along each of self's dimensions is 0 where it broadcasts.
// Adjacent dimensions then collapse whenever the mask index stays linear across
// them (outer stride == inner stride x inner extent), which is what turns a
// [B,H,L,S] tensor with an [L,S] mask into two dimensions rather than four. MLX
// carries g1_/g2_/g3_, so three groups is the limit — enough for same-shape (1),
// a leading broadcast (2) and an interior one like [B,1,L,S] (3).
inline bool try_vendor_masked_fill(at::Tensor& self, const at::Tensor& mask,
                                   const c10::Scalar& value) {
    constexpr const char* R = "masked_fill";
    route_enter(R, nullptr);
    if (!vendor_elementwise_route_enabled()) return decline(R, nullptr, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, nullptr, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, nullptr, "corpus_unavailable");

    if (mask.scalar_type() != c10::ScalarType::Bool)
        return decline(R, nullptr, "mask_not_bool");
    if (!self.is_contiguous()) return decline(R, nullptr, "self_not_contiguous");
    if (!mask.is_contiguous()) return decline(R, nullptr, "mask_not_contiguous");
    const int dt = hagane_vendor_dtype(self.scalar_type());
    if (dt < 0) return decline(R, nullptr, "dtype");
    if (self.numel() <= 0) return true;           // nothing to fill

    const auto ssz = self.sizes();
    const auto msz = mask.sizes();
    const int64_t n = static_cast<int64_t>(ssz.size());
    const int64_t mn = static_cast<int64_t>(msz.size());
    if (mn > n) return decline(R, nullptr, "mask_rank_gt_self");

    // Mask strides in the OUTPUT's index space, right-aligned. mask is
    // contiguous, so its own strides are the running products of its sizes.
    std::vector<int64_t> mstride_of_dim(n, 0);
    {
        int64_t acc = 1;
        for (int64_t j = mn - 1; j >= 0; --j) {
            const int64_t i = j + (n - mn);
            if (msz[j] == ssz[i])      mstride_of_dim[i] = acc;
            else if (msz[j] == 1)      mstride_of_dim[i] = 0;
            else                       return decline(R, nullptr, "mask_not_broadcastable");
            acc *= msz[j];
        }
    }

    // Collapse, outermost first. A dimension of extent 1 contributes nothing.
    int64_t ext[8], str[8];
    int ng = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (ssz[i] == 1) continue;
        const int64_t e = ssz[i], s = mstride_of_dim[i];
        if (ng > 0 && str[ng - 1] == s * e) {     // index stays linear: merge
            ext[ng - 1] *= e;
            str[ng - 1] = s;
        } else {
            if (ng == 8) return decline(R, nullptr, "groups_gt8");
            ext[ng] = e; str[ng] = s; ++ng;
        }
    }
    if (ng == 0) { ext[0] = 1; str[0] = 0; ng = 1; }   // all dims were 1
    if (ng > 3) return decline(R, nullptr, "groups_gt3");   // beyond g3_

    uint64_t bits = 0;
    if (!vendor_scalar_bytes_of(value, self.scalar_type(), &bits))
        return decline(R, nullptr, "scalar_dtype");

    if (haganeOpsVendorMaskedFill(dt, mask.const_data_ptr(), self.data_ptr(),
                                  bits, ext, str, ng) != HAGANE_OPS_SUCCESS)
        return decline(R, nullptr, "kernel_dispatch");
    note_native_launch("mlx:Select");
    return true;
}

// `where` through the SAME MLX Select kernels masked_fill_ uses — the general
// form, where all three operands carry their own strides instead of two of them
// being an immediate and a contiguous self.
//
// Geometry is taken the way try_vendor_binary_g takes it: each operand's stride
// along every OUTPUT dimension, right-aligned, then collapsed outermost-first
// wherever EVERY operand's index stays linear across the pair. That is what
// makes a broadcast operand free — and it is not a corner case here. ARDY's 71
// are 20 top-level `where.self` plus 51 nested inside hagane_index_on_device's
// negative-index fixup (HaganeOps.cpp:2651), whose third operand is an
// as_strided view of the caller's index tensor and may be broadcast.
//
// MLX instantiates Select by a SINGLE type name, so all three value operands
// and the output share one dtype and the condition is bool. torch guarantees
// exactly that above the stub — where_self_out casts self/other to result_type
// and TORCH_CHECKs the condition is kBool — but the stub is public, so this
// CHECKS rather than assumes and declines under its own name if it ever differs.
inline bool try_vendor_where(TensorIteratorBase& iter) {
    constexpr const char* R = "ternary";
    constexpr const char* OP = "where";
    route_enter(R, OP);
    if (!vendor_elementwise_route_enabled()) return decline(R, OP, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, OP, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, OP, "corpus_unavailable");
    if (iter.ninputs() != 3) return decline(R, OP, "ninputs");

    const auto out_st = iter.dtype(0);
    const int dt = hagane_vendor_dtype(out_st);
    if (dt < 0) return decline(R, OP, "dtype");
    if (iter.dtype(1) != c10::ScalarType::Bool)
        return decline(R, OP, "cond_not_bool");
    // One kernel, one tname: a mixed value pair has no Select instantiation and
    // must not be composed out of a cast the caller did not ask for.
    if (iter.dtype(2) != out_st || iter.dtype(3) != out_st)
        return decline_dtypes(R, OP, "operand_dtype_differ", iter.dtype(2), out_st);
    if (iter.is_cpu_scalar(1)) return decline(R, OP, "cond_is_cpu_scalar");

    const at::Tensor& out = iter.tensor(0);
    if (!out.is_contiguous()) return decline(R, OP, "out_not_contiguous");
    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to compute
    if (N > static_cast<int64_t>(INT32_MAX)) return decline(R, OP, "numel_too_big");

    const auto osz = out.sizes();
    const int64_t n = static_cast<int64_t>(osz.size());
    if (n > 16) return decline(R, OP, "rank_gt16");

    int64_t st[3][16];
    const void* ptrs[3] = {nullptr, nullptr, nullptr};
    bool is_scalar[3] = {false, false, false};
    uint64_t scalars[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        const int arg = i + 1;
        if (iter.is_cpu_scalar(arg)) {
            if (!vendor_scalar_bytes(iter, arg, out_st, &scalars[i]))
                return decline(R, OP, "scalar_dtype");
            is_scalar[i] = true;
            for (int64_t k = 0; k < n; ++k) st[i][k] = 0;
            continue;
        }
        const at::Tensor& t = iter.tensor(arg);
        const int64_t tn = t.dim();
        if (tn > n) return decline(R, OP, "operand_rank_gt_out");
        for (int64_t k = 0; k < n; ++k) {
            const int64_t j = k - (n - tn);
            if (j < 0)                    { st[i][k] = 0; continue; }
            if (t.size(j) == osz[k])        st[i][k] = t.stride(j);
            else if (t.size(j) == 1)        st[i][k] = 0;
            else return decline(R, OP, "not_broadcastable");
            if (st[i][k] < 0) return decline(R, OP, "negative_stride");
        }
        ptrs[i] = t.const_data_ptr();
    }

    int64_t ext[16], sc[16], sx[16], sy[16];
    int ng = 0;
    for (int64_t k = 0; k < n; ++k) {
        if (osz[k] == 1) continue;
        const int64_t e = osz[k], C = st[0][k], X = st[1][k], Y = st[2][k];
        if (ng > 0 && sc[ng - 1] == C * e && sx[ng - 1] == X * e &&
            sy[ng - 1] == Y * e) {
            ext[ng - 1] *= e; sc[ng - 1] = C; sx[ng - 1] = X; sy[ng - 1] = Y;
        } else {
            ext[ng] = e; sc[ng] = C; sx[ng] = X; sy[ng] = Y; ++ng;
        }
    }
    if (ng == 0) { ext[0] = 1; sc[0] = 0; sx[0] = 0; sy[0] = 0; ng = 1; }
    if (ng > 3) return decline(R, OP, "groups_gt3");   // beyond g3_

    if (haganeOpsVendorTernary(dt, ptrs[0], ptrs[1], ptrs[2], iter.data_ptr(0),
                               is_scalar[1] ? 1 : 0, is_scalar[2] ? 1 : 0,
                               scalars[1], scalars[2],
                               ext, sc, sx, sy, ng) != HAGANE_OPS_SUCCESS)
        return decline(R, OP, "kernel_dispatch");
    note_native_launch("mlx:g_Select");
    return true;
}

// A FULL reduction to one element through MLX's all_reduce kernels — torch's
// `max()`, `min()`, `all()`, `any()` over a whole tensor.
//
// PARTIAL reductions decline: `all(x, dim=-1)` is MLX's row_reduce family, a
// different kernel taking a whole reduction plan rather than a length, and it
// picks between three kernels by row size. That is its own slice; here it keeps
// the MLX fallback and is named in the decline table so the count is visible.
inline bool try_vendor_reduce_all(const char* torch_op, const char* mlx_op,
                                  const at::TensorBase& self,
                                  const at::TensorBase& result) {
    constexpr const char* R = "reduce_all";
    route_enter(R, torch_op);
    if (!mlx_op) return decline(R, torch_op, "no_mlx_reduce_op");
    if (!vendor_elementwise_route_enabled()) return decline(R, torch_op, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, torch_op, "corpus_unavailable");

    if (result.numel() != 1) return decline(R, torch_op, "not_full_reduction");
    if (!self.is_contiguous()) return decline(R, torch_op, "in_not_contiguous");
    // MLX remaps the in/out pair only for sum and prod, which are not routed
    // here; for max/min/and/or the two must already agree.
    if (self.scalar_type() != result.scalar_type())
        return decline_dtypes(R, torch_op, "in_out_dtype_differ",
                              self.scalar_type(), result.scalar_type());
    const int dt = hagane_vendor_dtype(self.scalar_type());
    if (dt < 0) return decline(R, torch_op, "dtype");
    const int64_t N = self.numel();
    if (N <= 0) return decline(R, torch_op, "empty");

    if (haganeOpsVendorReduceAll(mlx_op, dt, self.const_data_ptr(),
                                 result.data_ptr(), N) != HAGANE_OPS_SUCCESS)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(std::string("mlx:all_reduce_") + mlx_op);
    return true;
}

// The PARTIAL reduction — `all(dim=)`, `any(dim=)`, `amax(dim=)`, `amin(dim=)`
// — through MLX's row_reduce family, dispatched into the caller's own block.
// #1010 1.9b; 1.9 shipped the full-reduce arm and named this one as its own
// slice because the row kernels take a reduction PLAN rather than a length.
//
// Bit-identical to the path it replaces by construction: `haganeOpsAll`'s
// partial arm is `mx::all(in, axes, true)`, which reaches the same
// row_reduce_general_dispatch. Only who owns the output changes.
//
// `self` and `result` are the reduction iterator's operands, so `result`
// carries the INPUT's rank with extent 1 at every reduced axis — that is
// review_reduce_result's doing (ReduceOpsUtils.h:170), and it is what lets the
// reduced axes be identified here without the dim list being threaded through
// every stub signature.
//
// Scope is "contiguous input, reduced axes all TRAILING", which is exactly
// MLX's ContiguousReduce plan with one merged row and therefore two numbers.
// A reduced axis in the middle is a col_reduce with a whole different kernel
// family and declines by name.
inline bool try_vendor_reduce_dim(const char* torch_op, const char* mlx_op,
                                  const at::TensorBase& self,
                                  const at::TensorBase& result) {
    constexpr const char* R = "reduce_dim";
    route_enter(R, torch_op);
    if (!mlx_op) return decline(R, torch_op, "no_mlx_reduce_op");
    if (!vendor_elementwise_route_enabled()) return decline(R, torch_op, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, torch_op, "corpus_unavailable");

    // The full reduction belongs to try_vendor_reduce_all, which runs first.
    // Declining here keeps the two routes' ENTER counts disjoint.
    if (result.numel() <= 1) return decline(R, torch_op, "full_reduction");
    if (!self.is_contiguous()) return decline(R, torch_op, "in_not_contiguous");
    if (!result.is_contiguous()) return decline(R, torch_op, "out_not_contiguous");
    // `all`/`any` over a non-bool input lands here: get_allany_iter keeps the
    // INPUT's dtype on CUDA (ReduceOps.cpp) while the result is Bool, and MLX's
    // remap_reduce_types would map that pair to a bool-output kernel with its
    // own conversion rule. Not a numerics decision to take inside a stash
    // removal — decline, and the MLX fallback answers as it does today.
    if (self.scalar_type() != result.scalar_type())
        return decline_dtypes(R, torch_op, "in_out_dtype_differ",
                              self.scalar_type(), result.scalar_type());
    const int dt = hagane_vendor_dtype(self.scalar_type());
    if (dt < 0) return decline(R, torch_op, "dtype");

    const int64_t nd = self.dim();
    if (nd < 1) return decline(R, torch_op, "rank_0");
    if (result.dim() != nd) return decline(R, torch_op, "rank_differs");
    if (self.numel() <= 0) return decline(R, torch_op, "empty");

    // Walk in from the last axis while it is reduced (or extent 1, which
    // reduces to itself either way and so belongs to whichever side keeps the
    // block trailing). What is left below `p` must contain no reduced axis.
    int64_t row_size = 1;
    int64_t p = nd;
    while (p > 0) {
        const int64_t d = p - 1;
        const bool unit = self.size(d) == 1;
        const bool reduced = result.size(d) == 1 && !unit;
        if (!reduced && !unit) break;
        row_size *= self.size(d);
        --p;
    }
    for (int64_t d = 0; d < p; ++d)
        if (result.size(d) == 1 && self.size(d) != 1)
            return decline(R, torch_op, "reduced_axis_not_trailing");
    if (row_size <= 1) return decline(R, torch_op, "no_reduced_axis");

    const int64_t n_rows = self.numel() / row_size;
    if (n_rows != result.numel()) return decline(R, torch_op, "shape_mismatch");
    if (n_rows <= 0) return decline(R, torch_op, "empty");

    if (haganeOpsVendorReduceDim(mlx_op, dt, self.const_data_ptr(),
                                 result.data_ptr(), n_rows,
                                 row_size) != HAGANE_OPS_SUCCESS)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(std::string("mlx:row_reduce_") + mlx_op);
    return true;
}

// cumsum / cumprod through MLX's own scan kernels, dispatched into the caller's
// block. The fallback already runs these exact kernels via mx::cumsum /
// mx::cumprod, so this is bit-identical to it and changes only who owns the
// output — the same story as 1.7c's reciprocal.
//
// `dim` arrives already normalised to [0, ndim). The tensor is described to the
// runtime by the scanned axis alone: its extent, its stride, and how many
// independent scans there are.
inline bool try_vendor_scan(const char* torch_op, const char* mlx_op,
                            const at::TensorBase& self,
                            const at::TensorBase& result,
                            int64_t dim) {
    constexpr const char* R = "scan";
    route_enter(R, torch_op);
    if (!mlx_op) return decline(R, torch_op, "no_mlx_scan_op");
    if (!vendor_elementwise_route_enabled()) return decline(R, torch_op, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, torch_op, "corpus_unavailable");

    // MLX's Scan copies a non-contiguous input to a fresh contiguous array
    // first; doing that here would allocate, which is the thing being removed.
    if (!self.is_contiguous()) return decline(R, torch_op, "in_not_contiguous");
    if (!result.is_contiguous()) return decline(R, torch_op, "out_not_contiguous");
    if (self.scalar_type() != result.scalar_type())
        return decline_dtypes(R, torch_op, "in_out_dtype_differ",
                              self.scalar_type(), result.scalar_type());
    const int dt = hagane_vendor_dtype(self.scalar_type());
    if (dt < 0) return decline(R, torch_op, "dtype");
    if (self.dim() == 0 || self.numel() <= 0)
        return decline(R, torch_op, "empty_or_0d");
    if (dim < 0 || dim >= self.dim()) return decline(R, torch_op, "dim_range");
    if (self.sizes() != result.sizes()) return decline(R, torch_op, "shape_differ");

    // For a contiguous tensor the scanned axis's stride is the product of the
    // extents inside it, and `outer` is everything outside.
    const int64_t axis_size = self.size(dim);
    const int64_t stride = self.stride(dim);
    if (axis_size <= 0 || stride <= 0) return decline(R, torch_op, "degenerate_axis");
    const int64_t outer = self.numel() / (axis_size * stride);
    if (outer <= 0) return decline(R, torch_op, "degenerate_axis");

    if (haganeOpsVendorScan(mlx_op, dt, self.const_data_ptr(),
                            result.data_ptr(), axis_size, stride, outer,
                            /*reverse=*/0, /*inclusive=*/1) != HAGANE_OPS_SUCCESS)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(std::string("mlx:scan_") + mlx_op);
    return true;
}

// cat through MLX's copy_gg: every input strided-copied straight into the
// output torch already allocated, so nothing on this path is MLX-owned.
//
// Each input gets its OWN collapsed nest — its strides on one side, the output
// slice's on the other — which is what makes a strided input free rather than a
// decline. It was 32 stashes per ARDY step, and the previous shape of this
// route (one 2-D [outer, run] view per input) could only describe a contiguous
// one.
template <typename TensorList>
inline bool try_vendor_cat(const TensorList& tensors,
                           int64_t dim, const at::Tensor& result) {
    constexpr const char* R = "cat";
    route_enter(R, nullptr);
    if (!vendor_elementwise_route_enabled()) return decline(R, nullptr, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, nullptr, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, nullptr, "corpus_unavailable");
    if (!result.is_contiguous()) return decline(R, nullptr, "out_not_contiguous");
    if (result.numel() <= 0) return decline(R, nullptr, "out_empty");

    const int dt = hagane_vendor_dtype(result.scalar_type());
    if (dt < 0) return decline(R, nullptr, "dtype");
    const int64_t n = static_cast<int64_t>(result.dim());
    if (n > 16) return decline(R, nullptr, "rank_gt16");
    if (dim < 0) dim += n;
    if (dim < 0 || dim >= n) return decline(R, nullptr, "dim_out_of_range");

    // The output's own contiguous strides. Every input lands in a SLICE of the
    // output, and a slice's strides are the output's — only its base moves.
    int64_t out_stride[16];
    {
        int64_t acc = 1;
        for (int64_t i = n - 1; i >= 0; --i) { out_stride[i] = acc; acc *= result.size(i); }
    }

    std::vector<const void*> srcs;
    std::vector<int64_t> dst_offset;
    std::vector<int32_t> ndims;
    std::vector<int64_t> ext, ss, ds;
    int64_t at = 0;                  // running offset along `dim`, in elements of it
    for (const auto& t_ref : tensors) {
        const at::Tensor& t = t_ref;
        if (t.numel() == 0) continue;             // legacy empties add nothing
        if (t.scalar_type() != result.scalar_type())
            return decline_dtypes(R, nullptr, "input_dtype_differs",
                                  t.scalar_type(), result.scalar_type());
        if (t.dim() != n) return decline(R, nullptr, "input_rank_differs");
        for (int64_t i = 0; i < n; ++i)
            if (i != dim && t.size(i) != result.size(i))
                return decline(R, nullptr, "input_shape_mismatch");

        // Collapse this input against its destination slice, merging only where
        // BOTH sides stay linear. A contiguous input collapses to exactly the
        // [outer, run] pair this route used to require.
        int ng = 0;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t e = t.size(k);
            if (e == 1) continue;
            const int64_t S = t.stride(k), D = out_stride[k];
            if (S < 0) return decline_layout(R, "input_negative_stride", t);
            if (ng > 0 && ss.back() == S * e && ds.back() == D * e) {
                ext.back() *= e; ss.back() = S; ds.back() = D;
            } else {
                if (ng == 8) return decline_layout(R, "input_groups_gt8", t);
                ext.push_back(e); ss.push_back(S); ds.push_back(D); ++ng;
            }
        }
        if (ng == 0) { ext.push_back(1); ss.push_back(0); ds.push_back(1); ng = 1; }

        srcs.push_back(t.const_data_ptr());
        dst_offset.push_back(at * out_stride[dim]);
        ndims.push_back(static_cast<int32_t>(ng));
        at += t.size(dim);
    }
    if (srcs.empty()) return decline(R, nullptr, "all_inputs_empty");
    // The pieces must tile the output; the runtime rechecks by element count,
    // because superseding the output's stash depends on it.
    if (at != result.size(dim)) return decline(R, nullptr, "pieces_do_not_tile");

    if (haganeOpsVendorCatNest(dt, srcs.data(), dst_offset.data(), ndims.data(),
                               ext.data(), ss.data(), ds.data(),
                               static_cast<int32_t>(srcs.size()),
                               result.data_ptr(), result.numel())
        != HAGANE_OPS_SUCCESS)
        return decline(R, nullptr, "kernel_dispatch");
    note_native_launch("mlx:copy_gg");
    return true;
}

// fill_ — MLX's copy_s, whose source is one scalar read for every element.
//
// The largest single stash in ARDY: 648 per step (#1024), which is 37% of what
// is left. fill_stub is one funnel, so this covers zeros/ones/zeros_like/full
// AND every at::empty + output.fill_(v) nested inside a composite — torch's own
// constant_pad_nd, which is how upstream SDPA pads an attention mask, is one of
// those. The per-op stash table could not see any of the nested ones; they were
// charged to whatever aten op the dispatch mode last saw.
//
// The value crosses as a double for the same reason haganeOpsFill takes one, and
// the runtime converts it by exactly fill_scalar's rule so route-on and
// route-off produce identical bits.
inline bool try_vendor_fill(TensorIteratorBase& iter, const c10::Scalar& value) {
    constexpr const char* R = "fill";
    route_enter(R, nullptr);
    if (!vendor_elementwise_route_enabled()) return decline(R, nullptr, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, nullptr, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, nullptr, "corpus_unavailable");
    if (iter.noutputs() != 1) return decline(R, nullptr, "noutputs");

    const at::Tensor& out = iter.tensor(0);
    // copy_s indexes the output linearly. A strided fill would need a kernel MLX
    // does not carry, and composing one out of gg*_copy plus a scratch scalar
    // would be inventing it — decline instead.
    if (!out.is_contiguous()) return decline_layout(R, "out_not_contiguous", out);

    const int dt = hagane_vendor_dtype(out.scalar_type());
    if (dt < 0) return decline(R, nullptr, "dtype");

    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to fill
    if (N > static_cast<int64_t>(UINT32_MAX)) return decline(R, nullptr, "numel_too_big");

    if (haganeOpsVendorFill(dt, out.data_ptr(), N, value.toDouble())
        != HAGANE_OPS_SUCCESS)
        return decline(R, nullptr, "kernel_dispatch");
    note_native_launch("mlx:s_copy");
    return true;
}

// arange — MLX's arange<T>, out[i] = start + i * step.
//
// 84 stashes per ARDY step (#1024), and the entry it replaces is lossy as well
// as lazy: arange_cuda_out hands haganeOpsArange two doubles, so an int64 range
// past 2^53 loses its start AND its step (mx::arange derives the step as
// (start+step)-start in the output dtype, which is 0 once double cannot separate
// them). torch's own CUDA kernel never goes through double — it takes
// `start.to<accscalar_t>()` straight off the Scalar — so this route does the
// same and hands the runtime raw bits.
//
// accscalar_t is at::acc_type<scalar_t, true>, read off AccumulateType.h rather
// than recalled: float32 for Half/BFloat16, int64 for every integer width,
// itself for Float. The runtime accumulates in it and narrows once.
//
//   - Float and the integers run acc == out, one dispatch, the same kernel and
//     the same immediates mx::arange would have used.
//   - Half/BFloat16 accumulate in float32 into runtime scratch and narrow with
//     v_copyfloat32<out>. That is a PARITY FIX, not only a stash removal: MLX's
//     arangefloat16 accumulates in half, which disagrees with CUDA on 37% of
//     the elements of a 0.1 step.
//   - The narrow integers pass int64's value truncated to their own width,
//     which is what makes one dispatch correct: truncation to N bits is
//     reduction mod 2^N, and (a + b*c) mod 2^N does not care whether a, b and c
//     were reduced first. So accumulating in int8 and truncating every step is
//     the same series CUDA gets by accumulating in int64 and truncating once.
inline bool try_vendor_arange(at::Tensor& result, const c10::Scalar& start,
                              const c10::Scalar& step) {
    constexpr const char* R = "arange";
    route_enter(R, nullptr);
    if (!vendor_elementwise_route_enabled()) return decline(R, nullptr, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, nullptr, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, nullptr, "corpus_unavailable");

    // arange<T> indexes its output linearly and MLX carries no strided variant.
    if (!result.is_contiguous()) return decline_layout(R, "out_not_contiguous", result);

    const c10::ScalarType st = result.scalar_type();
    // No arange<bool> in the corpus, and torch's own AT_DISPATCH_ALL_TYPES_AND2
    // does not dispatch Bool either — so this is unreachable, not a gap.
    if (st == c10::ScalarType::Bool) return decline(R, nullptr, "dtype_bool");
    const int dt = hagane_vendor_dtype(st);
    if (dt < 0) return decline(R, nullptr, "dtype");

    const int64_t N = result.numel();
    if (N <= 0) return true;                      // nothing to write
    if (N > static_cast<int64_t>(UINT32_MAX)) return decline(R, nullptr, "numel_too_big");

    int acc_dt = dt;
    uint64_t start_bits = 0, step_bits = 0;
    if (st == c10::ScalarType::Float || st == c10::ScalarType::Half ||
        st == c10::ScalarType::BFloat16) {
        if (st != c10::ScalarType::Float) acc_dt = HAGANE_DTYPE_FLOAT32;
        const float s = start.to<float>();
        const float p = step.to<float>();
        std::memcpy(&start_bits, &s, sizeof(s));
        std::memcpy(&step_bits, &p, sizeof(p));
    } else {
        // accscalar_t is int64 for every integer width; the runtime reads the
        // low esize(dtype) bytes, which on a little-endian target IS
        // static_cast<scalar_t>. Same conversion torch does, same throw on a
        // Scalar that will not fit an int64.
        const int64_t s = start.to<int64_t>();
        const int64_t p = step.to<int64_t>();
        std::memcpy(&start_bits, &s, sizeof(s));
        std::memcpy(&step_bits, &p, sizeof(p));
    }

    if (haganeOpsVendorArange(dt, acc_dt, result.data_ptr(), N,
                              start_bits, step_bits) != HAGANE_OPS_SUCCESS)
        return decline(R, nullptr, "kernel_dispatch");
    note_native_launch("mlx:arange");
    return true;
}

// Raw bits of a clamp BOUND in the compute dtype, converted exactly the way
// upstream's CUDA kernel converts it.
//
// launch_clamp_scalar does `lim.to<opmath_t>()`, and both halves of that matter:
//
//   * `.to<T>()` is c10::checked_convert, so a bound outside the compute dtype
//     RAISES rather than truncating. clamp(int8_tensor, min=1000) throws
//     upstream; passing 1000 through toFloat() and letting it land as -24 is
//     the silent-wrong this replaces, and it was measured before this route
//     existed (cpu=RuntimeError, dev=None).
//   * opmath_t is FLOAT for Half/BFloat16, so a bound outside half's range but
//     inside float's must NOT throw. Convert through float exactly as upstream
//     does, then round to the tensor's dtype for the kernel immediate.
//
// Rounding the bound to the tensor's own dtype is measured equivalent to
// upstream's clamp-in-float-then-cast, not assumed: rounding is monotone and
// the result is either v (already representable) or a bound (which upstream
// rounds too), so no representable value crosses the boundary. The ulp sweep in
// scripts/test_scalar_ops.py pins it at 0/1/4/1/2/3/4/1 ulp either side of a
// tick for both half types, and pins torch's own CPU kernel — which likewise
// rounds the bound — against the float formula.
inline bool vendor_bound_bits(c10::ScalarType st, const c10::Scalar& s,
                              uint64_t* out) {
    *out = 0;
    switch (st) {
        case c10::ScalarType::Float:
            { float v = s.to<float>();  std::memcpy(out, &v, 4); return true; }
        case c10::ScalarType::Half:
            { auto v = static_cast<c10::Half>(s.to<float>());
              std::memcpy(out, &v, 2); return true; }
        case c10::ScalarType::BFloat16:
            { auto v = static_cast<c10::BFloat16>(s.to<float>());
              std::memcpy(out, &v, 2); return true; }
        case c10::ScalarType::Char:
            { int8_t v = s.to<int8_t>();   std::memcpy(out, &v, 1); return true; }
        case c10::ScalarType::Short:
            { int16_t v = s.to<int16_t>(); std::memcpy(out, &v, 2); return true; }
        case c10::ScalarType::Int:
            { int32_t v = s.to<int32_t>(); std::memcpy(out, &v, 4); return true; }
        case c10::ScalarType::Long:
            { int64_t v = s.to<int64_t>(); std::memcpy(out, &v, 8); return true; }
        case c10::ScalarType::Byte:
            { uint8_t v = s.to<uint8_t>(); std::memcpy(out, &v, 1); return true; }
        case c10::ScalarType::Bool:
            { bool v = s.to<bool>();       std::memcpy(out, &v, 1); return true; }
        default: return false;      // Double: Metal has no float64
    }
}

// clamp with scalar bounds — MLX's vs_Maximum then vs_Minimum, which IS
// upstream's `::min(::max(v, lower), upper)` and not an approximation of it.
// See haganeOpsVendorClamp for why composing the pair is faithful and why it is
// one runtime entry rather than two vendor_binary calls.
//
// 98 stashes per ARDY step (#1024) with ZERO route ENTERs — no route at all,
// which is a different fix from a decline. The path it replaces loses the bound
// twice (min_val.toFloat() at the call site, then a float32 mx::array inside
// haganeOpsClamp, which promotes the whole tensor), so this is a parity fix as
// well as a stash removal.
//
// A NaN bound never arrives: TORCH_IMPL_FUNC(clamp_out) fills the result with
// NaN above the stub and dispatches nothing. That matters because MLX's Maximum
// would NOT reproduce CUDA's fmaxf on one — see hagane_clamp_scalar_common.
inline bool try_vendor_clamp(at::TensorIteratorBase& iter,
                             bool has_min, const c10::Scalar& min_val,
                             bool has_max, const c10::Scalar& max_val) {
    constexpr const char* R = "clamp";
    route_enter(R, nullptr);
    if (!vendor_elementwise_route_enabled()) return decline(R, nullptr, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, nullptr, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, nullptr, "corpus_unavailable");
    // No bound left after normalisation is a copy, not a clamp; the caller
    // handles it rather than having this route quietly become something else.
    if (!has_min && !has_max) return decline(R, nullptr, "no_bounds");

    // MLX has no strided vs_ kernel, and composing one out of gg*_copy plus a
    // scratch would be inventing a kernel.
    if (!iter.is_contiguous()) return decline(R, nullptr, "iter_not_contiguous");

    const c10::ScalarType compute = iter.common_dtype();
    if (iter.dtype(0) != compute)
        return decline_dtypes(R, nullptr, "in_out_dtype_differ",
                              iter.dtype(0), compute);
    const int dt = hagane_vendor_dtype(compute);
    if (dt < 0) return decline(R, nullptr, "dtype");

    const int64_t N = iter.numel();
    if (N <= 0) return true;                      // nothing to write
    if (N > static_cast<int64_t>(UINT32_MAX)) return decline(R, nullptr, "numel_too_big");

    // Deliberately BEFORE the dispatch and deliberately able to throw: a bound
    // that will not fit the compute dtype raises upstream, and matching that is
    // parity, not an error path.
    uint64_t lo = 0, hi = 0;
    if (has_min && !vendor_bound_bits(compute, min_val, &lo))
        return decline(R, nullptr, "bound_dtype");
    if (has_max && !vendor_bound_bits(compute, max_val, &hi))
        return decline(R, nullptr, "bound_dtype");

    if (haganeOpsVendorClamp(dt, iter.data_ptr(1), iter.data_ptr(0), N,
                             has_min ? 1 : 0, lo,
                             has_max ? 1 : 0, hi) != HAGANE_OPS_SUCCESS)
        return decline(R, nullptr, "kernel_dispatch");
    note_native_launch("mlx:Maximum/Minimum");
    return true;
}

// copy_ — a strided read laid down contiguously in the caller's block, through
// MLX's copy corpus.
//
// This is the biggest single block of MLX-owned dispatches in a real workload,
// and it hides behind other ops' names. ARDY's scaled_dot_product_attention
// measured 960 MLX-owned dispatches over 160 calls; bisecting the same call by
// input layout showed six per call, of which the attention kernel is ONE — the
// other five are layout copies (q/k/v forced contiguous by the SDPA entry, plus
// the mask's preprocessing). `reshape` and the rest of `cat`'s declines are the
// same family. So the attention math was never the target: a transposed view
// being made contiguous was.
//
// Are these destination strides injective over these extents — i.e. does each
// element have exactly one thread writing it? Sufficient and structural: sort
// the dimensions by stride and require each to clear the span of everything
// inside it. That accepts a slice with a row pitch and a transposed
// destination, and rejects every genuine overlap.
//
// NOT at::has_internal_overlap: that answers TooHard for anything merely
// non-dense, which is every slice this path exists to route (it declined all
// 200 of ARDY's on the first attempt). The runtime derives the same rule
// independently — it is the boundary that must not be talked into a race; this
// copy exists so the decline has a NAME, because "the destination overlaps
// itself" and "the corpus has no such kernel" want different follow-ups.
inline bool dst_nest_injective(const int64_t* ext, const int64_t* ds, int ng) {
    int order[16];
    for (int i = 0; i < ng; ++i) order[i] = i;
    std::sort(order, order + ng, [&](int a, int b) { return ds[a] > ds[b]; });
    int64_t span = 1;   // elements already covered by the dimensions inside
    for (int i = ng - 1; i >= 0; --i) {
        if (ds[order[i]] < span) return false;
        span = ds[order[i]] * ext[order[i]];
    }
    return true;
}

// Routes when the source is a non-overlapping strided view of the same shape
// and both dtypes have a Metal spelling. A broadcast source (stride 0) is fine.
//
// The DESTINATION may be strided too, which was 200 of ARDY's stashes per step
// and all 160 of SDPA's: torch's own preprocess_mask aligns an attention mask
// with `pad(m,[0,k])[..., :n]`, and that lands here as a copy into a slice with
// a row pitch. MLX's gg*_ family derives BOTH indices from strides, so it is
// the same dispatch with one more stride array — but only for ONE dtype, and
// only when the destination is injective. Anything else declines to the
// existing path, which stays byte-for-byte what it was.
inline bool try_vendor_copy(const at::Tensor& dst, const at::Tensor& src) {
    constexpr const char* R = "copy";
    route_enter(R, nullptr);
    if (!vendor_elementwise_route_enabled()) return decline(R, nullptr, "route_off");
    if (haganeOpsTapeRecording()) return decline(R, nullptr, "tape_recording");
    if (!haganeOpsVendorElementwiseAvailable())
        return decline(R, nullptr, "corpus_unavailable");

    if (!dst.is_cuda() || !src.is_cuda()) return decline(R, nullptr, "not_both_device");
    const int64_t numel = dst.numel();
    if (numel <= 0) return decline(R, nullptr, "empty");
    if (src.sizes() != dst.sizes()) return decline(R, nullptr, "shape_mismatch");

    const bool dst_contig = dst.is_contiguous();
    if (!dst_contig) {
        // MLX instantiates the gg_ family for one dtype (instantiate_copy_same),
        // so a strided destination AND a cast is a kernel that does not exist.
        if (src.scalar_type() != dst.scalar_type())
            return decline_dtypes(R, nullptr, "dst_strided_cast",
                                  src.scalar_type(), dst.scalar_type());
    }

    const int dt_in = hagane_vendor_dtype(src.scalar_type());
    const int dt_out = hagane_vendor_dtype(dst.scalar_type());
    if (dt_in < 0) return decline(R, nullptr, "src_dtype");
    if (dt_out < 0) return decline(R, nullptr, "dst_dtype");

    const int64_t n = static_cast<int64_t>(dst.dim());
    if (n > 16) return decline(R, nullptr, "rank_gt16");

    // A parallel copy has no correct order when it reads what it writes, and no
    // settle can give it one. Decline on ANY byte overlap rather than reason
    // about which layouts happen to be safe.
    {
        const char* sb = static_cast<const char*>(src.const_data_ptr());
        const char* db = static_cast<const char*>(dst.const_data_ptr());
        // Each side's span from its own base, which is what the strides index.
        int64_t s_last = 0, d_last = 0;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t st = src.stride(k), dt = dst.stride(k);
            if (st < 0) return decline(R, nullptr, "negative_stride");
            if (dt < 0) return decline_layout(R, "dst_negative_stride", dst);
            s_last += (src.size(k) - 1) * st;
            d_last += (dst.size(k) - 1) * dt;
        }
        const char* se = sb + (s_last + 1) * src.element_size();
        const char* de = db + (d_last + 1) * dst.element_size();
        if (sb < de && db < se) return decline(R, nullptr, "src_dst_overlap");
    }

    // Collapse, outermost first. With a contiguous destination, merge an inner
    // dimension into the group outside it whenever the SOURCE stays linear
    // across the pair (the destination always does); with a strided one, only
    // when BOTH do — the same rule try_vendor_binary_g uses for two operands.
    int64_t ext[16], ss[16], ds[16];
    int ng = 0;
    for (int64_t k = 0; k < n; ++k) {
        const int64_t e = dst.size(k);
        if (e == 1) continue;
        const int64_t S = src.stride(k), D = dst.stride(k);
        if (ng > 0 && ss[ng - 1] == S * e && (dst_contig || ds[ng - 1] == D * e)) {
            ext[ng - 1] *= e; ss[ng - 1] = S; ds[ng - 1] = D;
        } else {
            ext[ng] = e; ss[ng] = S; ds[ng] = D; ++ng;
        }
    }
    if (ng == 0) { ext[0] = 1; ss[0] = 0; ds[0] = 1; ng = 1; }   // every dim was 1
    if (ng > 8) return decline(R, nullptr, "groups_gt8");  // beyond the corpus nest

    if (!dst_contig) {
        // Two threads writing one element is a race, not a copy, and no settle
        // can give it an order.
        if (!dst_nest_injective(ext, ds, ng))
            return decline_layout(R, "dst_not_injective", dst);
        if (haganeOpsVendorCopyGG(dt_in, dt_out, src.const_data_ptr(),
                                  dst.data_ptr(), ext, ss, ds, ng)
            != HAGANE_OPS_SUCCESS)
            return decline_layout(R, "gg_kernel_dispatch", dst);
        note_native_launch("mlx:gg_copy");
        return true;
    }

    // A single collapsed group of stride 1 is a flat copy — MLX's cheapest
    // kernel, and the shape a plain copy or a dtype cast actually has.
    const bool contiguous_src = (ng == 1 && ss[0] == 1);

    if (haganeOpsVendorCopy(dt_in, dt_out, src.const_data_ptr(), dst.data_ptr(),
                            contiguous_src ? 1 : 0, ext, ss, ng)
        != HAGANE_OPS_SUCCESS)
        return decline(R, nullptr, "kernel_dispatch");
    note_native_launch(contiguous_src ? "mlx:v_copy" : "mlx:g_copy");
    return true;
}

// ---- 1.10: the index family, Hagane-owned (see hagane/kernels/index.hip) -----
// PyTorch's real index kernels do not transpile — ScatterGatherKernel.hip 27
// errors / 0 kernels (an atomicCAS overload gap), IndexKernel.hip 25 errors / 1
// irrelevant kernel, Indexing.hip 1163 errors / 0 kernels — so we own generic
// gather/scatter primitives, exactly as reduce.hip owns the block reducer and
// for the same stated reason. Route-off (HAGANE_USE_METALLIB_ROUTE=0) skips
// registration → MLX (mx::take / take_along_axis / put_along_axis), the spine
// these replace.
inline bool index_metallib_available() {
    static const bool available = [] {
        const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
        if (route && route[0] == '0') return false;
        std::string path = metallib_dir() + "/index.metallib";
        if (haganeRegisterMetallibAll(path.c_str()) != hipSuccess) {
            std::fprintf(stderr,
                "[hagane-path-alpha] failed to register %s; index ops stay on MLX\n",
                path.c_str());
            return false;
        }
        std::fprintf(stderr,
            "[hagane-path-alpha] index metallib registered "
            "(gather/scatter/index_put/triangle/compact, 52 kernels)\n");
        return true;
    }();
    return available;
}

// Mirrors HaganeGatherDims in hagane/kernels/index.hip. Rides as ONE setBytes
// scalar, so the per-dim strides cost no device allocation and no upload.
// KEEP IN SYNC with the kernel source — a mismatch is a silent-wrong, which is
// why both sides name MAX_DIMS from the same literal and the gate runs every
// rank 1..8.
#define HAGANE_INDEX_MAX_DIMS 8
struct HaganeGatherDimsHost {
    int64_t shape[HAGANE_INDEX_MAX_DIMS];
    int64_t a_es[HAGANE_INDEX_MAX_DIMS];
    int64_t b_es[HAGANE_INDEX_MAX_DIMS];
    int64_t c_es[HAGANE_INDEX_MAX_DIMS];
};

// Element span an operand's strides actually reach over `shape`, so the input
// settle covers exactly the bytes the kernel reads and the output mark covers
// exactly the bytes it writes. Same quantity the vendor routes compute.
inline int64_t index_span_elems(const int64_t* shape, const int64_t* es, int ndim) {
    int64_t span = 1;
    for (int d = 0; d < ndim; ++d)
        if (shape[d] > 1) span += (shape[d] - 1) * es[d];
    return span;
}

// The element-size arm. Selection is a pure element copy, so b1/b2/b4/b8 cover
// every torch dtype EXACTLY — there is no per-dtype numerics here at all.
inline const char* index_bsuffix(int64_t esz) {
    switch (esz) {
        case 1: return "b1";
        case 2: return "b2";
        case 4: return "b4";
        case 8: return "b8";
        default: return nullptr;
    }
}

// A + C — gather along one axis. ONE route, because index_select and gather
// compute the same thing and differ only in how the index is addressed:
//   index_select / embedding : index is 1-D along `dim`  -> b_es[d]=0 for d!=dim
//   gather                   : index carries out's shape -> b_es = its strides
// Callers pass `index_1d_along_dim` to say which. Every operand is addressed
// through its own strides, so a non-contiguous input, output or (ARDY's real
// shape) a BROADCAST index all route rather than declining.
inline bool try_index_gather_axis(const char* torch_op, const at::Tensor& in,
                                  const at::Tensor& index, const at::Tensor& out,
                                  int64_t dim, bool index_1d_along_dim) {
    const char* R = "index_axis";
    if (!index_metallib_available()) return false;
    route_enter(R, torch_op);
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (!in.defined() || !index.defined() || !out.defined())
        return decline(R, torch_op, "undefined_operand");
    if (!in.is_cuda() || !index.is_cuda() || !out.is_cuda())
        return decline(R, torch_op, "not_device");
    // Raw element copy: a dtype mismatch would move bytes between
    // representations, which is a silent-wrong and not something the caller can
    // see. index_select and gather both preserve dtype, so this only fires if
    // an unexpected caller appears.
    if (in.scalar_type() != out.scalar_type())
        return decline_dtypes(R, torch_op, "in_out_dtype_differ",
                              in.scalar_type(), out.scalar_type());
    const int64_t esz = out.element_size();
    const char* bsfx = index_bsuffix(esz);
    if (!bsfx) return decline(R, torch_op, "element_size");

    const auto ist = index.scalar_type();
    const char* isfx = ist == at::kLong ? "i64" : (ist == at::kInt ? "i32" : nullptr);
    if (!isfx) return decline_dtypes(R, torch_op, "index_dtype", ist, out.scalar_type());

    const int ndim = static_cast<int>(out.dim());
    if (ndim <= 0 || ndim > HAGANE_INDEX_MAX_DIMS)
        return decline(R, torch_op, "ndim");
    if (dim < 0 || dim >= static_cast<int64_t>(in.dim()))
        return decline(R, torch_op, "dim_out_of_range");
    if (in.dim() != out.dim()) return decline(R, torch_op, "rank_mismatch");
    const int64_t total = out.numel();
    if (total <= 0) return decline(R, torch_op, "empty");
    const int64_t dim_size = in.size(dim);
    if (dim_size <= 0) return decline(R, torch_op, "empty_indexed_dim");

    HaganeGatherDimsHost d{};
    for (int k = 0; k < ndim; ++k) {
        d.shape[k] = out.size(k);
        d.a_es[k]  = in.stride(k);
        d.c_es[k]  = out.stride(k);
        if (index_1d_along_dim)
            d.b_es[k] = (k == dim) ? (index.dim() ? index.stride(0) : 0) : 0;
        else
            d.b_es[k] = index.stride(k);
        // A negative stride would index below data_ptr(); the buffer resolution
        // is base+offset and cannot promise that is inside the allocation.
        // Rare in torch (as_strided only) and honestly declined.
        if (d.a_es[k] < 0 || d.b_es[k] < 0 || d.c_es[k] < 0)
            return decline(R, torch_op, "negative_stride");
        // `in` is addressed at coord k for every k != dim, so out must not
        // exceed it there. gather guarantees this; assert rather than trust.
        if (k != dim && d.shape[k] > in.size(k))
            return decline(R, torch_op, "out_exceeds_in");
    }
    if (!index_1d_along_dim && index.dim() != out.dim())
        return decline(R, torch_op, "index_rank_mismatch");
    if (index_1d_along_dim && index.numel() != out.size(dim))
        return decline(R, torch_op, "index_len_mismatch");

    // Spans: `in` is read over its OWN extent along dim (any index may select
    // any row), the index over the iteration shape, the output over the same.
    int64_t in_shape[HAGANE_INDEX_MAX_DIMS];
    for (int k = 0; k < ndim; ++k) in_shape[k] = (k == dim) ? dim_size : d.shape[k];
    const int64_t in_span  = index_span_elems(in_shape, d.a_es, ndim);
    const int64_t idx_span = index_span_elems(d.shape, d.b_es, ndim);
    const int64_t out_span = index_span_elems(d.shape, d.c_es, ndim);

    void* p_in  = const_cast<void*>(in.const_data_ptr());
    void* p_idx = const_cast<void*>(index.const_data_ptr());
    void* p_out = out.data_ptr();

    flush_or_commit_metallib_input(p_in,  in_span  * esz);
    flush_or_commit_metallib_input(p_idx, idx_span * index.element_size());
    // Settle the OUTPUT's stale stash before writing its block. Which settle
    // depends on whether the write covers every byte of the span — the same
    // question vendor_settle_output asks, and getting it backwards is #1014:
    // superseding a strided write drops live values in the holes.
    //
    // J (#1080): the writer below is a KERNEL on the Hagane queue, so the
    // DEVICE-write entry, not the host-write one. The queue is totally ordered
    // and this dispatch is sequenced after every queued write to p_out by the
    // shared event — which is the same guarantee haganeOpsMarkMetallibWrite
    // relies on a dozen lines down.
    if (out_span == total) haganeOpsSettleForDeviceWrite(p_out, out_span * esz);
    else                   haganeOpsFlushRegion(p_out, out_span * esz);

    std::string kname = std::string("hagane_gather_axis_") + bsfx + "_" + isfx;
    int64_t c_ndim = ndim, c_total = total, c_dim = dim, c_dimsz = dim_size;
    void*  args[]  = {p_in, p_idx, p_out, &d, &c_ndim, &c_total, &c_dim, &c_dimsz};
    int    at_[]   = {0, 0, 0, 1, 1, 1, 1, 1};
    size_t as_[]   = {0, 0, 0, sizeof(d), sizeof(int64_t), sizeof(int64_t),
                      sizeof(int64_t), sizeof(int64_t)};
    int64_t nb = (total + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 65535) nb = 65535;          // the grid-stride loop covers the rest
    dim3 block(256, 1, 1), grid(static_cast<unsigned>(nb), 1, 1);
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, at_, as_, 8) != hipSuccess)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(p_out, static_cast<size_t>(out_span * esz));
    return true;
}

// D — scatter along one axis (torch `scatter.src`). The only WRITER in the
// family, so its risk surface is the settle, not the arithmetic:
//
//   `self` here IS the output (the structured op already cloned it for the
//   out-of-place variant), and the kernel writes ONLY the scattered positions.
//   Every other element must already be present in the block, so a stale stash
//   on `self` has to be MATERIALISED, never dropped — dropping it is exactly
//   #1014 / RW-8.0, and it would silently zero everything the scatter does not
//   touch. That is why this uses haganeOpsFlushRegion where the gather routes
//   use haganeOpsSettleForDeviceWrite (J/#1080; it was haganeOpsFlushForWrite
//   before J moved the full-span cases off the host-write entry).
//
// Duplicate indices are a plain write with an UNSPECIFIED winner in torch (it
// is scatter_ADD that accumulates), so no atomics are needed — which is also
// why the atomicCAS transpiler gap does not block this slice.
inline bool try_index_scatter_axis(const at::Tensor& self, int64_t dim,
                                   const at::Tensor& index, const at::Tensor& src) {
    const char* R = "index_axis";
    const char* OP = "scatter";
    if (!index_metallib_available()) return false;
    route_enter(R, OP);
    if (haganeOpsTapeRecording()) return decline(R, OP, "tape_recording");
    if (!self.defined() || !index.defined() || !src.defined())
        return decline(R, OP, "undefined_operand");
    if (!self.is_cuda() || !index.is_cuda() || !src.is_cuda())
        return decline(R, OP, "not_device");
    if (self.scalar_type() != src.scalar_type())
        return decline_dtypes(R, OP, "self_src_dtype_differ",
                              src.scalar_type(), self.scalar_type());
    if (index.scalar_type() != at::kLong)
        return decline_dtypes(R, OP, "index_dtype", index.scalar_type(),
                              self.scalar_type());
    const int64_t esz = self.element_size();
    const char* bsfx = index_bsuffix(esz);
    if (!bsfx) return decline(R, OP, "element_size");

    const int ndim = static_cast<int>(self.dim());
    if (ndim <= 0 || ndim > HAGANE_INDEX_MAX_DIMS) return decline(R, OP, "ndim");
    if (index.dim() != ndim || src.dim() != ndim)
        return decline(R, OP, "rank_mismatch");
    if (dim < 0 || dim >= ndim) return decline(R, OP, "dim_out_of_range");
    const int64_t total = index.numel();
    if (total <= 0) return decline(R, OP, "empty");
    const int64_t dim_size = self.size(dim);
    if (dim_size <= 0) return decline(R, OP, "empty_indexed_dim");

    HaganeGatherDimsHost d{};
    for (int k = 0; k < ndim; ++k) {
        d.shape[k] = index.size(k);
        d.a_es[k]  = self.stride(k);
        d.b_es[k]  = index.stride(k);
        d.c_es[k]  = src.stride(k);
        if (d.a_es[k] < 0 || d.b_es[k] < 0 || d.c_es[k] < 0)
            return decline(R, OP, "negative_stride");
        if (d.shape[k] > src.size(k)) return decline(R, OP, "index_exceeds_src");
        if (k != dim && d.shape[k] > self.size(k))
            return decline(R, OP, "index_exceeds_self");
    }

    int64_t self_shape[HAGANE_INDEX_MAX_DIMS];
    for (int k = 0; k < ndim; ++k) self_shape[k] = self.size(k);
    const int64_t self_span = index_span_elems(self_shape, d.a_es, ndim);
    const int64_t idx_span  = index_span_elems(d.shape, d.b_es, ndim);
    const int64_t src_span  = index_span_elems(d.shape, d.c_es, ndim);

    void* p_self = self.data_ptr();
    void* p_idx  = const_cast<void*>(index.const_data_ptr());
    void* p_src  = const_cast<void*>(src.const_data_ptr());

    flush_or_commit_metallib_input(p_idx, idx_span * index.element_size());
    flush_or_commit_metallib_input(p_src, src_span * esz);
    // MATERIALISE, never drop — see the header comment. The write is partial by
    // construction (that is what scatter IS), so there is no branch here.
    haganeOpsFlushRegion(p_self, self_span * esz);

    std::string kname = std::string("hagane_scatter_axis_") + bsfx;
    int64_t c_ndim = ndim, c_total = total, c_dim = dim, c_dimsz = dim_size;
    void*  args[] = {p_self, p_idx, p_src, &d, &c_ndim, &c_total, &c_dim, &c_dimsz};
    int    at_[]  = {0, 0, 0, 1, 1, 1, 1, 1};
    size_t as_[]  = {0, 0, 0, sizeof(d), sizeof(int64_t), sizeof(int64_t),
                     sizeof(int64_t), sizeof(int64_t)};
    int64_t nb = (total + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 65535) nb = 65535;
    dim3 block(256, 1, 1), grid(static_cast<unsigned>(nb), 1, 1);
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, at_, as_, 8) != hipSuccess)
        return decline(R, OP, "kernel_dispatch");
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(p_self, static_cast<size_t>(self_span * esz));
    return true;
}

// 1.11-C — triu / tril, into the caller's own block.
//
//   triu: out[..., r, c] = (r <= c - diagonal) ? in[..., r, c] : 0
//   tril: out[..., r, c] = (r >= c - diagonal) ? in[..., r, c] : 0
//
// What this replaces is not just a stash: haganeOpsTriu built the predicate out
// of FIVE MLX ops per call (arange, two reshape, subtract, less_equal, where,
// zeros_like), allocating an index ramp and a zeros tensor each time, and it
// cast the int64 `diagonal` through `int` on the way. The owned kernel takes
// the int64 directly and allocates nothing.
//
// Both operands are required CONTIGUOUS and same-shape, so the flat index
// carries any batch dims as pure outer stride — a batched (…, H, W) needs no
// extra arm. `out` may alias `in` (torch's triu_), and the SETTLE is the part
// that has to be right there, not the kernel: FlushForWrite ERASES a stash
// whose out_ptr is in range, which for an aliased operand would drop the value
// the kernel is about to read (#1014). So an alias takes FlushRegion, which
// materialises instead.
inline bool try_triangle(const char* torch_op, bool upper,
                         const at::Tensor& in, const at::Tensor& out,
                         int64_t diagonal) {
    const char* R = "triangle";
    if (!index_metallib_available()) return false;
    route_enter(R, torch_op);
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (!in.defined() || !out.defined())
        return decline(R, torch_op, "undefined_operand");
    if (!in.is_cuda() || !out.is_cuda()) return decline(R, torch_op, "not_device");
    // A raw element copy: a dtype mismatch would move bytes between
    // representations. triu/tril preserve dtype, so this only fires if an
    // unexpected caller appears.
    if (in.scalar_type() != out.scalar_type())
        return decline_dtypes(R, torch_op, "in_out_dtype_differ",
                              in.scalar_type(), out.scalar_type());
    const int64_t esz = out.element_size();
    const char* bsfx = index_bsuffix(esz);
    if (!bsfx) return decline(R, torch_op, "element_size");
    if (in.dim() < 2) return decline(R, torch_op, "rank_lt2");
    if (in.sizes() != out.sizes()) return decline(R, torch_op, "shape_differ");
    // The flat index assumes both sides are packed. A strided `out=` keeps the
    // MLX fallback, which addresses through copy_result's strides.
    if (!in.is_contiguous()) return decline(R, torch_op, "in_not_contiguous");
    if (!out.is_contiguous()) return decline(R, torch_op, "out_not_contiguous");
    const int64_t total = out.numel();
    if (total <= 0) return decline(R, torch_op, "empty");

    const int64_t H = in.size(in.dim() - 2), W = in.size(in.dim() - 1);
    void* p_in  = const_cast<void*>(in.const_data_ptr());
    void* p_out = out.data_ptr();
    const size_t bytes = static_cast<size_t>(total) * esz;

    flush_or_commit_metallib_input(p_in, bytes);
    // J (#1080): device write — see the note in try_index_gather_axis.
    if (p_in == p_out) haganeOpsFlushRegion(p_out, static_cast<int64_t>(bytes));
    else               haganeOpsSettleForDeviceWrite(p_out, static_cast<int64_t>(bytes));

    std::string kname = std::string(upper ? "hagane_triu_" : "hagane_tril_") + bsfx;
    int64_t c_total = total, c_H = H, c_W = W, c_diag = diagonal;
    void*  args[]  = {p_in, p_out, &c_total, &c_H, &c_W, &c_diag};
    int    at_[]   = {0, 0, 1, 1, 1, 1};
    size_t as_[]   = {0, 0, sizeof(int64_t), sizeof(int64_t), sizeof(int64_t),
                      sizeof(int64_t)};
    int64_t nb = (total + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 65535) nb = 65535;          // the grid-stride loop covers the rest
    dim3 block(256, 1, 1), grid(static_cast<unsigned>(nb), 1, 1);
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, at_, as_, 6) != hipSuccess)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(p_out, bytes);
    return true;
}

// 1.11-D — the Philox4x32-10 normal sampler. Its own metallib rather than a row
// in index.metallib: index.hip is pure selection with no numerics, and this is
// the opposite (a whole arithmetic spec), so sharing an artefact would make one
// gate's failure ambiguous about which family broke.
inline bool random_metallib_available() {
    static const bool available = [] {
        const char* route = std::getenv("HAGANE_USE_METALLIB_ROUTE");
        if (route && route[0] == '0') return false;
        std::string path = metallib_dir() + "/random.metallib";
        if (haganeRegisterMetallibAll(path.c_str()) != hipSuccess) {
            std::fprintf(stderr,
                "[hagane-path-alpha] failed to register %s; randn stays on MLX\n",
                path.c_str());
            return false;
        }
        std::fprintf(stderr,
            "[hagane-path-alpha] random metallib registered (normal, 3 kernels)\n");
        return true;
    }();
    return available;
}

// PyTorch's calc_execution_policy (native/hip/DistributionTemplates.h:52),
// UNCLAMPED. This is the one design decision in 1.11-D that departs from
// upstream's code, so here is the whole argument.
//
// The grid is part of the STREAM: it sets the grid-stride step that decides
// which Philox counter lands at which element. Upstream then clamps the grid by
// `multiProcessorCount * (maxThreadsPerMultiProcessor / block)`, which means an
// A100 and an H100 already produce different randn for the same seed once numel
// is large. There is no device-independent CUDA answer to reproduce — only a
// rule applied to a device.
//
// Applying that rule to Hagane's properties is what the plan called for, and it
// was tried and MEASURED: it produces grid == 1, because hipGetDeviceProperties
// writes a private 316-byte struct while callers read the public 1472-byte one,
// so multiProcessorCount reads 0 at the caller's offset and
// maxThreadsPerMultiProcessor was never filled at all (#1078). 16M randn took
// 11.8 ms against MLX's 0.218 ms — a 54x regression from generating the whole
// tensor with 256 threads. Building the stream on a property that lies would
// also mean the stream silently MOVES when #1078 is fixed.
//
// So: keep upstream's formula and its clamp, but take the cap from a FIXED
// CONSTANT instead of a device property. kGridCap = 2048 blocks (524288
// threads). The consequences are all good and all stated:
//   - Every tensor with numel <= 256*2048 = 524288 gets the IDENTICAL stream to
//     any CUDA device whose own cap is at least 2048 blocks — which is every
//     current datacentre GPU. That is the strong parity result, over the whole
//     range where CUDA itself is device-independent.
//   - Above that, CUDA devices already disagree with each other, and we differ
//     as one more device in that set. Same class of difference, no new one.
//   - The geometry depends only on (numel, block), so it is deterministic, it
//     needs no device property, and #1078 cannot silently move it.
//   - The clamp is also what makes the unroll pay: once it binds, each thread
//     loops and USES all four values of its hiprand_normal4 draw instead of
//     discarding three. Measured on 16M randn: grid=1 (the property bug) 11.8
//     ms, fully unclamped 1.595 ms, clamped here 0.243 ms.
// The cap is a CHOICE, so it is a constant with a name and a comment, not a
// number derived from something that happens to be lying.
//
// counter_offset must be handed to philox_engine_inputs() by the CALLER, so the
// generator advances by what the kernel actually consumes. `unroll` is 4 because
// hiprand_normal4 yields four values per draw; the trailing literal 4 is
// upstream's max_generator_offsets_per_curand_call and is NOT the same quantity.
struct HaganeDistPolicy {
    uint32_t block_size;
    uint32_t grid_x;
    uint64_t counter_offset;
};
inline HaganeDistPolicy hagane_dist_policy(int64_t numel) {
    const uint32_t block_size = 256;   // upstream block_size_bound
    const uint32_t unroll = 4;
    const uint64_t kGridCap = 2048;
    const uint64_t n = numel > 0 ? static_cast<uint64_t>(numel) : 1;
    uint64_t grid = (n + block_size - 1) / block_size;
    if (grid > kGridCap) grid = kGridCap;
    if (grid < 1) grid = 1;
    HaganeDistPolicy p;
    p.block_size = block_size;
    p.grid_x = static_cast<uint32_t>(grid);
    p.counter_offset =
        ((n - 1) / (static_cast<uint64_t>(block_size) * grid * unroll) + 1) * 4;
    return p;
}

// normal_ / randn onto the owned Philox kernel, writing the CALLER's block.
// Replaces mx::random::normal, whose threefry key was derived by hashing
// (seed, offset) — a stream with no spec at all, which is why this slice is a
// deliberate one-time re-baseline of sampled outputs and not a silent change.
inline bool try_normal_metallib(const char* torch_op, const at::TensorBase& out,
                                double mean, double std,
                                uint64_t seed, uint64_t offset,
                                const HaganeDistPolicy& policy) {
    const char* R = "normal";
    if (!random_metallib_available()) return false;
    route_enter(R, torch_op);
    if (haganeOpsTapeRecording()) return decline(R, torch_op, "tape_recording");
    if (!out.defined()) return decline(R, torch_op, "undefined_operand");
    if (!out.is_cuda()) return decline(R, torch_op, "not_device");
    const char* dsfx = nullptr;
    switch (out.scalar_type()) {
        case at::kFloat:    dsfx = "f32";  break;
        case at::kHalf:     dsfx = "f16";  break;
        case at::kBFloat16: dsfx = "bf16"; break;
        default: break;   // float64 has no Metal arm; complex/int never dispatch here
    }
    if (!dsfx) return decline_dtypes(R, torch_op, "dtype",
                                     out.scalar_type(), out.scalar_type());
    // The kernel addresses out[] flat. A strided destination keeps the MLX path,
    // which goes through copy_result's strides.
    if (!out.is_contiguous()) return decline(R, torch_op, "out_not_contiguous");
    const int64_t numel = out.numel();
    if (numel <= 0) return decline(R, torch_op, "empty");

    void* p_out = const_cast<void*>(out.const_data_ptr());
    const size_t bytes = static_cast<size_t>(numel) * out.element_size();
    // A sample covers every byte of a contiguous output and reads nothing, so
    // this is the unambiguous pre-write barrier — the fill/arange case, not
    // triu's aliased one.
    // J (#1080): device write — see the note in try_index_gather_axis.
    haganeOpsSettleForDeviceWrite(p_out, static_cast<int64_t>(bytes));

    std::string kname = std::string("hagane_normal_") + dsfx;
    int64_t  c_numel = numel;
    uint64_t c_seed = seed, c_off = offset;
    float    c_mean = static_cast<float>(mean), c_std = static_cast<float>(std);
    void*  args[]  = {p_out, &c_numel, &c_seed, &c_off, &c_mean, &c_std};
    int    at_[]   = {0, 1, 1, 1, 1, 1};
    size_t as_[]   = {0, sizeof(int64_t), sizeof(uint64_t), sizeof(uint64_t),
                      sizeof(float), sizeof(float)};
    dim3 block(policy.block_size, 1, 1), grid(policy.grid_x, 1, 1);
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, at_, as_, 6) != hipSuccess)
        return decline(R, torch_op, "kernel_dispatch");
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(p_out, bytes);
    return true;
}

// Mirrors HaganeIndexGatherDims in hagane/kernels/index.hip.
#define HAGANE_INDEX_MAX_IDX 3
struct HaganeIndexGatherDimsHost {
    int64_t shape[HAGANE_INDEX_MAX_DIMS];
    int64_t base_es[HAGANE_INDEX_MAX_DIMS];
    int64_t out_es[HAGANE_INDEX_MAX_DIMS];
    int64_t idx_es[HAGANE_INDEX_MAX_IDX * HAGANE_INDEX_MAX_DIMS];
    int64_t isize[HAGANE_INDEX_MAX_IDX];
    int64_t istride[HAGANE_INDEX_MAX_IDX];
};

// B — advanced indexing (`index.Tensor`) as ONE dispatch:
//   out[i] = base[ SUM_d coord_d*base_es[d] + SUM_j wrap(idx_j[coord])*istride[j] ]
//
// This is exactly the arithmetic haganeOpsIndexGather does with mx:: ops and
// hagane_index_on_device does with ~12 at:: dispatches. Do NOT "fix" this by
// forcing the at:: composition: every op in it is now routed (arange 1.6,
// clamp 1.7b, where 1.9, mul/add 1.4), so that would reach zero stashes while
// re-adding the ~1.9 ms/call — 13.2% of an ARDY denoise step — that
// haganeOpsIndexGather was written to remove. Trading a stash for a measured
// regression is the workaround this project forbids.
//
// Unlike index_select/gather, advanced indexing DOES wrap a negative index
// (torch semantics), which is what hagane_wrap_index's wrap is for. Out of
// range is clamped, not diagnosed — #1000, pre-existing on all three paths.
inline bool try_index_gather_advanced(TensorIteratorBase& iter,
                                      at::IntArrayRef indexed_sizes,
                                      at::IntArrayRef indexed_strides) {
    const char* R = "index_adv";
    const char* OP = "index";
    if (!index_metallib_available()) return false;
    route_enter(R, OP);
    if (haganeOpsTapeRecording()) return decline(R, OP, "tape_recording");

    const int n_idx = static_cast<int>(iter.ntensors()) - 2;
    if (n_idx <= 0 || n_idx > HAGANE_INDEX_MAX_IDX)
        return decline(R, OP, "n_idx");
    if ((int)indexed_sizes.size() != n_idx || (int)indexed_strides.size() != n_idx)
        return decline(R, OP, "indexed_arity_mismatch");

    const at::Tensor& out  = iter.tensor(0);
    const at::Tensor& base = iter.tensor(1);
    if (!out.defined() || !base.defined()) return decline(R, OP, "undefined_operand");
    if (!out.is_cuda() || !base.is_cuda()) return decline(R, OP, "not_device");
    if (base.scalar_type() != out.scalar_type())
        return decline_dtypes(R, OP, "in_out_dtype_differ",
                              base.scalar_type(), out.scalar_type());

    const int64_t esz = out.element_size();
    const char* bsfx = index_bsuffix(esz);
    if (!bsfx) return decline(R, OP, "element_size");

    const auto shape = iter.shape();
    const int ndim = static_cast<int>(shape.size());
    if (ndim <= 0 || ndim > HAGANE_INDEX_MAX_DIMS) return decline(R, OP, "ndim");
    const int64_t total = out.numel();
    if (total <= 0) return decline(R, OP, "empty");

    // The iterator reports BYTE strides; the kernel works in elements. A stride
    // that is not a whole element means this operand cannot be addressed in
    // element space at all — decline rather than round.
    auto elem_strides = [&](int arg, int64_t item, int64_t* dst) -> bool {
        if (item <= 0) return false;
        const auto st = iter.strides(arg);
        if ((int)st.size() != ndim) return false;
        for (int k = 0; k < ndim; ++k) {
            if (st[k] % item != 0) return false;
            dst[k] = st[k] / item;
            if (dst[k] < 0) return false;   // see try_index_gather_axis
        }
        return true;
    };

    HaganeIndexGatherDimsHost d{};
    for (int k = 0; k < ndim; ++k) d.shape[k] = shape[k];
    if (!elem_strides(0, esz, d.out_es)) return decline(R, OP, "out_strides");
    if (!elem_strides(1, esz, d.base_es)) return decline(R, OP, "base_strides");

    void* p_idx[HAGANE_INDEX_MAX_IDX] = {nullptr, nullptr, nullptr};
    int64_t idx_span[HAGANE_INDEX_MAX_IDX] = {0, 0, 0};
    for (int j = 0; j < n_idx; ++j) {
        const at::Tensor& ib = iter.tensor(2 + j);
        if (!ib.defined()) return decline(R, OP, "undefined_index");
        // The kernel has one index arm and it is int64, which is what the
        // caller's own guard already requires. An int32 index here would be
        // read as int64 garbage, so this is a silent-wrong guard, not a
        // capability limit.
        if (ib.scalar_type() != at::kLong)
            return decline_dtypes(R, OP, "index_dtype", ib.scalar_type(),
                                  out.scalar_type());
        if (!elem_strides(2 + j, ib.element_size(),
                          d.idx_es + j * HAGANE_INDEX_MAX_DIMS))
            return decline(R, OP, "index_strides");
        if (indexed_strides[j] % esz != 0) return decline(R, OP, "indexed_stride");
        d.isize[j]   = indexed_sizes[j];
        d.istride[j] = indexed_strides[j] / esz;
        if (d.isize[j] <= 0) return decline(R, OP, "empty_indexed_dim");
        if (d.istride[j] < 0) return decline(R, OP, "negative_indexed_stride");
        p_idx[j] = iter.data_ptr(2 + j);
        idx_span[j] = index_span_elems(d.shape, d.idx_es + j * HAGANE_INDEX_MAX_DIMS,
                                       ndim) * ib.element_size();
    }

    // `base` is read at the largest coordinate AND the largest wrapped index on
    // every indexed dim, so the settle must cover both terms.
    int64_t base_span = index_span_elems(d.shape, d.base_es, ndim);
    for (int j = 0; j < n_idx; ++j) base_span += (d.isize[j] - 1) * d.istride[j];
    const int64_t out_span = index_span_elems(d.shape, d.out_es, ndim);

    void* p_base = iter.data_ptr(1);
    void* p_out  = iter.data_ptr(0);
    flush_or_commit_metallib_input(p_base, base_span * esz);
    for (int j = 0; j < n_idx; ++j)
        flush_or_commit_metallib_input(p_idx[j], idx_span[j]);
    // J (#1080): device write — see the note in try_index_gather_axis.
    if (out_span == total) haganeOpsSettleForDeviceWrite(p_out, out_span * esz);
    else                   haganeOpsFlushRegion(p_out, out_span * esz);

    std::string kname = "hagane_index_gather" + std::to_string(n_idx) + "_" + bsfx;
    int64_t c_ndim = ndim, c_total = total;
    void*  args[8];
    int    at_[8];
    size_t as_[8];
    int n = 0;
    args[n] = p_base; at_[n] = 0; as_[n] = 0; ++n;
    for (int j = 0; j < n_idx; ++j) { args[n] = p_idx[j]; at_[n] = 0; as_[n] = 0; ++n; }
    args[n] = p_out;  at_[n] = 0; as_[n] = 0; ++n;
    args[n] = &d;      at_[n] = 1; as_[n] = sizeof(d); ++n;
    args[n] = &c_ndim; at_[n] = 1; as_[n] = sizeof(int64_t); ++n;
    args[n] = &c_total; at_[n] = 1; as_[n] = sizeof(int64_t); ++n;

    int64_t nb = (total + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 65535) nb = 65535;
    dim3 block(256, 1, 1), grid(static_cast<unsigned>(nb), 1, 1);
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, at_, as_, n) != hipSuccess)
        return decline(R, OP, "kernel_dispatch");
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(p_out, static_cast<size_t>(out_span * esz));
    return true;
}

// B' (S3 / #996) — advanced indexing as a WRITE, `index_put` with
// accumulate=false:
//
//   base[ SUM_d coord_d*base_es[d] + SUM_j wrap(idx_j[coord])*istride[j] ]
//       = value[value_es . coord]
//
// The mirror of try_index_gather_advanced. What it replaces is not a slower
// route, it is an UNCONDITIONAL HOP TO THE CPU — index_put_stub(CPU, ...) —
// which was the last op leaving the GPU on a real workload (11 calls per ARDY
// replan, the entire host_fallbacks count) and a hard blocker on deleting the
// stash: a host-side kernel cannot write the caller's block as a queued GPU
// write, by construction, so while one remains pending_/LazyGraph cannot go.
//
// OPERAND ROLES ARE THE OTHER WAY ROUND FROM THE GATHER, and that is the one
// thing to get right here. aten builds this iterator (make_index_put_iterator)
// as: operand 0 = `self` RESTRIDED with stride 0 on the indexed dims — the
// DESTINATION, and the tensor the index arithmetic applies to; operand 1 =
// `value` — the source; operands 2.. = the int64 indices. The shared
// HaganeIndexGatherDims keeps its meaning under that swap because `base_es`
// always strides the INDEXED operand and `out_es` always strides the other
// one, whichever happens to be written.
//
// THE SETTLE IS THE PART THAT IS NOT SYMMETRIC WITH THE GATHER. This write is
// PARTIAL by construction — it touches the indexed positions and nothing else
// — so a stash on `base` has to be MATERIALISED, never dropped. That is
// haganeOpsFlushRegion, unconditionally, exactly as try_index_scatter_axis
// does and for exactly the same reason: haganeOpsSettleForDeviceWrite ERASES a
// stash whose out_ptr is in range, which here would silently zero everything
// the put does not write (#1014 / RW-8.0). The gather's `out_span == total`
// branch has no analogue: we cannot know without reading the indices whether
// the put covers its base, so it is always treated as partial.
//
// Two behaviours are deliberately inherited rather than fixed here:
//   * Out-of-range indices WRAP then CLAMP (#1000), which is what every other
//     route in this family does. The CPU stub this replaces raised IndexError;
//     CUDA fires a device-side assert. Memory safety holds either way, and the
//     family needs one answer, not a special case on the writer.
//   * Duplicate indices race, with the winner one of the sources. torch
//     documents accumulate=false with duplicates as undefined and CUDA's own
//     kernel is an unordered store, so this IS the parity behaviour — the CPU
//     stub's last-wins was the accident.
inline bool try_index_put_advanced(TensorIterator& iter,
                                   at::IntArrayRef indexed_sizes,
                                   at::IntArrayRef indexed_strides,
                                   bool accumulate) {
    const char* R = "index_adv";
    const char* OP = "index_put";
    if (!index_metallib_available()) return false;
    route_enter(R, OP);
    if (haganeOpsTapeRecording()) return decline(R, OP, "tape_recording");
    // accumulate=true is a different op: it needs atomics and torch routes it
    // to index_put_with_sort_stub long before here. Guarded anyway, because
    // quietly overwriting instead of adding is the worst failure available.
    if (accumulate) return decline(R, OP, "accumulate");

    const int n_idx = static_cast<int>(iter.ntensors()) - 2;
    if (n_idx <= 0 || n_idx > HAGANE_INDEX_MAX_IDX)
        return decline(R, OP, "n_idx");
    if ((int)indexed_sizes.size() != n_idx || (int)indexed_strides.size() != n_idx)
        return decline(R, OP, "indexed_arity_mismatch");

    const at::Tensor& base  = iter.tensor(0);   // `self`, restrided — WRITTEN
    const at::Tensor& value = iter.tensor(1);   // the source
    if (!base.defined() || !value.defined()) return decline(R, OP, "undefined_operand");
    if (!base.is_cuda() || !value.is_cuda()) return decline(R, OP, "not_device");
    if (base.scalar_type() != value.scalar_type())
        return decline_dtypes(R, OP, "self_value_dtype_differ",
                              value.scalar_type(), base.scalar_type());

    const int64_t esz = base.element_size();
    const char* bsfx = index_bsuffix(esz);
    if (!bsfx) return decline(R, OP, "element_size");

    const auto shape = iter.shape();
    const int ndim = static_cast<int>(shape.size());
    if (ndim <= 0 || ndim > HAGANE_INDEX_MAX_DIMS) return decline(R, OP, "ndim");
    const int64_t total = iter.numel();
    if (total <= 0) return decline(R, OP, "empty");

    // Byte strides must be whole elements: the kernel works in element space,
    // and a partial-element stride cannot be expressed there at all.
    auto elem_strides = [&](int arg, int64_t item, int64_t* dst) -> bool {
        if (item <= 0) return false;
        const auto st = iter.strides(arg);
        if ((int)st.size() != ndim) return false;
        for (int k = 0; k < ndim; ++k) {
            if (st[k] % item != 0) return false;
            dst[k] = st[k] / item;
            if (dst[k] < 0) return false;   // see try_index_gather_axis
        }
        return true;
    };

    HaganeIndexGatherDimsHost d{};
    for (int k = 0; k < ndim; ++k) d.shape[k] = shape[k];
    if (!elem_strides(0, esz, d.base_es)) return decline(R, OP, "base_strides");
    if (!elem_strides(1, esz, d.out_es)) return decline(R, OP, "value_strides");

    void* p_idx[HAGANE_INDEX_MAX_IDX] = {nullptr, nullptr, nullptr};
    int64_t idx_span[HAGANE_INDEX_MAX_IDX] = {0, 0, 0};
    for (int j = 0; j < n_idx; ++j) {
        const at::Tensor& ib = iter.tensor(2 + j);
        if (!ib.defined()) return decline(R, OP, "undefined_index");
        // int64 is the kernel's only index arm and the caller's own guard
        // already requires it. An int32 index would be read as int64 garbage,
        // so this is a silent-wrong guard, not a capability limit.
        if (ib.scalar_type() != at::kLong)
            return decline_dtypes(R, OP, "index_dtype", ib.scalar_type(),
                                  base.scalar_type());
        if (!elem_strides(2 + j, ib.element_size(),
                          d.idx_es + j * HAGANE_INDEX_MAX_DIMS))
            return decline(R, OP, "index_strides");
        if (indexed_strides[j] % esz != 0) return decline(R, OP, "indexed_stride");
        d.isize[j]   = indexed_sizes[j];
        d.istride[j] = indexed_strides[j] / esz;
        if (d.isize[j] <= 0) return decline(R, OP, "empty_indexed_dim");
        if (d.istride[j] < 0) return decline(R, OP, "negative_indexed_stride");
        p_idx[j] = iter.data_ptr(2 + j);
        idx_span[j] = index_span_elems(d.shape, d.idx_es + j * HAGANE_INDEX_MAX_DIMS,
                                       ndim) * ib.element_size();
    }

    // `base` is WRITTEN at the largest coordinate AND the largest wrapped index
    // on every indexed dim, so both terms are in the span — the same quantity
    // the gather computes for its read, because it is the same address set.
    int64_t base_span = index_span_elems(d.shape, d.base_es, ndim);
    for (int j = 0; j < n_idx; ++j) base_span += (d.isize[j] - 1) * d.istride[j];
    const int64_t value_span = index_span_elems(d.shape, d.out_es, ndim);

    void* p_base  = iter.data_ptr(0);
    void* p_value = iter.data_ptr(1);
    flush_or_commit_metallib_input(p_value, value_span * esz);
    for (int j = 0; j < n_idx; ++j)
        flush_or_commit_metallib_input(p_idx[j], idx_span[j]);
    // MATERIALISE, never drop — see the header comment. Partial by
    // construction, so unlike the gather there is no full-overwrite branch.
    haganeOpsFlushRegion(p_base, base_span * esz);

    std::string kname = "hagane_index_put" + std::to_string(n_idx) + "_" + bsfx;
    int64_t c_ndim = ndim, c_total = total;
    void*  args[8];
    int    at_[8];
    size_t as_[8];
    int n = 0;
    args[n] = p_base; at_[n] = 0; as_[n] = 0; ++n;
    for (int j = 0; j < n_idx; ++j) { args[n] = p_idx[j]; at_[n] = 0; as_[n] = 0; ++n; }
    args[n] = p_value; at_[n] = 0; as_[n] = 0; ++n;
    args[n] = &d;       at_[n] = 1; as_[n] = sizeof(d); ++n;
    args[n] = &c_ndim;  at_[n] = 1; as_[n] = sizeof(int64_t); ++n;
    args[n] = &c_total; at_[n] = 1; as_[n] = sizeof(int64_t); ++n;

    int64_t nb = (total + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 65535) nb = 65535;
    dim3 block(256, 1, 1), grid(static_cast<unsigned>(nb), 1, 1);
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, at_, as_, n) != hipSuccess)
        return decline(R, OP, "kernel_dispatch");
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(p_base, static_cast<size_t>(base_span * esz));
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

// 1.11-A — the binary family's CPU-SCALAR arms (`_a1_sA` / `_a1_sB`).
//
// Distinct from try_launch_unary_scalar_metallib above in the one way that
// matters: **the scalar crosses as RAW BITS in the compute dtype**, not as a
// float. Pushing an int64 divisor through a float loses everything past 2^24 —
// the same defect class 1.6 removed from arange and 1.7b from clamp, and
// exactly what 1.11 step 0 measured as `16777217 // 1 -> 16777216`.
//
// `in_idx` is the TENSOR operand (1 or 2); the scalar is the other one. The
// KERNEL NAME already encodes which side the scalar is on, so this function
// never reasons about operand order — it binds the one buffer it is handed,
// and the caller's name choice is what makes the arithmetic right. The gate
// carries an operand-order case precisely because that pairing is the thing
// that can silently invert.
inline bool try_launch_binary_scalar_metallib(const std::string& kname,
                                              TensorIteratorBase& iter,
                                              int in_idx, uint64_t scalar_bits) {
    if (haganeOpsTapeRecording()) return false;  // record via MLX so replay is correct
    void* d_out = iter.data_ptr(0);
    void* d_in  = iter.data_ptr(in_idx);
    int N = static_cast<int>(iter.numel());
    if (N <= 0) return true;
    flush_or_commit_metallib_input(
        d_in, static_cast<int64_t>(N) * iter.element_size(in_idx));
    // The scalar param is declared at the STORAGE dtype in the harvested
    // kernel (gpu_kernel_with_scalars keeps the functor's arg at scalar_t), so
    // setBytes gets exactly element_size bytes — the low bytes of `sc` on this
    // little-endian target, which is where vendor_scalar_bytes wrote them.
    uint64_t sc = scalar_bits;
    const size_t esz = static_cast<size_t>(iter.element_size(0));
    const int wpt = metallib_work_per_thread(iter.dtype());
    const int nthreads = (N + wpt - 1) / wpt;
    dim3 block(256, 1, 1), grid((nthreads + 255) / 256, 1, 1);
    void*  args[]      = {d_in, d_out, &sc, &N};
    int    arg_types[] = {0, 0, 1, 1};
    size_t arg_sizes[] = {0, 0, esz, sizeof(int)};
    if (hagane_launch_kernel_mixed_tracked(kname.c_str(), grid, block, 0, nullptr,
                                           args, arg_types, arg_sizes, 4) != hipSuccess)
        return false;
    note_native_launch(kname);
    haganeOpsMarkMetallibWrite(d_out, static_cast<size_t>(N) * iter.element_size(0));
    return true;
}

// Run torch's own CPU kernel for a DECLINED binary op.
//
// A `c_abi_fn == nullptr` config declines onto the CPU stub, which is the
// correct answer by construction. But the CPU loops cannot be handed a
// CUDA-built iterator that still carries an unpromoted CPU SCALAR: TensorIterator
// leaves such an operand at Double and lets the CUDA kernel read it with a
// checked cast (that is what gpu_kernel_with_scalars is FOR), while upstream's
// CPU loops assert `!needs_dynamic_casting` and abort. Measured, not assumed —
// `torch.floor_divide(float_tensor, 0.4)` raised that internal assert the
// moment div_floor's C-ABI was removed, i.e. a crash where the contract says
// honest fallback.
//
// So hand the operands to a FRESH CPU iterator and let TensorIterator do its
// own promotion, exactly as a real CPU dispatch would.
//
// The cpu-scalar operand is passed THROUGH UNCHANGED. `iter.tensor(i)` is still
// the original wrapped-number tensor precisely because no cast was applied to
// it, and rebuilding it at the compute dtype would be a REAL divergence, not a
// detail: torch reads that scalar back at `accscalar_t` on BOTH backends — CUDA
// `scalar_value<accscalar_t>(2)`, CPU `original_scalar_value<accscalar_t>(2)`,
// each inside its own reciprocal branch. Measured: a first version that
// narrowed it gave `half(4.0) // 0.4 = 10` against the 9 that CUDA and CPU
// both produce.
//
// Everything else — no cpu scalar in play — runs the stub directly on the
// caller's iterator, unchanged, which keeps gcd/lcm byte-identical.
inline void run_binary_cpu_fallback(TensorIteratorBase& iter,
                                    void (*stub)(TensorIteratorBase&)) {
    if (iter.ninputs() != 2) { ::haganeOpsFlush(); stub(iter); return; }

    // Every operand is brought over with torch's own `.to(CPU)`, which is a
    // real D2H copy and therefore respects the deferred-write barrier. Handing
    // the CPU kernel the CUDA iterator directly — which is what this path used
    // to do, and still does for gcd/lcm's non-binary shapes — punning device
    // pointers as host ones because UMA makes that *appear* to work, reads
    // STALE BYTES for any lazily produced operand. Found by the 1.11-A gate: a
    // broadcast `floor_divide` raised "ZeroDivisionError" because the divisor
    // buffer still read as zeros, and haganeOpsFlush() alone did not settle it.
    // #1025's rule again — enumerate the readers of a deferred write by the
    // QUEUE they run on, not by function name.
    at::Tensor in[2];
    for (int i = 1; i <= 2; ++i)
        in[i - 1] = iter.tensor(i).to(c10::DeviceType::CPU);
    at::Tensor out_cpu = at::empty(
        iter.tensor(0).sizes(),
        at::TensorOptions().dtype(iter.dtype(0)).device(c10::DeviceType::CPU));
    auto cpu_iter = at::TensorIterator::binary_op(out_cpu, in[0], in[1]);
    stub(cpu_iter);
    iter.tensor(0).copy_(out_cpu);
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
            // The kernel is named by the OUTPUT dtype and binds data_ptr(1) raw,
            // so a PROMOTED INPUT would be read as the output's type: `iter.dtype()`
            // is Float for `sqrt(int64)` while the buffer holds int64. That
            // reinterpreted the bytes — sqrt(int64 1) came back 0.0, exp came back
            // 1.0, log came back -inf — a silent-wrong across the whole owned
            // unary family, and it predates 1.9. torch does this on purpose: it
            // promotes a unary op's OUTPUT and leaves the INPUT alone, because
            // upstream's gpu_kernel casts on the load and ours does not.
            //
            // Decline here and let try_vendor_unary below take it, which casts the
            // input with MLX's own v_copy first (#1031).
            if (MetallibState<Cfg>::available && iter.is_contiguous() &&
                iter.dtype(1) == iter.dtype(0)) {
                std::string kname =
                    metallib_kernel_name(Cfg.metallib_kernel, iter.dtype(), "");
                if (!kname.empty() && try_launch_unary_metallib(kname, iter)) return;
            }
        }

        // #1010: MLX's kernel, our output block. Same position as the binary
        // family — after our own metallib, before the MLX C-ABI that allocates.
        if (try_vendor_unary(Cfg.op_name, iter)) return;

        auto out = make_ops_tensor_local(iter, 0);
        auto in  = make_ops_tensor_local(iter, 1);
        if (Cfg.c_abi_fn(&in, &out) != HAGANE_OPS_SUCCESS) {
            ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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

    // 1.11-A — the CPU-SCALAR arms. A cpu scalar has stride 0, so
    // iter.is_contiguous() is FALSE and the `_a2` guard above can never take
    // one; that is precisely why all 12 of ARDY's floor_divides fell past the
    // metallib to the C-ABI, and why the stash survived every earlier slice.
    //
    // AUnaryFunctor holds the scalar as operand ONE, BUnaryFunctor as operand
    // TWO (Loops.cuh:140/154), so the arm is chosen from is_cpu_scalar rather
    // than inferred — get it backwards and every non-commutative op computes
    // `scalar OP tensor`.
    if constexpr (Cfg.metallib_kernel != nullptr) {
        if (MetallibState<Cfg>::available && iter.ninputs() == 2) {
            const bool s1 = iter.is_cpu_scalar(1);
            const bool s2 = iter.is_cpu_scalar(2);
            // Upstream's CUDA-only reciprocal branch for a FLOATING cpu scalar
            // is a different formula from the general one the harvest carries
            // (see BinaryOpConfig::float_cpu_scalar_differs), so those decline
            // here and are owned separately. Integral dtypes have no such
            // branch and the harvested arm is exactly what torch dispatches.
            const bool arm_faithful =
                !Cfg.float_cpu_scalar_differs ||
                c10::isIntegralType(iter.dtype(), /*includeBool=*/true);
            const int tin = s1 ? 2 : 1;
            if ((s1 != s2) && arm_faithful
                && iter.tensor(tin).scalar_type() == iter.dtype()
                && iter.tensor(tin).is_contiguous()
                && iter.tensor(0).is_contiguous()
                && iter.tensor(tin).sizes() == iter.tensor(0).sizes()) {
                uint64_t bits = 0;
                if (vendor_scalar_bytes(iter, s1 ? 1 : 2, iter.dtype(), &bits)) {
                    std::string kname = metallib_kernel_name(
                        Cfg.metallib_kernel, iter.dtype(), s1 ? "_a1_sA" : "_a1_sB");
                    if (!kname.empty() &&
                        try_launch_binary_scalar_metallib(kname, iter, tin, bits))
                        return;
                }
            }
        }
    }

    // #1010: MLX's kernel, our output block. Tried after our own metallib (that
    // one is already ours end-to-end) and before the MLX C-ABI (which is the
    // path that makes MLX allocate).
    if (try_vendor_binary(Cfg.op_name, iter)) return;

    if constexpr (Cfg.c_abi_fn == nullptr) {
        // No MLX path (e.g. integer gcd/lcm, and 1.11-A's div/mod family): the
        // metallib is the only device route; fall to torch's own CPU kernel on
        // fallthrough (route-off/unsupported). run_binary_cpu_fallback is the
        // plain stub call unless the iterator carries an unpromoted cpu scalar,
        // which the CPU loops abort on — see its comment.
        ::haganeOpsHostFallbackNote(Cfg.op_name, "no_device_route");
        run_binary_cpu_fallback(iter, Cfg.cpu_fallback);
    } else {
        auto out = make_ops_tensor_local(iter, 0);
        at::Tensor sa, sb;
        auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
        auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
        if (Cfg.c_abi_fn(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
            ::haganeOpsFlush();
            ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
    // Both routes take Cfg.op_name and decline by name what they do not own,
    // so which ops carry a `metallib_kernel` and which ops have an arm are two
    // independent facts. They used to be one — `op_name == "mean"` — which held
    // only while sum and mean were the whole sentinel set.
    if (Cfg.metallib_kernel != nullptr &&
        try_launch_reduction_metallib(iter, Cfg.op_name))
        return;

    // #1135: the PARTIAL sum/mean, which the line above has declined by name
    // since A9. Ordered after it so the full reduction keeps its 2-pass path
    // and the two routes' counts stay disjoint.
    if (Cfg.metallib_kernel != nullptr &&
        try_launch_reduction_dim_metallib(iter, Cfg.op_name))
        return;

    // #1010 1.9: MLX's all_reduce, our output block. A partial reduction
    // declines inside and falls through to the row_reduce route below; anything
    // neither of them covers falls through to the MLX C-ABI unchanged.
    if (Cfg.mlx_reduce_op != nullptr &&
        try_vendor_reduce_all(Cfg.op_name, Cfg.mlx_reduce_op,
                              iter.tensor(1), iter.tensor(0)))
        return;

    // #1010 1.9b: MLX's row_reduce, our output block.
    if (Cfg.mlx_reduce_op != nullptr &&
        try_vendor_reduce_dim(Cfg.op_name, Cfg.mlx_reduce_op,
                              iter.tensor(1), iter.tensor(0)))
        return;

    auto out = make_ops_tensor_local(iter, 0);
    auto in  = make_ops_tensor_local(iter, 1);
    if (Cfg.c_abi_fn(&in, &out) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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

    // #1010: alpha is part of the kernel. out = a + alpha*b is NOT MLX's Add
    // for a general alpha, and dropping it here would be exactly the
    // silent-wrong class the project forbids. But two alphas ARE another op
    // exactly: a + (-1)*b is Subtract, and a - (-1)*b is Add.
    //
    // That identity is not a nicety — it is the whole subtraction family.
    // torch's sub_out calls add_stub(-alpha) (BinaryOps.cpp), so EVERY
    // subtraction arrives here as add with alpha = -1 and sub_stub is never
    // reached. Without this, `a - b` could not route at all: ARDY's step showed
    // sub.Tensor at 187 MLX-owned dispatches and zero owned.
    if (const char* eop = alpha_effective_op(Cfg.op_name, alpha))
        if (try_vendor_binary(eop, iter)) return;

    // #1026 — everything past here evaluates `a + alpha*b` in FLOAT32, because
    // the C-ABI takes a `float alpha` and haganeOpsAdd builds it as a float32
    // mx::array. Faithful for the float dtypes (CUDA's accscalar_t is float
    // there too), not faithful for the integer ones. Take torch's own exact
    // implementation rather than return a plausible wrong answer — measured, the
    // wrong answer was torch.add(int64, 3, alpha=2**24+1) off by 4.
    //
    // This is a decline, not a route: the route is `tmp = alpha*b; out = a+tmp`
    // as two owned dispatches, which is 1.7b's shape and a follow-on. Keeping a
    // silent-wrong until then is not an option the project has.
    if (!alpha_expressible_in_float(iter.common_dtype(), alpha)) {
        (void)decline("bin_alpha", Cfg.op_name, "alpha_not_float_exact");
        ::haganeOpsFlush();
        ::haganeOpsHostFallbackNote(Cfg.op_name, "route_declined");
        Cfg.cpu_fallback(iter, alpha);
        return;
    }

    note_general_alpha(Cfg.op_name, iter, alpha);

    auto out = make_ops_tensor_local(iter, 0);
    at::Tensor sa, sb;
    auto a = make_ops_tensor_or_scalar_local(iter, 1, sa);
    auto b = make_ops_tensor_or_scalar_local(iter, 2, sb);
    if (Cfg.c_abi_fn(&a, &b, &out, alpha.toFloat()) != HAGANE_OPS_SUCCESS) {
        ::haganeOpsFlush();
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
    int32_t dim32 = static_cast<int32_t>(dim < 0 ? dim + self.dim() : dim);
    if (try_vendor_scan(Cfg.op_name, Cfg.mlx_scan_op, self, result, dim32)) return;
    haganeOpsTensor_t in  = make_ops_tensor_from_tensor(self);
    haganeOpsTensor_t out = make_ops_tensor_from_tensor(result);
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
        Cfg.cpu_fallback(iter, correction, take_sqrt);
        return;
    }
    if (nout == 2) {
        auto mean_out = make_ops_tensor_local(iter, 1);
        if (haganeOpsMean(&in, &mean_out) != HAGANE_OPS_SUCCESS) {
            ::haganeOpsFlush();
            ::haganeOpsHostFallbackNote(Cfg.op_name, "route_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
        ::haganeOpsHostFallbackNote(Cfg.op_name, "c_abi_declined");
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
