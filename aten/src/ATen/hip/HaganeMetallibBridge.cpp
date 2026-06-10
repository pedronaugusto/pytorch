// Copyright (c) 2026 Pedro Augusto
// SPDX-License-Identifier: MIT
//
// HaganeMetallibBridge.cpp — REGISTER_DISPATCH-only TU after Sprint X+21
// extraction. All helpers / structs / per-op trampolines / bridge templates
// live in `hagane_dispatch.h`. Each row here corresponds to one entry
// retired from `HaganeOps.cpp` (X+15 abs → X+21 max/min_elementwise).
//
// At Sprint EE the bridge itself is deleted alongside `HaganeOps.cpp`
// (ADR-036 §6 — transition artifact).
//
// REGISTER_DISPATCH (NOT __attribute__((constructor)))
// ====================================================
// X+15 Lane C diagnosed: constructor-time `set_hip_dispatch_ptr` corrupts
// adjacent DispatchStub init state. All registration is at static-init via
// REGISTER_DISPATCH, the same mechanism every other PyTorch op uses.

#include "hagane_dispatch.h"

namespace at::native {

using namespace at::native::hagane_dispatch::detail;

// ---- Unary `OpConfig` rows (37 ops) ---------------------------------------
REGISTER_DISPATCH(abs_stub,        &hagane_kernel_bridge<kAbsCfg>)
REGISTER_DISPATCH(neg_stub,        &hagane_kernel_bridge<kNegCfg>)
REGISTER_DISPATCH(sign_stub,       &hagane_kernel_bridge<kSignCfg>)
REGISTER_DISPATCH(sgn_stub,        &hagane_kernel_bridge<kSgnCfg>)
REGISTER_DISPATCH(exp_stub,        &hagane_kernel_bridge<kExpCfg>)
REGISTER_DISPATCH(log_stub,        &hagane_kernel_bridge<kLogCfg>)
REGISTER_DISPATCH(sqrt_stub,       &hagane_kernel_bridge<kSqrtCfg>)
REGISTER_DISPATCH(sin_stub,        &hagane_kernel_bridge<kSinCfg>)
REGISTER_DISPATCH(cos_stub,        &hagane_kernel_bridge<kCosCfg>)
REGISTER_DISPATCH(ceil_stub,       &hagane_kernel_bridge<kCeilCfg>)
REGISTER_DISPATCH(floor_stub,      &hagane_kernel_bridge<kFloorCfg>)
REGISTER_DISPATCH(tanh_stub,       &hagane_kernel_bridge<kTanhCfg>)
REGISTER_DISPATCH(sigmoid_stub,    &hagane_kernel_bridge<kSigmoidCfg>)

REGISTER_DISPATCH(reciprocal_stub, &hagane_kernel_bridge<kReciprocalCfg>)
REGISTER_DISPATCH(rsqrt_stub,      &hagane_kernel_bridge<kRsqrtCfg>)
REGISTER_DISPATCH(round_stub,      &hagane_kernel_bridge<kRoundCfg>)
REGISTER_DISPATCH(trunc_stub,      &hagane_kernel_bridge<kTruncCfg>)
REGISTER_DISPATCH(erf_stub,        &hagane_kernel_bridge<kErfCfg>)
REGISTER_DISPATCH(log2_stub,       &hagane_kernel_bridge<kLog2Cfg>)
REGISTER_DISPATCH(log10_stub,      &hagane_kernel_bridge<kLog10Cfg>)
REGISTER_DISPATCH(log1p_stub,      &hagane_kernel_bridge<kLog1pCfg>)
REGISTER_DISPATCH(exp2_stub,       &hagane_kernel_bridge<kExp2Cfg>)

REGISTER_DISPATCH(expm1_stub,       &hagane_kernel_bridge<kExpm1Cfg>)
REGISTER_DISPATCH(bitwise_not_stub, &hagane_kernel_bridge<kBitwiseNotCfg>)
REGISTER_DISPATCH(logical_not_stub, &hagane_kernel_bridge<kLogicalNotCfg>)

REGISTER_DISPATCH(tan_stub,        &hagane_kernel_bridge<kTanCfg>)
REGISTER_DISPATCH(acos_stub,       &hagane_kernel_bridge<kAcosCfg>)
REGISTER_DISPATCH(asin_stub,       &hagane_kernel_bridge<kAsinCfg>)
REGISTER_DISPATCH(atan_stub,       &hagane_kernel_bridge<kAtanCfg>)
REGISTER_DISPATCH(cosh_stub,       &hagane_kernel_bridge<kCoshCfg>)
REGISTER_DISPATCH(sinh_stub,       &hagane_kernel_bridge<kSinhCfg>)
REGISTER_DISPATCH(erfc_stub,       &hagane_kernel_bridge<kErfcCfg>)

// X+21 — lgamma re-migration (X+20 revert resolved via hagane_ops.cpp:1335
// `mx::eval(x)` → `::haganeOpsFlush()` flush fix).
REGISTER_DISPATCH(lgamma_stub,     &hagane_kernel_bridge<kLgammaCfg>)

// X+21 — frac brace-defect bug-fix-by-migration.
REGISTER_DISPATCH(frac_stub,       &hagane_kernel_bridge<kFracCfg>)

REGISTER_DISPATCH(hardsigmoid_stub, &hagane_kernel_bridge<kHardsigmoidCfg>)
REGISTER_DISPATCH(mish_stub,        &hagane_kernel_bridge<kMishCfg>)
REGISTER_DISPATCH(silu_stub,        &hagane_kernel_bridge<kSiluCfg>)

// ---- Binary `BinaryOpConfig` rows (25 ops) --------------------------------
REGISTER_DISPATCH(eq_stub,                &hagane_binary_bridge<kEqCfg>)
REGISTER_DISPATCH(ne_stub,                &hagane_binary_bridge<kNeCfg>)
REGISTER_DISPATCH(lt_stub,                &hagane_binary_bridge<kLtCfg>)
REGISTER_DISPATCH(gt_stub,                &hagane_binary_bridge<kGtCfg>)
REGISTER_DISPATCH(le_stub,                &hagane_binary_bridge<kLeCfg>)
REGISTER_DISPATCH(ge_stub,                &hagane_binary_bridge<kGeCfg>)
REGISTER_DISPATCH(mul_stub,               &hagane_binary_bridge<kMulCfg>)

REGISTER_DISPATCH(pow_tensor_tensor_stub, &hagane_binary_bridge<kPowTtCfg>)
REGISTER_DISPATCH(atan2_stub,             &hagane_binary_bridge<kAtan2Cfg>)
REGISTER_DISPATCH(remainder_stub,         &hagane_binary_bridge<kRemainderCfg>)
REGISTER_DISPATCH(fmod_stub,              &hagane_binary_bridge<kFmodCfg>)
REGISTER_DISPATCH(div_true_stub,          &hagane_binary_bridge<kDivTrueCfg>)
REGISTER_DISPATCH(div_floor_stub,         &hagane_binary_bridge<kDivFloorCfg>)
REGISTER_DISPATCH(div_trunc_stub,         &hagane_binary_bridge<kDivTruncCfg>)

// X+21 Lane D — 6 binary ops on `structured_binary_fn` (TensorIteratorBase&).
REGISTER_DISPATCH(bitwise_and_stub,       &hagane_binary_bridge<kBitwiseAndCfg>)
REGISTER_DISPATCH(bitwise_or_stub,        &hagane_binary_bridge<kBitwiseOrCfg>)
REGISTER_DISPATCH(bitwise_xor_stub,       &hagane_binary_bridge<kBitwiseXorCfg>)
REGISTER_DISPATCH(maximum_stub,           &hagane_binary_bridge<kMaximumCfg>)
REGISTER_DISPATCH(minimum_stub,           &hagane_binary_bridge<kMinimumCfg>)
REGISTER_DISPATCH(copysign_stub,          &hagane_binary_bridge<kCopysignCfg>)

// X+21 Lane D — 3 binary ops on `binary_fn` (TensorIterator&).
// max_elementwise_stub / min_elementwise_stub: declared in BinaryOps.h but
// never DEFINE_DISPATCH'd upstream — phantom stubs, no `::DEFAULT` symbol.
// HaganeOps.cpp definitions were orphaned dead code; removed in Lane E
// without a bridge registration.
REGISTER_DISPATCH(logical_and_stub,       &hagane_binary_iter_bridge<kLogicalAndCfg>)
REGISTER_DISPATCH(logical_or_stub,        &hagane_binary_iter_bridge<kLogicalOrCfg>)
REGISTER_DISPATCH(logical_xor_stub,       &hagane_binary_iter_bridge<kLogicalXorCfg>)

// ---- Binary-alpha `BinaryAlphaOpConfig` rows (2 ops) ----------------------
REGISTER_DISPATCH(add_stub,        &hagane_binary_alpha_bridge<kAddCfg>)
REGISTER_DISPATCH(sub_stub,        &hagane_binary_alpha_bridge<kSubCfg>)

// ---- Unary-scalar `UnaryScalarOpConfig` rows (5 ops) ----------------------
REGISTER_DISPATCH(pow_tensor_scalar_stub, &hagane_unary_scalar_bridge<kPowTsCfg>)
REGISTER_DISPATCH(leaky_relu_stub,        &hagane_unary_scalar_bridge<kLeakyReluCfg>)
REGISTER_DISPATCH(hardshrink_stub,        &hagane_unary_scalar_bridge<kHardshrinkCfg>)
REGISTER_DISPATCH(softshrink_stub,        &hagane_unary_scalar_bridge<kSoftshrinkCfg>)
REGISTER_DISPATCH(logit_stub,             &hagane_unary_scalar_bridge<kLogitCfg>)

// ---- Unary-iter `UnaryIterOpConfig` rows (X+22, 10 ops) -------------------
// 9 reductions via reduce_fn + 1 activation (hardswish) via hardswish_fn —
// both alias to `void(*)(TensorIterator&)`. C-ABIs are pure-MLX
// `copy_result(...)` (hagane/src/runtime/hagane_ops.cpp:909-1729 +
// :1594) — no flush-before-read fix needed.
REGISTER_DISPATCH(sum_stub,         &hagane_unary_iter_bridge<kSumCfg>)
REGISTER_DISPATCH(mean_stub,        &hagane_unary_iter_bridge<kMeanCfg>)
REGISTER_DISPATCH(prod_stub,        &hagane_unary_iter_bridge<kProdCfg>)
REGISTER_DISPATCH(argmax_stub,      &hagane_unary_iter_bridge<kArgmaxCfg>)
REGISTER_DISPATCH(argmin_stub,      &hagane_unary_iter_bridge<kArgminCfg>)
REGISTER_DISPATCH(max_values_stub,  &hagane_unary_iter_bridge<kMaxValuesCfg>)
REGISTER_DISPATCH(min_values_stub,  &hagane_unary_iter_bridge<kMinValuesCfg>)
REGISTER_DISPATCH(and_stub,         &hagane_unary_iter_bridge<kAndCfg>)
REGISTER_DISPATCH(or_stub,          &hagane_unary_iter_bridge<kOrCfg>)
REGISTER_DISPATCH(hardswish_stub,   &hagane_unary_iter_bridge<kHardswishCfg>)

// ---- Reduce-with-flag `ReduceFlagOpConfig` rows (X+23, 2 ops) -------------
// norm_stub / powsum_stub via reduce_fn_flag (TensorIterator&, const Scalar&).
// Bridge derives reduction dim from input/output shape and forwards
// (p.toDouble(), dim) to existing haganeOpsNormVal / haganeOpsPowsum C-ABIs.
REGISTER_DISPATCH(norm_stub,        &hagane_unary_iter_flag_bridge<kNormCfg>)
REGISTER_DISPATCH(powsum_stub,      &hagane_unary_iter_flag_bridge<kPowsumCfg>)

// ---- Binary fmax/fmin `BinaryOpConfig` rows (X+23, 2 ops) -----------------
// NaN-aware via X+23 runtime additions (haganeOpsFmax / haganeOpsFmin —
// mx::where(isnan(a), b, mx::where(isnan(b), a, mx::maximum/minimum(a, b)))).
REGISTER_DISPATCH(fmax_stub,        &hagane_binary_bridge<kFmaxCfg>)
REGISTER_DISPATCH(fmin_stub,        &hagane_binary_bridge<kFminCfg>)

// ---- X+24 sinc / signbit standard `OpConfig` rows (2 ops) -----------------
// Pure-MLX C-ABIs added in hagane/src/runtime/hagane_ops.cpp (sinc via
// mx::sin + double-where divide-by-zero guard; signbit via mx::less(x, 0)).
REGISTER_DISPATCH(sinc_stub,        &hagane_kernel_bridge<kSincCfg>)
REGISTER_DISPATCH(signbit_stub,     &hagane_kernel_bridge<kSignbitCfg>)

// ---- X+24 cumulative `CumOpConfig` rows (3 ops) ---------------------------
// Existing C-ABIs (hagane_ops.cpp:1842/1851/1936) are pure-MLX
// copy_result(mx::cumsum/cumprod/...). cumsum/cumprod use the
// structured_cum_fn signature; logcumsumexp uses cum_fn (mutable result).
REGISTER_DISPATCH(cumsum_stub,       &hagane_cum_bridge_structured<kCumsumCfg>)
REGISTER_DISPATCH(cumprod_stub,      &hagane_cum_bridge_structured<kCumprodCfg>)
REGISTER_DISPATCH(logcumsumexp_stub, &hagane_cum_bridge_mutable<kLogcumsumexpCfg>)

// ---- X+24 std_var `ReduceStdVarOpConfig` row (1 op) -----------------------
// std_var_stub takes (TensorIterator&, double correction, bool take_sqrt).
// Existing haganeOpsVar(in, out, double, int) — wrap shim casts bool→int.
// For 2-output std_var (var_mean), bridge additionally calls haganeOpsMean
// to populate output 1.
REGISTER_DISPATCH(std_var_stub,      &hagane_unary_iter_stdvar_bridge<kStdVarCfg>)

// ---- X+25 nan_to_num `UnaryOptionalTripleOpConfig` row (1 op) -------------
// nan_to_num_stub takes (TensorIteratorBase&, optional<double>×3). New
// haganeOpsNanToNum C-ABI (hagane_ops.cpp) is a pure-MLX mx::where chain over
// isnan/isinf masks. ADR-036 row 88.
REGISTER_DISPATCH(nan_to_num_stub,            &hagane_unary_optional_triple_bridge<kNanToNumCfg>)

// X+26 Lane B — hardsigmoid_backward retired (ADR-036 row 89). The bridge
// route + wrap_tensor pending_get fix delivers correct parity-vs-CPU under
// HAGANE_FP32_EAGER_EVAL=1.
REGISTER_DISPATCH(hardsigmoid_backward_stub,  &hagane_binary_bridge<kHardsigmoidBackwardCfg>)

// X+27 Lane B — 3 deferred activation backwards retired (ADR-036 rows 90-92).
// Bridge route bypasses the retired kernels' at::*-on-iter.tensor lazy-stash
// interaction by going pure-MLX through C-ABIs. leaky_relu_backward routes via
// leaky_relu_backward_wrap which inverts a/b to match Activation.cpp:190's
// build_borrowing_binary_op(output, self_or_result=INPUT, grad_output=GRAD)
// iter convention. hardswish_backward / mish_backward use the standard
// (grad_out=pos1, input=pos2) convention per Activation.cpp:509/565.
REGISTER_DISPATCH(leaky_relu_backward_stub,   &hagane_binary_alpha_bridge<kLeakyReluBackwardCfg>)
REGISTER_DISPATCH(hardswish_backward_stub,    &hagane_binary_iter_bridge<kHardswishBackwardCfg>)
REGISTER_DISPATCH(mish_backward_stub,         &hagane_binary_iter_bridge<kMishBackwardCfg>)

// X+28 Lane A — sigmoid/tanh/logit backwards retired (ADR-036 rows 93-95).
// Per BinaryOps.cpp:306-316 the TORCH_META_FUNC iter is built as
// (out, grad_output, output_or_input); bridges pass a=grad_output, b=output/input
// directly to (grad_out, output/input, grad_in[, eps]) — no swap shim.
REGISTER_DISPATCH(sigmoid_backward_stub,      &hagane_binary_bridge<kSigmoidBackwardCfg>)
REGISTER_DISPATCH(tanh_backward_stub,         &hagane_binary_bridge<kTanhBackwardCfg>)
REGISTER_DISPATCH(logit_backward_stub,        &hagane_binary_alpha_bridge<kLogitBackwardCfg>)

// X+28 Lane B — elu/softplus/hardtanh backwards retired (ADR-036 rows 96-98).
// Formula fixes vs the retired HaganeOps.cpp kernels: strict `<` / `>`
// inequalities at the threshold boundary (CPU uses strict; HaganeOps.cpp used
// inclusive, leaking gradient where CPU returned 0 or vice versa). The bridge
// route + pure-MLX wrap_tensor also bypasses the at::*-on-iter.tensor
// lazy-stash interaction that drove the gating FAILs at X+27 baseline.
REGISTER_DISPATCH(elu_backward_stub,           &hagane_elu_backward_bridge<kEluBackwardCfg>)
REGISTER_DISPATCH(softplus_backward_stub,      &hagane_binary_double_scalar_bridge<kSoftplusBackwardCfg>)
REGISTER_DISPATCH(hardtanh_backward_stub,      &hagane_binary_double_scalar_iter_bridge<kHardtanhBackwardCfg>)

// X+30 Lane C/D — upsample_nearest backward retirements (ADR-036 rows 99-101).
// Mechanism: the HaganeOps.cpp TORCH_IMPL_FUNC bodies for the 3 non-exact
// upsample_nearest backward stubs now call the DispatchStub (instead of the
// C-ABI haganeOps* directly). The bridge owns the stub via REGISTER_DISPATCH.
// The full `.cu`-patch retirement (compiling the hipified .hip under Hagane)
// is blocked by transitive `<<<>>>` inclusion through PersistentSoftmax.cuh;
// recorded as X+31+ entry condition.
REGISTER_DISPATCH(upsample_nearest1d_backward_kernel,
                  &hagane_upsample_scale1d_bridge<kUpsampleNearest1dBackwardCfg>)
REGISTER_DISPATCH(upsample_nearest2d_backward_kernel,
                  &hagane_upsample_scale2d_bridge<kUpsampleNearest2dBackwardCfg>)
REGISTER_DISPATCH(upsample_nearest3d_backward_kernel,
                  &hagane_upsample_scale3d_bridge<kUpsampleNearest3dBackwardCfg>)

// X+31 Lane A — align-corners upsample backward retirements (ADR-036 rows 102-104).
// Same stub-redirect mechanism as X+30. C-ABIs in hagane_ops.cpp already carry
// the X+29 fix template (haganeOpsFlush + scratch + copy_result) applied in
// Lane A.1. bicubic2d_backward omitted (no DispatchStub upstream — pending
// .cu-patch unblocker via Lane E header sweep).
REGISTER_DISPATCH(upsample_linear1d_backward_kernel,
                  &hagane_upsample_scale_alignc_1d_bridge<kUpsampleLinear1dBackwardCfg>)
REGISTER_DISPATCH(upsample_bilinear2d_backward_kernel,
                  &hagane_upsample_scale_alignc_2d_bridge<kUpsampleBilinear2dBackwardCfg>)
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel,
                  &hagane_upsample_scale_alignc_3d_bridge<kUpsampleTrilinear3dBackwardCfg>)

// X+31 Lane B — exact-nearest backward retirements (ADR-036 rows 105-107).
// C-ABI in hagane_ops.cpp extended with `int exact` (centered source-index
// for exact variants); same scratch + copy_result path as non-exact siblings.
REGISTER_DISPATCH(_upsample_nearest_exact1d_backward_kernel,
                  &hagane_upsample_scale1d_bridge<kUpsampleNearestExact1dBackwardCfg>)
REGISTER_DISPATCH(_upsample_nearest_exact2d_backward_kernel,
                  &hagane_upsample_scale2d_bridge<kUpsampleNearestExact2dBackwardCfg>)
REGISTER_DISPATCH(_upsample_nearest_exact3d_backward_kernel,
                  &hagane_upsample_scale3d_bridge<kUpsampleNearestExact3dBackwardCfg>)

// X+31 Lane C — pool backward retirements (ADR-036 rows 108-110).
// avg_pool2d/3d_backward are TORCH_IMPL_FUNCs routed through DispatchStub.
// max_pool3d_with_indices_backward is C10_EXPORT but routes through the
// max_pool3d_backward_kernel stub the bridge REGISTER_DISPATCHes.
REGISTER_DISPATCH(avg_pool2d_backward_kernel,
                  &hagane_avg_pool2d_backward_bridge<kAvgPool2dBackwardCfg>)
REGISTER_DISPATCH(max_pool3d_backward_kernel,
                  &hagane_max_pool3d_backward_bridge<kMaxPool3dBackwardCfg>)

// X+32 Lane A — max_pool2d_backward retirement (ADR-036 row 111).
// haganeOpsMaxPool2dBackward (hagane_ops.cpp:3270) now carries the X+29 fix
// template (flush + scratch + copy_result + flat_idx != -1 sentinel guard,
// mirroring the 3D sibling at line 3367). Resolves the X+31 Lane D lazy-stash
// race (regression FAIL 2.875 from bridge route).
REGISTER_DISPATCH(max_pool2d_backward_kernel,
                  &hagane_pool_backward_with_indices_bridge<kMaxPool2dBackwardCfg>)

// X+32 Lane C — anti-aliased upsample backward retirements (ADR-036 rows 112-113).
// New C-ABIs in hagane_ops.cpp implement the CPU AA scatter formula directly
// (PIL Resample.c style) with X+29 fix template. Replaces pre-X+32 pseudo-impl
// that silently routed AA backward to the non-AA C-ABIs.
REGISTER_DISPATCH(_upsample_bilinear2d_aa_backward_kernel,
                  &hagane_upsample_scale_alignc_2d_bridge<kUpsampleBilinear2dAABackwardCfg>)
REGISTER_DISPATCH(_upsample_bicubic2d_aa_backward_kernel,
                  &hagane_upsample_scale_alignc_2d_bridge<kUpsampleBicubic2dAABackwardCfg>)

// X+32 Lane D — bicubic2d_backward retirement via upstream DispatchStub patch
// (ADR-036 row 114). UpSample.h + UpSampleBicubic2d.cpp now expose
// upsample_bicubic2d_backward_kernel as a DispatchStub (was a free function).
REGISTER_DISPATCH(upsample_bicubic2d_backward_kernel,
                  &hagane_upsample_scale_alignc_2d_bridge<kUpsampleBicubic2dBackwardCfg>)

// X+37 Lane B — adaptive_max_pool{2,3}d_backward retirements (ADR-036 rows
// 115-116). C-ABIs haganeOpsAdaptiveMaxPool{2,3}dBackward already exist in
// libhagane-runtime; signatures match PoolBackwardWithIndicesOpConfig (Tensor& +
// gradOutput + indices), so the existing hagane_pool_backward_with_indices_bridge
// template instantiates cleanly with the new Cfg constants in
// hagane_dispatch.h. HaganeOps.cpp's adaptive_max_pool{2,3}d_backward_out_cuda
// TIFs migrated from direct-C-ABI to .cu-patch admission (per X+37 Lane B
// admission of AdaptiveMaxPool{2,3}d.hip).
REGISTER_DISPATCH(adaptive_max_pool2d_backward_kernel,
                  &hagane_pool_backward_with_indices_bridge<kAdaptiveMaxPool2dBackwardCfg>)
REGISTER_DISPATCH(adaptive_max_pool3d_backward_kernel,
                  &hagane_pool_backward_with_indices_bridge<kAdaptiveMaxPool3dBackwardCfg>)

// X+38 Lane A — avg_pool3d_backward retirement (ADR-036 row 117). Bridge
// infrastructure (AvgPool3dOpConfig + kAvgPool3dBackwardCfg + template)
// existed from X+31 Lane C; was held back by the missing upstream
// DEFINE_DISPATCH(avg_pool3d_backward_kernel). X+38 Lane A.1 added the
// DEFINE in AveragePool3d.cpp (parallel to AveragePool2d.cpp:254-255 + the
// X+37 adaptive_max_pool3d pattern). Now wired up.
REGISTER_DISPATCH(avg_pool3d_backward_kernel,
                  &hagane_avg_pool3d_backward_bridge<kAvgPool3dBackwardCfg>)

// X+40 Lane A — adaptive_avg_pool{2,3}d_backward retirements (ADR-036 rows
// 118-119). C-ABIs haganeOpsAdaptiveAvgPool{2,3}dBackward exist in
// libhagane-runtime; DispatchStub typedefs share the 2-arg signature
// `void(*)(Tensor& grad_input, const Tensor& grad_output)`. Single OpConfig
// (AdaptiveAvgPoolBackwardOpConfig) + single bridge template
// (hagane_adaptive_avg_pool_backward_bridge) reused across 2D + 3D. The 3D
// DEFINE_DISPATCH was activated by X+39 Lane A (AdaptiveAveragePooling.cpp:160);
// the 2D DEFINE_DISPATCH pre-existed (AdaptiveAveragePooling.cpp:148-149).
// No .cu admission — AdaptiveAveragePooling{,3d}.cu is C10_EXPORT-style;
// HaganeOps.cpp retains the C10_EXPORT entry points (now thin DispatchStub
// callers instead of direct C-ABI invokers).
REGISTER_DISPATCH(adaptive_avg_pool2d_backward_kernel,
                  &hagane_adaptive_avg_pool_backward_bridge<kAdaptiveAvgPool2dBackwardCfg>)
REGISTER_DISPATCH(adaptive_avg_pool3d_backward_kernel,
                  &hagane_adaptive_avg_pool_backward_bridge<kAdaptiveAvgPool3dBackwardCfg>)

// X+42 Lane B — adaptive_avg_pool{2,3}d forward retirement POC (ADR-036 rows
// 120-121). C-ABIs haganeOpsAdaptiveAvgPool{2,3}d exist in libhagane-runtime
// (forward variants, previously invoked direct in HaganeOps.cpp). DispatchStub
// typedefs share the 3-arg signature
// `void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size)`.
// Single OpConfig (AdaptiveAvgPoolForwardOpConfig) + single bridge template
// (hagane_adaptive_avg_pool_forward_bridge) reused across 2D + 3D. The
// HaganeOps.cpp entry points (adaptive_avg_pool{2,3}d_out_cuda + _cuda) become
// thin DispatchStub callers — proves Tier 2 forward retirement viability.
REGISTER_DISPATCH(adaptive_avg_pool2d_kernel,
                  &hagane_adaptive_avg_pool_forward_bridge<kAdaptiveAvgPool2dForwardCfg>)
REGISTER_DISPATCH(adaptive_avg_pool3d_kernel,
                  &hagane_adaptive_avg_pool_forward_bridge<kAdaptiveAvgPool3dForwardCfg>)

// X+44 Lane B — avg_pool{2,3}d forward retirement (ADR-036 rows 122-123).
// Mirrors X+42 Lane B pattern. C-ABIs haganeOpsAvgPool{2,3}d exist in
// libhagane-runtime. DispatchStub typedefs (Pool.h:21,30) use int64_t args.
// HaganeOps.cpp TIF entry points at lines 3436 + 3452 become thin
// DispatchStub callers — second forward-cohort retirement after X+42's
// adaptive_avg_pool pair. The .cu forward TIFs (AveragePool{2,3}d.cu:253+)
// are already gated by `#if !defined(__HIP_PLATFORM_HAGANE__)` so only
// HaganeOps.cpp's TIF is active under Hagane.
REGISTER_DISPATCH(avg_pool2d_kernel,
                  &hagane_avg_pool2d_forward_bridge<kAvgPool2dForwardCfg>)
REGISTER_DISPATCH(avg_pool3d_kernel,
                  &hagane_avg_pool3d_forward_bridge<kAvgPool3dForwardCfg>)

} // namespace at::native
