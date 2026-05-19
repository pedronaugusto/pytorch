// Copyright (c) 2026 Pedro Augusto
// SPDX-License-Identifier: MIT
//
// hagane_dispatch_detail.h — Sprint X+29 TU split extraction.
//
// What this is
// ============
// The helpers + per-op fp64/cpu-dispatch trampolines half of the dispatch
// header. Lives in `at::native::hagane_dispatch::detail` exactly like the
// rest of the header — `hagane_dispatch.h` `#include`s this file before
// opening the same namespace again to declare the OpConfig structs + rows
// + bridge templates that consume these symbols.
//
// Why extracted in X+29
// =====================
// X+22-X+28 grew the helper + trampoline section from ~100 LOC to ~350
// LOC while the OpConfig + bridge-template section grew from ~600 LOC to
// ~720 LOC. The combined header crossed the 1100 LOC soft cap at X+28
// (1139 LOC). The two halves are independent — trampolines only depend on
// the upstream DispatchStub declarations + C-ABI prototypes, while the
// OpConfig + bridges only depend on the trampolines via function-pointer
// rows. The natural seam is the `// ---- Per-op configuration ---` divider
// between the two halves. Extracting the upper half drops `dispatch.h` to
// ~790 LOC, well under the soft cap, and gives X+30+ cohort retirements
// headroom without re-splitting.
//
// Linkage rules
// =============
// Identical to `hagane_dispatch.h`: all helpers `inline`, all C-ABI
// trampolines `inline`. ODR safety is the same as if the symbols had
// remained inline in `hagane_dispatch.h`.

#pragma once

#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Pow.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
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
#include <string>

namespace at::native::hagane_dispatch::detail {

using at::Tensor;
using c10::Scalar;

// ---- Helpers --------------------------------------------------------------
// Mirrors the TU-local helper at HaganeOps.cpp:1752. The bridge has zero
// model-shaped logic and must not depend on HaganeOps.cpp internals
// (ADR-036 §6.1).

inline int32_t to_hagane_dtype_local(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Float:    return HAGANE_DTYPE_FLOAT32;
        case c10::ScalarType::Half:     return HAGANE_DTYPE_FLOAT16;
        case c10::ScalarType::BFloat16: return HAGANE_DTYPE_BFLOAT16;
        case c10::ScalarType::Double:   return HAGANE_DTYPE_FLOAT64;
        case c10::ScalarType::Int:      return HAGANE_DTYPE_INT32;
        case c10::ScalarType::Long:     return HAGANE_DTYPE_INT64;
        case c10::ScalarType::Short:    return HAGANE_DTYPE_INT16;
        case c10::ScalarType::Char:     return HAGANE_DTYPE_INT8;
        case c10::ScalarType::Byte:     return HAGANE_DTYPE_UINT8;
        case c10::ScalarType::Bool:     return HAGANE_DTYPE_BOOL;
        default:                        return HAGANE_DTYPE_FLOAT32;
    }
}

inline haganeOpsTensor_t make_ops_tensor_local(TensorIteratorBase& iter, int arg) {
    const at::Tensor& t = iter.tensor(arg);
    haganeOpsTensor_t desc;
    desc.data = iter.data_ptr(arg);
    desc.shape = t.sizes().data();
    desc.strides = t.strides().data();
    desc.ndim = static_cast<int32_t>(t.dim());
    desc.dtype = to_hagane_dtype_local(iter.dtype(arg));
    return desc;
}

// X+30 Lane C — Tensor-only overload for stub bridges (no TensorIterator).
inline haganeOpsTensor_t make_ops_tensor_local_t(const at::Tensor& t) {
    haganeOpsTensor_t desc;
    desc.data = t.data_ptr();
    desc.shape = t.sizes().data();
    desc.strides = t.strides().data();
    desc.ndim = static_cast<int32_t>(t.dim());
    desc.dtype = to_hagane_dtype_local(t.scalar_type());
    return desc;
}

// Mirrors HaganeOps.cpp:1834 `make_ops_tensor_or_scalar` minus its scalar
// cache. The cache is a HaganeOps.cpp-internal perf optimization; the bridge
// must not depend on `g_scalar_tensor_cache` (ADR-036 §6.1).
inline haganeOpsTensor_t make_ops_tensor_or_scalar_local(
    TensorIteratorBase& iter, int arg, at::Tensor& storage) {
    if (iter.is_cpu_scalar(arg)) {
        auto scalar_dtype = iter.dtype(arg);
        c10::ScalarType target_dtype;
        if (scalar_dtype == c10::ScalarType::Double) {
            target_dtype = iter.common_dtype();
            if (target_dtype == c10::ScalarType::Double) {
                target_dtype = c10::ScalarType::Float;
            }
        } else {
            target_dtype = scalar_dtype;
        }
        c10::Scalar scalar_val;
        if (target_dtype == c10::ScalarType::Bool) {
            scalar_val = (iter.scalar_value<int64_t>(arg) != 0);
        } else {
            scalar_val = iter.scalar_value<double>(arg);
        }
        storage = at::full({}, scalar_val,
                           iter.tensor(0).options().dtype(target_dtype));
        haganeOpsTensor_t desc;
        desc.data = storage.data_ptr();
        desc.shape = storage.sizes().data();
        desc.strides = storage.strides().data();
        desc.ndim = 0;
        desc.dtype = to_hagane_dtype_local(target_dtype);
        return desc;
    }
    return make_ops_tensor_local(iter, arg);
}

inline std::string metallib_dir() {
    if (const char* env = std::getenv("HAGANE_METALLIB_DIR")) return env;
    return "/Users/gusto/work/izumo/hagane-sdk/share/metallibs";
}

// ---- Per-op fp64 trampolines ----------------------------------------------
// Pin a uniform function-pointer signature for the OpConfig tables and
// sidestep `at::*` overload-resolution ambiguity.

inline Tensor fp64_abs(const Tensor& t) { return at::abs(t); }
inline Tensor fp64_neg(const Tensor& t) { return at::neg(t); }

inline Tensor fp64_eq (const Tensor& a, const Tensor& b) { return at::eq(a, b); }
inline Tensor fp64_ne (const Tensor& a, const Tensor& b) { return at::ne(a, b); }
inline Tensor fp64_lt (const Tensor& a, const Tensor& b) { return at::lt(a, b); }
inline Tensor fp64_gt (const Tensor& a, const Tensor& b) { return at::gt(a, b); }
inline Tensor fp64_le (const Tensor& a, const Tensor& b) { return at::le(a, b); }
inline Tensor fp64_ge (const Tensor& a, const Tensor& b) { return at::ge(a, b); }
inline Tensor fp64_mul(const Tensor& a, const Tensor& b) {
    return at::mul(a, b).to(c10::ScalarType::Double);
}

inline Tensor fp64_div(const Tensor& a, const Tensor& b) {
    return at::div(a, b).to(c10::ScalarType::Double);
}

inline Tensor fp64_add_alpha(const Tensor& a, const Tensor& b, const Scalar& alpha) {
    auto r = (alpha.toFloat() == 1.0f)
        ? at::add(a, b)
        : at::add(a, at::mul(b, at::full_like(b, alpha.toFloat())));
    return r.to(c10::ScalarType::Double);
}
inline Tensor fp64_sub_alpha(const Tensor& a, const Tensor& b, const Scalar& alpha) {
    auto r = (alpha.toFloat() == 1.0f)
        ? at::sub(a, b)
        : at::sub(a, at::mul(b, at::full_like(b, alpha.toFloat())));
    return r.to(c10::ScalarType::Double);
}

// ---- Per-op cpu-dispatch trampolines --------------------------------------

inline void cpu_dispatch_abs    (TensorIteratorBase& iter) { abs_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_neg    (TensorIteratorBase& iter) { neg_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_sign   (TensorIteratorBase& iter) { sign_stub   (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_sgn    (TensorIteratorBase& iter) { sgn_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_exp    (TensorIteratorBase& iter) { exp_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_log    (TensorIteratorBase& iter) { log_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_sqrt   (TensorIteratorBase& iter) { sqrt_stub   (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_sin    (TensorIteratorBase& iter) { sin_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_cos    (TensorIteratorBase& iter) { cos_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_ceil   (TensorIteratorBase& iter) { ceil_stub   (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_floor  (TensorIteratorBase& iter) { floor_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_tanh   (TensorIteratorBase& iter) { tanh_stub   (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_sigmoid(TensorIteratorBase& iter) { sigmoid_stub(c10::DeviceType::CPU, iter); }

inline void cpu_dispatch_reciprocal(TensorIteratorBase& iter) { reciprocal_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_rsqrt     (TensorIteratorBase& iter) { rsqrt_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_round     (TensorIteratorBase& iter) { round_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_trunc     (TensorIteratorBase& iter) { trunc_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_erf       (TensorIteratorBase& iter) { erf_stub       (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_log2      (TensorIteratorBase& iter) { log2_stub      (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_log10     (TensorIteratorBase& iter) { log10_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_log1p     (TensorIteratorBase& iter) { log1p_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_exp2      (TensorIteratorBase& iter) { exp2_stub      (c10::DeviceType::CPU, iter); }

inline void cpu_dispatch_expm1      (TensorIteratorBase& iter) { expm1_stub      (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_bitwise_not(TensorIteratorBase& iter) { bitwise_not_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_logical_not(TensorIteratorBase& iter) { logical_not_stub(c10::DeviceType::CPU, iter); }

inline void cpu_dispatch_tan   (TensorIteratorBase& iter) { tan_stub   (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_acos  (TensorIteratorBase& iter) { acos_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_asin  (TensorIteratorBase& iter) { asin_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_atan  (TensorIteratorBase& iter) { atan_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_cosh  (TensorIteratorBase& iter) { cosh_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_sinh  (TensorIteratorBase& iter) { sinh_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_erfc  (TensorIteratorBase& iter) { erfc_stub  (c10::DeviceType::CPU, iter); }

// X+21 — lgamma re-migration after the `haganeOpsLgamma` flush fix
// (hagane/src/runtime/hagane_ops.cpp:1335 mx::eval → ::haganeOpsFlush; X+20
// lesson #15 resolved).
inline void cpu_dispatch_lgamma(TensorIteratorBase& iter) { lgamma_stub(c10::DeviceType::CPU, iter); }

// X+21 — frac brace-defect bug-fix-by-migration. Pure-MLX C-ABI uses
// copy_result(mx::op(...), out) which evaluates the subgraph (no flush
// needed).
inline void cpu_dispatch_frac(TensorIteratorBase& iter) { frac_stub(c10::DeviceType::CPU, iter); }

inline void cpu_dispatch_hardsigmoid(TensorIteratorBase& iter) { hardsigmoid_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_mish       (TensorIteratorBase& iter) { mish_stub       (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_silu       (TensorIteratorBase& iter) { silu_stub       (c10::DeviceType::CPU, iter); }

inline void cpu_dispatch_eq (TensorIteratorBase& iter) { eq_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_ne (TensorIteratorBase& iter) { ne_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_lt (TensorIteratorBase& iter) { lt_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_gt (TensorIteratorBase& iter) { gt_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_le (TensorIteratorBase& iter) { le_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_ge (TensorIteratorBase& iter) { ge_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_mul(TensorIteratorBase& iter) { mul_stub(c10::DeviceType::CPU, iter); }

inline void cpu_dispatch_pow_tt   (TensorIteratorBase& iter) { pow_tensor_tensor_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_atan2    (TensorIteratorBase& iter) { atan2_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_remainder(TensorIteratorBase& iter) { remainder_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_fmod     (TensorIteratorBase& iter) { fmod_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_div_true (TensorIteratorBase& iter) { div_true_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_div_floor(TensorIteratorBase& iter) { div_floor_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_div_trunc(TensorIteratorBase& iter) { div_trunc_stub(c10::DeviceType::CPU, iter); }

// X+21 Lane D binary cpu-dispatch trampolines (11 ops).
// `structured_binary_fn` (TensorIteratorBase&) stubs:
inline void cpu_dispatch_bitwise_and (TensorIteratorBase& iter) { bitwise_and_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_bitwise_or  (TensorIteratorBase& iter) { bitwise_or_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_bitwise_xor (TensorIteratorBase& iter) { bitwise_xor_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_maximum     (TensorIteratorBase& iter) { maximum_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_minimum     (TensorIteratorBase& iter) { minimum_stub     (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_copysign    (TensorIteratorBase& iter) { copysign_stub    (c10::DeviceType::CPU, iter); }
// `binary_fn` (TensorIterator&) stubs — distinct signature; routed through
// the BinaryIterOpConfig + hagane_binary_iter_bridge template below.
inline void cpu_dispatch_logical_and (TensorIterator& iter) { logical_and_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_logical_or  (TensorIterator& iter) { logical_or_stub  (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_logical_xor (TensorIterator& iter) { logical_xor_stub (c10::DeviceType::CPU, iter); }
// max_elementwise_stub / min_elementwise_stub: declared in BinaryOps.h but
// never `DEFINE_DISPATCH`d in upstream PyTorch — phantom stubs with no
// `::DEFAULT` symbol. The HaganeOps.cpp brace-defect kernel bodies were
// orphaned dead code (never wired via REGISTER_DISPATCH); X+21 Lane E
// removes them. No bridge registration possible.

// X+22 Lane A — Unary-iter cpu-dispatch trampolines.
// reduce_fn (ReduceOps.h) and hardswish_fn (Activation.h) both alias to
// `void(*)(TensorIterator&)`. Routed through UnaryIterOpConfig +
// hagane_unary_iter_bridge below.
inline void cpu_dispatch_sum       (TensorIterator& iter) { sum_stub       (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_mean      (TensorIterator& iter) { mean_stub      (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_prod      (TensorIterator& iter) { prod_stub      (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_argmax    (TensorIterator& iter) { argmax_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_argmin    (TensorIterator& iter) { argmin_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_max_values(TensorIterator& iter) { max_values_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_min_values(TensorIterator& iter) { min_values_stub(c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_and       (TensorIterator& iter) { and_stub       (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_or        (TensorIterator& iter) { or_stub        (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_hardswish (TensorIterator& iter) { hardswish_stub (c10::DeviceType::CPU, iter); }

// X+23 Lane A — reduce_fn_flag (TensorIterator&, const Scalar&). Routed
// through ReduceFlagOpConfig + hagane_unary_iter_flag_bridge below.
inline void cpu_dispatch_norm   (TensorIterator& iter, const Scalar& p) { norm_stub   (c10::DeviceType::CPU, iter, p); }
inline void cpu_dispatch_powsum (TensorIterator& iter, const Scalar& p) { powsum_stub (c10::DeviceType::CPU, iter, p); }

// X+23 Lane B — fmax/fmin via existing BinaryOpConfig (structured_binary_fn).
// fmax_stub/fmin_stub CPU registration exists in BinaryOpsKernel.cpp.
inline void cpu_dispatch_fmax (TensorIteratorBase& iter) { fmax_stub (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_fmin (TensorIteratorBase& iter) { fmin_stub (c10::DeviceType::CPU, iter); }

// X+24 Lane B — sinc/signbit via new haganeOpsSinc / haganeOpsSignbit
// (pure-MLX, hagane/src/runtime/hagane_ops.cpp). CPU registration:
// pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:817-818.
inline void cpu_dispatch_sinc    (TensorIteratorBase& iter) { sinc_stub    (c10::DeviceType::CPU, iter); }
inline void cpu_dispatch_signbit (TensorIteratorBase& iter) { signbit_stub (c10::DeviceType::CPU, iter); }

// X+24 Lane A — cumulative reductions (cumsum/cumprod/logcumsumexp). Signature
// asymmetry: structured_cum_fn = void(*)(const Tensor&, const Tensor&, int64_t),
// cum_fn = void(*)(Tensor&, const Tensor&, int64_t). One CumOpConfig + two
// bridge templates carry the asymmetry.
inline void cpu_dispatch_cumsum       (const Tensor& self, const Tensor& result, int64_t dim) {
    cumsum_stub      (c10::DeviceType::CPU, self, result, dim);
}
inline void cpu_dispatch_cumprod      (const Tensor& self, const Tensor& result, int64_t dim) {
    cumprod_stub     (c10::DeviceType::CPU, self, result, dim);
}
inline void cpu_dispatch_logcumsumexp (Tensor& result, const Tensor& self, int64_t dim) {
    logcumsumexp_stub(c10::DeviceType::CPU, result, self, dim);
}

// X+24 Lane C — std_var on (TensorIterator&, double correction, bool take_sqrt).
// CPU registration: pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:548.
inline void cpu_dispatch_std_var(TensorIterator& iter, double correction, bool take_sqrt) {
    std_var_stub(c10::DeviceType::CPU, iter, correction, take_sqrt);
}

// X+25 Lane B — nan_to_num on (TensorIteratorBase&, optional<double>×3).
// CPU registration: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:447+.
inline void cpu_dispatch_nan_to_num(TensorIteratorBase& iter,
                                     std::optional<double> a,
                                     std::optional<double> b,
                                     std::optional<double> c) {
    nan_to_num_stub(c10::DeviceType::CPU, iter, a, b, c);
}

// X+26 Lane B — 4 activation backward retirements after wrap_tensor
// contiguous-fast-path pending_get fix (hagane_ops.cpp wrap_tensor). The C-ABIs
// are pure-MLX compositions; bridge avoids the lazy-stash interaction that
// breaks the retired at::*-on-iter.tensor kernels.
//
// Iter layouts (verified via Activation.cpp TORCH_META_FUNC / binary_op sites):
//   leaky_relu_backward (TORCH_META_FUNC:177-191): build_borrowing_binary_op(
//     output, self_or_result, grad_output) — pos 1 = INPUT, pos 2 = grad_out.
//     C-ABI signature (grad_out, input, grad_in, negval) doesn't match —
//     leaky_relu_backward_wrap swaps a/b before forwarding.
//   hardsigmoid_backward (TORCH_META_FUNC:197): pos 1 = grad_out, pos 2 = input.
//   hardswish_backward (Activation.cpp:509 unstructured): pos 1 = grad_out, pos 2 = self.
//   mish_backward (Activation.cpp:565 unstructured): pos 1 = grad_out, pos 2 = input.
inline int leaky_relu_backward_wrap(const haganeOpsTensor_t* a,
                                     const haganeOpsTensor_t* b,
                                     const haganeOpsTensor_t* out,
                                     float negval) {
    // X+27 — swap a/b before forwarding. Per Activation.cpp:190
    // build_borrowing_binary_op(maybe_get_output(), self_or_result, grad_output)
    // → bridge maps a = iter.tensor(1) = INPUT, b = iter.tensor(2) = GRADIENT.
    // C-ABI signature is (grad_out, input, grad_in, negval), so pass (b, a, ...).
    // The X+26 "no-swap" path landed on a misread of an empirical test
    // (test_harness was using leaky_relu(slope=0) which masks the swap bug).
    return haganeOpsLeakyReluBackward(b, a, out, negval);
}

inline void cpu_dispatch_leaky_relu_backward(TensorIteratorBase& iter, const Scalar& s) {
    leaky_relu_backward_stub(c10::DeviceType::CPU, iter, s);
}
inline void cpu_dispatch_hardswish_backward(TensorIterator& iter) {
    hardswish_backward_stub(c10::DeviceType::CPU, iter);
}
inline void cpu_dispatch_hardsigmoid_backward(TensorIteratorBase& iter) {
    hardsigmoid_backward_stub(c10::DeviceType::CPU, iter);
}
inline void cpu_dispatch_mish_backward(TensorIterator& iter) {
    mish_backward_stub(c10::DeviceType::CPU, iter);
}

// X+28 Lane A — sigmoid/tanh/logit backwards via bridge.
inline void cpu_dispatch_sigmoid_backward(TensorIteratorBase& iter) {
    sigmoid_backward_stub(c10::DeviceType::CPU, iter);
}
inline void cpu_dispatch_tanh_backward(TensorIteratorBase& iter) {
    tanh_backward_stub(c10::DeviceType::CPU, iter);
}
inline void cpu_dispatch_logit_backward(TensorIteratorBase& iter, const Scalar& eps) {
    logit_backward_stub(c10::DeviceType::CPU, iter, eps);
}

// X+28 Lane B — elu/softplus/hardtanh backwards via bridge.
inline void cpu_dispatch_elu_backward(TensorIteratorBase& iter,
                                       const Scalar& alpha, const Scalar& scale,
                                       const Scalar& input_scale, bool is_result) {
    elu_backward_stub(c10::DeviceType::CPU, iter, alpha, scale, input_scale, is_result);
}
inline void cpu_dispatch_softplus_backward(TensorIteratorBase& iter,
                                            const Scalar& beta, const Scalar& threshold) {
    softplus_backward_stub(c10::DeviceType::CPU, iter, beta, threshold);
}
inline void cpu_dispatch_hardtanh_backward(TensorIterator& iter,
                                            const Scalar& min_val, const Scalar& max_val) {
    hardtanh_backward_stub(c10::DeviceType::CPU, iter, min_val, max_val);
}

// ---- Binary-alpha bespoke CPU fallbacks -----------------------------------
// add_stub / sub_stub have no CPU kernel registration; the retired form ran
// lift + clone + in-place + copy directly.

inline void cpu_fallback_add(TensorIteratorBase& iter, const Scalar& alpha) {
    auto cpu_a = iter.tensor(1).cpu();
    auto cpu_b = iter.tensor(2).cpu();
    auto r = cpu_a.clone();
    r.add_(cpu_b, alpha);
    iter.tensor(0).copy_(r);
}
inline void cpu_fallback_sub(TensorIteratorBase& iter, const Scalar& alpha) {
    auto cpu_a = iter.tensor(1).cpu();
    auto cpu_b = iter.tensor(2).cpu();
    auto r = cpu_a.clone();
    r.sub_(cpu_b, alpha);
    iter.tensor(0).copy_(r);
}

// ---- Unary-scalar C-ABI trampolines ---------------------------------------
// haganeOpsPowScalar's C-ABI arg order is (in, scalar, out), different from
// the other 4 (in, out, scalar); reorder so the bridge template uses a
// uniform fn-pointer signature.
inline int pow_scalar_wrap(const haganeOpsTensor_t* in, const haganeOpsTensor_t* out, float exp) {
    return haganeOpsPowScalar(in, exp, out);
}

inline void cpu_dispatch_pow_ts     (TensorIteratorBase& iter, const Scalar& exp)   { pow_tensor_scalar_stub(c10::DeviceType::CPU, iter, exp); }
inline void cpu_dispatch_leaky_relu (TensorIteratorBase& iter, const Scalar& slope) { leaky_relu_stub(c10::DeviceType::CPU, iter, slope); }
inline void cpu_dispatch_hardshrink (TensorIteratorBase& iter, const Scalar& lambd) { hardshrink_stub(c10::DeviceType::CPU, iter, lambd); }
inline void cpu_dispatch_softshrink (TensorIteratorBase& iter, const Scalar& lambd) { softshrink_stub(c10::DeviceType::CPU, iter, lambd); }
inline void cpu_dispatch_logit      (TensorIteratorBase& iter, const Scalar& eps)   { logit_stub(c10::DeviceType::CPU, iter, eps); }

// X+30 Lane C/D — upsample backward CPU-fallback trampolines.
inline void cpu_dispatch_upsample_nearest1d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_w) {
    upsample_nearest1d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, scales_w);
}
inline void cpu_dispatch_upsample_nearest2d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    upsample_nearest2d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, scales_h, scales_w);
}
inline void cpu_dispatch_upsample_nearest3d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w) {
    upsample_nearest3d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, scales_d, scales_h, scales_w);
}

// X+31 Lane B — exact-nearest backward CPU fallback trampolines.
inline void cpu_dispatch_upsample_nearest_exact1d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_w) {
    _upsample_nearest_exact1d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, scales_w);
}
inline void cpu_dispatch_upsample_nearest_exact2d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    _upsample_nearest_exact2d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, scales_h, scales_w);
}
inline void cpu_dispatch_upsample_nearest_exact3d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w) {
    _upsample_nearest_exact3d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, scales_d, scales_h, scales_w);
}

// X+31 Lane C — pool backward CPU-fallback trampolines.
inline void cpu_dispatch_avg_pool2d_backward(
    const Tensor& gradInput, const Tensor& gradOutput,
    int kW, int kH, int dW, int dH, int padW, int padH,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
    avg_pool2d_backward_kernel(c10::DeviceType::CPU,
        gradInput, gradOutput, kW, kH, dW, dH, padW, padH,
        count_include_pad, divisor_override);
}
inline void cpu_dispatch_avg_pool3d_backward(
    const Tensor& gradInput, const Tensor& gradOutput,
    int kW, int kH, int kD, int dW, int dH, int dD,
    int padW, int padH, int padD,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
    avg_pool3d_backward_kernel(c10::DeviceType::CPU,
        gradInput, gradOutput, kW, kH, kD, dW, dH, dD, padW, padH, padD,
        count_include_pad, divisor_override);
}
inline void cpu_dispatch_max_pool3d_backward(
    Tensor& gradInput, const Tensor& gradOutput, const Tensor& indices) {
    max_pool3d_backward_kernel(c10::DeviceType::CPU,
        gradInput, gradOutput, indices);
}
inline void cpu_dispatch_max_pool2d_backward(
    const Tensor& gradInput, const Tensor& gradOutput, const Tensor& indices) {
    max_pool2d_backward_kernel(c10::DeviceType::CPU,
        gradInput, gradOutput, indices);
}

// X+31 Lane A — align-corners upsample backward CPU-fallback trampolines.
inline void cpu_dispatch_upsample_linear1d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners, std::optional<double> scales_w) {
    upsample_linear1d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, align_corners, scales_w);
}
inline void cpu_dispatch_upsample_bilinear2d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    upsample_bilinear2d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, align_corners, scales_h, scales_w);
}
inline void cpu_dispatch_upsample_trilinear3d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w) {
    upsample_trilinear3d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, align_corners, scales_d, scales_h, scales_w);
}
// X+32 Lane D — bicubic2d_backward via upstream-patched DispatchStub.
inline void cpu_dispatch_upsample_bicubic2d_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    upsample_bicubic2d_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, align_corners, scales_h, scales_w);
}

// X+32 Lane C — anti-aliased upsample backwards.
inline void cpu_dispatch_upsample_bilinear2d_aa_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    _upsample_bilinear2d_aa_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, align_corners, scales_h, scales_w);
}
inline void cpu_dispatch_upsample_bicubic2d_aa_backward(
    const Tensor& grad_input, const Tensor& grad_output,
    bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    _upsample_bicubic2d_aa_backward_kernel(c10::DeviceType::CPU,
        grad_input, grad_output, align_corners, scales_h, scales_w);
}

} // namespace at::native::hagane_dispatch::detail
