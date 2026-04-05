// Hagane: real op implementations for Apple Silicon (UMA).
// Phase 3: Element-wise ops run on Metal GPU via MLX (through hagane_ops C API).
// Falls back to CPU delegation on UMA for non-contiguous or unsupported dtypes.

#if defined(__HIP_PLATFORM_HAGANE__)

#include <ATen/core/Tensor.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Fill.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/ops/add_native.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Gelu.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/Pow.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <c10/macros/Export.h>

#include <hagane_ops.h>

#include <cstring>

// ---------------------------------------------------------------------------
// Misc CUDA API stubs (defined in .cu files excluded by hipify)
// ---------------------------------------------------------------------------

namespace at::cuda {
C10_EXPORT void sleep(int64_t) {
  // No-op on Hagane — GPU sleep is used for testing only.
}
C10_EXPORT void flush_icache() {
  // No-op on Hagane — instruction cache flush is ROCm-specific.
}
} // namespace at::cuda

// ---------------------------------------------------------------------------
// Level 1: at::native:: functions called directly from RegisterCUDA_*.cpp
// These need C10_EXPORT because PyTorch builds with -fvisibility=hidden.
// ---------------------------------------------------------------------------

namespace at::native {

C10_EXPORT Tensor empty_cuda(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_cuda(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

C10_EXPORT Tensor empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_cuda(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

} // namespace at::native

// ---------------------------------------------------------------------------
// Level 1b: Structured kernel impls for CUDA ufuncs
// These are defined in generated .cu files (excluded) — we provide thin
// wrappers that call the DispatchStub, which then delegates to MLX via C API.
// ---------------------------------------------------------------------------

namespace at::native {

TORCH_IMPL_FUNC(ufunc_add_CUDA)(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, const at::Tensor& out) {
  add_stub(device_type(), *this, alpha);
}

// ---------------------------------------------------------------------------
// Direct function implementations from excluded .hip kernel files
// These are called directly (not through DispatchStub) from compiled .cpp files.
// ---------------------------------------------------------------------------

C10_EXPORT void GeluCUDAKernelImpl(TensorIteratorBase& iter, GeluType approximate) {
  // Route through hagane_ops MLX GPU implementation
  haganeOpsTensor_t out, in;
  const at::Tensor& out_t = iter.tensor(0);
  const at::Tensor& in_t = iter.tensor(1);
  out = { iter.data_ptr(0), out_t.sizes().data(), out_t.strides().data(),
          static_cast<int32_t>(out_t.dim()), HAGANE_DTYPE_FLOAT32 };
  in = { iter.data_ptr(1), in_t.sizes().data(), in_t.strides().data(),
         static_cast<int32_t>(in_t.dim()), HAGANE_DTYPE_FLOAT32 };
  // Map ScalarType
  auto st = iter.dtype();
  int32_t dt = HAGANE_DTYPE_FLOAT32;
  if (st == c10::ScalarType::Half) dt = HAGANE_DTYPE_FLOAT16;
  else if (st == c10::ScalarType::BFloat16) dt = HAGANE_DTYPE_BFLOAT16;
  out.dtype = dt;
  in.dtype = dt;
  if (haganeOpsGelu(&in, &out) != HAGANE_OPS_SUCCESS) {
    // CPU fallback via GeluKernel DispatchStub
    GeluKernel(c10::DeviceType::CPU, iter, approximate);
  }
}

C10_EXPORT void GeluBackwardCUDAKernelImpl(TensorIteratorBase& iter, GeluType approximate) {
  // For now, fall back to CPU for backward pass (training only)
  GeluBackwardKernel(c10::DeviceType::CPU, iter, approximate);
}

static int32_t to_hagane_dtype_ext(c10::ScalarType st) {
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

static haganeOpsTensor_t make_ops_tensor_ext(TensorIteratorBase& iter, int arg) {
  const at::Tensor& t = iter.tensor(arg);
  haganeOpsTensor_t desc;
  desc.data = iter.data_ptr(arg);
  desc.shape = t.sizes().data();
  desc.strides = t.strides().data();
  desc.ndim = static_cast<int32_t>(t.dim());
  desc.dtype = to_hagane_dtype_ext(iter.dtype(arg));
  return desc;
}

static int32_t hagane_dtype(c10::ScalarType st) {
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

C10_EXPORT void max_all_launch_kernel(TensorIterator& iter) {
  // Use the same make_ops_tensor pattern that works for sum/argmax
  auto out = make_ops_tensor_ext(iter, 0);
  auto in = make_ops_tensor_ext(iter, 1);
  if (haganeOpsMaxValues(&in, &out) != HAGANE_OPS_SUCCESS) {
    // Fallback: compute on CPU
    const at::Tensor& in_t = iter.tensor(1);
    auto cpu_in = in_t.cpu();
    auto cpu_max = cpu_in.max();
    auto* out_ptr = static_cast<float*>(iter.data_ptr(0));
    *out_ptr = cpu_max.item<float>();
  }
}

C10_EXPORT void min_all_launch_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor_ext(iter, 0);
  auto in = make_ops_tensor_ext(iter, 1);
  if (haganeOpsMinValues(&in, &out) != HAGANE_OPS_SUCCESS) {
    const at::Tensor& in_t = iter.tensor(1);
    auto cpu_in = in_t.cpu();
    auto cpu_min = cpu_in.min();
    auto* out_ptr = static_cast<float*>(iter.data_ptr(0));
    *out_ptr = cpu_min.item<float>();
  }
}

} // namespace at::native

// ---------------------------------------------------------------------------
// Level 2: DispatchStub registrations for CUDA device
// On Hagane (Apple Silicon), ops run on Metal GPU via MLX through the
// hagane_ops C API. Non-contiguous or unsupported dtypes fall back to CPU
// delegation on UMA (still correct, just slower).
// ---------------------------------------------------------------------------

namespace at::native {
namespace {

// ---------------------------------------------------------------------------
// Helpers: build haganeOpsTensor_t from TensorIterator
// ---------------------------------------------------------------------------

static int32_t to_hagane_dtype(c10::ScalarType st) {
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

static haganeOpsTensor_t make_ops_tensor(TensorIteratorBase& iter, int arg) {
  const at::Tensor& t = iter.tensor(arg);
  haganeOpsTensor_t desc;
  desc.data = iter.data_ptr(arg);
  desc.shape = t.sizes().data();
  desc.strides = t.strides().data();
  desc.ndim = static_cast<int32_t>(t.dim());
  desc.dtype = to_hagane_dtype(iter.dtype(arg));
  return desc;
}

// ---------------------------------------------------------------------------
// Copy + Fill (memcpy is optimal on UMA for copy)
// ---------------------------------------------------------------------------

void hagane_copy_kernel(TensorIterator& iter, bool non_blocking) {
  // Fast path: contiguous, same dtype -> memcpy
  if (iter.is_contiguous() && iter.dtype(0) == iter.dtype(1)) {
    void* dst = iter.data_ptr(0);
    void* src = iter.data_ptr(1);
    int64_t nbytes = iter.numel() * iter.element_size(0);
    if (nbytes > 0) {
      std::memcpy(dst, src, nbytes);
    }
    return;
  }
  // Slow path: CPU copy handles dtype conversion and non-contiguous layouts.
  copy_stub(c10::DeviceType::CPU, iter, non_blocking);
}

void hagane_fill_kernel(TensorIterator& iter, const c10::Scalar& value) {
  fill_stub(c10::DeviceType::CPU, iter, value);
}

// ---------------------------------------------------------------------------
// Binary ops — Metal GPU via MLX, CPU fallback
// ---------------------------------------------------------------------------

void hagane_add_kernel(TensorIteratorBase& iter, const Scalar& alpha) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsAdd(&a, &b, &out, alpha.toFloat()) != HAGANE_OPS_SUCCESS) {
    add_stub(c10::DeviceType::CPU, iter, alpha);
  }
}

void hagane_mul_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsMul(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    mul_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_div_true_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsDiv(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    div_true_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_div_trunc_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsDivTrunc(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    div_trunc_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_div_floor_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsDivFloor(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    div_floor_stub(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Comparison ops — Metal GPU via MLX, CPU fallback
// ---------------------------------------------------------------------------

void hagane_eq_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsEq(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    eq_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_ne_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsNe(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    ne_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_lt_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsLt(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    lt_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_gt_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsGt(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    gt_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_le_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsLe(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    le_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_ge_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsGe(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    ge_stub(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Unary ops — Metal GPU via MLX, CPU fallback
// ---------------------------------------------------------------------------

void hagane_neg_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsNeg(&in, &out) != HAGANE_OPS_SUCCESS) {
    neg_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_abs_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsAbs(&in, &out) != HAGANE_OPS_SUCCESS) {
    abs_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_exp_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsExp(&in, &out) != HAGANE_OPS_SUCCESS) {
    exp_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_log_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLog(&in, &out) != HAGANE_OPS_SUCCESS) {
    log_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_sqrt_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSqrt(&in, &out) != HAGANE_OPS_SUCCESS) {
    sqrt_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_tanh_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsTanh(&in, &out) != HAGANE_OPS_SUCCESS) {
    tanh_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_sigmoid_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSigmoid(&in, &out) != HAGANE_OPS_SUCCESS) {
    sigmoid_stub(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Activation ops — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_silu_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSilu(&in, &out) != HAGANE_OPS_SUCCESS) {
    silu_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_silu_backward_kernel(TensorIteratorBase& iter) {
  // Training only — CPU fallback
  silu_backward_stub(c10::DeviceType::CPU, iter);
}

// ---------------------------------------------------------------------------
// Additional unary ops — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_reciprocal_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsReciprocal(&in, &out) != HAGANE_OPS_SUCCESS) {
    reciprocal_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_rsqrt_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsRsqrt(&in, &out) != HAGANE_OPS_SUCCESS) {
    rsqrt_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_sin_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSin(&in, &out) != HAGANE_OPS_SUCCESS) {
    sin_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_cos_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsCos(&in, &out) != HAGANE_OPS_SUCCESS) {
    cos_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_floor_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsFloor(&in, &out) != HAGANE_OPS_SUCCESS) {
    floor_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_ceil_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsCeil(&in, &out) != HAGANE_OPS_SUCCESS) {
    ceil_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_round_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsRound(&in, &out) != HAGANE_OPS_SUCCESS) {
    round_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_trunc_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsTrunc(&in, &out) != HAGANE_OPS_SUCCESS) {
    trunc_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_sign_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSign(&in, &out) != HAGANE_OPS_SUCCESS) {
    sign_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_erf_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsErf(&in, &out) != HAGANE_OPS_SUCCESS) {
    erf_stub(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Additional unary ops — Batch 1 completions
// ---------------------------------------------------------------------------

void hagane_log2_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLog2(&in, &out) != HAGANE_OPS_SUCCESS) {
    log2_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_log10_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLog10(&in, &out) != HAGANE_OPS_SUCCESS) {
    log10_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_log1p_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLog1p(&in, &out) != HAGANE_OPS_SUCCESS) {
    log1p_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_exp2_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsExp2(&in, &out) != HAGANE_OPS_SUCCESS) {
    exp2_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_expm1_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsExpm1(&in, &out) != HAGANE_OPS_SUCCESS) {
    expm1_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_bitwise_not_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsBitwiseNot(&in, &out) != HAGANE_OPS_SUCCESS) {
    bitwise_not_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_logical_not_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLogicalNot(&in, &out) != HAGANE_OPS_SUCCESS) {
    logical_not_stub(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Additional binary ops — Batch 1
// ---------------------------------------------------------------------------

void hagane_sub_kernel(TensorIteratorBase& iter, const Scalar& alpha) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsSub(&a, &b, &out, alpha.toFloat()) != HAGANE_OPS_SUCCESS) {
    sub_stub(c10::DeviceType::CPU, iter, alpha);
  }
}

void hagane_atan2_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsAtan2(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    atan2_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_pow_tt_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsPow(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    pow_tensor_tensor_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_pow_ts_kernel(TensorIteratorBase& iter, const Scalar& exp) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  if (haganeOpsPowScalar(&a, exp.toFloat(), &out) != HAGANE_OPS_SUCCESS) {
    pow_tensor_scalar_stub(c10::DeviceType::CPU, iter, exp);
  }
}

void hagane_remainder_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsRemainder(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    remainder_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_fmod_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsFmod(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    fmod_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_bitwise_and_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsBitwiseAnd(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    bitwise_and_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_bitwise_or_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsBitwiseOr(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    bitwise_or_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_bitwise_xor_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsBitwiseXor(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    bitwise_xor_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_logical_and_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsLogicalAnd(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    logical_and_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_logical_or_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsLogicalOr(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    logical_or_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_logical_xor_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsLogicalXor(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    logical_xor_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_maximum_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsMaximum(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    maximum_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_minimum_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsMinimum(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    minimum_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_copysign_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto a = make_ops_tensor(iter, 1);
  auto b = make_ops_tensor(iter, 2);
  if (haganeOpsCopysign(&a, &b, &out) != HAGANE_OPS_SUCCESS) {
    copysign_stub(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Activation ops — Batch 1
// ---------------------------------------------------------------------------

void hagane_threshold_kernel(TensorIteratorBase& iter, const Scalar& threshold, const Scalar& value) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsThreshold(&in, &out, threshold.toFloat(), value.toFloat()) != HAGANE_OPS_SUCCESS) {
    threshold_stub(c10::DeviceType::CPU, iter, threshold, value);
  }
}

void hagane_elu_kernel(TensorIteratorBase& iter, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsElu(&in, &out, alpha.toFloat(), scale.toFloat(), input_scale.toFloat()) != HAGANE_OPS_SUCCESS) {
    elu_stub(c10::DeviceType::CPU, iter, alpha, scale, input_scale);
  }
}

void hagane_softplus_kernel(TensorIteratorBase& iter, const Scalar& beta, const Scalar& threshold) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSoftplus(&in, &out, beta.toFloat(), threshold.toFloat()) != HAGANE_OPS_SUCCESS) {
    softplus_stub(c10::DeviceType::CPU, iter, beta, threshold);
  }
}

void hagane_leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLeakyRelu(&in, &out, negative_slope.toFloat()) != HAGANE_OPS_SUCCESS) {
    leaky_relu_stub(c10::DeviceType::CPU, iter, negative_slope);
  }
}

void hagane_hardsigmoid_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsHardsigmoid(&in, &out) != HAGANE_OPS_SUCCESS) {
    hardsigmoid_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_hardswish_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsHardswish(&in, &out) != HAGANE_OPS_SUCCESS) {
    hardswish_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_mish_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsMish(&in, &out) != HAGANE_OPS_SUCCESS) {
    mish_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsHardshrink(&in, &out, lambd.toFloat()) != HAGANE_OPS_SUCCESS) {
    hardshrink_stub(c10::DeviceType::CPU, iter, lambd);
  }
}

void hagane_softshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSoftshrink(&in, &out, lambd.toFloat()) != HAGANE_OPS_SUCCESS) {
    softshrink_stub(c10::DeviceType::CPU, iter, lambd);
  }
}

// ---------------------------------------------------------------------------
// Reductions — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_sum_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsSum(&in, &out) != HAGANE_OPS_SUCCESS) {
    sum_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_mean_kernel(TensorIterator& iter) {
  // Compute sum via the working sum dispatch, then divide by reduction count
  sum_stub(c10::DeviceType::CUDA, iter);
  // Divide output by N = in_numel / out_numel
  const at::Tensor& in_t = iter.tensor(1);
  const at::Tensor& out_t = iter.tensor(0);
  int64_t out_numel = out_t.numel();
  float N = (out_numel > 0) ? static_cast<float>(in_t.numel()) / static_cast<float>(out_numel) : 1.0f;
  if (N <= 1.0f) return;
  // Use out_t.data_ptr() — the actual tensor memory
  if (out_t.scalar_type() == c10::ScalarType::Float) {
    float* ptr = static_cast<float*>(out_t.data_ptr());
    for (int64_t i = 0; i < out_numel; i++) ptr[i] /= N;
  } else if (out_t.scalar_type() == c10::ScalarType::Half) {
    auto* ptr = static_cast<c10::Half*>(out_t.data_ptr());
    for (int64_t i = 0; i < out_numel; i++) ptr[i] = static_cast<c10::Half>(static_cast<float>(ptr[i]) / N);
  } else if (out_t.scalar_type() == c10::ScalarType::BFloat16) {
    auto* ptr = static_cast<c10::BFloat16*>(out_t.data_ptr());
    for (int64_t i = 0; i < out_numel; i++) ptr[i] = static_cast<c10::BFloat16>(static_cast<float>(ptr[i]) / N);
  }
}

void hagane_prod_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsProd(&in, &out) != HAGANE_OPS_SUCCESS) {
    prod_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_argmax_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsArgmax(&in, &out) != HAGANE_OPS_SUCCESS) {
    argmax_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_argmin_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsArgmin(&in, &out) != HAGANE_OPS_SUCCESS) {
    argmin_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_max_values_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsMaxValues(&in, &out) != HAGANE_OPS_SUCCESS) {
    max_values_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_min_values_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsMinValues(&in, &out) != HAGANE_OPS_SUCCESS) {
    min_values_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_and_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsAll(&in, &out) != HAGANE_OPS_SUCCESS) {
    and_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_or_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsAny(&in, &out) != HAGANE_OPS_SUCCESS) {
    or_stub(c10::DeviceType::CPU, iter);
  }
}

void hagane_norm_kernel(TensorIterator& iter, const Scalar& p) {
  // norm not yet in C API — CPU delegation
  norm_stub(c10::DeviceType::CPU, iter, p);
}

// ReduceAllOps: max_all, min_all — different signature: (Tensor& result, const Tensor& self)
void hagane_max_all_kernel(Tensor& result, const Tensor& self) {
  haganeOpsTensor_t out_d = { result.data_ptr(), result.sizes().data(), result.strides().data(),
    static_cast<int32_t>(result.dim()), to_hagane_dtype(result.scalar_type()) };
  haganeOpsTensor_t in_d = { const_cast<void*>(self.const_data_ptr()), self.sizes().data(), self.strides().data(),
    static_cast<int32_t>(self.dim()), to_hagane_dtype(self.scalar_type()) };
  if (haganeOpsMaxValues(&in_d, &out_d) != HAGANE_OPS_SUCCESS) {
    max_all_stub(c10::DeviceType::CPU, result, self);
  }
}

void hagane_min_all_kernel(Tensor& result, const Tensor& self) {
  haganeOpsTensor_t out_d = { result.data_ptr(), result.sizes().data(), result.strides().data(),
    static_cast<int32_t>(result.dim()), to_hagane_dtype(result.scalar_type()) };
  haganeOpsTensor_t in_d = { const_cast<void*>(self.const_data_ptr()), self.sizes().data(), self.strides().data(),
    static_cast<int32_t>(self.dim()), to_hagane_dtype(self.scalar_type()) };
  if (haganeOpsMinValues(&in_d, &out_d) != HAGANE_OPS_SUCCESS) {
    min_all_stub(c10::DeviceType::CPU, result, self);
  }
}

// ---------------------------------------------------------------------------
// Clamp ops — needed for relu (clamp_min), clamp, etc.
// ---------------------------------------------------------------------------

void hagane_clamp_kernel(TensorIteratorBase& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  auto min_t = make_ops_tensor(iter, 2);
  auto max_t = make_ops_tensor(iter, 3);
  // tensor-based clamp — use where(x < min, min, where(x > max, max, x))
  // For now CPU delegation since this needs 4-operand support
  clamp_stub(c10::DeviceType::CPU, iter);
}

void hagane_clamp_scalar_kernel(TensorIteratorBase& iter, const Scalar& min_val, const Scalar& max_val) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsClamp(&in, &out, 1, min_val.toFloat(), 1, max_val.toFloat()) != HAGANE_OPS_SUCCESS) {
    clamp_scalar_stub(c10::DeviceType::CPU, iter, min_val, max_val);
  }
}

void hagane_clamp_min_scalar_kernel(TensorIteratorBase& iter, Scalar min_val) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsClamp(&in, &out, 1, min_val.toFloat(), 0, 0.0f) != HAGANE_OPS_SUCCESS) {
    clamp_min_scalar_stub(c10::DeviceType::CPU, iter, min_val);
  }
}

void hagane_clamp_max_scalar_kernel(TensorIteratorBase& iter, Scalar max_val) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsClamp(&in, &out, 0, 0.0f, 1, max_val.toFloat()) != HAGANE_OPS_SUCCESS) {
    clamp_max_scalar_stub(c10::DeviceType::CPU, iter, max_val);
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// REGISTER_DISPATCH calls — Phase 1 (memory)
// ---------------------------------------------------------------------------

REGISTER_DISPATCH(copy_stub, &hagane_copy_kernel)
REGISTER_DISPATCH(fill_stub, &hagane_fill_kernel)

// ---------------------------------------------------------------------------
// REGISTER_DISPATCH calls — Phase 3 (Metal GPU via MLX)
// ---------------------------------------------------------------------------

// Binary
REGISTER_DISPATCH(add_stub, &hagane_add_kernel)
REGISTER_DISPATCH(mul_stub, &hagane_mul_kernel)
REGISTER_DISPATCH(div_true_stub, &hagane_div_true_kernel)
REGISTER_DISPATCH(div_trunc_stub, &hagane_div_trunc_kernel)
REGISTER_DISPATCH(div_floor_stub, &hagane_div_floor_kernel)

// Comparison
REGISTER_DISPATCH(eq_stub, &hagane_eq_kernel)
REGISTER_DISPATCH(ne_stub, &hagane_ne_kernel)
REGISTER_DISPATCH(lt_stub, &hagane_lt_kernel)
REGISTER_DISPATCH(gt_stub, &hagane_gt_kernel)
REGISTER_DISPATCH(le_stub, &hagane_le_kernel)
REGISTER_DISPATCH(ge_stub, &hagane_ge_kernel)

// Unary
REGISTER_DISPATCH(neg_stub, &hagane_neg_kernel)
REGISTER_DISPATCH(abs_stub, &hagane_abs_kernel)
REGISTER_DISPATCH(exp_stub, &hagane_exp_kernel)
REGISTER_DISPATCH(log_stub, &hagane_log_kernel)
REGISTER_DISPATCH(sqrt_stub, &hagane_sqrt_kernel)
REGISTER_DISPATCH(tanh_stub, &hagane_tanh_kernel)
REGISTER_DISPATCH(sigmoid_stub, &hagane_sigmoid_kernel)

// Activations
REGISTER_DISPATCH(silu_stub, &hagane_silu_kernel)
REGISTER_DISPATCH(silu_backward_stub, &hagane_silu_backward_kernel)

// Additional unary
REGISTER_DISPATCH(reciprocal_stub, &hagane_reciprocal_kernel)
REGISTER_DISPATCH(rsqrt_stub, &hagane_rsqrt_kernel)
REGISTER_DISPATCH(sin_stub, &hagane_sin_kernel)
REGISTER_DISPATCH(cos_stub, &hagane_cos_kernel)
REGISTER_DISPATCH(floor_stub, &hagane_floor_kernel)
REGISTER_DISPATCH(ceil_stub, &hagane_ceil_kernel)
REGISTER_DISPATCH(round_stub, &hagane_round_kernel)
REGISTER_DISPATCH(trunc_stub, &hagane_trunc_kernel)
REGISTER_DISPATCH(sign_stub, &hagane_sign_kernel)
REGISTER_DISPATCH(erf_stub, &hagane_erf_kernel)

// Batch 1: additional unary
REGISTER_DISPATCH(log2_stub, &hagane_log2_kernel)
REGISTER_DISPATCH(log10_stub, &hagane_log10_kernel)
REGISTER_DISPATCH(log1p_stub, &hagane_log1p_kernel)
REGISTER_DISPATCH(exp2_stub, &hagane_exp2_kernel)
REGISTER_DISPATCH(expm1_stub, &hagane_expm1_kernel)
REGISTER_DISPATCH(bitwise_not_stub, &hagane_bitwise_not_kernel)
REGISTER_DISPATCH(logical_not_stub, &hagane_logical_not_kernel)

// Batch 1: additional binary
REGISTER_DISPATCH(sub_stub, &hagane_sub_kernel)
REGISTER_DISPATCH(atan2_stub, &hagane_atan2_kernel)
REGISTER_DISPATCH(pow_tensor_tensor_stub, &hagane_pow_tt_kernel)
REGISTER_DISPATCH(pow_tensor_scalar_stub, &hagane_pow_ts_kernel)
REGISTER_DISPATCH(remainder_stub, &hagane_remainder_kernel)
REGISTER_DISPATCH(fmod_stub, &hagane_fmod_kernel)
REGISTER_DISPATCH(bitwise_and_stub, &hagane_bitwise_and_kernel)
REGISTER_DISPATCH(bitwise_or_stub, &hagane_bitwise_or_kernel)
REGISTER_DISPATCH(bitwise_xor_stub, &hagane_bitwise_xor_kernel)
REGISTER_DISPATCH(logical_and_stub, &hagane_logical_and_kernel)
REGISTER_DISPATCH(logical_or_stub, &hagane_logical_or_kernel)
REGISTER_DISPATCH(logical_xor_stub, &hagane_logical_xor_kernel)
REGISTER_DISPATCH(maximum_stub, &hagane_maximum_kernel)
REGISTER_DISPATCH(minimum_stub, &hagane_minimum_kernel)
REGISTER_DISPATCH(copysign_stub, &hagane_copysign_kernel)

// Batch 1: activations
REGISTER_DISPATCH(threshold_stub, &hagane_threshold_kernel)
REGISTER_DISPATCH(elu_stub, &hagane_elu_kernel)
REGISTER_DISPATCH(softplus_stub, &hagane_softplus_kernel)
REGISTER_DISPATCH(leaky_relu_stub, &hagane_leaky_relu_kernel)
REGISTER_DISPATCH(hardsigmoid_stub, &hagane_hardsigmoid_kernel)
REGISTER_DISPATCH(hardswish_stub, &hagane_hardswish_kernel)
REGISTER_DISPATCH(mish_stub, &hagane_mish_kernel)
REGISTER_DISPATCH(hardshrink_stub, &hagane_hardshrink_kernel)
REGISTER_DISPATCH(softshrink_stub, &hagane_softshrink_kernel)

// Reductions (MLX GPU)
REGISTER_DISPATCH(sum_stub, &hagane_sum_kernel)
REGISTER_DISPATCH(mean_stub, &hagane_mean_kernel)
REGISTER_DISPATCH(prod_stub, &hagane_prod_kernel)
REGISTER_DISPATCH(argmax_stub, &hagane_argmax_kernel)
REGISTER_DISPATCH(argmin_stub, &hagane_argmin_kernel)
REGISTER_DISPATCH(max_values_stub, &hagane_max_values_kernel)
REGISTER_DISPATCH(min_values_stub, &hagane_min_values_kernel)
REGISTER_DISPATCH(and_stub, &hagane_and_kernel)
REGISTER_DISPATCH(or_stub, &hagane_or_kernel)
// norm_stub registered by hip/ReduceOps.cpp

// Clamp
REGISTER_DISPATCH(clamp_stub, &hagane_clamp_kernel)
REGISTER_DISPATCH(clamp_scalar_stub, &hagane_clamp_scalar_kernel)
REGISTER_DISPATCH(clamp_min_scalar_stub, &hagane_clamp_min_scalar_kernel)
REGISTER_DISPATCH(clamp_max_scalar_stub, &hagane_clamp_max_scalar_kernel)

// max_all_stub and min_all_stub are registered by hip/ReduceOps.cpp
// which calls our C10_EXPORT max_all_launch_kernel/min_all_launch_kernel

} // namespace at::native

#endif // __HIP_PLATFORM_HAGANE__
