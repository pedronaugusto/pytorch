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
// Reductions — Metal GPU via MLX, CPU fallback
// ---------------------------------------------------------------------------

void hagane_sum_kernel(TensorIterator& iter) {
  sum_stub(c10::DeviceType::CPU, iter);
}

void hagane_argmax_kernel(TensorIterator& iter) {
  argmax_stub(c10::DeviceType::CPU, iter);
}

void hagane_argmin_kernel(TensorIterator& iter) {
  argmin_stub(c10::DeviceType::CPU, iter);
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

// Reductions
REGISTER_DISPATCH(sum_stub, &hagane_sum_kernel)
REGISTER_DISPATCH(argmax_stub, &hagane_argmax_kernel)
REGISTER_DISPATCH(argmin_stub, &hagane_argmin_kernel)

} // namespace at::native

#endif // __HIP_PLATFORM_HAGANE__
