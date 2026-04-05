// Hagane: real op implementations for Apple Silicon (UMA).
// Replaces auto-generated stubs from hagane_gen_stubs.py for ops that
// can be implemented using CPU delegation on unified memory.

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
#include <ATen/native/ReduceOps.h>
#include <ATen/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <c10/macros/Export.h>

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
// wrappers that call the DispatchStub, which then delegates to CPU on UMA.
// ---------------------------------------------------------------------------

namespace at::native {

TORCH_IMPL_FUNC(ufunc_add_CUDA)(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, const at::Tensor& out) {
  add_stub(device_type(), *this, alpha);
}

} // namespace at::native

// ---------------------------------------------------------------------------
// Level 2: DispatchStub registrations for CUDA device
// On Hagane (Apple Silicon UMA), all device memory is CPU-accessible.
// We delegate to CPU kernels which operate on raw data pointers via
// TensorIterator without device assertions.
// ---------------------------------------------------------------------------

namespace at::native {
namespace {

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
  // Works on UMA because all CUDA tensor data pointers are CPU-accessible.
  copy_stub(c10::DeviceType::CPU, iter, non_blocking);
}

void hagane_fill_kernel(TensorIterator& iter, const c10::Scalar& value) {
  // CPU fill kernel operates on raw pointers via TensorIterator — works on UMA.
  fill_stub(c10::DeviceType::CPU, iter, value);
}

// ---------------------------------------------------------------------------
// Binary ops — delegate to CPU kernel on UMA
// ---------------------------------------------------------------------------

void hagane_add_kernel(TensorIteratorBase& iter, const Scalar& alpha) {
  add_stub(c10::DeviceType::CPU, iter, alpha);
}
void hagane_mul_kernel(TensorIteratorBase& iter) {
  mul_stub(c10::DeviceType::CPU, iter);
}
void hagane_div_true_kernel(TensorIteratorBase& iter) {
  div_true_stub(c10::DeviceType::CPU, iter);
}
void hagane_div_trunc_kernel(TensorIteratorBase& iter) {
  div_trunc_stub(c10::DeviceType::CPU, iter);
}
void hagane_div_floor_kernel(TensorIteratorBase& iter) {
  div_floor_stub(c10::DeviceType::CPU, iter);
}

// ---------------------------------------------------------------------------
// Comparison ops
// ---------------------------------------------------------------------------

void hagane_eq_kernel(TensorIteratorBase& iter) {
  eq_stub(c10::DeviceType::CPU, iter);
}
void hagane_ne_kernel(TensorIteratorBase& iter) {
  ne_stub(c10::DeviceType::CPU, iter);
}
void hagane_lt_kernel(TensorIteratorBase& iter) {
  lt_stub(c10::DeviceType::CPU, iter);
}
void hagane_gt_kernel(TensorIteratorBase& iter) {
  gt_stub(c10::DeviceType::CPU, iter);
}
void hagane_le_kernel(TensorIteratorBase& iter) {
  le_stub(c10::DeviceType::CPU, iter);
}
void hagane_ge_kernel(TensorIteratorBase& iter) {
  ge_stub(c10::DeviceType::CPU, iter);
}

// ---------------------------------------------------------------------------
// Unary ops
// ---------------------------------------------------------------------------

void hagane_neg_kernel(TensorIteratorBase& iter) {
  neg_stub(c10::DeviceType::CPU, iter);
}
void hagane_abs_kernel(TensorIteratorBase& iter) {
  abs_stub(c10::DeviceType::CPU, iter);
}
void hagane_exp_kernel(TensorIteratorBase& iter) {
  exp_stub(c10::DeviceType::CPU, iter);
}
void hagane_log_kernel(TensorIteratorBase& iter) {
  log_stub(c10::DeviceType::CPU, iter);
}
void hagane_sqrt_kernel(TensorIteratorBase& iter) {
  sqrt_stub(c10::DeviceType::CPU, iter);
}
void hagane_tanh_kernel(TensorIteratorBase& iter) {
  tanh_stub(c10::DeviceType::CPU, iter);
}
void hagane_sigmoid_kernel(TensorIteratorBase& iter) {
  sigmoid_stub(c10::DeviceType::CPU, iter);
}

// ---------------------------------------------------------------------------
// Reductions
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
// REGISTER_DISPATCH calls — Phase 2 (compute)
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

// Reductions
REGISTER_DISPATCH(sum_stub, &hagane_sum_kernel)
REGISTER_DISPATCH(argmax_stub, &hagane_argmax_kernel)
REGISTER_DISPATCH(argmin_stub, &hagane_argmin_kernel)

} // namespace at::native

#endif // __HIP_PLATFORM_HAGANE__
