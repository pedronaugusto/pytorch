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

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &hagane_copy_kernel)
REGISTER_DISPATCH(fill_stub, &hagane_fill_kernel)

} // namespace at::native

#endif // __HIP_PLATFORM_HAGANE__
