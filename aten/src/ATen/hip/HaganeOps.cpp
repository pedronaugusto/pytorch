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
#include <ATen/native/Sorting.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/hip/Sort.h>
#include <ATen/native/hip/SortStable.h>
#include <ATen/native/hip/TensorTopK.h>
#include <ATen/native/hip/ScanKernels.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/IListRef.h>
#include <ATen/Functions.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/native/UnaryOps.h>
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

// ---------------------------------------------------------------------------
// Helper: build haganeOpsTensor_t from TensorBase (for Level B functions)
// ---------------------------------------------------------------------------

static haganeOpsTensor_t make_tensor_desc(const TensorBase& t) {
  haganeOpsTensor_t desc;
  desc.data = const_cast<void*>(t.const_data_ptr());
  desc.shape = t.sizes().data();
  desc.strides = t.strides().data();
  desc.ndim = static_cast<int32_t>(t.dim());
  desc.dtype = hagane_dtype(t.scalar_type());
  return desc;
}

// ---------------------------------------------------------------------------
// Level B: Sort/TopK launch_kernel functions (called from compiled Sort.cpp,
// TensorTopK.cpp). These replace implementations from excluded .hip files.
// ---------------------------------------------------------------------------

C10_EXPORT void sortKeyValueInplace(
    const TensorBase& key, const TensorBase& value, int64_t dim,
    bool descending, bool stable) {
  auto key_d = make_tensor_desc(key);
  auto val_d = make_tensor_desc(value);
  if (haganeOpsSort(&key_d, &val_d, static_cast<int32_t>(dim), descending ? 1 : 0) != HAGANE_OPS_SUCCESS) {
    Tensor key_t(key), val_t(value);
    auto key_cpu = key_t.cpu();
    auto [sorted_key, sorted_idx] = key_cpu.sort(dim, descending, stable);
    std::memcpy(const_cast<void*>(key.const_data_ptr()), sorted_key.const_data_ptr(),
                key.numel() * key.itemsize());
    std::memcpy(const_cast<void*>(value.const_data_ptr()), sorted_idx.const_data_ptr(),
                value.numel() * value.itemsize());
  }
}

C10_EXPORT void launch_stable_sort_kernel(
    const TensorBase& self, int64_t dim, bool descending,
    const TensorBase& values, const TensorBase& indices) {
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  std::memcpy(const_cast<void*>(values.const_data_ptr()),
              self.const_data_ptr(), self.numel() * self.itemsize());
  if (haganeOpsSort(&val_d, &idx_d, static_cast<int32_t>(dim), descending ? 1 : 0) != HAGANE_OPS_SUCCESS) {
    Tensor self_t(self);
    auto self_cpu = self_t.cpu();
    auto [sorted, sorted_idx] = self_cpu.sort(dim, descending, /*stable=*/true);
    std::memcpy(const_cast<void*>(values.const_data_ptr()), sorted.const_data_ptr(),
                values.numel() * values.itemsize());
    std::memcpy(const_cast<void*>(indices.const_data_ptr()), sorted_idx.const_data_ptr(),
                indices.numel() * indices.itemsize());
  }
}

C10_EXPORT void launch_gather_topk_kernel(
    const TensorBase& self, int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices) {
  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  if (haganeOpsTopk(&in_d, &val_d, &idx_d, k, static_cast<int32_t>(dim), largest ? 1 : 0) != HAGANE_OPS_SUCCESS) {
    Tensor self_t(self);
    auto [topk_vals, topk_idx] = self_t.cpu().topk(k, dim, largest, /*sorted=*/true);
    std::memcpy(const_cast<void*>(values.const_data_ptr()), topk_vals.const_data_ptr(),
                values.numel() * values.itemsize());
    std::memcpy(const_cast<void*>(indices.const_data_ptr()), topk_idx.const_data_ptr(),
                indices.numel() * indices.itemsize());
  }
}

// ---------------------------------------------------------------------------
// Level B: Scan launch_kernel functions (called from compiled ScanKernels.cpp)
// ---------------------------------------------------------------------------

C10_EXPORT void launch_cumsum_cuda_kernel(
    const TensorBase& result, const TensorBase& self, int64_t dim) {
  auto in_d = make_tensor_desc(self);
  auto out_d = make_tensor_desc(result);
  if (haganeOpsCumsum(&in_d, &out_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto result_cpu = Tensor(self).cpu().cumsum(dim);
    std::memcpy(const_cast<void*>(result.const_data_ptr()), result_cpu.const_data_ptr(),
                result.numel() * result.itemsize());
  }
}

C10_EXPORT void launch_cumprod_cuda_kernel(
    const TensorBase& result, const TensorBase& self, int64_t dim) {
  auto in_d = make_tensor_desc(self);
  auto out_d = make_tensor_desc(result);
  if (haganeOpsCumprod(&in_d, &out_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto result_cpu = Tensor(self).cpu().cumprod(dim);
    std::memcpy(const_cast<void*>(result.const_data_ptr()), result_cpu.const_data_ptr(),
                result.numel() * result.itemsize());
  }
}

C10_EXPORT void launch_cummax_cuda_kernel(
    const TensorBase& self, const TensorBase& values,
    const TensorBase& indices, int64_t dim) {
  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  haganeOpsCummax(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim));
}

C10_EXPORT void launch_cummin_cuda_kernel(
    const TensorBase& self, const TensorBase& values,
    const TensorBase& indices, int64_t dim) {
  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  haganeOpsCummin(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim));
}

C10_EXPORT void launch_logcumsumexp_cuda_kernel(
    const TensorBase& result, const TensorBase& self, int64_t dim) {
  auto in_d = make_tensor_desc(self);
  auto out_d = make_tensor_desc(result);
  haganeOpsLogcumsumexp(&in_d, &out_d, static_cast<int32_t>(dim));
}

// ---------------------------------------------------------------------------
// Level B: Reduce-dim launch_kernel functions (called from compiled ReduceOps.cpp)
// ---------------------------------------------------------------------------

// Helper: find the reduction dim by comparing input vs output shapes.
// The TensorIterator from make_reduction keeps dims (keepdim internally),
// so output has same ndim with 1 at the reduced dim.
static int64_t find_reduce_dim(const TensorBase& input, const TensorBase& output) {
  for (int64_t d = 0; d < input.dim(); d++) {
    if (input.size(d) > 1 && (d >= output.dim() || output.size(d) == 1))
      return d;
  }
  return input.dim() - 1;  // fallback: last dim
}

C10_EXPORT void max_launch_kernel(TensorIterator& iter) {
  // iter: output 0 = values, output 1 = indices (Long), input = self
  const Tensor& self = iter.tensor(2);
  const Tensor& values = iter.tensor(0);
  const Tensor& indices = iter.tensor(1);
  int64_t dim = find_reduce_dim(self, values);

  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  if (haganeOpsMaxDim(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = Tensor(self).cpu();
    auto [cpu_vals, cpu_idx] = cpu_self.max(dim, /*keepdim=*/true);
    std::memcpy(values.data_ptr(), cpu_vals.const_data_ptr(), values.numel() * values.itemsize());
    std::memcpy(indices.data_ptr(), cpu_idx.const_data_ptr(), indices.numel() * indices.itemsize());
  }
}

C10_EXPORT void min_launch_kernel(TensorIterator& iter) {
  const Tensor& self = iter.tensor(2);
  const Tensor& values = iter.tensor(0);
  const Tensor& indices = iter.tensor(1);
  int64_t dim = find_reduce_dim(self, values);

  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  if (haganeOpsMinDim(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = Tensor(self).cpu();
    auto [cpu_vals, cpu_idx] = cpu_self.min(dim, /*keepdim=*/true);
    std::memcpy(values.data_ptr(), cpu_vals.const_data_ptr(), values.numel() * values.itemsize());
    std::memcpy(indices.data_ptr(), cpu_idx.const_data_ptr(), indices.numel() * indices.itemsize());
  }
}

C10_EXPORT void aminmax_launch_kernel(TensorIterator& iter) {
  // iter: output 0 = min_result, output 1 = max_result, input = self
  const Tensor& self = iter.tensor(2);
  const Tensor& min_result = iter.tensor(0);
  const Tensor& max_result = iter.tensor(1);
  int64_t dim = find_reduce_dim(self, min_result);

  auto in_d = make_tensor_desc(self);
  auto min_d = make_tensor_desc(min_result);
  auto max_d = make_tensor_desc(max_result);
  if (haganeOpsAminmax(&in_d, &min_d, &max_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = Tensor(self).cpu();
    auto [cpu_min, cpu_max] = cpu_self.aminmax(dim, /*keepdim=*/true);
    std::memcpy(min_result.data_ptr(), cpu_min.const_data_ptr(), min_result.numel() * min_result.itemsize());
    std::memcpy(max_result.data_ptr(), cpu_max.const_data_ptr(), max_result.numel() * max_result.itemsize());
  }
}

C10_EXPORT void aminmax_allreduce_launch_kernel(TensorIterator& iter) {
  const Tensor& self = iter.tensor(2);
  const Tensor& min_result = iter.tensor(0);
  const Tensor& max_result = iter.tensor(1);

  auto in_d = make_tensor_desc(self);
  auto min_d = make_tensor_desc(min_result);
  auto max_d = make_tensor_desc(max_result);
  if (haganeOpsAminmaxAll(&in_d, &min_d, &max_d) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = Tensor(self).cpu();
    auto [cpu_min, cpu_max] = cpu_self.aminmax();
    std::memcpy(min_result.data_ptr(), cpu_min.const_data_ptr(), min_result.numel() * min_result.itemsize());
    std::memcpy(max_result.data_ptr(), cpu_max.const_data_ptr(), max_result.numel() * max_result.itemsize());
  }
}

C10_EXPORT void powsum_launch_kernel(TensorIterator& iter, double p) {
  const Tensor& self = iter.tensor(1);
  const Tensor& result = iter.tensor(0);
  int64_t dim = find_reduce_dim(self, result);

  auto in_d = make_tensor_desc(self);
  auto out_d = make_tensor_desc(result);
  if (haganeOpsPowsum(&in_d, &out_d, p, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = Tensor(self).cpu().abs().pow(p);
    auto cpu_result = cpu_self.sum(dim, /*keepdim=*/true);
    std::memcpy(result.data_ptr(), cpu_result.const_data_ptr(), result.numel() * result.itemsize());
  }
}

C10_EXPORT void norm_launch_kernel(TensorIterator& iter, double p) {
  const Tensor& self = iter.tensor(1);
  const Tensor& result = iter.tensor(0);
  int64_t dim = find_reduce_dim(self, result);

  auto in_d = make_tensor_desc(self);
  auto out_d = make_tensor_desc(result);
  if (haganeOpsNormVal(&in_d, &out_d, p, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = Tensor(self).cpu();
    auto cpu_result = cpu_self.norm(p, dim, /*keepdim=*/true);
    std::memcpy(result.data_ptr(), cpu_result.const_data_ptr(), result.numel() * result.itemsize());
  }
}

// ---------------------------------------------------------------------------
// Level B: Mode/Kthvalue/Median launch_kernel functions
// (called from compiled TensorModeKernel.cpp, Sorting.cpp)
// ---------------------------------------------------------------------------

C10_EXPORT void launch_fused_mode_kernel(
    const TensorBase& values, const TensorBase& indices,
    const TensorBase& self, int64_t slice_size, int64_t slices) {
  // self is transposed+contiguous; values/indices are transposed views
  // Mode is always along the last dim of the transposed tensor
  int32_t dim = static_cast<int32_t>(self.dim() - 1);
  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  if (haganeOpsMode(&in_d, &val_d, &idx_d, dim) != HAGANE_OPS_SUCCESS) {
    Tensor self_cpu = Tensor(self).cpu();
    auto [mode_vals, mode_idx] = self_cpu.mode(dim);
    std::memcpy(const_cast<void*>(values.const_data_ptr()),
                mode_vals.const_data_ptr(), values.numel() * values.itemsize());
    std::memcpy(const_cast<void*>(indices.const_data_ptr()),
                mode_idx.const_data_ptr(), indices.numel() * indices.itemsize());
  }
}

C10_EXPORT void launch_apply_mode_kernel(
    const TensorBase& values, const TensorBase& indices,
    const TensorBase& self, int64_t dim, int64_t ndim) {
  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  if (haganeOpsMode(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    Tensor self_cpu = Tensor(self).cpu();
    auto [mode_vals, mode_idx] = self_cpu.mode(dim);
    std::memcpy(const_cast<void*>(values.const_data_ptr()),
                mode_vals.const_data_ptr(), values.numel() * values.itemsize());
    std::memcpy(const_cast<void*>(indices.const_data_ptr()),
                mode_idx.const_data_ptr(), indices.numel() * indices.itemsize());
  }
}

C10_EXPORT void launch_kthvalue_kernel(
    const TensorBase& values, const TensorBase& indices,
    const TensorBase& self, int64_t dim, int64_t k) {
  auto in_d = make_tensor_desc(self);
  auto val_d = make_tensor_desc(values);
  auto idx_d = make_tensor_desc(indices);
  if (haganeOpsKthvalue(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim), k) != HAGANE_OPS_SUCCESS) {
    Tensor self_cpu = Tensor(self).cpu();
    auto [kth_vals, kth_idx] = self_cpu.kthvalue(k, dim, /*keepdim=*/true);
    std::memcpy(const_cast<void*>(values.const_data_ptr()),
                kth_vals.const_data_ptr(), values.numel() * values.itemsize());
    std::memcpy(const_cast<void*>(indices.const_data_ptr()),
                kth_idx.const_data_ptr(), indices.numel() * indices.itemsize());
  }
}

C10_EXPORT void launch_median_kernel(
    const TensorBase& vals, const TensorBase& inds,
    const TensorBase& in, int64_t dim, bool ignore_nan) {
  auto in_d = make_tensor_desc(in);
  auto val_d = make_tensor_desc(vals);
  auto idx_d = make_tensor_desc(inds);
  if (haganeOpsMedian(&in_d, &val_d, &idx_d, static_cast<int32_t>(dim),
                      ignore_nan ? 1 : 0) != HAGANE_OPS_SUCCESS) {
    Tensor in_cpu = Tensor(in).cpu();
    auto [med_vals, med_idx] = in_cpu.median(dim, /*keepdim=*/true);
    std::memcpy(const_cast<void*>(vals.const_data_ptr()),
                med_vals.const_data_ptr(), vals.numel() * vals.itemsize());
    std::memcpy(const_cast<void*>(inds.const_data_ptr()),
                med_idx.const_data_ptr(), inds.numel() * inds.itemsize());
  }
}

// ---------------------------------------------------------------------------
// Level B: Tensor creation (from excluded .hip files)
// ---------------------------------------------------------------------------

C10_EXPORT Tensor& arange_cuda_out(
    const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  double sd = start.toDouble(), ed = end.toDouble(), st = step.toDouble();
  TORCH_CHECK(st != 0, "step must be nonzero");
  TORCH_CHECK((st > 0 && sd <= ed) || (st < 0 && sd >= ed),
              "upper bound and larger bound inconsistent with step sign");
  int64_t size = static_cast<int64_t>(std::ceil((ed - sd) / st));
  if (size < 0) size = 0;
  result.resize_({size});
  if (size > 0) {
    auto out_d = make_tensor_desc(result);
    if (haganeOpsArange(&out_d, sd, st) != HAGANE_OPS_SUCCESS) {
      auto cpu_r = at::arange(start, end, step, result.options().device(c10::kCPU));
      std::memcpy(result.data_ptr(), cpu_r.const_data_ptr(), result.numel() * result.itemsize());
    }
  }
  return result;
}

C10_EXPORT Tensor& linspace_cuda_out(
    const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  result.resize_({steps});
  if (steps > 0) {
    auto out_d = make_tensor_desc(result);
    if (haganeOpsLinspace(&out_d, start.toDouble(), end.toDouble(), steps) != HAGANE_OPS_SUCCESS) {
      auto cpu_r = at::linspace(start, end, steps, result.options().device(c10::kCPU));
      std::memcpy(result.data_ptr(), cpu_r.const_data_ptr(), result.numel() * result.itemsize());
    }
  }
  return result;
}

C10_EXPORT Tensor& eye_out_cuda(int64_t n, Tensor& result) {
  result.resize_({n, n});
  result.zero_();
  auto out_d = make_tensor_desc(result);
  if (haganeOpsEye(&out_d, n, n) != HAGANE_OPS_SUCCESS) {
    auto cpu_r = at::eye(n, result.options().device(c10::kCPU));
    std::memcpy(result.data_ptr(), cpu_r.const_data_ptr(), result.numel() * result.itemsize());
  }
  return result;
}

C10_EXPORT Tensor& eye_out_cuda(int64_t n, int64_t m, Tensor& result) {
  result.resize_({n, m});
  result.zero_();
  auto out_d = make_tensor_desc(result);
  if (haganeOpsEye(&out_d, n, m) != HAGANE_OPS_SUCCESS) {
    auto cpu_r = at::eye(n, m, result.options().device(c10::kCPU));
    std::memcpy(result.data_ptr(), cpu_r.const_data_ptr(), result.numel() * result.itemsize());
  }
  return result;
}

// ---------------------------------------------------------------------------
// Level B: Index/manipulation (from excluded .hip files)
// ---------------------------------------------------------------------------

C10_EXPORT Tensor index_select_cuda(
    const Tensor& self, int64_t dim, const Tensor& index) {
  auto result_sizes = self.sizes().vec();
  result_sizes[dim] = index.numel();
  auto result = at::empty(result_sizes, self.options());

  auto in_d = make_tensor_desc(self);
  auto idx_d = make_tensor_desc(index);
  auto out_d = make_tensor_desc(result);
  if (haganeOpsIndexSelect(&in_d, &idx_d, &out_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    auto cpu_result = self.cpu().index_select(dim, index.cpu());
    std::memcpy(result.data_ptr(), cpu_result.const_data_ptr(),
                result.numel() * result.itemsize());
  }
  return result;
}

C10_EXPORT Tensor& masked_fill__cuda(
    Tensor& self, const Tensor& mask, const Scalar& value) {
  auto self_d = make_tensor_desc(self);
  auto mask_d = make_tensor_desc(mask);
  if (haganeOpsMaskedFill(&self_d, &mask_d, value.toFloat()) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = self.cpu();
    cpu_self.masked_fill_(mask.cpu(), value);
    std::memcpy(self.data_ptr(), cpu_self.const_data_ptr(),
                self.numel() * self.itemsize());
  }
  return self;
}

C10_EXPORT Tensor& masked_fill__cuda(
    Tensor& self, const Tensor& mask, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ expects a 0-dimensional value tensor");
  return masked_fill__cuda(self, mask, value.item());
}

C10_EXPORT Tensor roll_cuda(
    const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.empty()) {
    // No dims specified: flatten, roll, reshape back
    auto flat = self.contiguous().view(-1);
    auto result_flat = at::empty_like(flat);
    auto in_d = make_tensor_desc(flat);
    auto out_d = make_tensor_desc(result_flat);
    int64_t s = shifts[0];
    int32_t d = 0;
    if (haganeOpsRoll(&in_d, &out_d, &s, &d, 1) != HAGANE_OPS_SUCCESS) {
      auto cpu_result = self.cpu().roll(shifts, dims);
      return cpu_result.to(self.device());
    }
    return result_flat.view(self.sizes());
  }

  auto result = at::empty_like(self);
  auto in_d = make_tensor_desc(self);
  auto out_d = make_tensor_desc(result);
  std::vector<int64_t> shifts_vec(shifts.begin(), shifts.end());
  std::vector<int32_t> dims_vec;
  for (auto d : dims) dims_vec.push_back(static_cast<int32_t>(d));

  if (haganeOpsRoll(&in_d, &out_d, shifts_vec.data(), dims_vec.data(),
                     static_cast<int32_t>(shifts.size())) != HAGANE_OPS_SUCCESS) {
    auto cpu_result = self.cpu().roll(shifts, dims);
    std::memcpy(result.data_ptr(), cpu_result.const_data_ptr(),
                result.numel() * result.itemsize());
  }
  return result;
}

C10_EXPORT Tensor repeat_interleave_cuda(
    const Tensor& repeat, std::optional<int64_t> output_size) {
  TORCH_CHECK(repeat.dim() == 1, "repeat_interleave: repeats must be 1-d");
  int64_t out_sz;
  if (output_size.has_value()) {
    out_sz = *output_size;
  } else {
    out_sz = repeat.sum().item<int64_t>();
  }
  auto result = at::empty({out_sz}, repeat.options().dtype(c10::kLong));
  auto rep_d = make_tensor_desc(repeat);
  if (haganeOpsRepeatInterleave(&rep_d, result.data_ptr(), out_sz,
                                 HAGANE_DTYPE_INT64) != HAGANE_OPS_SUCCESS) {
    auto cpu_result = at::repeat_interleave(repeat.cpu(), output_size);
    std::memcpy(result.data_ptr(), cpu_result.const_data_ptr(),
                result.numel() * result.itemsize());
  }
  return result;
}

C10_EXPORT Tensor& nonzero_out_cuda(const Tensor& self, Tensor& out) {
  int64_t num_nonzero = 0;
  auto in_d = make_tensor_desc(self);

  // First pass: get count
  if (haganeOpsNonzero(&in_d, nullptr, &num_nonzero) != HAGANE_OPS_SUCCESS) {
    auto cpu_result = self.cpu().nonzero();
    out.resize_({cpu_result.size(0), cpu_result.size(1)});
    std::memcpy(out.data_ptr(), cpu_result.const_data_ptr(),
                out.numel() * out.itemsize());
    return out;
  }

  out.resize_({num_nonzero, self.dim()});
  if (num_nonzero > 0) {
    haganeOpsNonzero(&in_d, out.data_ptr(), &num_nonzero);
  }
  return out;
}

C10_EXPORT Tensor nonzero_cuda(const Tensor& self) {
  auto out = at::empty({0}, self.options().dtype(c10::kLong));
  nonzero_out_cuda(self, out);
  return out;
}

// ---------------------------------------------------------------------------
// Batch 5: Foreach ops — Metal GPU via MLX
// All foreach ops iterate tensor lists and apply element-wise MLX operations.
// ---------------------------------------------------------------------------

// --- Foreach unary ops ---

#define FOREACH_UNARY_IMPL(name, cfunc) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_cuda(TensorList tensors) { \
  std::vector<Tensor> r; r.reserve(tensors.size()); \
  for (const auto& t : tensors) { \
    auto out = at::empty_like(t); \
    auto id = make_tensor_desc(t), od = make_tensor_desc(out); \
    cfunc(&id, &od); \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_cuda_(TensorList tensors) { \
  for (const auto& t : tensors) { \
    auto d = make_tensor_desc(t); \
    cfunc(&d, &d); \
  } \
}

FOREACH_UNARY_IMPL(abs, haganeOpsAbs)
FOREACH_UNARY_IMPL(neg, haganeOpsNeg)
FOREACH_UNARY_IMPL(cos, haganeOpsCos)
FOREACH_UNARY_IMPL(sin, haganeOpsSin)
FOREACH_UNARY_IMPL(tan, haganeOpsTan)
FOREACH_UNARY_IMPL(acos, haganeOpsAcos)
FOREACH_UNARY_IMPL(asin, haganeOpsAsin)
FOREACH_UNARY_IMPL(atan, haganeOpsAtan)
FOREACH_UNARY_IMPL(cosh, haganeOpsCosh)
FOREACH_UNARY_IMPL(sinh, haganeOpsSinh)
FOREACH_UNARY_IMPL(tanh, haganeOpsTanh)
FOREACH_UNARY_IMPL(exp, haganeOpsExp)
FOREACH_UNARY_IMPL(expm1, haganeOpsExpm1)
FOREACH_UNARY_IMPL(log, haganeOpsLog)
FOREACH_UNARY_IMPL(log2, haganeOpsLog2)
FOREACH_UNARY_IMPL(log10, haganeOpsLog10)
FOREACH_UNARY_IMPL(log1p, haganeOpsLog1p)
FOREACH_UNARY_IMPL(sqrt, haganeOpsSqrt)
FOREACH_UNARY_IMPL(rsqrt, haganeOpsRsqrt)
FOREACH_UNARY_IMPL(ceil, haganeOpsCeil)
FOREACH_UNARY_IMPL(floor, haganeOpsFloor)
FOREACH_UNARY_IMPL(round, haganeOpsRound)
FOREACH_UNARY_IMPL(trunc, haganeOpsTrunc)
FOREACH_UNARY_IMPL(frac, haganeOpsFrac)
FOREACH_UNARY_IMPL(sign, haganeOpsSign)
FOREACH_UNARY_IMPL(sigmoid, haganeOpsSigmoid)
FOREACH_UNARY_IMPL(erf, haganeOpsErf)
FOREACH_UNARY_IMPL(erfc, haganeOpsErfc)
FOREACH_UNARY_IMPL(reciprocal, haganeOpsReciprocal)
FOREACH_UNARY_IMPL(lgamma, haganeOpsLgamma)

#undef FOREACH_UNARY_IMPL

// zero_ is inplace-only
C10_EXPORT void foreach_tensor_zero_cuda_(TensorList tensors) {
  for (const auto& t : tensors) {
    auto d = make_tensor_desc(t);
    haganeOpsFill(&d, 0.0);
  }
}

// clone returns new tensors (with optional memory format)
C10_EXPORT std::vector<Tensor> foreach_tensor_clone_cuda(
    TensorList tensors, std::optional<MemoryFormat> memory_format) {
  std::vector<Tensor> r; r.reserve(tensors.size());
  for (const auto& t : tensors) {
    r.push_back(t.clone(memory_format.value_or(MemoryFormat::Preserve)));
  }
  return r;
}

// --- Foreach binary scalar ops ---

#define FOREACH_BINARY_SCALAR_IMPL(name, op_expr) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_scalar_kernel_cuda( \
    TensorList tensors, const Scalar& scalar) { \
  std::vector<Tensor> r; r.reserve(tensors.size()); \
  for (const auto& t : tensors) { \
    auto out = at::empty_like(t); \
    auto s = at::scalar_tensor(scalar, t.options()); \
    auto id = make_tensor_desc(t), sd = make_tensor_desc(s), od = make_tensor_desc(out); \
    op_expr; \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_scalar_kernel_cuda_( \
    TensorList tensors, const Scalar& scalar) { \
  for (const auto& t : tensors) { \
    auto s = at::scalar_tensor(scalar, t.options()); \
    auto id = make_tensor_desc(t), sd = make_tensor_desc(s); \
    auto od = id; od.data = const_cast<void*>(t.const_data_ptr()); \
    op_expr; \
  } \
}

FOREACH_BINARY_SCALAR_IMPL(add, haganeOpsAdd(&id, &sd, &od, 1.0f))
FOREACH_BINARY_SCALAR_IMPL(sub, haganeOpsSub(&id, &sd, &od, 1.0f))
FOREACH_BINARY_SCALAR_IMPL(mul, haganeOpsMul(&id, &sd, &od))
FOREACH_BINARY_SCALAR_IMPL(div, haganeOpsDiv(&id, &sd, &od))
FOREACH_BINARY_SCALAR_IMPL(pow, haganeOpsPow(&id, &sd, &od))
FOREACH_BINARY_SCALAR_IMPL(clamp_max, haganeOpsMinimum(&id, &sd, &od))
FOREACH_BINARY_SCALAR_IMPL(clamp_min, haganeOpsMaximum(&id, &sd, &od))

#undef FOREACH_BINARY_SCALAR_IMPL

// scalar^tensor (pow with scalar base)
C10_EXPORT std::vector<Tensor> foreach_scalar_pow_list_kernel_cuda(
    const Scalar& scalar, TensorList exponent) {
  std::vector<Tensor> r; r.reserve(exponent.size());
  for (const auto& t : exponent) {
    auto base = at::scalar_tensor(scalar, t.options());
    auto out = at::empty_like(t);
    auto bd = make_tensor_desc(base), td = make_tensor_desc(t), od = make_tensor_desc(out);
    haganeOpsPow(&bd, &td, &od);
    r.push_back(std::move(out));
  }
  return r;
}

// --- Foreach binary list ops ---

#define FOREACH_BINARY_LIST_IMPL(name, op_expr) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_list_kernel_cuda( \
    TensorList t1, TensorList t2) { \
  std::vector<Tensor> r; r.reserve(t1.size()); \
  for (size_t i = 0; i < t1.size(); i++) { \
    auto out = at::empty_like(t1[i]); \
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]), od = make_tensor_desc(out); \
    op_expr; \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_list_kernel_cuda_( \
    TensorList t1, TensorList t2) { \
  for (size_t i = 0; i < t1.size(); i++) { \
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]); \
    auto od = ad; od.data = const_cast<void*>(t1[i].const_data_ptr()); \
    op_expr; \
  } \
}

FOREACH_BINARY_LIST_IMPL(mul, haganeOpsMul(&ad, &bd, &od))
FOREACH_BINARY_LIST_IMPL(div, haganeOpsDiv(&ad, &bd, &od))
FOREACH_BINARY_LIST_IMPL(pow, haganeOpsPow(&ad, &bd, &od))
FOREACH_BINARY_LIST_IMPL(clamp_max, haganeOpsMinimum(&ad, &bd, &od))
FOREACH_BINARY_LIST_IMPL(clamp_min, haganeOpsMaximum(&ad, &bd, &od))

#undef FOREACH_BINARY_LIST_IMPL

// add/sub list with alpha parameter
C10_EXPORT std::vector<Tensor> foreach_tensor_add_list_kernel_cuda(
    TensorList t1, TensorList t2, const Scalar& alpha) {
  std::vector<Tensor> r; r.reserve(t1.size());
  for (size_t i = 0; i < t1.size(); i++) {
    auto out = at::empty_like(t1[i]);
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]), od = make_tensor_desc(out);
    haganeOpsAdd(&ad, &bd, &od, alpha.toFloat());
    r.push_back(std::move(out));
  }
  return r;
}
C10_EXPORT void foreach_tensor_add_list_kernel_cuda_(
    TensorList t1, TensorList t2, const Scalar& alpha) {
  for (size_t i = 0; i < t1.size(); i++) {
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]);
    auto od = ad; od.data = const_cast<void*>(t1[i].const_data_ptr());
    haganeOpsAdd(&ad, &bd, &od, alpha.toFloat());
  }
}

C10_EXPORT std::vector<Tensor> foreach_tensor_sub_list_kernel_cuda(
    TensorList t1, TensorList t2, const Scalar& alpha) {
  std::vector<Tensor> r; r.reserve(t1.size());
  for (size_t i = 0; i < t1.size(); i++) {
    auto out = at::empty_like(t1[i]);
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]), od = make_tensor_desc(out);
    haganeOpsSub(&ad, &bd, &od, alpha.toFloat());
    r.push_back(std::move(out));
  }
  return r;
}
C10_EXPORT void foreach_tensor_sub_list_kernel_cuda_(
    TensorList t1, TensorList t2, const Scalar& alpha) {
  for (size_t i = 0; i < t1.size(); i++) {
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]);
    auto od = ad; od.data = const_cast<void*>(t1[i].const_data_ptr());
    haganeOpsSub(&ad, &bd, &od, alpha.toFloat());
  }
}

// copy list (inplace only)
C10_EXPORT void foreach_tensor_copy_list_kernel_cuda_(
    TensorList self, TensorList src, bool /*non_blocking*/) {
  for (size_t i = 0; i < self.size(); i++) {
    auto sd = make_tensor_desc(src[i]);
    auto dd = make_tensor_desc(self[i]);
    dd.data = const_cast<void*>(self[i].const_data_ptr());
    std::memcpy(dd.data, sd.data, self[i].numel() * self[i].itemsize());
  }
}

// --- Foreach binary scalarlist ops ---

#define FOREACH_BINARY_SCALARLIST_IMPL(name, op_expr) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_scalarlist_kernel_cuda( \
    TensorList tensors, at::ArrayRef<Scalar> scalars) { \
  std::vector<Tensor> r; r.reserve(tensors.size()); \
  for (size_t i = 0; i < tensors.size(); i++) { \
    auto out = at::empty_like(tensors[i]); \
    auto s = at::scalar_tensor(scalars[i], tensors[i].options()); \
    auto td = make_tensor_desc(tensors[i]), sd = make_tensor_desc(s), od = make_tensor_desc(out); \
    op_expr; \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_scalarlist_kernel_cuda_( \
    TensorList tensors, at::ArrayRef<Scalar> scalars) { \
  for (size_t i = 0; i < tensors.size(); i++) { \
    auto s = at::scalar_tensor(scalars[i], tensors[i].options()); \
    auto td = make_tensor_desc(tensors[i]), sd = make_tensor_desc(s); \
    auto od = td; od.data = const_cast<void*>(tensors[i].const_data_ptr()); \
    op_expr; \
  } \
}

FOREACH_BINARY_SCALARLIST_IMPL(add, haganeOpsAdd(&td, &sd, &od, 1.0f))
FOREACH_BINARY_SCALARLIST_IMPL(sub, haganeOpsSub(&td, &sd, &od, 1.0f))
FOREACH_BINARY_SCALARLIST_IMPL(mul, haganeOpsMul(&td, &sd, &od))
FOREACH_BINARY_SCALARLIST_IMPL(div, haganeOpsDiv(&td, &sd, &od))
FOREACH_BINARY_SCALARLIST_IMPL(pow, haganeOpsPow(&td, &sd, &od))
FOREACH_BINARY_SCALARLIST_IMPL(clamp_max, haganeOpsMinimum(&td, &sd, &od))
FOREACH_BINARY_SCALARLIST_IMPL(clamp_min, haganeOpsMaximum(&td, &sd, &od))

#undef FOREACH_BINARY_SCALARLIST_IMPL

// --- Foreach binary tensor ops ---

C10_EXPORT std::vector<Tensor> foreach_tensor_add_tensor_kernel_cuda(
    TensorList tensors, const Tensor& tensor, const Scalar& alpha) {
  std::vector<Tensor> r; r.reserve(tensors.size());
  auto td2 = make_tensor_desc(tensor);
  for (const auto& t : tensors) {
    auto out = at::empty_like(t);
    auto td1 = make_tensor_desc(t), od = make_tensor_desc(out);
    haganeOpsAdd(&td1, &td2, &od, alpha.toFloat());
    r.push_back(std::move(out));
  }
  return r;
}
C10_EXPORT void foreach_tensor_add_tensor_kernel_cuda_(
    TensorList tensors, const Tensor& tensor, const Scalar& alpha) {
  auto td2 = make_tensor_desc(tensor);
  for (const auto& t : tensors) {
    auto td1 = make_tensor_desc(t);
    auto od = td1; od.data = const_cast<void*>(t.const_data_ptr());
    haganeOpsAdd(&td1, &td2, &od, alpha.toFloat());
  }
}

#define FOREACH_BINARY_TENSOR_IMPL(name, op_expr) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_tensor_kernel_cuda( \
    TensorList tensors, const Tensor& tensor) { \
  std::vector<Tensor> r; r.reserve(tensors.size()); \
  auto td2 = make_tensor_desc(tensor); \
  for (const auto& t : tensors) { \
    auto out = at::empty_like(t); \
    auto td1 = make_tensor_desc(t), od = make_tensor_desc(out); \
    op_expr; \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_tensor_kernel_cuda_( \
    TensorList tensors, const Tensor& tensor) { \
  auto td2 = make_tensor_desc(tensor); \
  for (const auto& t : tensors) { \
    auto td1 = make_tensor_desc(t); \
    auto od = td1; od.data = const_cast<void*>(t.const_data_ptr()); \
    op_expr; \
  } \
}

FOREACH_BINARY_TENSOR_IMPL(mul, haganeOpsMul(&td1, &td2, &od))
FOREACH_BINARY_TENSOR_IMPL(div, haganeOpsDiv(&td1, &td2, &od))

#undef FOREACH_BINARY_TENSOR_IMPL

// --- Foreach ternary (lerp) ops ---

// lerp(a, b, weight) = a + weight * (b - a)
C10_EXPORT std::vector<Tensor> foreach_tensor_lerp_ternary_cuda(
    TensorList t1, TensorList t2, TensorList t3) {
  std::vector<Tensor> r; r.reserve(t1.size());
  for (size_t i = 0; i < t1.size(); i++) {
    auto out = at::empty_like(t1[i]);
    // lerp(a, b, w) = a + w * (b - a)
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]);
    auto wd = make_tensor_desc(t3[i]), od = make_tensor_desc(out);
    // diff = b - a
    auto diff = at::empty_like(t1[i]);
    auto dd = make_tensor_desc(diff);
    haganeOpsSub(&bd, &ad, &dd, 1.0f);
    // w_diff = w * diff
    auto w_diff = at::empty_like(t1[i]);
    auto wdd = make_tensor_desc(w_diff);
    haganeOpsMul(&wd, &dd, &wdd);
    // out = a + w_diff
    haganeOpsAdd(&ad, &wdd, &od, 1.0f);
    r.push_back(std::move(out));
  }
  return r;
}
C10_EXPORT void foreach_tensor_lerp_ternary_cuda_(
    TensorList t1, TensorList t2, TensorList t3) {
  for (size_t i = 0; i < t1.size(); i++) {
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]), wd = make_tensor_desc(t3[i]);
    auto diff = at::empty_like(t1[i]);
    auto dd = make_tensor_desc(diff);
    haganeOpsSub(&bd, &ad, &dd, 1.0f);
    auto w_diff = at::empty_like(t1[i]);
    auto wdd = make_tensor_desc(w_diff);
    haganeOpsMul(&wd, &dd, &wdd);
    auto od = ad; od.data = const_cast<void*>(t1[i].const_data_ptr());
    haganeOpsAdd(&ad, &wdd, &od, 1.0f);
  }
}

C10_EXPORT std::vector<Tensor> foreach_tensor_lerp_list_cuda(
    TensorList t1, TensorList t2, const Scalar& weight) {
  std::vector<Tensor> r; r.reserve(t1.size());
  for (size_t i = 0; i < t1.size(); i++) {
    auto out = at::empty_like(t1[i]);
    auto w = at::scalar_tensor(weight, t1[i].options());
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]);
    auto wd = make_tensor_desc(w), od = make_tensor_desc(out);
    auto diff = at::empty_like(t1[i]);
    auto dd = make_tensor_desc(diff);
    haganeOpsSub(&bd, &ad, &dd, 1.0f);
    auto w_diff = at::empty_like(t1[i]);
    auto wdd = make_tensor_desc(w_diff);
    haganeOpsMul(&wd, &dd, &wdd);
    haganeOpsAdd(&ad, &wdd, &od, 1.0f);
    r.push_back(std::move(out));
  }
  return r;
}
C10_EXPORT void foreach_tensor_lerp_list_cuda_(
    TensorList t1, TensorList t2, const Scalar& weight) {
  for (size_t i = 0; i < t1.size(); i++) {
    auto w = at::scalar_tensor(weight, t1[i].options());
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]), wd = make_tensor_desc(w);
    auto diff = at::empty_like(t1[i]);
    auto dd = make_tensor_desc(diff);
    haganeOpsSub(&bd, &ad, &dd, 1.0f);
    auto w_diff = at::empty_like(t1[i]);
    auto wdd = make_tensor_desc(w_diff);
    haganeOpsMul(&wd, &dd, &wdd);
    auto od = ad; od.data = const_cast<void*>(t1[i].const_data_ptr());
    haganeOpsAdd(&ad, &wdd, &od, 1.0f);
  }
}

C10_EXPORT std::vector<Tensor> foreach_tensor_lerp_scalarlist_cuda(
    TensorList t1, TensorList t2, at::ArrayRef<Scalar> scalars) {
  std::vector<Tensor> r; r.reserve(t1.size());
  for (size_t i = 0; i < t1.size(); i++) {
    auto out = at::empty_like(t1[i]);
    auto w = at::scalar_tensor(scalars[i], t1[i].options());
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]);
    auto wd = make_tensor_desc(w), od = make_tensor_desc(out);
    auto diff = at::empty_like(t1[i]);
    auto dd = make_tensor_desc(diff);
    haganeOpsSub(&bd, &ad, &dd, 1.0f);
    auto w_diff = at::empty_like(t1[i]);
    auto wdd = make_tensor_desc(w_diff);
    haganeOpsMul(&wd, &dd, &wdd);
    haganeOpsAdd(&ad, &wdd, &od, 1.0f);
    r.push_back(std::move(out));
  }
  return r;
}
C10_EXPORT void foreach_tensor_lerp_scalarlist_cuda_(
    TensorList t1, TensorList t2, at::ArrayRef<Scalar> scalars) {
  for (size_t i = 0; i < t1.size(); i++) {
    auto w = at::scalar_tensor(scalars[i], t1[i].options());
    auto ad = make_tensor_desc(t1[i]), bd = make_tensor_desc(t2[i]), wd = make_tensor_desc(w);
    auto diff = at::empty_like(t1[i]);
    auto dd = make_tensor_desc(diff);
    haganeOpsSub(&bd, &ad, &dd, 1.0f);
    auto w_diff = at::empty_like(t1[i]);
    auto wdd = make_tensor_desc(w_diff);
    haganeOpsMul(&wd, &dd, &wdd);
    auto od = ad; od.data = const_cast<void*>(t1[i].const_data_ptr());
    haganeOpsAdd(&ad, &wdd, &od, 1.0f);
  }
}

// --- Foreach pointwise ops (addcmul, addcdiv) ---
// addcmul(input, t1, t2, value) = input + value * t1 * t2
// addcdiv(input, t1, t2, value) = input + value * t1 / t2

#define FOREACH_POINTWISE_SCALAR_IMPL(name, inner_op) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_scalar_cuda( \
    TensorList input, TensorList t1, TensorList t2, const Scalar& scalar) { \
  std::vector<Tensor> r; r.reserve(input.size()); \
  for (size_t i = 0; i < input.size(); i++) { \
    auto tmp = at::empty_like(input[i]); \
    auto t1d = make_tensor_desc(t1[i]), t2d = make_tensor_desc(t2[i]), td = make_tensor_desc(tmp); \
    inner_op(&t1d, &t2d, &td); \
    auto out = at::empty_like(input[i]); \
    auto v = at::scalar_tensor(scalar, input[i].options()); \
    auto vd = make_tensor_desc(v); \
    auto scaled = at::empty_like(input[i]); \
    auto scd = make_tensor_desc(scaled); \
    haganeOpsMul(&td, &vd, &scd); \
    auto ind = make_tensor_desc(input[i]), od = make_tensor_desc(out); \
    haganeOpsAdd(&ind, &scd, &od, 1.0f); \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_scalar_cuda_( \
    TensorList input, TensorList t1, TensorList t2, const Scalar& scalar) { \
  for (size_t i = 0; i < input.size(); i++) { \
    auto tmp = at::empty_like(input[i]); \
    auto t1d = make_tensor_desc(t1[i]), t2d = make_tensor_desc(t2[i]), td = make_tensor_desc(tmp); \
    inner_op(&t1d, &t2d, &td); \
    auto v = at::scalar_tensor(scalar, input[i].options()); \
    auto vd = make_tensor_desc(v); \
    auto scaled = at::empty_like(input[i]); \
    auto scd = make_tensor_desc(scaled); \
    haganeOpsMul(&td, &vd, &scd); \
    auto ind = make_tensor_desc(input[i]); \
    auto od = ind; od.data = const_cast<void*>(input[i].const_data_ptr()); \
    haganeOpsAdd(&ind, &scd, &od, 1.0f); \
  } \
}

FOREACH_POINTWISE_SCALAR_IMPL(addcmul, haganeOpsMul)
FOREACH_POINTWISE_SCALAR_IMPL(addcdiv, haganeOpsDiv)

#undef FOREACH_POINTWISE_SCALAR_IMPL

#define FOREACH_POINTWISE_TENSOR_IMPL(name, inner_op) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_tensor_cuda( \
    TensorList input, TensorList t1, TensorList t2, const Tensor& scalars_) { \
  std::vector<Tensor> r; r.reserve(input.size()); \
  for (size_t i = 0; i < input.size(); i++) { \
    auto s = scalars_[static_cast<int64_t>(i)]; \
    auto tmp = at::empty_like(input[i]); \
    auto t1d = make_tensor_desc(t1[i]), t2d = make_tensor_desc(t2[i]), td = make_tensor_desc(tmp); \
    inner_op(&t1d, &t2d, &td); \
    auto sv = at::scalar_tensor(s.item(), input[i].options()); \
    auto svd = make_tensor_desc(sv); \
    auto scaled = at::empty_like(input[i]); \
    auto scd = make_tensor_desc(scaled); \
    haganeOpsMul(&td, &svd, &scd); \
    auto out = at::empty_like(input[i]); \
    auto ind = make_tensor_desc(input[i]), od = make_tensor_desc(out); \
    haganeOpsAdd(&ind, &scd, &od, 1.0f); \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_tensor_cuda_( \
    TensorList input, TensorList t1, TensorList t2, const Tensor& scalars_) { \
  for (size_t i = 0; i < input.size(); i++) { \
    auto s = scalars_[static_cast<int64_t>(i)]; \
    auto tmp = at::empty_like(input[i]); \
    auto t1d = make_tensor_desc(t1[i]), t2d = make_tensor_desc(t2[i]), td = make_tensor_desc(tmp); \
    inner_op(&t1d, &t2d, &td); \
    auto sv = at::scalar_tensor(s.item(), input[i].options()); \
    auto svd = make_tensor_desc(sv); \
    auto scaled = at::empty_like(input[i]); \
    auto scd = make_tensor_desc(scaled); \
    haganeOpsMul(&td, &svd, &scd); \
    auto ind = make_tensor_desc(input[i]); \
    auto od = ind; od.data = const_cast<void*>(input[i].const_data_ptr()); \
    haganeOpsAdd(&ind, &scd, &od, 1.0f); \
  } \
}

FOREACH_POINTWISE_TENSOR_IMPL(addcmul, haganeOpsMul)
FOREACH_POINTWISE_TENSOR_IMPL(addcdiv, haganeOpsDiv)

#undef FOREACH_POINTWISE_TENSOR_IMPL

#define FOREACH_POINTWISE_SCALARLIST_IMPL(name, inner_op) \
C10_EXPORT std::vector<Tensor> foreach_tensor_##name##_scalarlist_cuda( \
    TensorList input, TensorList t1, TensorList t2, at::ArrayRef<Scalar> scalars) { \
  std::vector<Tensor> r; r.reserve(input.size()); \
  for (size_t i = 0; i < input.size(); i++) { \
    auto tmp = at::empty_like(input[i]); \
    auto t1d = make_tensor_desc(t1[i]), t2d = make_tensor_desc(t2[i]), td = make_tensor_desc(tmp); \
    inner_op(&t1d, &t2d, &td); \
    auto v = at::scalar_tensor(scalars[i], input[i].options()); \
    auto vd = make_tensor_desc(v); \
    auto scaled = at::empty_like(input[i]); \
    auto scd = make_tensor_desc(scaled); \
    haganeOpsMul(&td, &vd, &scd); \
    auto out = at::empty_like(input[i]); \
    auto ind = make_tensor_desc(input[i]), od = make_tensor_desc(out); \
    haganeOpsAdd(&ind, &scd, &od, 1.0f); \
    r.push_back(std::move(out)); \
  } \
  return r; \
} \
C10_EXPORT void foreach_tensor_##name##_scalarlist_cuda_( \
    TensorList input, TensorList t1, TensorList t2, at::ArrayRef<Scalar> scalars) { \
  for (size_t i = 0; i < input.size(); i++) { \
    auto tmp = at::empty_like(input[i]); \
    auto t1d = make_tensor_desc(t1[i]), t2d = make_tensor_desc(t2[i]), td = make_tensor_desc(tmp); \
    inner_op(&t1d, &t2d, &td); \
    auto v = at::scalar_tensor(scalars[i], input[i].options()); \
    auto vd = make_tensor_desc(v); \
    auto scaled = at::empty_like(input[i]); \
    auto scd = make_tensor_desc(scaled); \
    haganeOpsMul(&td, &vd, &scd); \
    auto ind = make_tensor_desc(input[i]); \
    auto od = ind; od.data = const_cast<void*>(input[i].const_data_ptr()); \
    haganeOpsAdd(&ind, &scd, &od, 1.0f); \
  } \
}

FOREACH_POINTWISE_SCALARLIST_IMPL(addcmul, haganeOpsMul)
FOREACH_POINTWISE_SCALARLIST_IMPL(addcdiv, haganeOpsDiv)

#undef FOREACH_POINTWISE_SCALARLIST_IMPL

// --- Foreach reduce ops ---

C10_EXPORT std::vector<Tensor> foreach_tensor_max_cuda(TensorList tensors) {
  std::vector<Tensor> r; r.reserve(tensors.size());
  for (const auto& t : tensors) {
    auto out = at::empty({}, t.options());
    auto id = make_tensor_desc(t), od = make_tensor_desc(out);
    haganeOpsMaxValues(&id, &od);
    r.push_back(std::move(out));
  }
  return r;
}

C10_EXPORT std::vector<Tensor> foreach_tensor_norm_cuda(
    TensorList tensors, const Scalar& ord, std::optional<ScalarType> /*dtype*/) {
  std::vector<Tensor> r; r.reserve(tensors.size());
  double p = ord.toDouble();
  for (const auto& t : tensors) {
    auto flat = t.contiguous().view({-1});
    auto out = at::empty({}, t.options());
    auto id = make_tensor_desc(flat), od = make_tensor_desc(out);
    haganeOpsNormVal(&id, &od, p, 0);
    r.push_back(std::move(out));
  }
  return r;
}

C10_EXPORT std::vector<Tensor> foreach_tensor_powsum_cuda(
    TensorList tensors, const Scalar& ord, std::optional<ScalarType> /*dtype*/) {
  std::vector<Tensor> r; r.reserve(tensors.size());
  double p = ord.toDouble();
  for (const auto& t : tensors) {
    auto flat = t.contiguous().view({-1});
    auto out = at::empty({}, t.options());
    auto id = make_tensor_desc(flat), od = make_tensor_desc(out);
    haganeOpsPowsum(&id, &od, p, 0);
    r.push_back(std::move(out));
  }
  return r;
}

// --- AMP: check for non-finite and unscale ---

C10_EXPORT void _amp_foreach_non_finite_check_and_unscale_cuda_(
    TensorList scaled_grads, Tensor& found_inf, const Tensor& inv_scale) {
  float* found_inf_ptr = found_inf.data_ptr<float>();
  float inv_scale_val = inv_scale.item<float>();
  for (const auto& t : scaled_grads) {
    // Check for inf/nan
    auto td = make_tensor_desc(t);
    // Simple check: multiply by inv_scale, check for non-finite
    int64_t n = t.numel();
    float* data = static_cast<float*>(const_cast<void*>(t.const_data_ptr()));
    for (int64_t j = 0; j < n; j++) {
      if (!std::isfinite(data[j])) {
        *found_inf_ptr = 1.0f;
        return;
      }
      data[j] *= inv_scale_val;
    }
  }
}

// ---------------------------------------------------------------------------
// Batch 5: Non-foreach ops — Random, Creation, Dropout
// ---------------------------------------------------------------------------

// randperm: generate random permutation
C10_EXPORT Tensor& randperm_out_cuda(
    int64_t n, std::optional<Generator> generator, Tensor& result) {
  result.resize_({n});
  if (n == 0) return result;
  // Generate random keys and argsort for permutation
  auto keys = at::empty({n}, result.options().dtype(kFloat));
  auto kd = make_tensor_desc(keys);
  haganeOpsUniform(&kd, 0.0, 1.0);
  // Argsort the random keys to get permutation
  auto rd = make_tensor_desc(result);
  auto sorted_keys = at::empty_like(keys);
  auto skd = make_tensor_desc(sorted_keys);
  // Copy keys for sorting
  std::memcpy(sorted_keys.data_ptr(), keys.data_ptr(), n * sizeof(float));
  haganeOpsSort(&skd, &rd, 0, 0);
  return result;
}

// logspace_out
C10_EXPORT Tensor& logspace_cuda_out(
    const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  result.resize_({steps});
  if (steps == 0) return result;
  if (steps == 1) {
    result.fill_(std::pow(base, start.toDouble()));
    return result;
  }
  // linspace in log domain, then pow
  auto lin = at::linspace(start, end, steps, result.options());
  auto ld = make_tensor_desc(lin), rd = make_tensor_desc(result);
  // result = base^lin
  if (base == 10.0) {
    // 10^x = exp(x * ln(10))
    auto scale = at::scalar_tensor(std::log(10.0), result.options());
    auto tmp = at::empty_like(result);
    auto sd = make_tensor_desc(scale), td = make_tensor_desc(tmp);
    haganeOpsMul(&ld, &sd, &td);
    haganeOpsExp(&td, &rd);
  } else if (base == 2.0) {
    haganeOpsExp2(&ld, &rd);
  } else {
    auto scale = at::scalar_tensor(std::log(base), result.options());
    auto tmp = at::empty_like(result);
    auto sd = make_tensor_desc(scale), td = make_tensor_desc(tmp);
    haganeOpsMul(&ld, &sd, &td);
    haganeOpsExp(&td, &rd);
  }
  return result;
}

// range_out
C10_EXPORT Tensor& range_cuda_out(
    const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  // Compute size
  double s = start.toDouble(), e = end.toDouble(), st = step.toDouble();
  int64_t size = static_cast<int64_t>(std::floor((e - s) / st)) + 1;
  result.resize_({size});
  auto rd = make_tensor_desc(result);
  haganeOpsArange(&rd, s, st);
  return result;
}

// _chunk_cat: concatenate chunks of tensors
C10_EXPORT Tensor _chunk_cat_cuda(TensorList tensors, int64_t dim, int64_t num_chunks) {
  // Collect the relevant chunks and concatenate
  std::vector<Tensor> chunks;
  chunks.reserve(tensors.size());
  for (const auto& t : tensors) {
    chunks.push_back(t);
  }
  return at::cat(chunks, dim);
}

C10_EXPORT Tensor& _chunk_cat_out_cuda(
    TensorList tensors, int64_t dim, int64_t num_chunks, Tensor& out) {
  auto result = _chunk_cat_cuda(tensors, dim, num_chunks);
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}

// split_with_sizes_copy_out
C10_EXPORT void split_with_sizes_copy_out_cuda(
    const Tensor& self, c10::ArrayRef<int64_t> split_sizes, int64_t dim,
    TensorList out) {
  auto splits = self.split_with_sizes(split_sizes, dim);
  for (size_t i = 0; i < out.size(); i++) {
    const_cast<Tensor&>(out[i]).copy_(splits[i]);
  }
}

// rrelu_with_noise
C10_EXPORT Tensor rrelu_with_noise_cuda(
    const Tensor& self, Tensor& noise, const Scalar& lower,
    const Scalar& upper, bool training, std::optional<Generator> generator) {
  auto output = at::empty_like(self);
  if (training) {
    // Fill noise with uniform random in [lower, upper]
    auto noise_d = make_tensor_desc(noise);
    haganeOpsUniform(&noise_d, lower.toDouble(), upper.toDouble());
    // output = self * noise where self < 0, else self
    auto mask = self.lt(0);
    auto scaled = at::mul(self, noise);
    output = at::where(mask, scaled, self);
  } else {
    double neg_slope = (lower.toDouble() + upper.toDouble()) / 2.0;
    auto neg_scaled = at::mul(self, at::scalar_tensor(neg_slope, self.options()));
    auto pos_mask = self.ge(at::scalar_tensor(0, self.options()));
    output = at::where(pos_mask, self, neg_scaled);
  }
  return output;
}

C10_EXPORT Tensor& rrelu_with_noise_cuda_(
    Tensor& self, Tensor& noise, const Scalar& lower,
    const Scalar& upper, bool training, std::optional<Generator> generator) {
  auto result = rrelu_with_noise_cuda(self, noise, lower, upper, training, generator);
  self.copy_(result);
  return self;
}

C10_EXPORT Tensor& rrelu_with_noise_out_cuda(
    const Tensor& self, Tensor& noise, const Scalar& lower,
    const Scalar& upper, bool training, std::optional<Generator> generator,
    Tensor& output) {
  auto result = rrelu_with_noise_cuda(self, noise, lower, upper, training, generator);
  output.resize_(result.sizes());
  output.copy_(result);
  return output;
}

// Philox RNG ops (thin wrappers — MLX manages its own RNG state)
C10_EXPORT Tensor& _philox_normal_cuda_(
    Tensor& self, const Tensor& /*philox_key*/, double mean, double std) {
  auto d = make_tensor_desc(self);
  haganeOpsNormal(&d, mean, std);
  return self;
}

C10_EXPORT Tensor& _philox_uniform_cuda_(
    Tensor& self, const Tensor& /*philox_key*/, double low, double high) {
  auto d = make_tensor_desc(self);
  haganeOpsUniform(&d, low, high);
  return self;
}

C10_EXPORT Tensor _philox_key_split_cuda(const Tensor& key, int64_t /*n*/) {
  // MLX manages RNG internally; return dummy key tensor
  return at::zeros_like(key);
}

C10_EXPORT Tensor _philox_key_fold_in_cuda(const Tensor& key, int64_t /*data*/) {
  return at::zeros_like(key);
}

// Dropout ops
C10_EXPORT std::tuple<Tensor, Tensor> native_dropout_cuda(
    const Tensor& input, double p, std::optional<bool> train) {
  bool is_train = train.value_or(true);
  if (!is_train || p == 0.0) {
    return {input.clone(), at::ones_like(input, input.options().dtype(kBool))};
  }
  if (p == 1.0) {
    return {at::zeros_like(input), at::zeros_like(input, input.options().dtype(kBool))};
  }
  auto mask = at::empty_like(input, input.options().dtype(kBool));
  auto md = make_tensor_desc(mask);
  haganeOpsBernoulliScalar(&md, 1.0 - p);
  double scale = 1.0 / (1.0 - p);
  auto mask_f = mask.to(input.dtype());
  auto scaled_mask = at::mul(mask_f, at::scalar_tensor(scale, input.options()));
  auto output = at::mul(input, scaled_mask);
  return {output, mask};
}

C10_EXPORT Tensor native_dropout_backward_cuda(
    const Tensor& grad_output, const Tensor& mask, double scale) {
  auto mask_f = mask.to(grad_output.dtype());
  auto scaled_mask = at::mul(mask_f, at::scalar_tensor(scale, grad_output.options()));
  return at::mul(grad_output, scaled_mask);
}

C10_EXPORT std::tuple<Tensor, Tensor> fused_dropout_cuda(
    const Tensor& self, double p, std::optional<Generator> /*gen*/) {
  return native_dropout_cuda(self, 1.0 - p, true);
}

C10_EXPORT void _fill_mem_eff_dropout_mask_(
    Tensor& mask, double dropout_p, int64_t /*seed*/, int64_t /*offset*/) {
  auto md = make_tensor_desc(mask);
  haganeOpsBernoulliScalar(&md, 1.0 - dropout_p);
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
  auto out = make_ops_tensor(iter, 0);
  if (haganeOpsFill(&out, value.toFloat()) != HAGANE_OPS_SUCCESS) {
    fill_stub(c10::DeviceType::CPU, iter, value);
  }
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
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  double pval = p.toDouble();
  // Find reduction dim from shapes
  const Tensor& in_t = iter.tensor(1);
  const Tensor& out_t = iter.tensor(0);
  int32_t dim = -1;
  for (int64_t d = 0; d < in_t.dim(); d++) {
    if (in_t.size(d) > 1 && (d >= out_t.dim() || out_t.size(d) == 1)) {
      dim = static_cast<int32_t>(d);
      break;
    }
  }
  if (dim < 0) dim = static_cast<int32_t>(in_t.dim() - 1);
  if (haganeOpsNormVal(&in, &out, pval, dim) != HAGANE_OPS_SUCCESS) {
    norm_stub(c10::DeviceType::CPU, iter, p);
  }
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

// ---------------------------------------------------------------------------
// Logit — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_logit_kernel(TensorIteratorBase& iter, const Scalar& eps) {
  auto out = make_ops_tensor(iter, 0);
  auto in = make_ops_tensor(iter, 1);
  if (haganeOpsLogit(&in, &out, eps.toFloat()) != HAGANE_OPS_SUCCESS) {
    logit_stub(c10::DeviceType::CPU, iter, eps);
  }
}

// ---------------------------------------------------------------------------
// Where — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_where_kernel(TensorIterator& iter) {
  auto out = make_ops_tensor(iter, 0);
  auto cond = make_ops_tensor(iter, 1);
  auto a = make_ops_tensor(iter, 2);
  auto b = make_ops_tensor(iter, 3);
  if (haganeOpsWhere(&cond, &a, &b, &out) != HAGANE_OPS_SUCCESS) {
    where_kernel(c10::DeviceType::CPU, iter);
  }
}

// ---------------------------------------------------------------------------
// Batch 2: Gather/Scatter — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_gather_kernel(const Tensor& result, const Tensor& self,
                          int64_t dim, const Tensor& index) {
  auto in_d = make_tensor_desc(self);
  auto idx_d = make_tensor_desc(index);
  auto out_d = make_tensor_desc(result);
  if (haganeOpsGather(&in_d, &idx_d, &out_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    gather_stub(c10::DeviceType::CPU, result, self, dim, index);
  }
}

void hagane_scatter_kernel(const Tensor& self, int64_t dim,
                           const Tensor& index, const Tensor& src) {
  auto self_d = make_tensor_desc(self);
  auto idx_d = make_tensor_desc(index);
  auto src_d = make_tensor_desc(src);
  if (haganeOpsScatter(&self_d, &idx_d, &src_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    scatter_stub(c10::DeviceType::CPU, self, dim, index, src);
  }
}

void hagane_scatter_fill_kernel(const Tensor& self, int64_t dim,
                                const Tensor& index, const Scalar& src) {
  auto self_d = make_tensor_desc(self);
  auto idx_d = make_tensor_desc(index);
  if (haganeOpsScatterFill(&self_d, &idx_d, src.toFloat(), static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    scatter_fill_stub(c10::DeviceType::CPU, self, dim, index, src);
  }
}

void hagane_scatter_add_kernel(const Tensor& self, int64_t dim,
                               const Tensor& index, const Tensor& src) {
  auto self_d = make_tensor_desc(self);
  auto idx_d = make_tensor_desc(index);
  auto src_d = make_tensor_desc(src);
  if (haganeOpsScatterAdd(&self_d, &idx_d, &src_d, static_cast<int32_t>(dim)) != HAGANE_OPS_SUCCESS) {
    scatter_add_stub(c10::DeviceType::CPU, self, dim, index, src);
  }
}

void hagane_scatter_reduce_kernel(const Tensor& self, int64_t dim,
                                  const Tensor& index, const Tensor& src,
                                  const ReductionType& reduce) {
  // Scatter with reduce — UMA allows CPU path on same memory
  scatter_reduce_stub(c10::DeviceType::CPU, self, dim, index, src, reduce);
}

void hagane_scatter_scalar_reduce_kernel(const Tensor& self, int64_t dim,
                                         const Tensor& index, const Scalar& value,
                                         const ReductionType& reduce) {
  scatter_scalar_reduce_stub(c10::DeviceType::CPU, self, dim, index, value, reduce);
}

void hagane_scatter_reduce_two_kernel(const Tensor& self, int64_t dim,
                                       const Tensor& index, const Tensor& src,
                                       const ReductionType& reduce) {
  scatter_reduce_two_stub(c10::DeviceType::CPU, self, dim, index, src, reduce);
}

// ---------------------------------------------------------------------------
// Batch 2: Index ops — CPU delegation (complex TensorIterator patterns)
// ---------------------------------------------------------------------------

void hagane_index_kernel(TensorIteratorBase& iter, IntArrayRef indexed_sizes,
                         IntArrayRef indexed_strides) {
  // Advanced indexing uses complex TensorIterator patterns
  // UMA allows CPU path to operate on the same memory directly
  index_stub(c10::DeviceType::CPU, iter, indexed_sizes, indexed_strides);
}

void hagane_index_fill_kernel(TensorIterator& iter, int64_t dim,
                              int64_t self_dim_size, int64_t self_dim_stride,
                              const Scalar& source) {
  index_fill_stub(c10::DeviceType::CPU, iter, dim, self_dim_size, self_dim_stride, source);
}

void hagane_index_copy_kernel(TensorIterator& iter, int64_t dim,
                              int64_t self_dim_size, int64_t self_dim_stride) {
  index_copy_stub(c10::DeviceType::CPU, iter, dim, self_dim_size, self_dim_stride);
}

void hagane_index_put_kernel(TensorIterator& iter, IntArrayRef indexed_sizes,
                             IntArrayRef indexed_strides, bool accumulate) {
  index_put_stub(c10::DeviceType::CPU, iter, indexed_sizes, indexed_strides, accumulate);
}

// ---------------------------------------------------------------------------
// Batch 2: Flip — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_flip_kernel(TensorIterator& iter, const bool quantized) {
  const auto& self = iter.tensor(1);
  const auto& result = iter.tensor(0);
  // Flip through TensorIterator — the iterator handles the flip logic,
  // we just need to do the copy. UMA makes CPU path work on same memory.
  flip_stub(c10::DeviceType::CPU, iter, quantized);
}

// ---------------------------------------------------------------------------
// Batch 2: Masked fill — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  // The primary path goes through masked_fill__cuda C10_EXPORT (MLX GPU).
  // This stub path handles edge cases from TensorIterator dispatch.
  const auto& self = iter.tensor(0);
  auto self_d = make_tensor_desc(self);
  // iter.tensor(1) is the mask
  const auto& mask = iter.tensor(1);
  auto mask_d = make_tensor_desc(mask);
  if (haganeOpsMaskedFill(&self_d, &mask_d, value.toFloat()) != HAGANE_OPS_SUCCESS) {
    masked_fill_stub(c10::DeviceType::CPU, iter, value);
  }
}

// ---------------------------------------------------------------------------
// Batch 2: Cat — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_cat_serial_kernel(const Tensor& result,
                              const MaterializedITensorListRef& tensors,
                              int64_t dim) {
  std::vector<haganeOpsTensor_t> descs;
  std::vector<const haganeOpsTensor_t*> desc_ptrs;
  descs.reserve(tensors.size());

  bool all_ok = true;
  for (const auto& t : tensors) {
    if (t.get().numel() == 0) continue;
    descs.push_back(make_tensor_desc(t));
    if (descs.back().dtype == HAGANE_DTYPE_FLOAT64 || !t.get().is_contiguous())
      all_ok = false;
  }
  for (auto& d : descs) desc_ptrs.push_back(&d);

  auto out_d = make_tensor_desc(result);
  if (all_ok && !desc_ptrs.empty() &&
      haganeOpsCat(desc_ptrs.data(), static_cast<int32_t>(desc_ptrs.size()),
                   &out_d, static_cast<int32_t>(dim)) == HAGANE_OPS_SUCCESS) {
    return;
  }
  cat_serial_stub(c10::DeviceType::CPU, result, tensors, dim);
}

// ---------------------------------------------------------------------------
// Batch 4: Random/Distribution ops — Metal GPU via MLX
// ---------------------------------------------------------------------------

void hagane_normal_kernel(const TensorBase& self, double mean, double std,
                          std::optional<Generator> gen) {
  auto out_d = make_tensor_desc(self);
  if (haganeOpsNormal(&out_d, mean, std) != HAGANE_OPS_SUCCESS) {
    // CPU fallback: generate on CPU, copy to "GPU" (UMA)
    auto cpu_t = at::empty(self.sizes(), self.options().device(c10::kCPU));
    cpu_t.normal_(mean, std);
    std::memcpy(const_cast<void*>(self.const_data_ptr()),
                cpu_t.const_data_ptr(), self.numel() * self.itemsize());
  }
}

void hagane_uniform_kernel(TensorIteratorBase& iter, double from, double to,
                           std::optional<Generator> gen) {
  auto out = make_ops_tensor(iter, 0);
  if (haganeOpsUniform(&out, from, to) != HAGANE_OPS_SUCCESS) {
    uniform_stub(c10::DeviceType::CPU, iter, from, to, gen);
  }
}

void hagane_bernoulli_tensor_kernel(const TensorBase& self, const TensorBase& p_,
                                    std::optional<Generator> gen) {
  auto out_d = make_tensor_desc(self);
  auto p_d = make_tensor_desc(p_);
  if (haganeOpsBernoulliTensor(&out_d, &p_d) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = at::empty(self.sizes(), self.options().device(c10::kCPU));
    cpu_self.bernoulli_(Tensor(p_).cpu());
    std::memcpy(const_cast<void*>(self.const_data_ptr()),
                cpu_self.const_data_ptr(), self.numel() * self.itemsize());
  }
}

void hagane_bernoulli_scalar_kernel(const TensorBase& self, double p,
                                    std::optional<Generator> gen) {
  auto out_d = make_tensor_desc(self);
  if (haganeOpsBernoulliScalar(&out_d, p) != HAGANE_OPS_SUCCESS) {
    auto cpu_self = at::empty(self.sizes(), self.options().device(c10::kCPU));
    cpu_self.bernoulli_(p);
    std::memcpy(const_cast<void*>(self.const_data_ptr()),
                cpu_self.const_data_ptr(), self.numel() * self.itemsize());
  }
}

void hagane_random_from_to_kernel(TensorIteratorBase& iter, uint64_t range,
                                  int64_t base, std::optional<Generator> gen) {
  auto out = make_ops_tensor(iter, 0);
  if (haganeOpsRandomFromTo(&out, base, base + static_cast<int64_t>(range)) != HAGANE_OPS_SUCCESS) {
    random_from_to_stub(c10::DeviceType::CPU, iter, range, base, gen);
  }
}

void hagane_random_full_kernel(TensorIteratorBase& iter, std::optional<Generator> gen) {
  auto out = make_ops_tensor(iter, 0);
  if (haganeOpsRandom(&out) != HAGANE_OPS_SUCCESS) {
    random_full_64_bits_range_stub(c10::DeviceType::CPU, iter, gen);
  }
}

void hagane_random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen) {
  auto out = make_ops_tensor(iter, 0);
  if (haganeOpsRandom(&out) != HAGANE_OPS_SUCCESS) {
    random_stub(c10::DeviceType::CPU, iter, gen);
  }
}

void hagane_log_normal_kernel(TensorIteratorBase& iter, double mean, double std,
                              std::optional<Generator> gen) {
  // log_normal = exp(normal(mean, std))
  auto out = make_ops_tensor(iter, 0);
  if (haganeOpsNormal(&out, mean, std) == HAGANE_OPS_SUCCESS) {
    // Apply exp in-place via MLX
    auto in = make_ops_tensor(iter, 0);
    haganeOpsExp(&in, &out);
  } else {
    log_normal_stub(c10::DeviceType::CPU, iter, mean, std, gen);
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

// Logit, Where (wiring existing hagane_ops C API)
REGISTER_DISPATCH(logit_stub, &hagane_logit_kernel)
REGISTER_DISPATCH(where_kernel, &hagane_where_kernel)

// Batch 2: Gather/Scatter
REGISTER_DISPATCH(gather_stub, &hagane_gather_kernel)
REGISTER_DISPATCH(scatter_stub, &hagane_scatter_kernel)
REGISTER_DISPATCH(scatter_fill_stub, &hagane_scatter_fill_kernel)
REGISTER_DISPATCH(scatter_add_stub, &hagane_scatter_add_kernel)
REGISTER_DISPATCH(scatter_reduce_stub, &hagane_scatter_reduce_kernel)
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &hagane_scatter_scalar_reduce_kernel)
REGISTER_DISPATCH(scatter_reduce_two_stub, &hagane_scatter_reduce_two_kernel)

// Batch 2: Index ops
REGISTER_DISPATCH(index_stub, &hagane_index_kernel)
REGISTER_DISPATCH(index_fill_stub, &hagane_index_fill_kernel)
REGISTER_DISPATCH(index_copy_stub, &hagane_index_copy_kernel)
REGISTER_DISPATCH(index_put_stub, &hagane_index_put_kernel)

// Batch 2: Flip, Masked fill
REGISTER_DISPATCH(flip_stub, &hagane_flip_kernel)
REGISTER_DISPATCH(masked_fill_stub, &hagane_masked_fill_kernel)

// Batch 2: Cat
REGISTER_DISPATCH(cat_serial_stub, &hagane_cat_serial_kernel)

// sort_stub registered by hip/Sort.cpp → calls our C10_EXPORT sortKeyValueInplace
// cumsum_stub, cumprod_stub registered by hip/ScanKernels.cpp → calls our C10_EXPORT launch_*_kernel

// Batch 4: Random/Distribution
REGISTER_DISPATCH(normal_stub, &hagane_normal_kernel)
REGISTER_DISPATCH(uniform_stub, &hagane_uniform_kernel)
REGISTER_DISPATCH(bernoulli_tensor_stub, &hagane_bernoulli_tensor_kernel)
REGISTER_DISPATCH(bernoulli_scalar_stub, &hagane_bernoulli_scalar_kernel)
REGISTER_DISPATCH(random_from_to_stub, &hagane_random_from_to_kernel)
REGISTER_DISPATCH(random_full_64_bits_range_stub, &hagane_random_full_kernel)
REGISTER_DISPATCH(random_stub, &hagane_random_kernel)
REGISTER_DISPATCH(log_normal_stub, &hagane_log_normal_kernel)

} // namespace at::native

#endif // __HIP_PLATFORM_HAGANE__
