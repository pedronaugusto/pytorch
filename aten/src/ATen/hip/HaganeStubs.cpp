// Hagane: stub implementations for functions from excluded .hip kernel files.
// These symbols are referenced by PyTorch but their implementations require
// hipcc (<<<>>> syntax). On Hagane, these ops dispatch through MLX instead.

#if defined(__HIP_PLATFORM_HAGANE__)

#include <ATen/jit_macros.h>
#if AT_USE_JITERATOR()
#include <ATen/hip/jiterator.h>
namespace at::cuda {
c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
    const std::string&, const std::string&, const int,
    const c10::SmallVector<at::Tensor>&, const c10::SmallVector<at::Scalar>&, bool) {
  TORCH_CHECK(false, "JIT kernel compilation not supported on Hagane/Metal");
}
} // namespace at::cuda
#endif

#endif // __HIP_PLATFORM_HAGANE__
