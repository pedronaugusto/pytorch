#!/usr/bin/env python3
"""Generate C++ stub implementations for undefined at::native::* symbols.

Reads undefined symbols from libtorch_hip.dylib and generates a .cpp file
where each function stub throws "not implemented on Hagane/Metal" and each
data symbol (DispatchStub instances) is allocated as zeroed storage.

Symbols already defined in libtorch_cpu.dylib are excluded — they will be
resolved at load time via dynamic linking.

Usage:
    python tools/hagane_gen_stubs.py build/lib/libtorch_hip.dylib \
        --cpu-lib build/lib/libtorch_cpu.dylib \
        -o build/aten/src/ATen/HaganeKernelStubs.cpp
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_defined_symbols(dylib_path: str) -> set[str]:
    """Return set of mangled symbol names defined (exported) in a dylib."""
    result = subprocess.run(
        ["nm", "-gU", dylib_path], capture_output=True, text=True, check=True
    )
    symbols = set()
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            symbols.add(parts[2])
    return symbols


def get_undefined_native_symbols(
    dylib_path: str, cpu_defined: set[str] | None = None
) -> list[tuple[str, str]]:
    """Return (mangled, demangled) pairs for undefined symbols from excluded kernel files.

    Catches at::native::*, at::hip::detail::*, and at::cuda::* — anything that
    lives in excluded .hip sources rather than in libtorch_cpu or hagane-sdk libs.
    """
    result = subprocess.run(
        ["nm", "-u", dylib_path], capture_output=True, text=True, check=True
    )
    mangled = []
    for line in result.stdout.splitlines():
        sym = line.strip()
        if not sym:
            continue
        mangled.append(sym)

    # Demangle all at once
    demangle = subprocess.run(
        ["c++filt"], input="\n".join(mangled), capture_output=True, text=True, check=True
    )
    demangled = demangle.stdout.splitlines()

    # Namespaces that contain symbols from excluded .hip kernel files.
    # Use 'in' not 'startswith' because template instantiations have return types
    # prepended (e.g., "void at::native::foo<float>(...)")
    STUB_PATTERNS = ("at::native::", "at::hip::detail::", "at::cuda::jit::")

    # Symbols with real implementations in HaganeOps.cpp — do not stub these.
    HAGANE_IMPLEMENTED = (
        # Batch 0: core
        "at::native::empty_cuda(",
        "at::native::empty_strided_cuda(",
        "at::native::structured_ufunc_add_CUDA::impl(",
        "at::native::GeluCUDAKernelImpl(",
        "at::native::GeluBackwardCUDAKernelImpl(",
        "at::native::max_all_launch_kernel(",
        "at::native::min_all_launch_kernel(",
        # Batch 2: sort/topk/scan
        "at::native::sortKeyValueInplace(",
        "at::native::launch_stable_sort_kernel(",
        "at::native::launch_gather_topk_kernel(",
        "at::native::launch_cumsum_cuda_kernel(",
        "at::native::launch_cumprod_cuda_kernel(",
        "at::native::launch_cummax_cuda_kernel(",
        "at::native::launch_cummin_cuda_kernel(",
        "at::native::launch_logcumsumexp_cuda_kernel(",
        # Batch 2: tensor creation
        "at::native::arange_cuda_out(",
        "at::native::linspace_cuda_out(",
        "at::native::eye_out_cuda(",
        # Batch 2: index/manipulation
        "at::native::index_select_cuda(",
        "at::native::masked_fill__cuda(",
        "at::native::roll_cuda(",
        "at::native::repeat_interleave_cuda(",
        "at::native::nonzero_out_cuda(",
        "at::native::nonzero_cuda(",
        # Batch 3: reduce-dim
        "at::native::max_launch_kernel(",
        "at::native::min_launch_kernel(",
        "at::native::aminmax_launch_kernel(",
        "at::native::aminmax_allreduce_launch_kernel(",
        "at::native::powsum_launch_kernel(",
        "at::native::norm_launch_kernel(",
        # Batch 3: mode/kthvalue/median
        "at::native::launch_fused_mode_kernel(",
        "at::native::launch_apply_mode_kernel(",
        "at::native::launch_kthvalue_kernel(",
        "at::native::launch_median_kernel(",
        # Batch 5: foreach ops (all patterns)
        "at::native::foreach_tensor_",
        "at::native::foreach_scalar_pow_list_kernel_cuda(",
        "at::native::_amp_foreach_non_finite_check_and_unscale_cuda_(",
        # Batch 5: random/distribution
        "at::native::randperm_out_cuda(",
        "at::native::_philox_normal_cuda_(",
        "at::native::_philox_uniform_cuda_(",
        "at::native::_philox_key_split_cuda(",
        "at::native::_philox_key_fold_in_cuda(",
        "at::native::rrelu_with_noise_cuda(",
        "at::native::rrelu_with_noise_cuda_(",
        "at::native::rrelu_with_noise_out_cuda(",
        # Batch 5: creation/manipulation
        "at::native::logspace_cuda_out(",
        "at::native::range_cuda_out(",
        "at::native::_chunk_cat_cuda(",
        "at::native::_chunk_cat_out_cuda(",
        "at::native::split_with_sizes_copy_out_cuda(",
        # Batch 5: dropout
        "at::native::native_dropout_cuda(",
        "at::native::native_dropout_backward_cuda(",
        "at::native::fused_dropout_cuda(",
        "at::native::_fill_mem_eff_dropout_mask_(",
        # Batch 6: softmax
        "at::native::structured_softmax_cuda_out::impl(",
        "at::native::structured_log_softmax_cuda_out::impl(",
        "at::native::structured_softmax_backward_cuda_out::impl(",
        "at::native::structured_log_softmax_backward_cuda_out::impl(",
        "at::native::masked_softmax_cuda(",
        "at::native::masked_softmax_backward_cuda(",
        "at::native::softmax_sparse_cuda(",
        "at::native::log_softmax_sparse_cuda(",
        "at::native::softmax_backward_sparse_cuda(",
        "at::native::log_softmax_backward_sparse_cuda(",
        # Batch 6: avg pool
        "at::native::structured_avg_pool2d_out_cuda::impl(",
        "at::native::structured_avg_pool2d_backward_out_cuda::impl(",
        "at::native::structured_avg_pool3d_out_cuda::impl(",
        "at::native::structured_avg_pool3d_backward_out_cuda::impl(",
        "at::native::adaptive_avg_pool2d_cuda(",
        "at::native::adaptive_avg_pool2d_out_cuda(",
        "at::native::adaptive_avg_pool2d_backward_cuda(",
        "at::native::adaptive_avg_pool3d_cuda(",
        "at::native::adaptive_avg_pool3d_out_cuda(",
        "at::native::adaptive_avg_pool3d_backward_cuda(",
        "at::native::adaptive_avg_pool3d_backward_out_cuda(",
        # Batch 6: max pool
        "at::native::structured_max_pool2d_with_indices_out_cuda::impl(",
        "at::native::structured_max_pool2d_with_indices_backward_out_cuda::impl(",
        "at::native::max_pool3d_with_indices_cuda(",
        "at::native::max_pool3d_with_indices_out_cuda(",
        "at::native::max_pool3d_with_indices_backward_cuda(",
        "at::native::max_pool3d_with_indices_backward_out_cuda(",
        # Batch 6: adaptive max pool
        "at::native::structured_adaptive_max_pool2d_out_cuda::impl(",
        "at::native::structured_adaptive_max_pool2d_backward_out_cuda::impl(",
        "at::native::structured_adaptive_max_pool3d_out_cuda::impl(",
        "at::native::structured_adaptive_max_pool3d_backward_out_cuda::impl(",
        # Batch 6: fractional max pool
        "at::native::structured_fractional_max_pool2d_out_cuda::impl(",
        "at::native::structured_fractional_max_pool2d_backward_cuda::impl(",
        "at::native::structured_fractional_max_pool3d_out_cuda::impl(",
        "at::native::fractional_max_pool3d_backward_cuda(",
        "at::native::fractional_max_pool3d_backward_out_cuda(",
        # Batch 6: max unpooling
        "at::native::max_unpooling2d_forward_cuda(",
        "at::native::max_unpooling2d_forward_out_cuda(",
        "at::native::max_unpooling3d_forward_cuda(",
        "at::native::max_unpooling3d_forward_out_cuda(",
        # Batch 6: upsample
        "at::native::structured_upsample_nearest1d_out_cuda::impl(",
        "at::native::structured_upsample_nearest2d_out_cuda::impl(",
        "at::native::structured_upsample_nearest3d_out_cuda::impl(",
        "at::native::structured__upsample_nearest_exact1d_out_cuda::impl(",
        "at::native::structured__upsample_nearest_exact2d_out_cuda::impl(",
        "at::native::structured__upsample_nearest_exact3d_out_cuda::impl(",
        "at::native::structured_upsample_nearest1d_backward_out_cuda::impl(",
        "at::native::structured_upsample_nearest2d_backward_out_cuda::impl(",
        "at::native::structured_upsample_nearest3d_backward_out_cuda::impl(",
        "at::native::structured__upsample_nearest_exact1d_backward_out_cuda::impl(",
        "at::native::structured__upsample_nearest_exact2d_backward_out_cuda::impl(",
        "at::native::structured__upsample_nearest_exact3d_backward_out_cuda::impl(",
        "at::native::structured_upsample_linear1d_out_cuda::impl(",
        "at::native::structured_upsample_linear1d_backward_out_cuda::impl(",
        "at::native::structured_upsample_bilinear2d_out_cuda::impl(",
        "at::native::structured_upsample_bilinear2d_backward_out_cuda::impl(",
        "at::native::structured__upsample_bilinear2d_aa_out_cuda::impl(",
        "at::native::structured__upsample_bilinear2d_aa_backward_out_cuda::impl(",
        "at::native::structured_upsample_bicubic2d_out_cuda::impl(",
        "at::native::structured_upsample_bicubic2d_backward_out_cuda::impl(",
        "at::native::structured__upsample_bicubic2d_aa_out_cuda::impl(",
        "at::native::structured__upsample_bicubic2d_aa_backward_out_cuda::impl(",
        "at::native::structured_upsample_trilinear3d_out_cuda::impl(",
        "at::native::structured_upsample_trilinear3d_backward_out_cuda::impl(",
        # Batch 6: convolution
        "at::native::conv_depthwise2d_cuda(",
        "at::native::conv_depthwise2d_cuda_out(",
        "at::native::conv_depthwise3d_cuda(",
        "at::native::slow_conv2d_forward_cuda(",
        "at::native::slow_conv2d_forward_out_cuda(",
        "at::native::slow_conv2d_backward_cuda(",
        "at::native::slow_conv2d_backward_out_cuda(",
        "at::native::slow_conv_dilated2d_cuda(",
        "at::native::slow_conv_dilated3d_cuda(",
        "at::native::slow_conv_transpose3d_cuda(",
        "at::native::slow_conv_transpose3d_out_cuda(",
        "at::native::structured_slow_conv_transpose2d_structured_cuda::impl(",
        # Batch 6: padding
        "at::native::structured_reflection_pad1d_out_cuda::impl(",
        "at::native::structured_reflection_pad1d_backward_out_cuda::impl(",
        "at::native::structured_reflection_pad3d_out_cuda::impl(",
        "at::native::structured_reflection_pad3d_backward_out_cuda::impl(",
        "at::native::reflection_pad2d_cuda(",
        "at::native::reflection_pad2d_out_cuda(",
        "at::native::reflection_pad2d_backward_cuda(",
        "at::native::reflection_pad2d_backward_out_cuda(",
        "at::native::structured_replication_pad1d_out_cuda::impl(",
        "at::native::structured_replication_pad1d_backward_out_cuda::impl(",
        "at::native::structured_replication_pad2d_out_cuda::impl(",
        "at::native::structured_replication_pad3d_out_cuda::impl(",
        "at::native::replication_pad2d_backward_cuda(",
        "at::native::replication_pad2d_backward_out_cuda(",
        "at::native::replication_pad3d_backward_cuda(",
        "at::native::replication_pad3d_backward_out_cuda(",
        # Batch 6: tril/triu
        "at::native::structured_tril_cuda::impl(",
        "at::native::structured_triu_cuda::impl(",
        "at::native::tril_indices_cuda(",
        "at::native::triu_indices_cuda(",
        # Batch 6: index
        "at::native::structured_index_add_cuda_out::impl(",
        "at::native::structured_index_reduce_cuda_out::impl(",
        # Batch 6: im2col/col2im
        "at::native::im2col_cuda(",
        "at::native::im2col_out_cuda(",
        "at::native::col2im_cuda(",
        "at::native::col2im_out_cuda(",
        # Batch 6: padding kernel launchers
        "at::native::add_padding_kernelLauncher",
        "at::native::remove_padding_kernelLauncher",
        "at::native::remove_padding_transform0213_kernelLauncher",
        # Batch 6: sparse index conversions
        "at::native::structured__convert_indices_from_coo_to_csr_structured_cuda::impl(",
        "at::native::structured__convert_indices_from_csr_to_coo_structured_cuda::impl(",
    )

    if cpu_defined is None:
        cpu_defined = set()

    pairs = []
    skipped_cpu = 0
    for m, d in zip(mangled, demangled):
        if any(pat in d for pat in STUB_PATTERNS):
            if any(impl in d for impl in HAGANE_IMPLEMENTED):
                continue
            # Skip symbols defined in libtorch_cpu — resolved at load time
            if m in cpu_defined:
                skipped_cpu += 1
                continue
            pairs.append((m, d))

    if skipped_cpu:
        print(f"  Skipped {skipped_cpu} symbols (defined in libtorch_cpu)")
    return pairs


def classify_symbols(
    pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split into (functions, data) based on demangled signature."""
    functions = []
    data = []
    for mangled, demangled in pairs:
        # Functions have a closing paren (possibly followed by qualifiers like const)
        # Data symbols don't have parentheses at all, OR have them only inside template args
        # Heuristic: if ')' appears AND is after the last '>' (or there's no '>'), it's a function
        last_paren = demangled.rfind(")")
        last_angle = demangled.rfind(">")
        if last_paren > last_angle:
            functions.append((mangled, demangled))
        else:
            data.append((mangled, demangled))
    return functions, data


def extract_function_name(demangled: str) -> str:
    """Extract a short readable name from demangled symbol for error messages."""
    # e.g. "at::native::foo_cuda(at::Tensor const&)" -> "foo_cuda"
    paren = demangled.find("(")
    if paren == -1:
        name = demangled
    else:
        name = demangled[:paren]
    # Take last component after ::
    parts = name.split("::")
    return parts[-1] if parts else name


def generate_cpp(
    functions: list[tuple[str, str]],
    data: list[tuple[str, str]],
) -> str:
    lines = []
    lines.append("// Auto-generated by tools/hagane_gen_stubs.py")
    lines.append("// Stub implementations for at::native::* symbols from excluded .hip kernel files.")
    lines.append("// Each function throws 'not implemented on Hagane/Metal' at runtime.")
    lines.append("")
    lines.append('#if defined(__HIP_PLATFORM_HAGANE__)')
    lines.append("")
    lines.append("#include <stdexcept>")
    lines.append("#include <string>")
    lines.append("")
    lines.append("// Use a plain throw to avoid pulling in TORCH_CHECK dependencies")
    lines.append("// that might create circular includes.")
    lines.append('#define HAGANE_STUB(name) \\')
    lines.append('    throw std::runtime_error( \\')
    lines.append('        std::string(name) + " is not implemented on Hagane/Metal")')
    lines.append("")

    # Function stubs: use __asm__ label to force the mangled name
    lines.append("// ---- Function stubs ({} symbols) ----".format(len(functions)))
    lines.append("")
    for i, (mangled, demangled) in enumerate(functions):
        fname = extract_function_name(demangled)
        # __asm__ label forces the compiler to emit this symbol with the given mangled name.
        # We declare as void since TORCH_CHECK throws before returning — return type
        # is not part of C++ name mangling for non-template functions.
        lines.append(
            f'__attribute__((visibility("default"))) extern "C" void _hagane_stub_{i}() __asm__("{mangled}");'
        )
        lines.append(f"void _hagane_stub_{i}() {{ HAGANE_STUB(\"{fname}\"); }}")
        lines.append("")

    # Data stubs: allocate zeroed storage with the right symbol name
    lines.append("// ---- Data stubs ({} symbols) ----".format(len(data)))
    lines.append("// DispatchStub instances and static members — zeroed storage.")
    lines.append("")
    for i, (mangled, demangled) in enumerate(data):
        # 256 bytes is generous for DispatchStub (typically ~64 bytes on ARM64)
        lines.append(
            f'__attribute__((visibility("default"))) extern "C" char _hagane_data_{i}[256] __asm__("{mangled}");'
        )
        lines.append(f"char _hagane_data_{i}[256] = {{}};")
        lines.append("")

    lines.append("#endif // __HIP_PLATFORM_HAGANE__")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dylib", help="Path to libtorch_hip.dylib")
    parser.add_argument("--cpu-lib", help="Path to libtorch_cpu.dylib (exclude its symbols)")
    parser.add_argument("-o", "--output", required=True, help="Output .cpp path")
    args = parser.parse_args()

    cpu_defined = None
    if args.cpu_lib:
        print(f"Reading defined symbols from {args.cpu_lib}...")
        cpu_defined = get_defined_symbols(args.cpu_lib)
        print(f"  {len(cpu_defined)} symbols in libtorch_cpu")

    pairs = get_undefined_native_symbols(args.dylib, cpu_defined)
    print(f"Found {len(pairs)} symbols needing stubs")

    functions, data = classify_symbols(pairs)
    print(f"  Functions: {len(functions)}")
    print(f"  Data:      {len(data)}")

    cpp = generate_cpp(functions, data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cpp)
    print(f"Generated {out} ({len(functions) + len(data)} stubs)")


if __name__ == "__main__":
    main()
