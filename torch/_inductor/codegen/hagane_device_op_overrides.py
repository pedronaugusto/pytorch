"""Inductor device-op overrides for the Hagane platform fork.

The Hagane PyTorch fork builds with ``__HIP_PLATFORM_HAGANE__=1`` and
exposes the runtime as ``device='cuda'`` (HIP=CUDA convention). On the
Inductor side this means ``get_gpu_type()`` returns ``"cuda"`` and
``get_device_op_overrides("cuda")`` would otherwise resolve to
:class:`CUDADeviceOpOverrides`, which emits HIP/Triton kernel-driver
boilerplate Hagane cannot execute.

This module replaces the "cuda" registration with
:class:`HaganeDeviceOpOverrides` *only when the Hagane runtime DSO is
loadable* — gated via :mod:`hagane.inductor_backend._target_marker`.

The overrides intentionally emit **no HIP kernel-driver source** —
Hagane has no Triton/HIP kernel emission path; the eager dispatch model
is the entry point. Stream/device-guard hooks remain ``torch.cuda.*``
calls because PyTorch's user-facing CUDA module is the surface used by
both real CUDA and the Hagane fork.
"""
from __future__ import annotations

from .common import DeviceOpOverrides, register_device_op_overrides


class HaganeDeviceOpOverrides(DeviceOpOverrides):
    """Hagane override of the "cuda" device-op overrides.

    Inductor calls these from the generated wrapper to set up device
    context and import the raw stream getter. Hagane reuses the
    ``torch.cuda.*`` Python surface (HIP=CUDA convention) so the Python
    wrapper hooks are identical to the real CUDA path. The C++ wrapper
    hooks are intentionally absent — Sprint IX MVP only exercises the
    Python wrapper.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    def set_device(self, device_idx: int) -> str:
        return f"torch.cuda.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.cuda.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.cuda._DeviceGuard({device_idx})"

    def cpp_kernel_type(self) -> str:
        return "void*"


def maybe_register_hagane_device_op_overrides() -> bool:
    """Register Hagane overrides for "cuda" if the Hagane runtime is loadable.

    Returns True if registration succeeded, False otherwise. Vanilla
    PyTorch builds (no Hagane runtime DSO) leave the existing
    :class:`CUDADeviceOpOverrides` registration untouched.
    """
    try:
        import hagane.inductor_backend._target_marker  # noqa: F401
    except Exception:
        return False
    register_device_op_overrides("cuda", HaganeDeviceOpOverrides())
    return True
