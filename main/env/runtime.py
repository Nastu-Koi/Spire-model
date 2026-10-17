"""Explicit resource controls and reproducibility manifest."""

import importlib.metadata
import platform

import torch


def configure_runtime(device, *, threads=4, gpu=True):
    """Thread count, matmul precision and the manifest of this process's runtime.
    With `gpu`, the GPU's budget is reserved as well (`configure_gpu`), which
    initializes CUDA in this process; a caller that must fork first (`iterate
    --overlap`) passes `gpu=False` and configures the GPU afterwards."""
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_build": torch.version.cuda,
    }
    if gpu:
        # Asking cuDNN its version initializes CUDA.
        manifest["cudnn"] = torch.backends.cudnn.version()
    for package in ("accelerate", "numpy", "triton"):
        try:
            manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if gpu and str(device).startswith("cuda"):
        manifest.update(configure_gpu(device))
    return manifest


def configure_gpu(device):
    """Reserve this process's share of the GPU (at most 80 GiB, leaving 8 GiB free)
    and describe it; initializes CUDA. Nothing to do on another device."""
    if not str(device).startswith("cuda"):
        return {}
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use --device cpu for small validation runs"
        )
    device = torch.device(device)
    device = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    free, total = torch.cuda.mem_get_info(device)
    budget = min(80 * 1024**3, free - 8 * 1024**3)
    if budget <= 0:
        raise RuntimeError("Insufficient free GPU memory after 8 GiB reserve")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    return dict(
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device),
        capability=torch.cuda.get_device_capability(device),
        gpu_budget_bytes=budget,
    )
