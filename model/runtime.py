"""Explicit resource controls and reproducibility manifest."""
import importlib.metadata
import platform

import torch


def configure_runtime(device, *, threads=4):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
                "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    for package in ("accelerate", "numpy", "triton"):
        try:
            manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for small validation runs")
        device = torch.device(device)
        device = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        budget = min(80 * 1024**3, free - 8 * 1024**3)
        if budget <= 0:
            raise RuntimeError("Insufficient free GPU memory after 8 GiB reserve")
        torch.cuda.set_per_process_memory_fraction(budget / total, device)
        manifest.update(gpu=torch.cuda.get_device_name(device), capability=torch.cuda.get_device_capability(device),
                        gpu_budget_bytes=budget)
    return manifest
