"""Hardware and software fingerprint captured at run time.

The methods section has to state what the experiments ran on and how long they took.
That is easy to record while the job is running and near-impossible to reconstruct
honestly months later, so every run writes it into the results JSON.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import torch


def _sysctl(key: str) -> str | None:
    """Read a macOS sysctl value; None anywhere else or on failure."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _total_ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def accelerator_name(device: torch.device) -> str:
    """Human-readable name of the chip actually doing the matmuls."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        # Apple silicon: the CPU brand string names the SoC (e.g. "Apple M2 Pro").
        return _sysctl("machdep.cpu.brand_string") or "Apple silicon GPU (MPS)"
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown CPU"


def hardware_info(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "accelerator": accelerator_name(device),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
        info["cuda"] = torch.version.cuda
    return info


def software_info() -> dict:
    import datasets
    import numpy
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": numpy.__version__,
    }


def describe(hardware: dict, total_compute_s: float) -> str:
    """One-line hardware + compute summary, ready to paste into the methods section."""
    minutes = total_compute_s / 60
    return (
        f"All baseline inference ran on {hardware['accelerator']} "
        f"({hardware['device']}), {hardware['total_ram_gb']} GB RAM, "
        f"for a total of {minutes:.1f} min of compute."
    )
