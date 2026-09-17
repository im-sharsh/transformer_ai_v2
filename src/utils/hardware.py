"""Hardware detection and honest recommendations (no GPU is required for the basic workflow)."""
from __future__ import annotations

import os
import platform


def detect_hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "ram_gb": None, "torch": None, "cuda": False, "gpu_name": None, "gpu_memory_gb": None, "device": "cpu",
            "note": ""}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({"cuda": True, "gpu_name": p.name, "gpu_memory_gb": round(p.total_memory / 1024**3, 1), "device": "cuda"})
    except ImportError:
        info["note"] = "PyTorch is not installed, so CUDA cannot be detected; running on CPU"
    return info


def recommendations(hw: dict) -> list[tuple[str, str]]:
    """(level, message) pairs, level in {'ok', 'warn'}."""
    recs = [("ok", "The built-in small transformer (Phase 5) is designed to run on CPU with demo-sized subsets")]
    if hw["cuda"]:
        recs.append(("ok", f"CUDA GPU detected: {hw['gpu_name']} ({hw['gpu_memory_gb']} GB). Larger subsets and model "
                           "configurations will be possible"))
    else:
        recs.append(("warn", "No CUDA GPU: use a demo-sized subset (for example 10,000–50,000 rows). Fine-tuning large "
                             "language models on CPU is not practical"))
    if hw["ram_gb"] and hw["ram_gb"] < 8:
        recs.append(("warn", f"{hw['ram_gb']} GB RAM: a dataset of ~1 GB in memory may not fit; load a smaller file"))
    return recs
