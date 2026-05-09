"""Device abstraction: pick + configure cuda / mps / cpu uniformly.

Keep all device-conditional logic here so train/eval/runner code stays
device-agnostic. CUDA is preferred when present, then MPS, then CPU.
"""

from __future__ import annotations

import os

import torch


def pick_device(prefer: str | None = None) -> str:
    """Return the best available device, or honor an explicit preference."""
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def setup_env(device: str) -> None:
    """Set env vars / backend flags required for the chosen device.

    Call before model + tensor allocation. Must precede heavy torch use
    on MPS because the CTC fallback flag is read at op-dispatch time.
    """
    if device == 'mps':
        # torchaudio's CTC has no MPS kernel — fall back to CPU.
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def empty_cache(device: str) -> None:
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def synchronize(device: str) -> None:
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()


def peak_memory_gib(device: str) -> float:
    """Driver-allocated memory in GiB (for OOM diagnostics)."""
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    if device == 'mps' and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / (1024 ** 3)
    return 0.0


def reset_peak_memory(device: str) -> None:
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()


__all__ = [
    'pick_device', 'setup_env', 'empty_cache', 'synchronize',
    'peak_memory_gib', 'reset_peak_memory',
]
