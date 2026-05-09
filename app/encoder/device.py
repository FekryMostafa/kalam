"""Device abstraction: pick + configure cuda / mps / cpu uniformly.

Keep all device-conditional logic here so train/eval/runner code stays
device-agnostic. CUDA is preferred when present, then MPS, then CPU.
"""

from __future__ import annotations

import os

import psutil
import torch

_PROCESS = psutil.Process(os.getpid())


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
    """Peak memory used since the last reset, in GiB."""
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    if device == 'mps' and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / (1024 ** 3)
    # CPU: process RSS, no high-water-mark API → return current usage.
    return _PROCESS.memory_info().rss / (1024 ** 3)


def reset_peak_memory(device: str) -> None:
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def total_memory_gib(device: str) -> float:
    """Total memory available for our use, in GiB. Queried from the runtime."""
    if device == 'cuda':
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if device == 'mps' and torch.backends.mps.is_available():
        # Apple's recommended max working set size — what Metal will let
        # PyTorch allocate before evicting. Varies with system RAM and load.
        return torch.mps.recommended_max_memory() / (1024 ** 3)
    return psutil.virtual_memory().total / (1024 ** 3)


def probe_token_budget(
    device: str,
    target_fraction: float = 0.7,
    probe_b: int = 2,
    probe_t: int = 30_000,
    bf16: bool = False,
) -> int:
    """Probe the device with a synthetic worst-case batch and return a safe
    token budget for `TokenBudgetPairedSampler`.

    Runs forward+backward on a (probe_b, probe_t) batch, measures peak
    memory, and extrapolates linearly to `target_fraction * total_memory`.

    Probe shape (B=2, T=30000) is biased toward long T because attention is
    O(T²) on MPS's math kernel — extrapolating from a longer-T probe
    captures more of the quadratic cost, so the resulting budget is safer
    on MPS. On cuda+flash, attention is O(T) so the choice of probe shape
    barely matters.

    Returns: token budget in EMG samples @ 1 kHz (i.e. B * T_max).
    """
    setup_env(device)

    # Avoid circular import at module load.
    import torch.nn.functional as F

    from .model import ConformerCTC

    model = ConformerCTC().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    empty_cache(device)
    reset_peak_memory(device)

    emg = torch.randn(probe_b, probe_t, 8, device=device)
    lengths = torch.full((probe_b,), probe_t, dtype=torch.long, device=device)

    use_amp = bf16 and device == 'cuda'
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
        log_probs, _, out_lens = model(emg, lengths)
    if use_amp:
        log_probs = log_probs.float()

    targets = torch.randint(1, 32, (probe_b * 30,), dtype=torch.long, device=device)
    target_lens = torch.full((probe_b,), 30, dtype=torch.long, device=device)
    log_probs_t = log_probs.transpose(0, 1).contiguous()
    loss = F.ctc_loss(
        log_probs_t, targets, out_lens, target_lens, blank=0, zero_infinity=True,
    )
    loss.backward()
    opt.step()
    synchronize(device)

    peak_gib = peak_memory_gib(device)
    target_gib = total_memory_gib(device) * target_fraction

    if peak_gib <= 0:
        budget = probe_b * probe_t  # measurement failed; return what we tested
    else:
        budget = int(probe_b * probe_t * target_gib / peak_gib)

    del model, opt, emg, log_probs, loss
    empty_cache(device)
    return budget


__all__ = [
    'empty_cache',
    'peak_memory_gib',
    'pick_device',
    'probe_token_budget',
    'reset_peak_memory',
    'setup_env',
    'synchronize',
    'total_memory_gib',
]
