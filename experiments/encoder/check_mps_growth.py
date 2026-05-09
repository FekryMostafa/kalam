"""Diagnose whether the MPS memory growth is a leak in our code or MPS
allocator behaviour.

Three probes, all batch=8, 50 steps:

   1. FIXED SHAPE        same T every step. eliminates shape-variability
                          fragmentation. if driver memory still grows here,
                          it's NOT shape-driven.
   2. VARYING SHAPE       cycle through 5 different T values. matches the
                          real-data scenario.
   3. VARYING + CLEAR     same as (2) but call empty_cache() every step.
                          if this stays flat, the growth is allocator
                          caching that empty_cache can release.

Logs CURRENT (live tensors) and DRIVER (OS-reserved) memory each step.

   current memory grows           → real leak
   current flat, driver grows     → MPS caching/fragmentation
   both flat                      → no growth at all
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn.functional as F

from app.encoder.device import empty_cache, pick_device, setup_env
from app.encoder.model import ConformerCTC
from app.encoder.vocab import BLANK_IDX


def _mem_gib(device):
    if device != 'mps' or not torch.backends.mps.is_available():
        return 0.0, 0.0
    cur = torch.mps.current_allocated_memory() / (1024 ** 3)
    drv = torch.mps.driver_allocated_memory() / (1024 ** 3)
    return cur, drv


def _step(model, opt, T, B, device):
    emg = torch.randn(B, T, 8, device=device)
    emg_lens = torch.full((B,), T, dtype=torch.long, device=device)
    targets = torch.randint(1, 32, (B * 30,), dtype=torch.long, device=device)
    target_lens = torch.full((B,), 30, dtype=torch.long, device=device)

    opt.zero_grad()
    log_probs, _, out_lens = model(emg, emg_lens)
    log_probs_T = log_probs.transpose(0, 1).contiguous()
    loss = F.ctc_loss(
        log_probs_T, targets, out_lens, target_lens,
        blank=BLANK_IDX, zero_infinity=True,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return float(loss.detach())


def probe(name, T_func, device, n_steps=50, clear_each_step=False, B=8):
    print('\n' + '=' * 70)
    print(f'PROBE: {name}')
    print('=' * 70)

    empty_cache(device)
    torch.manual_seed(0)
    model = ConformerCTC().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # baseline after model + optimizer materialised
    cur0, drv0 = _mem_gib(device)
    print(f'  baseline:                cur={cur0:.2f} GiB  drv={drv0:.2f} GiB')

    sample = (0, 5, 10, 20, 30, 40, 49)
    print(f'\n  {"step":>4}  {"T":>6}  {"current":>10}  {"driver":>10}  '
          f'{"d_cur":>7}  {"d_drv":>7}')
    base_cur, base_drv = None, None
    for step in range(n_steps):
        T = T_func(step)
        _step(model, opt, T, B, device)
        if clear_each_step:
            empty_cache(device)
        if step in sample:
            cur, drv = _mem_gib(device)
            if base_cur is None:
                base_cur, base_drv = cur, drv
            print(f'  {step:>4}  {T:>6}  {cur:>7.2f}GiB  {drv:>7.2f}GiB  '
                  f'{cur - base_cur:>+6.2f}  {drv - base_drv:>+6.2f}')

    cur, drv = _mem_gib(device)
    print(f'  total growth from step 0:  current Δ={cur - base_cur:+.2f} GiB,  '
          f'driver Δ={drv - base_drv:+.2f} GiB')

    del model, opt
    empty_cache(device)
    cur, drv = _mem_gib(device)
    print(f'  after teardown:          cur={cur:.2f} GiB  drv={drv:.2f} GiB')


def main():
    device = pick_device()
    setup_env(device)
    print(f'device={device}')

    if device != 'mps':
        print('this diagnostic is MPS-specific; nothing to learn elsewhere.')
        return

    # Fixed-shape probe: T=10000 every step
    probe('1. FIXED SHAPE  T=10000 every step',
          T_func=lambda step: 10000,
          device=device, n_steps=50, clear_each_step=False)

    # Varying-shape probe: cycle through 5 T values
    Ts = [3000, 12000, 7000, 18000, 5000]
    probe('2. VARYING SHAPE  T cycles in {3k, 12k, 7k, 18k, 5k}',
          T_func=lambda step: Ts[step % 5],
          device=device, n_steps=50, clear_each_step=False)

    # Varying-shape + empty_cache every step
    probe('3. VARYING + empty_cache() every step',
          T_func=lambda step: Ts[step % 5],
          device=device, n_steps=50, clear_each_step=True)


if __name__ == '__main__':
    main()
