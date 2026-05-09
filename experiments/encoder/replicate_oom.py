"""Replicate the original MPS OOM and identify the root cause.

Three controlled probes:

   1. STATIC CAPACITY      synthetic batches at growing T (no fragmentation,
                           no varying shapes). Tells us the largest single
                           batch the model can fit on MPS.

   2. WORST-CASE REAL      sample real Gaddy data, build a batch from the
                           longest utterances. Confirms whether real long
                           utterances OOM in isolation.

   3. SUSTAINED RUN        ~300 steps of realistic training. Track peak
                           memory per step. Monotonic growth = allocator
                           fragmentation. Spikes correlated with batch
                           T_max = padding-driven.

Hypothesis: the OOM was driven by (a) worst-case padding within batches
(one 31s utterance pads everything to 31s) and (b) MPS allocator
fragmentation accumulating across variable-length batches — NOT the
attention implementation itself.
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.encoder.data import GaddyEMGDataset, make_loader  # noqa: E402
from app.encoder.device import (  # noqa: E402
    pick_device, setup_env, empty_cache, peak_memory_gib, reset_peak_memory,
)
from app.encoder.model import ConformerCTC  # noqa: E402
from app.encoder.vocab import BLANK_IDX  # noqa: E402


def _build_model(device):
    torch.manual_seed(0)
    return ConformerCTC().to(device)


def _ctc_step(model, opt, emg, emg_lens, targets, target_lens):
    opt.zero_grad()
    log_probs, features, out_lens = model(emg, emg_lens)
    log_probs_T = log_probs.transpose(0, 1).contiguous()
    loss = F.ctc_loss(
        log_probs_T, targets, out_lens, target_lens,
        blank=BLANK_IDX, zero_infinity=True,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return float(loss.detach())


def static_capacity(device, batch_size: int = 8):
    """Probe single-batch capacity at synthetic lengths."""
    print('\n' + '=' * 70)
    print(f'STATIC CAPACITY  batch={batch_size}')
    print('=' * 70)
    print(f'{"T":>7}  {"sec(real)":>10}  {"after_fwd":>10}  {"after_bwd":>10}  status')

    for T_emg in [3_000, 5_000, 10_000, 20_000, 31_000]:
        empty_cache(device)
        reset_peak_memory(device)
        try:
            model = _build_model(device)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

            emg = torch.randn(batch_size, T_emg, 8, device=device)
            emg_lens = torch.full((batch_size,), T_emg, dtype=torch.long, device=device)
            targets = torch.randint(1, 32, (batch_size * 30,), dtype=torch.long, device=device)
            target_lens = torch.full((batch_size,), 30, dtype=torch.long, device=device)

            opt.zero_grad()
            log_probs, features, out_lens = model(emg, emg_lens)
            mem_fwd = peak_memory_gib(device)

            log_probs_T = log_probs.transpose(0, 1).contiguous()
            loss = F.ctc_loss(
                log_probs_T, targets, out_lens, target_lens,
                blank=BLANK_IDX, zero_infinity=True,
            )
            loss.backward()
            mem_bwd = peak_memory_gib(device)

            opt.step()
            print(f'{T_emg:>7}  {T_emg/1000:>10.1f}  {mem_fwd:>8.2f}GiB  {mem_bwd:>8.2f}GiB  OK')
            del model, opt, emg, log_probs, features, loss
            empty_cache(device)
        except RuntimeError as e:
            msg = str(e).split('\n')[0]
            print(f'{T_emg:>7}  {T_emg/1000:>10.1f}                            OOM  {msg[:50]}')
            break


def worst_case_real(device, batch_size: int = 8, top_n: int = 8):
    """Build a batch from the longest real Gaddy utterances."""
    print('\n' + '=' * 70)
    print(f'WORST-CASE REAL  batch={batch_size} (longest {top_n} utterances)')
    print('=' * 70)

    print('Scanning Gaddy lengths...')
    ds = GaddyEMGDataset(
        modes=['voiced', 'silent'], with_frame_labels=False, max_emg_length=None,
    )
    lengths = []
    for idx, (sess, ex_idx) in enumerate(ds.examples):
        emg_path = os.path.join(sess.directory, f'{ex_idx}_emg.npy')
        try:
            arr = np.load(emg_path, mmap_mode='r')
            lengths.append((arr.shape[0], idx))
        except (OSError, ValueError):
            continue
    lengths.sort(reverse=True)
    longest = lengths[:top_n]
    print(f'top {top_n} lengths: {[L for L, _ in longest]}')
    print(f'global max: {longest[0][0]} ({longest[0][0]/1000:.1f}s)')

    empty_cache(device)
    reset_peak_memory(device)
    try:
        model = _build_model(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

        items = [ds[idx] for _, idx in longest[:batch_size]]
        # custom collate (skip _loc_to_examples / pair / frame stuff)
        Tmax = max(it['emg'].shape[0] for it in items)
        Lmax = max(it['phon_target'].shape[0] for it in items)
        emg = torch.zeros(batch_size, Tmax, 8, dtype=torch.float32, device=device)
        emg_lens = torch.tensor([it['emg'].shape[0] for it in items], dtype=torch.long, device=device)
        targets_flat = torch.cat([it['phon_target'] for it in items]).to(device)
        target_lens = torch.tensor([it['phon_target'].shape[0] for it in items], dtype=torch.long, device=device)
        for i, it in enumerate(items):
            emg[i, :it['emg'].shape[0]] = it['emg']

        loss = _ctc_step(model, opt, emg, emg_lens, targets_flat, target_lens)
        peak = peak_memory_gib(device)
        print(f'  Tmax={Tmax} ({Tmax/1000:.1f}s)  loss={loss:.3f}  peak={peak:.2f}GiB  OK')
        del model, opt, emg
        empty_cache(device)
    except RuntimeError as e:
        msg = str(e).split('\n')[0]
        print(f'  OOM  {msg[:80]}')


def sustained_run(device, batch_size: int = 8, n_steps: int = 200,
                  max_emg_length: int | None = None, sort_by_length: bool = False,
                  empty_cache_every: int | None = None):
    """Sustained training to detect fragmentation."""
    label = (
        f'cap={max_emg_length}'
        f'  sorted={sort_by_length}'
        f'  empty_cache_every={empty_cache_every}'
    )
    print('\n' + '=' * 70)
    print(f'SUSTAINED  batch={batch_size}  n_steps={n_steps}  {label}')
    print('=' * 70)

    ds = GaddyEMGDataset(
        modes=['voiced', 'silent'], with_frame_labels=False,
        max_emg_length=max_emg_length, mask_channels=[3],
    )
    print(f'  {len(ds)} examples  (max_emg_length={max_emg_length})')
    loader = make_loader(
        ds, batch_size=batch_size, balanced=False, paired=False,
        shuffle=True, num_workers=0, sampler_seed=0,
    )

    empty_cache(device)
    reset_peak_memory(device)
    model = _build_model(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f'{"step":>5}  {"T_max":>6}  {"pad%":>5}  {"loss":>6}  '
          f'{"peak(running)":>14}  {"sec":>5}')

    sample_idx = [0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    crashed_at = None
    for step, b in enumerate(loader):
        if not b:
            continue
        if step >= n_steps:
            break
        emg = b['emg'].to(device)
        emg_lens = b['emg_lengths'].to(device)
        if sort_by_length:
            order = torch.argsort(emg_lens, descending=True)
            emg = emg[order]
            emg_lens = emg_lens[order]
            phon = b['phon_targets']  # flat — sort by reordering target lengths
            tgt_lens = b['target_lengths']
            # reorder flat targets per sorted order
            offsets = torch.cumsum(torch.cat([torch.tensor([0]), tgt_lens]), dim=0)
            new_phon = []
            for i in order.tolist():
                new_phon.append(phon[offsets[i]:offsets[i + 1]])
            phon = torch.cat(new_phon).to(device)
            tgt_lens = tgt_lens[order].to(device)
        else:
            phon = b['phon_targets'].to(device)
            tgt_lens = b['target_lengths'].to(device)

        T_max = emg.shape[1]
        true_total = int(emg_lens.sum().item())
        pad_pct = 100 * (1 - true_total / (emg.shape[0] * T_max))

        try:
            t0 = time.time()
            loss = _ctc_step(model, opt, emg, emg_lens, phon, tgt_lens)
            dt = time.time() - t0
            peak = peak_memory_gib(device)
        except RuntimeError as e:
            msg = str(e).split('\n')[0]
            print(f'{step:>5}  {T_max:>6}                                       OOM {msg[:40]}')
            crashed_at = step
            break

        if step in sample_idx or step == n_steps - 1:
            print(f'{step:>5}  {T_max:>6}  {pad_pct:>4.0f}%  {loss:>6.3f}  '
                  f'{peak:>11.2f}GiB  {dt:>4.1f}')

        if empty_cache_every and step > 0 and step % empty_cache_every == 0:
            empty_cache(device)

    if crashed_at is None:
        print(f'completed {min(n_steps, step + 1)} steps  final_peak={peak_memory_gib(device):.2f}GiB')
    else:
        print(f'CRASHED at step {crashed_at}')


def main():
    device = pick_device()
    setup_env(device)
    print(f'device={device}')

    static_capacity(device, batch_size=8)
    worst_case_real(device, batch_size=8, top_n=8)

    print('\n\n>>> Comparison run A: random ordering, no length cap, no empty_cache')
    sustained_run(device, batch_size=8, n_steps=200,
                  max_emg_length=None, sort_by_length=False, empty_cache_every=None)

    print('\n\n>>> Comparison run B: random ordering, 20s cap, periodic empty_cache')
    sustained_run(device, batch_size=8, n_steps=200,
                  max_emg_length=20000, sort_by_length=False, empty_cache_every=25)

    print('\n\n>>> Comparison run C: length-sorted within batch, no cap')
    sustained_run(device, batch_size=8, n_steps=200,
                  max_emg_length=None, sort_by_length=True, empty_cache_every=None)


if __name__ == '__main__':
    main()
