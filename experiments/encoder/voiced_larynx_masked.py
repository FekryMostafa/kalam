"""Encoder run: voiced + silent joint multi-loss training, larynx (ch3) masked.

Builds a ConformerCTC and a GaddyEMGDataset (both modes, with frame labels
on voiced via forced alignment). Trains with CTC + frame-CE + contrastive.

Run:
    python experiments/encoder/voiced_larynx_masked.py --dry-run
    python experiments/encoder/voiced_larynx_masked.py
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# CTC has no MPS kernel — fall back to CPU. Must be set before torch import.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from app.encoder.data import GaddyEMGDataset, make_loader  # noqa: E402
from app.encoder.device import (  # noqa: E402
    pick_device, setup_env, probe_token_budget, total_memory_gib,
)
from app.encoder.model import ConformerCTC, count_params  # noqa: E402
from app.encoder.train import TrainConfig, train_ctc  # noqa: E402


LARYNX_CH = 3

# Token-budget sampler defaults. min keeps InfoNCE supplied with negatives;
# max keeps short-utterance batches from running away.
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 64

LENGTH_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'cache', 'encoder', 'lengths.json',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dry-run-steps', type=int, default=3)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader parallel workers')
    parser.add_argument('--ckpt-every-n-steps', type=int, default=500,
                        help='save latest checkpoint every N steps (OOM safety)')
    parser.add_argument('--no-frame-labels', action='store_true',
                        help='skip forced alignment (CTC + contrastive only)')
    parser.add_argument('--voiced-only', action='store_true',
                        help='train on voiced data only (no silent, no contrastive). '
                             'used to validate the architecture converges on the supervised case '
                             'before adding silent + contrastive.')
    parser.add_argument('--token-budget', type=int, default=None,
                        help='override auto-probe (padded EMG samples per batch)')
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    setup_env(device)
    bf16 = (device == 'cuda')  # autocast on cuda; MPS/CPU stay fp32

    # Don't truncate the dataset in dry-run mode — truncation by index
    # is mode-skewed (silent sessions come first in discovery), which
    # starves the balanced sampler. Forced alignment runs lazily, so only
    # utterances actually sampled pay the cost.
    max_examples = args.max_examples

    modes = ['voiced'] if args.voiced_only else ['voiced', 'silent']
    balanced = not args.voiced_only  # nothing to balance against in voiced-only

    if args.voiced_only:
        default_log = 'voiced_only_dryrun.json' if args.dry_run else 'voiced_only.json'
        default_ckpt = 'encoder_voiced_only.pt'
    else:
        default_log = 'voiced_larynx_masked_dryrun.json' if args.dry_run else 'voiced_larynx_masked.json'
        default_ckpt = 'encoder_voiced_larynx_masked.pt'

    log_path = args.log_path or os.path.join(_PROJECT_ROOT, 'logs', 'encoder', default_log)
    ckpt_path = args.ckpt_path or os.path.join(_PROJECT_ROOT, 'cache', default_ckpt)

    # In dry-run we don't filter by split (we want broad coverage to exercise
    # the loop). In real training we use Gaddy's predefined dev/test splits
    # for leakage prevention: voiced_train excludes book_locations of
    # silent val + test.
    train_subset = None if args.dry_run else 'train'
    val_subset = None if args.dry_run else 'val'

    # No utterance-length cap. Token-budget sampler bounds per-batch memory
    # by packing long utts into smaller batches (B=1 for the longest 54s
    # utterance, which is well under the auto-probed budget on either device).
    max_emg_length = None

    print('Building train dataset...', flush=True)
    train_dataset = GaddyEMGDataset(
        modes=modes,
        mask_channels=[LARYNX_CH],
        max_examples=max_examples,
        with_frame_labels=not args.no_frame_labels,
        subset=train_subset,
        max_emg_length=max_emg_length,
    )
    print(f'  train: {len(train_dataset)} examples ({sum(1 for s, _ in train_dataset.examples if s.silent)} silent / '
          f'{sum(1 for s, _ in train_dataset.examples if not s.silent)} voiced) '
          f'mask_channels={train_dataset.mask_channels} '
          f'with_frame_labels={train_dataset.with_frame_labels} '
          f'balanced_sampler={balanced}')

    # Token budget: auto-probe the device unless overridden. Probe runs a
    # synthetic worst-case forward+backward and extrapolates to ~70% of
    # device memory.
    if args.token_budget is None:
        print(f'Probing {device} for safe token budget '
              f'(total memory ≈ {total_memory_gib(device):.1f} GiB)...', flush=True)
        token_budget = probe_token_budget(device, target_fraction=0.7, bf16=bf16)
        print(f'  auto-probed token_budget = {token_budget:,}', flush=True)
    else:
        token_budget = args.token_budget
        print(f'Using --token-budget = {token_budget:,}', flush=True)

    # TokenBudgetPairedSampler: variable-B batches packed under the budget.
    # Pair-aware (paired voiced+silent always co-occur). Voiced-only mode
    # falls back to balanced sampling since pairs need both modes.
    if args.voiced_only:
        train_loader = make_loader(
            train_dataset, batch_size=MIN_BATCH_SIZE, balanced=False,
            shuffle=True, num_workers=args.num_workers,
        )
    else:
        train_loader = make_loader(
            train_dataset,
            token_budget=token_budget,
            min_batch_size=MIN_BATCH_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            length_cache_path=LENGTH_CACHE_PATH,
            num_workers=args.num_workers,
        )

    val_loader = None
    if not args.dry_run:
        print('Building val dataset...', flush=True)
        val_dataset = GaddyEMGDataset(
            modes=modes,
            mask_channels=[LARYNX_CH],
            with_frame_labels=False,  # val PER doesn't need frame labels
            subset=val_subset,
            max_emg_length=max_emg_length,
        )
        n_v_silent = sum(1 for s, _ in val_dataset.examples if s.silent)
        n_v_voiced = sum(1 for s, _ in val_dataset.examples if not s.silent)
        print(f'  val: {len(val_dataset)} examples ({n_v_silent} silent / {n_v_voiced} voiced)')
        if len(val_dataset) > 0:
            # Val also uses token-budget sampling — keeps memory bounded
            # without us having to pick a val batch size.
            val_loader = make_loader(
                val_dataset,
                token_budget=token_budget,
                min_batch_size=1,
                max_batch_size=MAX_BATCH_SIZE,
                length_cache_path=LENGTH_CACHE_PATH.replace('.json', '_val.json'),
                num_workers=args.num_workers,
            )

    print('Building model...', flush=True)
    model = ConformerCTC()
    print(f'  params={count_params(model):,}')

    cfg = TrainConfig(
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        bf16=bf16,
        dry_run=args.dry_run,
        dry_run_steps=args.dry_run_steps,
        contrast_weight=0.0 if args.voiced_only else 0.3,
        early_stop_patience=5,
        early_stop_metric='silent_per' if not args.voiced_only else 'voiced_per',
        ckpt_every_n_steps=args.ckpt_every_n_steps,
        log_path=log_path,
        ckpt_path=None if args.dry_run else ckpt_path,
    )

    ok = train_ctc(model, train_loader, val_loader=val_loader, cfg=cfg)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
