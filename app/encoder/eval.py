"""CTC decoding + phoneme-error-rate (PER) evaluation."""

from __future__ import annotations

import torch
from jiwer import wer as _jiwer_wer

from .vocab import BLANK_IDX, IDX_TO_CLASS


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """Greedy CTC decode: argmax per frame, collapse repeats, drop blanks.

    Args:
        log_probs: (B, T, C) — encoder output
        lengths:   (B,)      — valid length per example (encoder out_lengths)

    Returns:
        list of class-index lists (no blanks), one per batch item.
    """
    preds = log_probs.argmax(dim=-1)  # (B, T)
    out: list[list[int]] = []
    for b in range(preds.shape[0]):
        seq = preds[b, : int(lengths[b])].tolist()
        collapsed: list[int] = []
        prev = -1
        for s in seq:
            if s != prev:
                collapsed.append(s)
            prev = s
        out.append([s for s in collapsed if s != BLANK_IDX])
    return out


def indices_to_classstr(indices: list[int]) -> str:
    """Collapsed-class indices -> space-separated label string (for jiwer.wer).

    Voicing-pair classes look like "B/P" or "T/D" — kept as a single token.
    """
    return ' '.join(IDX_TO_CLASS[i] for i in indices if i in IDX_TO_CLASS)


def per(refs: list[list[int]], hyps: list[list[int]]) -> float:
    """Phoneme/class error rate. Treats each class as a 'word' for jiwer.wer."""
    ref_strs = [indices_to_classstr(r) for r in refs]
    hyp_strs = [indices_to_classstr(h) if h else 'pad' for h in hyps]
    return float(_jiwer_wer(ref_strs, hyp_strs))


__all__ = ['ctc_greedy_decode', 'indices_to_classstr', 'per']
