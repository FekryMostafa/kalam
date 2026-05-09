"""Alignment quality check: verify that forced alignment produces sensible
word/phoneme spans before relying on its output for frame-CE supervision.

Two outputs:

  1. logs/encoder/alignment_check.json — quantitative report.

  2. cache/encoder/align_audio/<phoneme_class>/<idx>.wav — per-phoneme
     audio clips extracted from the alignment. Listen to a few from each
     phoneme directory to verify "this K clip sounds like a K". If the
     clips don't match the directory name, the alignment is wrong.

Pass criterion: ≥ 0.9 fraction of utterances align cleanly (no fallback).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np  # noqa: E402
import soundfile  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from app.encoder.data.align import (  # noqa: E402
    _audio_emissions, _normalise_text, _tokens_for_text,
    _word_spans_from_alignment, _get_asr, load_audio,
)
from app.encoder.vocab import text_to_phonemes  # noqa: E402


OUT_PATH = os.path.join(_PROJECT_ROOT, 'logs', 'encoder', 'alignment_check.json')
AUDIO_OUT_DIR = os.path.join(_PROJECT_ROOT, 'cache', 'encoder', 'align_audio')


def _audio_rms_at_span(waveform: torch.Tensor, sr: int,
                       start_frac: float, end_frac: float) -> float:
    """RMS of the audio waveform within a fractional span of its length."""
    n = waveform.shape[-1]
    s = int(start_frac * n)
    e = int(end_frac * n)
    if e <= s:
        return 0.0
    chunk = waveform[..., s:e].float()
    return float(chunk.pow(2).mean().sqrt().item())


def _save_phoneme_audio(
    waveform: torch.Tensor, sr: int,
    word_spans: list[tuple[float, float, str]],
    norm_words: list[str], g2p, audio_idx: int,
    counts_per_class: dict[str, int], max_per_class: int,
):
    """Slice audio into per-phoneme clips and save under cache/.../<class>/.

    For each word span: get phonemes for that word, distribute the span
    uniformly, slice waveform, save .wav into cache/encoder/align_audio/
    <class>/. Skips classes that already have max_per_class samples.
    """
    n_audio = waveform.shape[-1]
    for (start_frac, end_frac, _aligned_word), text_word in zip(
        word_spans, norm_words[:len(word_spans)],
    ):
        phons = text_to_phonemes(text_word, g2p)
        if not phons:
            continue
        a0 = int(start_frac * n_audio)
        a1 = int(end_frac * n_audio)
        if a1 <= a0:
            continue
        n_phon = len(phons)
        for k, ph in enumerate(phons):
            cls = ph
            if counts_per_class.get(cls, 0) >= max_per_class:
                continue
            f0 = a0 + k * (a1 - a0) // n_phon
            f1 = a0 + (k + 1) * (a1 - a0) // n_phon
            if f1 - f0 < int(0.02 * sr):  # skip clips shorter than 20ms
                continue
            chunk = waveform[..., f0:f1]
            cls_dir = os.path.join(AUDIO_OUT_DIR, cls)
            os.makedirs(cls_dir, exist_ok=True)
            out_path = os.path.join(cls_dir, f'{cls}_{audio_idx}_{k}.wav')
            # soundfile expects (T,) mono or (T, C); shape from chunk is (1, T)
            soundfile.write(out_path, chunk.squeeze(0).numpy(), sr)
            counts_per_class[cls] = counts_per_class.get(cls, 0) + 1


def check_one(audio_path: str, text: str, audio_idx: int, g2p,
              counts_per_class: dict, max_per_class: int) -> dict:
    """Align a single utterance, return diagnostics + write audio clips."""
    _, asr_labels = _get_asr()
    waveform, sr = load_audio(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    norm_words = _normalise_text(text).split(' ')
    n_words_text = len(norm_words)

    log_probs, _ = _audio_emissions(audio_path)
    tokens = _tokens_for_text(text, asr_labels)
    word_spans = _word_spans_from_alignment(log_probs, tokens, asr_labels)

    n_words_aligned = len(word_spans)
    aligned_cleanly = (n_words_aligned == n_words_text)

    word_table = []
    for (start_frac, end_frac, aligned_word), text_word in zip(
        word_spans, norm_words[:n_words_aligned],
    ):
        rms = _audio_rms_at_span(waveform, sr, start_frac, end_frac)
        word_table.append({
            'text_word': text_word,
            'aligned_word': aligned_word,
            'start_frac': round(start_frac, 4),
            'end_frac': round(end_frac, 4),
            'duration_frac': round(end_frac - start_frac, 4),
            'audio_rms': round(rms, 6),
        })

    span_durations = [w['duration_frac'] for w in word_table]
    spans_collapsed_to_zero = sum(1 for d in span_durations if d <= 0.001)

    if aligned_cleanly:
        _save_phoneme_audio(
            waveform, sr, word_spans, norm_words, g2p,
            audio_idx, counts_per_class, max_per_class,
        )

    return {
        'audio_path': audio_path,
        'text': text,
        'n_words_text': n_words_text,
        'n_words_aligned': n_words_aligned,
        'aligned_cleanly': aligned_cleanly,
        'spans_collapsed_to_zero': spans_collapsed_to_zero,
        'mean_word_duration_frac': round(float(np.mean(span_durations)), 4) if span_durations else 0.0,
        'min_word_duration_frac': round(float(min(span_durations)), 4) if span_durations else 0.0,
        'max_word_duration_frac': round(float(max(span_durations)), 4) if span_durations else 0.0,
        'words': word_table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='utterances to check')
    parser.add_argument('--max-per-class', type=int, default=5,
                        help='max audio clips to save per phoneme class')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    from g2p_en import G2p
    from app.encoder.data.dataset import GaddyEMGDataset

    print(f'Output paths:', flush=True)
    print(f'  report:      {OUT_PATH}', flush=True)
    print(f'  audio clips: {AUDIO_OUT_DIR}/<phoneme_class>/', flush=True)

    print('\nLoading voiced dataset (no frame labels)...', flush=True)
    ds = GaddyEMGDataset(
        modes=('voiced',),
        with_frame_labels=False,
    )
    print(f'  {len(ds)} voiced utterances available', flush=True)

    rng = random.Random(args.seed)
    pick_indices = rng.sample(range(len(ds)), min(args.n, len(ds)))
    print(f'  picked {len(pick_indices)} utterances (seed={args.seed})', flush=True)

    print('\nLoading wav2vec2 ASR (one-time, downloads ~360 MB on first run)...',
          flush=True)
    _get_asr()  # warm-load before timing
    print('  ready', flush=True)

    print('\nLoading g2p_en...', flush=True)
    g2p = G2p()
    print('  ready', flush=True)

    counts_per_class: dict[str, int] = {}
    results = []
    for k, ex_i in enumerate(pick_indices):
        sess, idx = ds.examples[ex_i]
        audio_path = os.path.join(sess.directory, f'{idx}_audio_clean.flac')
        info_path = os.path.join(sess.directory, f'{idx}_info.json')
        with open(info_path) as f:
            text = json.load(f).get('text', '')
        if not os.path.exists(audio_path) or not text.strip():
            print(f'\n[{k+1}/{len(pick_indices)}] SKIP: missing audio or empty text',
                  flush=True)
            continue

        print(f'\n[{k+1}/{len(pick_indices)}] {os.path.basename(sess.directory)}/{idx}: '
              f'"{text[:60]}{"..." if len(text) > 60 else ""}"', flush=True)
        try:
            res = check_one(audio_path, text, k, g2p, counts_per_class, args.max_per_class)
        except Exception as e:  # pragma: no cover
            print(f'  ERROR: {e}', flush=True)
            results.append({
                'audio_path': audio_path,
                'text': text,
                'error': str(e),
                'aligned_cleanly': False,
            })
            continue

        ok = 'OK' if res['aligned_cleanly'] else 'FAIL'
        print(f'  [{ok}] text_words={res["n_words_text"]}  '
              f'aligned_words={res["n_words_aligned"]}  '
              f'collapsed_spans={res["spans_collapsed_to_zero"]}  '
              f'word_dur min/mean/max = '
              f'{res["min_word_duration_frac"]}/'
              f'{res["mean_word_duration_frac"]}/'
              f'{res["max_word_duration_frac"]}', flush=True)
        results.append(res)

    n_total = len(results)
    n_clean = sum(1 for r in results if r.get('aligned_cleanly'))
    pass_threshold = 0.9
    pass_ratio = n_clean / max(n_total, 1)
    passed = pass_ratio >= pass_threshold

    summary = {
        'n_checked': n_total,
        'n_aligned_cleanly': n_clean,
        'pass_ratio': round(pass_ratio, 3),
        'pass_threshold': pass_threshold,
        'passed': passed,
        'audio_clips_per_class': dict(sorted(counts_per_class.items())),
        'audio_output_dir': AUDIO_OUT_DIR,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump({'summary': summary, 'per_utterance': results}, f, indent=2)

    print('\n' + '=' * 60)
    print('ALIGNMENT QUALITY CHECK')
    print('=' * 60)
    for k, v in summary.items():
        if k == 'audio_clips_per_class':
            print(f'  {k}: {dict(list(v.items())[:8])}...' if len(v) > 8 else f'  {k}: {v}')
        else:
            print(f'  {k}: {v}')
    print(f'\nWrote {OUT_PATH}')
    print(f'Audio clips: {AUDIO_OUT_DIR}/<phoneme_class>/')
    print('Listen to a few from each class — they should sound like the class name.')

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
