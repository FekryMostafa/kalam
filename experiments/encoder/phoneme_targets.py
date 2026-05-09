"""Inventory + leakage audit for the encoder's phoneme targets.

Reports for every utterance: text, raw ARPAbet phonemes, collapsed-class
indices, voicing-pair count. Aggregates inventory coverage across the
whole dataset and flags train/test sentence overlap (the leakage risk
flagged by the design review).

Run before any training. Validates that:
  1. every utterance produces a non-empty target sequence
  2. the collapsed inventory covers all phonemes in the corpus
  3. the silent test set's utterances don't appear in the voiced train set
     (would inflate any reported number)
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.encoder.data.dataset import GaddyEMGDataset  # noqa: E402
from app.encoder.vocab import (  # noqa: E402
    ARPABET, COLLAPSED_CLASSES, IDX_TO_CLASS, N_CLASSES,
    VOICING_PAIRS, text_to_phonemes, phonemes_to_indices,
)


OUT_PATH = os.path.join(_PROJECT_ROOT, 'logs', 'encoder', 'phoneme_targets.json')


def main():
    print('Loading both modes...', flush=True)
    ds = GaddyEMGDataset(modes=('voiced', 'silent'))
    print(f'  {len(ds)} utterances across {len(ds.sessions)} sessions')

    from g2p_en import G2p
    g2p = G2p()

    per_utterance = []
    voicing_pair_count = 0
    total_phon = 0
    classes_seen: set[int] = set()
    voiced_book_locations: set[tuple] = set()
    silent_book_locations: set[tuple] = set()
    empty_target_count = 0

    voicing_pair_indices = {phonemes_to_indices([a])[0] for a, b in VOICING_PAIRS}

    print('Walking utterances...', flush=True)
    for i, (sess, idx) in enumerate(ds.examples):
        info_path = os.path.join(sess.directory, f'{idx}_info.json')
        with open(info_path) as f:
            info = json.load(f)
        text = info.get('text', '')
        book_loc = (info.get('book', ''), info.get('sentence_index', -1))

        phons = text_to_phonemes(text, g2p)
        target_indices = phonemes_to_indices(phons)

        if not target_indices:
            empty_target_count += 1

        for cls_idx in target_indices:
            classes_seen.add(cls_idx)

        n_voicing = sum(1 for c in target_indices if c in voicing_pair_indices)
        voicing_pair_count += n_voicing
        total_phon += len(target_indices)

        if sess.silent:
            silent_book_locations.add(book_loc)
        else:
            voiced_book_locations.add(book_loc)

        per_utterance.append({
            'session': os.path.basename(sess.directory),
            'idx': idx,
            'mode': 'silent' if sess.silent else 'voiced',
            'text': text,
            'book_location': list(book_loc),
            'n_phonemes': len(target_indices),
            'n_voicing_pair_classes': n_voicing,
        })

        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(ds)}', flush=True)

    overlap = silent_book_locations & voiced_book_locations
    classes_unseen = set(range(1, N_CLASSES)) - classes_seen

    summary = {
        'n_utterances': len(per_utterance),
        'n_voiced': sum(1 for p in per_utterance if p['mode'] == 'voiced'),
        'n_silent': sum(1 for p in per_utterance if p['mode'] == 'silent'),
        'empty_target_count': empty_target_count,
        'total_phonemes_across_corpus': total_phon,
        'total_voicing_pair_class_instances': voicing_pair_count,
        'voicing_pair_share': round(voicing_pair_count / max(total_phon, 1), 4),
        'inventory': {
            'arpabet_size': len(ARPABET),
            'collapsed_classes': len(COLLAPSED_CLASSES),
            'total_ctc_classes_with_blank': N_CLASSES,
            'classes_seen_in_corpus': len(classes_seen),
            'classes_unseen': sorted(IDX_TO_CLASS[i] for i in classes_unseen),
        },
        'silent_voiced_overlap': {
            'silent_utterances': len(silent_book_locations),
            'voiced_utterances': len(voiced_book_locations),
            'overlap_count': len(overlap),
            'note': ('overlap = utterances recorded in both modes (parallel pairs). '
                     'expected to be > 0 for Gaddy. when defining the silent test '
                     'set, exclude its book_locations from voiced training.'),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump({'summary': summary, 'per_utterance': per_utterance}, f, indent=2)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for k, v in summary.items():
        if k == 'per_utterance':
            continue
        print(f'  {k}: {v}')
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
