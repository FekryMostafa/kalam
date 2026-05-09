"""Train / val / test splits with leakage prevention.

Uses Gaddy's predefined dev/test split (book_locations) so our numbers
are comparable to MONA LISA and other published work.

Leakage prevention: silent val/test book_locations are excluded from
the voiced training set. Otherwise the model could memorise sentence
content from voiced training and apply it to silent at test time —
inflating reported WER.

Splits returned by `compute_splits()`:
    silent_train     silent utterances NOT in val/test book_locations
    silent_val       silent utterances at val book_locations (~200)
    silent_test      silent utterances at test book_locations (~100)
    voiced_train     ALL voiced utterances NOT in val/test book_locations
                     (so the model doesn't see voiced versions of val/test
                     sentences during training)
    voiced_val       voiced utterances at val book_locations
                     (used to monitor voiced-side performance, not for
                     early-stopping decisions — silent is the deployment
                     target.)
"""

from __future__ import annotations

import json
import os

SPLITS_FILE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'dataset', 'testset_largedev.json',
))


def _load_split_locations() -> tuple[set, set]:
    """Read Gaddy's split file. Returns (val_locs, test_locs) as sets of
    (book, sentence_index) tuples.
    """
    with open(SPLITS_FILE) as f:
        j = json.load(f)
    val = {tuple(loc) for loc in j['dev']}
    test = {tuple(loc) for loc in j['test']}
    return val, test


def compute_splits(examples) -> dict:
    """Partition `examples` (list of (_Sess, idx) from GaddyEMGDataset) into
    splits. Returns dict mapping split name -> list of indices into
    `examples`.

    Args:
        examples: iterable of (sess, idx) where each has metadata
                  available via `sess.directory + idx_info.json`.
    """
    val_locs, test_locs = _load_split_locations()

    silent_train: list[int] = []
    silent_val: list[int] = []
    silent_test: list[int] = []
    voiced_train: list[int] = []
    voiced_val: list[int] = []
    voiced_test: list[int] = []  # parallel to silent_test (excluded from train)

    for ex_i, (sess, idx) in enumerate(examples):
        info_path = os.path.join(sess.directory, f'{idx}_info.json')
        try:
            with open(info_path) as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        loc = (info.get('book', ''), info.get('sentence_index', -1))
        in_val = loc in val_locs
        in_test = loc in test_locs

        if sess.silent:
            if in_test:
                silent_test.append(ex_i)
            elif in_val:
                silent_val.append(ex_i)
            else:
                silent_train.append(ex_i)
        else:
            if in_test:
                voiced_test.append(ex_i)
            elif in_val:
                voiced_val.append(ex_i)
            else:
                voiced_train.append(ex_i)

    return {
        'silent_train': silent_train,
        'silent_val': silent_val,
        'silent_test': silent_test,
        'voiced_train': voiced_train,
        'voiced_val': voiced_val,
        'voiced_test': voiced_test,
    }
