"""Phoneme inventory for the encoder. Two views:

ARPABET           the standard 39-phoneme ARPAbet set. Used by g2p_en.
                  Each phoneme has its own integer.

COLLAPSED_CLASSES the encoder's actual output inventory. Voicing pairs
                  (B/P, T/D, K/G, S/Z, F/V, SH/ZH, CH/JH, DH/TH) collapse
                  into single classes because they are physically
                  indistinguishable from silent EMG (claim 15 in
                  first_principles.md). Forcing the encoder to discriminate
                  them would be asking the impossible — the inventory bakes
                  the physical constraint into the model.

Convention:
    index 0       CTC blank
    index 1..N    phoneme classes (collapsed or unambiguous)

The collapsed classes are exposed at decode time as `(X:0.5/Y:0.5)` text
in the LLM prompt (see app/decoder_prompt.md).
"""

ARPABET = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
]

# Voicing pairs — phonemes that differ only in voicing. Physically
# indistinguishable from silent EMG.
VOICING_PAIRS = [
    ('B', 'P'),
    ('D', 'T'),
    ('G', 'K'),
    ('Z', 'S'),
    ('V', 'F'),
    ('ZH', 'SH'),
    ('JH', 'CH'),
    ('DH', 'TH'),
]

# Build the collapsed inventory. Each voicing pair becomes a single class
# named e.g. "B/P". All other phonemes (vowels, sonorants, /HH/) keep their
# own class.
_VOICING_LOOKUP = {}
for a, b in VOICING_PAIRS:
    label = f'{a}/{b}'
    _VOICING_LOOKUP[a] = label
    _VOICING_LOOKUP[b] = label

COLLAPSED_CLASSES = []
_seen: set[str] = set()
for ph in ARPABET:
    cls = _VOICING_LOOKUP.get(ph, ph)
    if cls not in _seen:
        _seen.add(cls)
        COLLAPSED_CLASSES.append(cls)

BLANK_IDX = 0
N_CLASSES = len(COLLAPSED_CLASSES) + 1  # phonemes + 1 blank

# Map ARPAbet phoneme strings to collapsed class indices in [1, N_CLASSES-1].
PHON_TO_IDX = {ph: COLLAPSED_CLASSES.index(_VOICING_LOOKUP.get(ph, ph)) + 1
               for ph in ARPABET}

# Map collapsed class indices back to class label strings.
IDX_TO_CLASS = {i + 1: c for i, c in enumerate(COLLAPSED_CLASSES)}


def text_to_phonemes(text, g2p):
    """Run text through a g2p_en G2p() instance, return ARPAbet strings.

    Strips stress digits (AA1 -> AA). Drops anything not in ARPABET.
    """
    raw = g2p(text)
    out = []
    for tok in raw:
        clean = ''.join(c for c in tok if not c.isdigit())
        if clean in PHON_TO_IDX:
            out.append(clean)
    return out


def phonemes_to_indices(phonemes):
    """ARPAbet strings -> collapsed-class indices in [1, N_CLASSES-1]."""
    return [PHON_TO_IDX[p] for p in phonemes]


def indices_to_classes(indices):
    """Collapsed-class indices -> class label strings (drops the blank)."""
    return [IDX_TO_CLASS[i] for i in indices if i != BLANK_IDX]
