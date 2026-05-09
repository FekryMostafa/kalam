"""ARPAbet 39-phoneme vocabulary with CTC blank at index 0.

Why blank at 0: torch.nn.CTCLoss requires a designated blank class; index 0
is the conventional choice. Real phonemes occupy indices 1..39.
"""

ARPABET = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
]

BLANK_IDX = 0
N_CLASSES = len(ARPABET) + 1  # 39 phonemes + 1 blank

PHON_TO_IDX = {p: i + 1 for i, p in enumerate(ARPABET)}
IDX_TO_PHON = {i + 1: p for i, p in enumerate(ARPABET)}


def text_to_phonemes(text, g2p):
    """Run text through a g2p_en G2p() instance, return ARPAbet strings.

    Strips stress digits (AA1 -> AA). Drops anything that isn't an ARPAbet
    phoneme (whitespace, punctuation tokens that g2p emits).
    """
    raw = g2p(text)
    out = []
    for tok in raw:
        clean = ''.join(c for c in tok if not c.isdigit())
        if clean in PHON_TO_IDX:
            out.append(clean)
    return out


def phonemes_to_indices(phonemes):
    """ARPAbet strings -> CTC class indices in [1, 39]."""
    return [PHON_TO_IDX[p] for p in phonemes]


def indices_to_phonemes(indices):
    """CTC class indices -> ARPAbet strings (drops the blank class)."""
    return [IDX_TO_PHON[i] for i in indices if i != BLANK_IDX]
