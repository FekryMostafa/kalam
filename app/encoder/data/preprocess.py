"""EMG cleanup pipeline (Gaddy 2020 + 2024 surface-EMG best practices).

Per utterance:
  1. concat with neighbor utterances     filter on a wider window so the
                                          notch + highpass don't ring at
                                          utterance boundaries (Gaddy 2020).
  2. notch at 60 Hz + harmonics 1..7     removes US power-line noise.
  3. butterworth(3) highpass at 2 Hz     removes slow electrode drift.
  4. crop neighbors back off
  5. per-channel z-score                 zero mean, unit variance. replaces
                                          ad-hoc tanh squash; principled and
                                          handles inter-session amplitude
                                          variance.

Sampling rate is 1 kHz; Nyquist is 500 Hz, so no lowpass is needed —
the signal can't have content above 500 Hz by construction.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

NOTCH_FREQ_HZ = 60.0
NOTCH_HARMONICS = 7  # 60, 120, ..., 420 Hz
NOTCH_Q = 30.0
HIGHPASS_HZ = 2.0
HIGHPASS_ORDER = 3
ZSCORE_EPS = 1e-8


def _notch(signal, freq, fs):
    b, a = scipy.signal.iirnotch(freq, NOTCH_Q, fs)
    return scipy.signal.filtfilt(b, a, signal)


def _notch_harmonics(signal, fs):
    out = signal
    for h in range(1, NOTCH_HARMONICS + 1):
        out = _notch(out, NOTCH_FREQ_HZ * h, fs)
    return out


def _highpass(signal, fs):
    b, a = scipy.signal.butter(HIGHPASS_ORDER, HIGHPASS_HZ, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def _per_channel(fn, x, *args):
    return np.stack([fn(x[:, c], *args) for c in range(x.shape[1])], axis=1)


def _zscore_per_channel(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + ZSCORE_EPS)


def preprocess_emg(
    raw: np.ndarray,
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
    fs: float = 1000.0,
) -> np.ndarray:
    """Notch + highpass + z-score, with optional neighbor buffering.

    Args:
        raw:    (T, C) EMG samples for one utterance.
        before: (T_b, C) preceding utterance, or None. Used only to absorb
                filter edge artifacts; not returned.
        after:  (T_a, C) following utterance, or None. Same role.

    Returns:
        (T, C) float32 — cleaned + z-scored, same length as raw.
    """
    n_before = before.shape[0] if before is not None and before.shape[0] > 0 else 0
    n_after = after.shape[0] if after is not None and after.shape[0] > 0 else 0

    if n_before or n_after:
        parts = []
        if n_before:
            parts.append(before)
        parts.append(raw)
        if n_after:
            parts.append(after)
        x = np.concatenate(parts, axis=0)
    else:
        x = raw

    x = _per_channel(_notch_harmonics, x, fs)
    x = _per_channel(_highpass, x, fs)

    if n_before or n_after:
        end = x.shape[0] - n_after if n_after else x.shape[0]
        x = x[n_before:end]

    x = _zscore_per_channel(x)
    return x.astype(np.float32)
