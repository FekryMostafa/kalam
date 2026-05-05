"""Test: are word boundaries detectable in EMG envelope?

Method:
  1. Load paired audio (16 kHz FLAC) + EMG (8-ch @ 1000 Hz) for voiced utterances.
  2. Use audio energy to identify ground-truth inter-word silences.
  3. Compute EMG envelope (rectified, smoothed).
  4. Ask: at the audio-silence times, does the EMG envelope also dip?

Outputs:
  - logs/word_boundary_test.json
  - visuals/word_boundaries/00_*.png  (a few example overlays)
"""
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = '/Users/fekrymostafa/Desktop/kalam/experiments/dataset'
OUT_VIS = os.path.join(PROJECT_ROOT, 'visuals', 'word_boundaries')
OUT_LOG = os.path.join(PROJECT_ROOT, 'logs', 'word_boundary_test.json')
os.makedirs(OUT_VIS, exist_ok=True)
os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)

EMG_SR = 1000
AUDIO_SR = 16000
ANALYSIS_SR = 100   # common rate after envelope smoothing
SESSIONS = ['5-4', '5-5', '5-6', '5-10', '5-11']  # exclude val sessions

# Gaddy preprocessing
_NOTCH = [sps.iirnotch(hz, Q=30, fs=EMG_SR) for hz in [60, 120, 180, 240, 300, 360, 420]]
_HP = sps.butter(4, 2.0, btype='high', fs=EMG_SR)


def emg_preprocess(emg):
    out = emg.astype(np.float32).copy()
    for b, a in _NOTCH:
        out = sps.filtfilt(b, a, out, axis=0)
    b, a = _HP
    out = sps.filtfilt(b, a, out, axis=0)
    return out.astype(np.float32)


def envelope(x_1d, win):
    return np.convolve(np.abs(x_1d), np.ones(win)/win, mode='same')


def downsample_to(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    ratio = dst_sr / src_sr
    new_len = int(round(len(x) * ratio))
    return sps.resample(x, new_len)


def find_silences(envelope_at_100hz, threshold_pctile=15, min_dur_ms=80):
    """Return list of (start_sample, end_sample) intervals where envelope is low."""
    thr = np.percentile(envelope_at_100hz, threshold_pctile)
    below = envelope_at_100hz < thr
    # min duration in samples at 100Hz
    min_samples = int(min_dur_ms / 1000 * ANALYSIS_SR)
    intervals = []
    start = None
    for i, b in enumerate(below):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= min_samples:
                intervals.append((start, i))
            start = None
    if start is not None and len(below) - start >= min_samples:
        intervals.append((start, len(below)))
    return intervals


def list_pairs():
    out = []
    for s in SESSIONS:
        d = f'{DATA_ROOT}/voiced_parallel_data/{s}'
        for info_path in sorted(glob.glob(f'{d}/*_info.json')):
            try: text = json.load(open(info_path)).get('text', '').strip()
            except Exception: continue
            if not text: continue
            idx = os.path.basename(info_path).replace('_info.json', '')
            emg = f'{d}/{idx}_emg.npy'
            audio = f'{d}/{idx}_audio_clean.flac'
            if os.path.exists(emg) and os.path.exists(audio):
                out.append((text, emg, audio))
    return out


def envelope_at_analysis_rate(emg, audio, audio_sr_actual):
    """Return (audio_env, emg_env_per_channel) both at ANALYSIS_SR."""
    emg_pp = emg_preprocess(emg)
    # EMG envelope: per-channel rectified + 50ms smooth, then downsample 1000 -> 100
    emg_env_1khz = np.stack([envelope(emg_pp[:, c], 50) for c in range(8)], axis=1)
    emg_env = np.stack([downsample_to(emg_env_1khz[:, c], EMG_SR, ANALYSIS_SR) for c in range(8)], axis=1)
    # Audio envelope: rectified + 50ms smooth, then downsample to 100Hz
    audio_env_full = envelope(audio, int(0.050 * audio_sr_actual))
    audio_env = downsample_to(audio_env_full, audio_sr_actual, ANALYSIS_SR)
    # Truncate to common length
    n = min(len(audio_env), len(emg_env))
    return audio_env[:n], emg_env[:n]


def overlap_score(audio_silences, emg_dips, tol_samples):
    """Precision/recall: how many audio silences have an EMG dip within tolerance?"""
    if not audio_silences:
        return None, None, None
    matched = 0
    for (a_start, a_end) in audio_silences:
        a_mid = (a_start + a_end) // 2
        for (e_start, e_end) in emg_dips:
            e_mid = (e_start + e_end) // 2
            if abs(a_mid - e_mid) <= tol_samples:
                matched += 1
                break
    recall = matched / len(audio_silences)
    if not emg_dips:
        precision = 0.0
    else:
        emg_matched = 0
        for (e_start, e_end) in emg_dips:
            e_mid = (e_start + e_end) // 2
            for (a_start, a_end) in audio_silences:
                a_mid = (a_start + a_end) // 2
                if abs(a_mid - e_mid) <= tol_samples:
                    emg_matched += 1
                    break
        precision = emg_matched / len(emg_dips)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    return precision, recall, f1


# ---------------- run ----------------
print('Indexing pairs...')
pairs = list_pairs()
print(f'  found {len(pairs)} voiced utterances with audio + EMG')

# Limit to a sample
SAMPLE = 80
rng = np.random.default_rng(42)
idxs = rng.choice(len(pairs), size=min(SAMPLE, len(pairs)), replace=False)

# Per-utterance metrics
results = []
TOL_MS = 100  # match within 100ms
TOL_SAMPLES = int(TOL_MS / 1000 * ANALYSIS_SR)

# Pick 4 utterances for visualization across length quartiles
viz_idxs_picked = sorted(idxs.tolist(), key=lambda i: len(pairs[i][0].split()))
viz_picks = [
    viz_idxs_picked[len(viz_idxs_picked)//8],
    viz_idxs_picked[len(viz_idxs_picked)//3],
    viz_idxs_picked[2*len(viz_idxs_picked)//3],
    viz_idxs_picked[-2],
]

for n, i in enumerate(idxs):
    text, emg_path, audio_path = pairs[i]
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(1)
    emg = np.load(emg_path).astype(np.float32)
    if emg.shape[1] != 8:
        continue
    audio_env, emg_env = envelope_at_analysis_rate(emg, audio, sr)
    # Sum envelope across face/jaw/lip channels (skip larynx ch3 which goes silent in silent speech anyway)
    face_chans = [0, 1, 2, 4, 5, 6, 7]
    emg_face_env = emg_env[:, face_chans].sum(axis=1)

    # Audio silences (ground truth word boundaries)
    audio_silences = find_silences(audio_env, threshold_pctile=15, min_dur_ms=80)
    # EMG dips
    emg_dips = find_silences(emg_face_env, threshold_pctile=20, min_dur_ms=80)

    n_words = len(text.split())
    p, r, f1 = overlap_score(audio_silences, emg_dips, TOL_SAMPLES)
    results.append({
        'text_words': n_words,
        'audio_silences': len(audio_silences),
        'emg_dips': len(emg_dips),
        'precision': p, 'recall': r, 'f1': f1,
        'duration_s': len(emg_face_env) / ANALYSIS_SR,
    })

    if i in viz_picks:
        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        t = np.arange(len(emg_face_env)) / ANALYSIS_SR
        axes[0].plot(t, audio_env / max(audio_env.max(), 1e-9), color='steelblue', lw=1)
        for s_i, e_i in audio_silences:
            axes[0].axvspan(s_i / ANALYSIS_SR, e_i / ANALYSIS_SR, alpha=0.2, color='gray')
        axes[0].set_ylabel('audio env (norm)')
        axes[0].set_title(f'"{text[:80]}"  ({n_words} words)')

        axes[1].plot(t, emg_face_env / max(emg_face_env.max(), 1e-9), color='crimson', lw=1)
        for s_i, e_i in emg_dips:
            axes[1].axvspan(s_i / ANALYSIS_SR, e_i / ANALYSIS_SR, alpha=0.2, color='red')
        axes[1].set_ylabel('EMG face env (norm)')
        axes[1].set_xlabel('time (s)')

        fig.suptitle(f'Audio silences (gray) vs EMG dips (red).  P={p:.2f} R={r:.2f} F1={f1:.2f}', fontsize=10)
        plt.tight_layout()
        fname = f'{OUT_VIS}/{n:02d}_w{n_words}_overlay.png'
        plt.savefig(fname, dpi=110)
        plt.close()
        print(f'  saved {fname}')

# Aggregate
res_arr = [r for r in results if r['precision'] is not None]
arr = lambda k: np.array([r[k] for r in res_arr])
summary = {
    'n_utterances': len(res_arr),
    'mean_precision': float(arr('precision').mean()),
    'median_precision': float(np.median(arr('precision'))),
    'mean_recall': float(arr('recall').mean()),
    'median_recall': float(np.median(arr('recall'))),
    'mean_f1': float(arr('f1').mean()),
    'median_f1': float(np.median(arr('f1'))),
    'mean_audio_silences_per_word': float((arr('audio_silences') / arr('text_words')).mean()),
    'mean_emg_dips_per_word': float((arr('emg_dips') / arr('text_words')).mean()),
    'mean_duration_s': float(arr('duration_s').mean()),
    'tolerance_ms': TOL_MS,
    'method': 'audio silences (gnd truth) vs EMG face-channel envelope dips',
    'envelope_window_ms': 50,
    'silence_min_dur_ms': 80,
    'audio_threshold_pctile': 15,
    'emg_threshold_pctile': 20,
}

with open(OUT_LOG, 'w') as f:
    json.dump(summary, f, indent=2)

print('\n' + '=' * 60)
print(f'n_utterances              : {summary["n_utterances"]}')
print(f'precision (median / mean) : {summary["median_precision"]:.3f} / {summary["mean_precision"]:.3f}')
print(f'recall    (median / mean) : {summary["median_recall"]:.3f} / {summary["mean_recall"]:.3f}')
print(f'F1        (median / mean) : {summary["median_f1"]:.3f} / {summary["mean_f1"]:.3f}')
print(f'audio silences / word     : {summary["mean_audio_silences_per_word"]:.3f} (target ~1.0)')
print(f'EMG dips / word           : {summary["mean_emg_dips_per_word"]:.3f}')
print(f'mean duration             : {summary["mean_duration_s"]:.2f}s')
print('=' * 60)
print(f'\nlogs/word_boundary_test.json + visuals/word_boundaries/ written')
