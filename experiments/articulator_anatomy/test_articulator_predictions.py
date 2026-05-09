"""Group EMG channels by anatomical placement (Gaddy 2020 Table 3) and report
silent/voiced ratios for amplitude, spectral centroid, and activity ratio.

Anatomy mapping (0-indexed):
  ch0 left cheek above mouth     -> lips
  ch1 left corner of chin        -> chin/lower-lip
  ch2 below chin, back 3 cm      -> floor-of-mouth/tongue
  ch3 throat 3 cm L of Adam's    -> larynx
  ch4 mid-jaw right              -> jaw
  ch5 right cheek below mouth    -> lips
  ch6 right cheek near nose      -> upper face
  ch7 right cheek 4 cm from ear  -> jaw/masseter
"""
import glob
import json
import os

import numpy as np
from scipy import signal as sps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = os.environ.get('KALAM_DATA', os.path.join(PROJECT_ROOT, 'dataset'))
SR = 1000
NPERSEG = 1024
SESSIONS = ['5-4', '5-5', '5-6', '5-8', '5-9', '5-10', '5-11']

GROUPS = {
    'larynx':            [3],
    'lips':              [0, 5],
    'chin_lower_lip':    [1],
    'floor_of_mouth':    [2],
    'jaw_masseter':      [4, 7],
    'upper_face':        [6],
}

def list_pairs():
    pairs = []
    for s in SESSIONS:
        v_dir = f'{DATA_ROOT}/voiced_parallel_data/{s}'
        si_dir = f'{DATA_ROOT}/silent_parallel_data/{s}_silent'
        v = {}; si = {}
        for d, store in [(v_dir, v), (si_dir, si)]:
            if not os.path.isdir(d): continue
            for info in glob.glob(f'{d}/*_info.json'):
                try: text = json.load(open(info)).get('text', '').strip()
                except Exception: continue
                if not text: continue
                idx = os.path.basename(info).replace('_info.json', '')
                emg = f'{d}/{idx}_emg.npy'
                if os.path.exists(emg): store[text] = emg
        for t in set(v) & set(si):
            pairs.append((v[t], si[t]))
    return pairs

# Gaddy preprocessing: notch 60 Hz harmonics + 2 Hz highpass
_NOTCH = [sps.iirnotch(hz, Q=30, fs=SR) for hz in [60, 120, 180, 240, 300, 360, 420]]
_HP = sps.butter(4, 2.0, btype='high', fs=SR)

def preprocess(emg):
    out = emg.astype(np.float32).copy()
    for b, a in _NOTCH:
        out = sps.filtfilt(b, a, out, axis=0)
    b, a = _HP
    out = sps.filtfilt(b, a, out, axis=0)
    return out.astype(np.float32)

def envelope(x, win=50):
    return np.convolve(np.abs(x), np.ones(win)/win, mode='same')

def per_channel_features(emg_pp):
    """Returns dict of (8,) arrays: rms, centroid_hz, activity_ratio."""
    T = emg_pp.shape[0]
    rms = np.sqrt((emg_pp ** 2).mean(0))
    # Spectral centroid per channel
    centroid = np.zeros(8, dtype=np.float32)
    for c in range(8):
        if T < NPERSEG: continue
        f, P = sps.welch(emg_pp[:, c], fs=SR, nperseg=min(NPERSEG, T))
        centroid[c] = (f * P).sum() / max(P.sum(), 1e-12)
    # Activity ratio: envelope > 2 * its own median
    activity = np.zeros(8, dtype=np.float32)
    for c in range(8):
        env = envelope(emg_pp[:, c], 50)
        thr = 2 * np.median(env)
        activity[c] = float((env > thr).mean())
    return rms, centroid, activity

print('Indexing pairs...')
pairs = list_pairs()
print(f'  {len(pairs)} paired utterances')

rms_v_all = []; rms_s_all = []
cent_v_all = []; cent_s_all = []
act_v_all = []; act_s_all = []
dur_v = []; dur_s = []

for i, (vp, sp) in enumerate(pairs):
    v = np.load(vp).astype(np.float32)
    si = np.load(sp).astype(np.float32)
    if v.shape[1] != 8 or si.shape[1] != 8: continue
    if min(len(v), len(si)) < NPERSEG: continue
    v_pp = preprocess(v); si_pp = preprocess(si)
    rv, cv, av = per_channel_features(v_pp)
    rs, cs, as_ = per_channel_features(si_pp)
    rms_v_all.append(rv); rms_s_all.append(rs)
    cent_v_all.append(cv); cent_s_all.append(cs)
    act_v_all.append(av); act_s_all.append(as_)
    dur_v.append(len(v)); dur_s.append(len(si))
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(pairs)}')

rms_v = np.array(rms_v_all); rms_s = np.array(rms_s_all)
cent_v = np.array(cent_v_all); cent_s = np.array(cent_s_all)
act_v = np.array(act_v_all); act_s = np.array(act_s_all)

# Per-channel silent/voiced ratios (median across pairs)
rms_ratio = np.median(rms_s / np.maximum(rms_v, 1e-9), axis=0)
act_ratio = np.median(act_s / np.maximum(act_v, 1e-9), axis=0)
cent_v_med = np.median(cent_v, axis=0)
cent_s_med = np.median(cent_s, axis=0)

def group_agg(per_ch):
    return {g: float(np.mean([per_ch[c] for c in chs])) for g, chs in GROUPS.items()}

out = {
    'method': 'paired voiced/silent EMG, same text, 7 sessions, Gaddy preprocessed (notch+2Hz HP)',
    'n_pairs': len(rms_v_all),
    'channel_to_anatomy': {
        '0': 'lips_L', '1': 'chin_L', '2': 'floor_of_mouth', '3': 'larynx',
        '4': 'jaw_R', '5': 'lips_R', '6': 'upper_face', '7': 'masseter',
    },
    'silent_over_voiced_rms_per_group': group_agg(rms_ratio),
    'silent_over_voiced_activity_ratio_per_group': group_agg(act_ratio),
    'voiced_centroid_hz_per_group': group_agg(cent_v_med),
    'silent_centroid_hz_per_group': group_agg(cent_s_med),
    'centroid_shift_hz_per_group': group_agg(cent_s_med - cent_v_med),
    'duration_silent_over_voiced_median': float(np.median(np.array(dur_s) / np.maximum(np.array(dur_v), 1))),
}

LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
with open(os.path.join(LOGS_DIR, 'articulator_test.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('saved logs/articulator_test.json\n')

print(f'{"group":<18s} {"rms_ratio":>10s} {"act_ratio":>10s} {"voiced_Hz":>11s} {"silent_Hz":>11s} {"shift_Hz":>10s}')
for g in GROUPS:
    print(f'{g:<18s} '
          f'{out["silent_over_voiced_rms_per_group"][g]:>10.3f} '
          f'{out["silent_over_voiced_activity_ratio_per_group"][g]:>10.3f} '
          f'{out["voiced_centroid_hz_per_group"][g]:>11.1f} '
          f'{out["silent_centroid_hz_per_group"][g]:>11.1f} '
          f'{out["centroid_shift_hz_per_group"][g]:>+10.1f}')
print(f'\nDuration silent/voiced median: {out["duration_silent_over_voiced_median"]:.3f}')
