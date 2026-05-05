"""Visual: voiced EMG traces for a few utterances.

Applies Gaddy-style preprocessing (notch 60 Hz harmonics + 2 Hz highpass)
and z-scores per channel before plotting so speech activity is visible.
"""
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps

# Paths are resolved relative to where you run this from.
DATA_ROOT = os.environ.get('KALAM_DATA', '/Users/fekrymostafa/Desktop/kalam/experiments/dataset')
OUT = os.environ.get('KALAM_VISUALS', '/Users/fekrymostafa/Desktop/kalam/Project kalam/visuals/voiced_examples')
os.makedirs(OUT, exist_ok=True)
SR = 1000

def preprocess(emg):
    out = emg.astype(np.float32).copy()
    for hz in [60, 120, 180, 240, 300, 360, 420]:
        b, a = sps.iirnotch(hz, Q=30, fs=SR)
        out = sps.filtfilt(b, a, out, axis=0)
    b, a = sps.butter(4, 2.0, btype='high', fs=SR)
    out = sps.filtfilt(b, a, out, axis=0)
    return out.astype(np.float32)

def envelope(x, win=50):
    return np.convolve(np.abs(x), np.ones(win)/win, mode='same')

def gather_items():
    items = []
    for s in ['5-4', '5-5', '5-6', '5-10', '5-11', '5-8', '5-9']:
        d = f'{DATA_ROOT}/voiced_parallel_data/{s}'
        for info in glob.glob(f'{d}/*_info.json'):
            try:
                text = json.load(open(info)).get('text', '').strip()
            except Exception:
                continue
            if not text: continue
            idx = os.path.basename(info).replace('_info.json', '')
            emg = f'{d}/{idx}_emg.npy'
            if os.path.exists(emg):
                items.append((text, emg, s))
    return items

def plot_grid(triplets, fname, title):
    n = len(triplets)
    fig, axes = plt.subplots(8, n, figsize=(5.5*n, 9))
    if n == 1:
        axes = axes.reshape(8, 1)
    for col, (label, text, emg_path) in enumerate(triplets):
        emg = preprocess(np.load(emg_path).astype(np.float32))
        emg = (emg - emg.mean(0)) / (emg.std(0) + 1e-6)
        t = np.arange(emg.shape[0]) / SR
        for c in range(8):
            ax = axes[c, col]
            ax.plot(t, emg[:, c], lw=0.4, color='gray', alpha=0.7)
            ax.plot(t, envelope(emg[:, c], 50), lw=1.2, color='crimson')
            ax.set_ylim(-5, 5)
            ax.grid(alpha=0.25)
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'ch{c}', rotation=0, ha='right', va='center')
        axes[0, col].set_title(f'{label}\n"{text[:60]}"', fontsize=9)
        axes[-1, col].set_xlabel('s')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()

def plot_single_envelope(text, emg_path, fname):
    emg = preprocess(np.load(emg_path).astype(np.float32))
    emg = (emg - emg.mean(0)) / (emg.std(0) + 1e-6)
    t = np.arange(emg.shape[0]) / SR
    fig, axes = plt.subplots(8, 1, figsize=(13, 9), sharex=True)
    for c in range(8):
        axes[c].plot(t, emg[:, c], color='lightgray', lw=0.4)
        axes[c].plot(t, envelope(emg[:, c], 50), color='crimson', lw=1.0)
        axes[c].set_ylim(-5, 5)
        axes[c].grid(alpha=0.25)
        axes[c].set_ylabel(f'ch{c}', rotation=0, ha='right', va='center')
        axes[c].set_yticks([])
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(f'Voiced EMG (preprocessed, z-scored) — "{text}"', fontsize=11)
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()

if __name__ == '__main__':
    items = gather_items()
    print(f'{len(items)} voiced items')
    short = [it for it in items if 2 <= len(it[0].split()) <= 5]
    long_ = [it for it in items if 8 <= len(it[0].split()) <= 14]

    seen = set(); picks = []
    for it in short:
        if it[0].lower() not in seen:
            picks.append(it); seen.add(it[0].lower())
        if len(picks) == 3: break

    plot_grid([(f'short {i+1}', t, p) for i,(t,p,_) in enumerate(picks)],
              f'{OUT}/01_three_short_utterances.png',
              'Voiced EMG (preprocessed, z-scored) — three different short utterances')
    print('saved 01')

    text_to_paths = {}
    for text, p, s in items:
        text_to_paths.setdefault(text.lower(), []).append((text, p, s))
    repeated = [(t, lst) for t, lst in text_to_paths.items()
                if len(lst) >= 2 and 2 <= len(t.split()) <= 6]
    if repeated:
        text, lst = repeated[0]
        plot_grid([(f'session {lst[0][2]}', lst[0][0], lst[0][1]),
                   (f'session {lst[1][2]}', lst[1][0], lst[1][1])],
                  f'{OUT}/02_same_text_two_sessions.png',
                  'Voiced EMG — same words spoken on different days')
        print('saved 02')

    plot_grid([('SHORT', picks[0][0], picks[0][1]),
               ('LONG',  long_[0][0],  long_[0][1])],
              f'{OUT}/03_short_vs_long.png',
              'Voiced EMG — short utterance vs long utterance')
    print('saved 03')

    def has(text, chars): return sum(text.lower().count(c) for c in chars)
    lippy = sorted(short, key=lambda x: -has(x[0], 'bpmf'))
    tonguey = sorted(short, key=lambda x: -has(x[0], 'tkg'))
    plot_grid([('"lippy" (b/p/m/f)', lippy[0][0], lippy[0][1]),
               ('"tonguey" (t/k/g)', tonguey[0][0], tonguey[0][1])],
              f'{OUT}/04_lippy_vs_tonguey.png',
              'Voiced EMG — phonetically different short utterances')
    print('saved 04')

    plot_single_envelope(picks[0][0], picks[0][1], f'{OUT}/05_envelope_overlay.png')
    print('saved 05')
