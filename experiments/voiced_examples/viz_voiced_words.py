"""Visual: voiced EMG traces for a handful of utterances to build intuition.

Picks short/similar/different texts and draws all 8 channels stacked.
"""
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt

OUT = 'analysis/voiced_examples'
os.makedirs(OUT, exist_ok=True)
SR = 1000

# ---- gather voiced items with short texts ----
items = []
for s in ['5-4', '5-5', '5-6', '5-10', '5-11', '5-8', '5-9']:
    d = f'dataset/voiced_parallel_data/{s}'
    for info in glob.glob(f'{d}/*_info.json'):
        try:
            text = json.load(open(info)).get('text', '').strip()
        except Exception:
            continue
        if not text:
            continue
        idx = os.path.basename(info).replace('_info.json', '')
        emg = f'{d}/{idx}_emg.npy'
        if os.path.exists(emg):
            items.append((text, emg, s))

print(f'{len(items)} voiced items')

# Also check closed_vocab voiced — it has short dates/times
for s in os.listdir('dataset/closed_vocab/voiced'):
    d = f'dataset/closed_vocab/voiced/{s}'
    if not os.path.isdir(d):
        continue
    for info in glob.glob(f'{d}/*_info.json'):
        try:
            text = json.load(open(info)).get('text', '').strip()
        except Exception:
            continue
        if not text:
            continue
        idx = os.path.basename(info).replace('_info.json', '')
        emg = f'{d}/{idx}_emg.npy'
        if os.path.exists(emg):
            items.append((text, emg, f'cv/{s}'))

print(f'{len(items)} total (incl. closed_vocab)')

short = [it for it in items if len(it[0].split()) <= 5]
long_ = [it for it in items if 8 <= len(it[0].split()) <= 14]
print(f'short (<=5 words): {len(short)}, medium (8-14): {len(long_)}')

# ---- helper plots ----
def plot_one(text, emg_path, fname, title_suffix=''):
    emg = np.load(emg_path).astype(np.float32)
    t = np.arange(emg.shape[0]) / SR
    fig, axes = plt.subplots(8, 1, figsize=(13, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, 8))
    for c in range(8):
        axes[c].plot(t, emg[:, c], color=colors[c], lw=0.5)
        axes[c].set_ylabel(f'ch{c}', rotation=0, ha='right', va='center')
        axes[c].grid(alpha=0.25)
        axes[c].set_yticks([])
    axes[-1].set_xlabel('time (seconds)')
    fig.suptitle(f'"{text}"{title_suffix}    [{emg.shape[0]/SR:.2f}s]', fontsize=11)
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()

def plot_grid(triplets, fname, title):
    """triplets: list of (label, text, emg_path). Plot 8 channels x N columns."""
    n = len(triplets)
    fig, axes = plt.subplots(8, n, figsize=(5*n, 9), sharey='row')
    if n == 1:
        axes = axes.reshape(8, 1)
    for col, (label, text, emg_path) in enumerate(triplets):
        emg = np.load(emg_path).astype(np.float32)
        t = np.arange(emg.shape[0]) / SR
        for c in range(8):
            ax = axes[c, col]
            ax.plot(t, emg[:, c], lw=0.4, color='steelblue')
            ax.grid(alpha=0.25)
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'ch{c}', rotation=0, ha='right', va='center')
        axes[0, col].set_title(f'{label}\n"{text[:60]}"', fontsize=9)
        axes[-1, col].set_xlabel('s')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()

# ---- 1. Three different short utterances side-by-side ----
rng = np.random.default_rng(0)
picks = []
seen_texts = set()
for it in short:
    if it[0].lower() not in seen_texts:
        picks.append(it); seen_texts.add(it[0].lower())
    if len(picks) >= 3: break
plot_grid([(f'short {i+1}', t, p) for i,(t,p,_) in enumerate(picks)],
          f'{OUT}/01_three_short_utterances.png',
          'Voiced EMG — three different short utterances')
print('saved 01_three_short_utterances.png')

# ---- 2. Same text, two different sessions (does the same word look the same?) ----
# Find a text spoken in 2 different sessions (closed_vocab is small, often repeats)
text_to_paths = {}
for text, p, s in items:
    text_to_paths.setdefault(text.lower(), []).append((text, p, s))
repeated = [(t, lst) for t, lst in text_to_paths.items() if len(lst) >= 2 and len(t.split()) <= 5]
if repeated:
    text, lst = repeated[0]
    plot_grid([(f'session {lst[0][2]}', lst[0][0], lst[0][1]),
               (f'session {lst[1][2]}', lst[1][0], lst[1][1])],
              f'{OUT}/02_same_text_two_sessions.png',
              f'Voiced EMG — same words spoken on different days')
    print('saved 02_same_text_two_sessions.png')

# ---- 3. Long-vs-short side-by-side (intuition: longer = more sustained activity) ----
short_pick = next(it for it in short if len(it[0]) > 5)
long_pick = next(it for it in long_)
plot_grid([('SHORT', short_pick[0], short_pick[1]),
           ('LONG',  long_pick[0],  long_pick[1])],
          f'{OUT}/03_short_vs_long.png',
          'Voiced EMG — short utterance vs long utterance')
print('saved 03_short_vs_long.png')

# ---- 4. Phonetically different short utterances ----
# Try to pick one heavy on lip closures (b/p/m) vs one heavy on tongue/throat (t/k/g)
def has_phonemes(text, chars):
    t = text.lower()
    return sum(t.count(c) for c in chars)

lippy = sorted(short, key=lambda x: -has_phonemes(x[0], 'bpmf'))[:5]
tonguey = sorted(short, key=lambda x: -has_phonemes(x[0], 'tkg'))[:5]
if lippy and tonguey:
    plot_grid([('"lippy" (b/p/m/f)', lippy[0][0], lippy[0][1]),
               ('"tonguey" (t/k/g)', tonguey[0][0], tonguey[0][1])],
              f'{OUT}/04_lippy_vs_tonguey.png',
              'Voiced EMG — phonetically different short utterances')
    print('saved 04_lippy_vs_tonguey.png')

# ---- 5. Single closer-look plot of one utterance with envelope ----
text_pick, emg_path_pick, _ = picks[0]
emg = np.load(emg_path_pick).astype(np.float32)
t = np.arange(emg.shape[0]) / SR
fig, axes = plt.subplots(8, 1, figsize=(13, 9), sharex=True)
for c in range(8):
    raw = emg[:, c]
    # smooth envelope: rectify + 50ms moving avg
    env = np.convolve(np.abs(raw), np.ones(50)/50, mode='same')
    axes[c].plot(t, raw, color='lightgray', lw=0.4, label='raw')
    axes[c].plot(t, env, color='crimson', lw=1.0, label='envelope (50ms)')
    axes[c].grid(alpha=0.25)
    axes[c].set_ylabel(f'ch{c}', rotation=0, ha='right', va='center')
    axes[c].set_yticks([])
axes[0].legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('time (s)')
fig.suptitle(f'Voiced EMG with rectified envelope — "{text_pick}"', fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/05_envelope_overlay.png', dpi=110); plt.close()
print('saved 05_envelope_overlay.png')

print('\nAll plots in', OUT)
