"""Side-by-side analysis of voiced vs silent EMG for the SAME utterance text.

Outputs PNGs + a summary.md to analysis/silent_vs_voiced/.
"""
import os, json, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps

OUT = 'analysis/silent_vs_voiced'
os.makedirs(OUT, exist_ok=True)
SR = 1000  # raw EMG sample rate
CHANNELS = ['lips_L', 'chin_L', 'floor_of_mouth', 'larynx', 'jaw_R', 'lips_R', 'upper_face', 'masseter']

def load_text(info_path):
    try:
        with open(info_path) as f:
            return json.load(f).get('text', '').strip()
    except Exception:
        return ''

def index_session(dir_path):
    """Return {text: emg_path} for a session directory."""
    out = {}
    for info_path in glob.glob(os.path.join(dir_path, '*_info.json')):
        t = load_text(info_path)
        if not t:
            continue
        idx = os.path.basename(info_path).replace('_info.json', '')
        emg = os.path.join(dir_path, f'{idx}_emg.npy')
        if os.path.exists(emg):
            out[t] = emg
    return out

# ---------- find pairs ----------
sessions = ['5-4', '5-5', '5-6', '5-8', '5-9', '5-10', '5-11']
pairs = []  # (text, voiced_path, silent_path, session)
for s in sessions:
    v = index_session(f'dataset/voiced_parallel_data/{s}')
    si = index_session(f'dataset/silent_parallel_data/{s}_silent')
    common = set(v) & set(si)
    for t in common:
        pairs.append((t, v[t], si[t], s))
print(f'Found {len(pairs)} paired voiced/silent utterances across {len(sessions)} sessions')

# Sort by text length for sampling
pairs.sort(key=lambda p: len(p[0]))
sys.stdout.flush()

# ---------- aggregate stats over all pairs ----------
def basic_stats(arr):
    return dict(
        per_ch_std=arr.std(axis=0),       # (8,)
        per_ch_mean=arr.mean(axis=0),
        per_ch_absmean=np.abs(arr).mean(axis=0),
        T=arr.shape[0],
    )

agg_v_std, agg_s_std = [], []
agg_v_abs, agg_s_abs = [], []
durations_v, durations_s = [], []

for text, vp, sp, s in pairs:
    v = np.load(vp).astype(np.float32)
    si = np.load(sp).astype(np.float32)
    if v.shape[1] != 8 or si.shape[1] != 8:
        continue
    sv, ss = basic_stats(v), basic_stats(si)
    agg_v_std.append(sv['per_ch_std']); agg_s_std.append(ss['per_ch_std'])
    agg_v_abs.append(sv['per_ch_absmean']); agg_s_abs.append(ss['per_ch_absmean'])
    durations_v.append(sv['T']); durations_s.append(ss['T'])

agg_v_std = np.array(agg_v_std); agg_s_std = np.array(agg_s_std)
agg_v_abs = np.array(agg_v_abs); agg_s_abs = np.array(agg_s_abs)
durations_v = np.array(durations_v); durations_s = np.array(durations_s)

print(f'Aggregated {len(agg_v_std)} pairs')

# ---------- PLOT 1: Per-channel amplitude (std) ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(8)
v_med = np.median(agg_v_std, 0); s_med = np.median(agg_s_std, 0)
v_q1, v_q3 = np.percentile(agg_v_std, [25, 75], 0)
s_q1, s_q3 = np.percentile(agg_s_std, [25, 75], 0)
ax = axes[0]
w = 0.35
ax.bar(x - w/2, v_med, w, yerr=[v_med - v_q1, v_q3 - v_med], label='Voiced', color='steelblue', capsize=3)
ax.bar(x + w/2, s_med, w, yerr=[s_med - s_q1, s_q3 - s_med], label='Silent', color='crimson', capsize=3)
ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
ax.set_ylabel('std of raw EMG (a.u.)'); ax.set_title('Per-channel amplitude (median ± IQR over pairs)')
ax.legend(); ax.grid(alpha=0.3)
# ratio
ax = axes[1]
ratio = agg_s_std / np.maximum(agg_v_std, 1e-6)
ax.boxplot([ratio[:, i] for i in range(8)], labels=CHANNELS, showfliers=False)
ax.axhline(1.0, color='k', ls='--', alpha=0.5)
ax.set_ylabel('silent / voiced std ratio'); ax.set_title('Silent-to-voiced amplitude ratio per channel')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/01_per_channel_amplitude.png', dpi=120); plt.close()
print('  saved 01_per_channel_amplitude.png')

# ---------- PLOT 2: Duration distribution ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(durations_v / SR, bins=40, alpha=0.6, label='Voiced', color='steelblue')
ax.hist(durations_s / SR, bins=40, alpha=0.6, label='Silent', color='crimson')
ax.set_xlabel('duration (seconds)'); ax.set_ylabel('count')
ax.set_title('Utterance durations'); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
ratio_dur = durations_s / np.maximum(durations_v, 1)
ax.hist(ratio_dur, bins=40, color='purple', alpha=0.7)
ax.axvline(1.0, color='k', ls='--')
ax.axvline(np.median(ratio_dur), color='red', ls='-', label=f'median={np.median(ratio_dur):.3f}')
ax.set_xlabel('silent / voiced duration ratio'); ax.set_ylabel('count')
ax.set_title('Same-text duration ratio (silent vs voiced)'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/02_duration.png', dpi=120); plt.close()
print('  saved 02_duration.png')

# ---------- PLOT 3 & 4: example pair traces and spectra ----------
# pick 3 pairs of varying length
sample_pairs = [pairs[len(pairs)//4], pairs[len(pairs)//2], pairs[3*len(pairs)//4]]

def time_align(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

def plot_traces(text, v, si, fname):
    fig, axes = plt.subplots(8, 2, figsize=(14, 12), sharex='col')
    t_v = np.arange(len(v)) / SR
    t_s = np.arange(len(si)) / SR
    for c in range(8):
        axes[c, 0].plot(t_v, v[:, c], color='steelblue', lw=0.5)
        axes[c, 0].set_ylabel(CHANNELS[c])
        axes[c, 1].plot(t_s, si[:, c], color='crimson', lw=0.5)
        for ax in axes[c]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_title(f'VOICED — {len(v)/SR:.2f}s')
    axes[0, 1].set_title(f'SILENT — {len(si)/SR:.2f}s')
    axes[-1, 0].set_xlabel('time (s)'); axes[-1, 1].set_xlabel('time (s)')
    fig.suptitle(f'"{text[:80]}"', fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()

def plot_spectra(text, v, si, fname):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for c in range(8):
        ax = axes[c // 4, c % 4]
        fv, Pv = sps.welch(v[:, c], fs=SR, nperseg=min(1024, len(v)))
        fs_, Ps = sps.welch(si[:, c], fs=SR, nperseg=min(1024, len(si)))
        ax.semilogy(fv, Pv, color='steelblue', label='voiced', lw=1)
        ax.semilogy(fs_, Ps, color='crimson', label='silent', lw=1)
        ax.set_title(CHANNELS[c]); ax.set_xlim(0, 500); ax.grid(alpha=0.3)
        if c == 0: ax.legend()
        if c // 4 == 1: ax.set_xlabel('Hz')
        if c % 4 == 0: ax.set_ylabel('PSD')
    fig.suptitle(f'Power spectrum — "{text[:80]}"', fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()

for i, (text, vp, sp, s) in enumerate(sample_pairs):
    v = np.load(vp).astype(np.float32)
    si = np.load(sp).astype(np.float32)
    plot_traces(text, v, si, f'{OUT}/03_traces_pair{i}.png')
    plot_spectra(text, v, si, f'{OUT}/04_spectra_pair{i}.png')
    print(f'  saved traces+spectra pair{i}')

# ---------- PLOT 5: Aggregated PSD across pairs ----------
print('Computing aggregated PSDs...')
N_FFT = 1024
freqs = None
psd_v = np.zeros((8, N_FFT // 2 + 1))
psd_s = np.zeros((8, N_FFT // 2 + 1))
n_used = 0
for text, vp, sp, s in pairs[:300]:  # cap for speed
    v = np.load(vp).astype(np.float32)
    si = np.load(sp).astype(np.float32)
    if min(len(v), len(si)) < N_FFT:
        continue
    for c in range(8):
        f, Pv = sps.welch(v[:, c], fs=SR, nperseg=N_FFT)
        _, Ps = sps.welch(si[:, c], fs=SR, nperseg=N_FFT)
        psd_v[c] += Pv; psd_s[c] += Ps
    if freqs is None: freqs = f
    n_used += 1
psd_v /= n_used; psd_s /= n_used
print(f'  averaged over {n_used} pairs')

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for c in range(8):
    ax = axes[c // 4, c % 4]
    ax.semilogy(freqs, psd_v[c], color='steelblue', label='voiced', lw=1.2)
    ax.semilogy(freqs, psd_s[c], color='crimson', label='silent', lw=1.2)
    ax.set_title(CHANNELS[c]); ax.set_xlim(0, 500); ax.grid(alpha=0.3)
    if c == 0: ax.legend()
    if c // 4 == 1: ax.set_xlabel('Hz')
    if c % 4 == 0: ax.set_ylabel('avg PSD')
fig.suptitle(f'Average power spectrum across {n_used} paired utterances', fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/05_avg_spectrum.png', dpi=120); plt.close()
print('  saved 05_avg_spectrum.png')

# Spectral ratio
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for c in range(8):
    ax.plot(freqs, psd_s[c] / np.maximum(psd_v[c], 1e-9), label=CHANNELS[c], lw=1)
ax.axhline(1.0, color='k', ls='--')
ax.set_xlim(0, 500); ax.set_yscale('log')
ax.set_xlabel('Hz'); ax.set_ylabel('silent / voiced PSD')
ax.set_title('Spectral ratio (>1 means silent has more power at that frequency)')
ax.legend(ncol=4); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/06_spectral_ratio.png', dpi=120); plt.close()
print('  saved 06_spectral_ratio.png')

# ---------- PLOT 7: Cross-channel correlation ----------
def corr_mat(x):
    return np.corrcoef(x.T)
corrs_v, corrs_s = [], []
for text, vp, sp, s in pairs[:300]:
    v = np.load(vp).astype(np.float32)
    si = np.load(sp).astype(np.float32)
    corrs_v.append(corr_mat(v)); corrs_s.append(corr_mat(si))
corrs_v = np.mean(corrs_v, 0); corrs_s = np.mean(corrs_s, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
im0 = axes[0].imshow(corrs_v, vmin=-1, vmax=1, cmap='RdBu_r')
axes[0].set_title('Voiced cross-ch correlation'); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(corrs_s, vmin=-1, vmax=1, cmap='RdBu_r')
axes[1].set_title('Silent cross-ch correlation'); plt.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(corrs_s - corrs_v, vmin=-0.3, vmax=0.3, cmap='RdBu_r')
axes[2].set_title('Silent − Voiced'); plt.colorbar(im2, ax=axes[2])
for ax in axes:
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
plt.tight_layout(); plt.savefig(f'{OUT}/07_cross_channel_corr.png', dpi=120); plt.close()
print('  saved 07_cross_channel_corr.png')

# ---------- PLOT 8: Spectrogram side-by-side ----------
text, vp, sp, s = sample_pairs[1]
v = np.load(vp).astype(np.float32); si = np.load(sp).astype(np.float32)
fig, axes = plt.subplots(8, 2, figsize=(14, 14), sharey=True)
for c in range(8):
    f, t, Sv = sps.spectrogram(v[:, c], fs=SR, nperseg=256, noverlap=192)
    f, t2, Ss = sps.spectrogram(si[:, c], fs=SR, nperseg=256, noverlap=192)
    axes[c, 0].pcolormesh(t, f, 10*np.log10(Sv + 1e-12), shading='auto', cmap='viridis')
    axes[c, 0].set_ylabel(f'{CHANNELS[c]}\nHz'); axes[c, 0].set_ylim(0, 400)
    axes[c, 1].pcolormesh(t2, f, 10*np.log10(Ss + 1e-12), shading='auto', cmap='viridis')
    axes[c, 1].set_ylim(0, 400)
axes[0, 0].set_title('VOICED'); axes[0, 1].set_title('SILENT')
axes[-1, 0].set_xlabel('s'); axes[-1, 1].set_xlabel('s')
fig.suptitle(f'Spectrograms — "{text[:80]}"', fontsize=10)
plt.tight_layout(); plt.savefig(f'{OUT}/08_spectrogram_pair.png', dpi=120); plt.close()
print('  saved 08_spectrogram_pair.png')

# ---------- PLOT 9: Pearson correlation of paired raw signals (after length match) ----------
print('Per-pair Pearson r per channel...')
prs = []
for text, vp, sp, s in pairs[:300]:
    v = np.load(vp).astype(np.float32); si = np.load(sp).astype(np.float32)
    a, b = time_align(v, si)
    rs = []
    for c in range(8):
        if a[:, c].std() < 1e-6 or b[:, c].std() < 1e-6:
            rs.append(np.nan)
        else:
            rs.append(np.corrcoef(a[:, c], b[:, c])[0, 1])
    prs.append(rs)
prs = np.array(prs)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.boxplot([prs[~np.isnan(prs[:, c]), c] for c in range(8)], labels=CHANNELS, showfliers=False)
ax.axhline(0, color='k', ls='--')
ax.set_ylabel('Pearson r (paired voiced vs silent, head-aligned)')
ax.set_title('Per-channel correlation of paired raw signals (NOT a fair test of similarity, just informative)')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/09_paired_pearson.png', dpi=120); plt.close()
print('  saved 09_paired_pearson.png')

# ---------- summary.md ----------
v_amp_med = np.median(agg_v_std, 0)
s_amp_med = np.median(agg_s_std, 0)
amp_ratio = s_amp_med / v_amp_med
dur_ratio = np.median(durations_s / durations_v)

# spectral band powers
def band(psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return psd[:, m].mean(1)
bands = [(0, 30, 'low'), (30, 150, 'mid'), (150, 300, 'high')]
band_lines = []
for lo, hi, name in bands:
    bv = band(psd_v, lo, hi); bs = band(psd_s, lo, hi)
    band_lines.append(f'  - {name} ({lo}-{hi}Hz): silent/voiced ratio per ch = ' +
                      ', '.join(f'{x:.2f}' for x in bs/bv))

with open(f'{OUT}/summary.md', 'w') as f:
    f.write(f'''# Voiced vs Silent EMG — Same-Text Comparison

**Pairs analyzed:** {len(pairs)} utterances spoken both voiced and silently across sessions {sessions}.

## Headline numbers

- **Amplitude (std) per channel — silent / voiced median ratio:**
  lips_L={amp_ratio[0]:.2f}, chin_L={amp_ratio[1]:.2f}, floor_of_mouth={amp_ratio[2]:.2f}, larynx={amp_ratio[3]:.2f}, jaw_R={amp_ratio[4]:.2f}, lips_R={amp_ratio[5]:.2f}, upper_face={amp_ratio[6]:.2f}, masseter={amp_ratio[7]:.2f}
- **Duration ratio (silent/voiced):** median = **{dur_ratio:.3f}** ({(1-dur_ratio)*100:.1f}% shorter on silent on average)
- **Cross-channel correlation matrix difference (silent−voiced):** mean abs = {np.mean(np.abs(corrs_s-corrs_v)):.3f}
- **Paired Pearson r per channel (head-aligned, raw):** ''' + ', '.join(f'{CHANNELS[c]}={np.nanmedian(prs[:,c]):.2f}' for c in range(8)) + '''

## Spectral band powers (silent/voiced ratio)

''' + '\n'.join(band_lines) + f'''

## Files

| file | what it shows |
|---|---|
| 01_per_channel_amplitude.png | bar chart of per-channel std + box of silent/voiced ratio |
| 02_duration.png | duration histograms + same-text duration ratio |
| 03_traces_pair*.png | raw 8-channel time traces, voiced vs silent, same text |
| 04_spectra_pair*.png | per-channel Welch PSD for that example pair |
| 05_avg_spectrum.png | PSD averaged over {n_used} pairs |
| 06_spectral_ratio.png | silent/voiced PSD ratio per channel — where the spectral gap lives |
| 07_cross_channel_corr.png | cross-channel correlation matrices, voiced vs silent vs diff |
| 08_spectrogram_pair.png | spectrograms for one pair, 8 channels stacked |
| 09_paired_pearson.png | per-channel Pearson r of paired raw signals (head-aligned) |

## How to read this

The encoder hits a wall going voiced→silent. Three things to look for:
1. **Amplitude shift** (plot 01): if silent is uniformly louder/quieter, BatchNorm running stats explain part of the gap.
2. **Spectral shift** (plots 05, 06): if silent has different power at certain frequency bands, the convolutional stem will respond differently.
3. **Cross-channel structure** (plot 07): if silent has different inter-electrode correlations, it means a different motor program — the model can't fix that with normalization.

The per-pair Pearson (plot 09) is informative but NOT a fair similarity score because voiced and silent have different durations and onsets.
''')
print(f'\nDone. See {OUT}/')
