"""Spectral equalization on raw silent EMG -> voiced PSD profile.

Baseline:  voiced-trained encoder on silent val features (no preprocessing)
Equalized: voiced-trained encoder on silent val features extracted from
           silent raw EMG that has been spectrally reshaped to match
           voiced's average PSD per channel.
"""
import os, sys, json, glob, time
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps
sys.path.insert(0, 'src')
from features import extract_features
from model import EMGEncoder

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SR = 1000
NPERSEG = 1024
CKPT = 'models/best_voiced.pt'
AUDIO_CACHE = 'dataset/.audio_embed_cache'

VAL_SILENT_SESSIONS = ['silent_parallel_data/5-8_silent', 'silent_parallel_data/5-9_silent']
TRAIN_PAIRED_SESSIONS = ['5-4', '5-5', '5-6', '5-10', '5-11']

HIDDEN, N_LAYERS, N_HEADS, TIME_STRIDE = 384, 4, 6, 4

def list_session(d):
    out = []
    if not os.path.isdir(d):
        return out
    for info_path in sorted(glob.glob(os.path.join(d, '*_info.json'))):
        try:
            text = json.load(open(info_path)).get('text', '').strip()
        except Exception:
            continue
        if not text:
            continue
        idx = os.path.basename(info_path).replace('_info.json', '')
        emg_path = os.path.join(d, f'{idx}_emg.npy')
        audio_path = os.path.join(d, f'{idx}_audio_clean.flac')
        ac = os.path.join(AUDIO_CACHE, os.path.relpath(audio_path, 'dataset').replace(os.sep, '_') + '.npy')
        if os.path.exists(emg_path) and os.path.exists(ac):
            out.append((emg_path, ac, text))
    return out

def avg_psd(emg_paths, label):
    print(f'  averaging PSD over {len(emg_paths)} {label} utterances...')
    sys.stdout.flush()
    acc = None; n = 0
    freqs = None
    for p in emg_paths:
        emg = np.load(p).astype(np.float32)
        if emg.shape[0] < NPERSEG:
            continue
        for c in range(8):
            f, P = sps.welch(emg[:, c], fs=SR, nperseg=NPERSEG)
            if acc is None:
                acc = np.zeros((8, len(P)), dtype=np.float64)
                freqs = f
            acc[c] += P
        n += 1
    return freqs, acc / max(n, 1), n

def spectral_equalize(emg, filter_response_freqs, filter_response):
    T = emg.shape[0]
    spec = np.fft.rfft(emg, axis=0)
    rfft_freqs = np.fft.rfftfreq(T, d=1.0/SR)
    eq = np.zeros((spec.shape[0], 8), dtype=np.float32)
    for c in range(8):
        eq[:, c] = np.interp(rfft_freqs, filter_response_freqs, filter_response[c])
    spec_eq = spec * eq
    return np.fft.irfft(spec_eq, n=T, axis=0).astype(np.float32)

print('Indexing sessions...'); sys.stdout.flush()
voiced_paths_train, silent_paths_train = [], []
for s in TRAIN_PAIRED_SESSIONS:
    voiced_paths_train.extend([x[0] for x in list_session(f'dataset/voiced_parallel_data/{s}')])
    silent_paths_train.extend([x[0] for x in list_session(f'dataset/silent_parallel_data/{s}_silent')])
print(f'  voiced train EMG files: {len(voiced_paths_train)}')
print(f'  silent train EMG files: {len(silent_paths_train)}')
sys.stdout.flush()

silent_val = []
for s in VAL_SILENT_SESSIONS:
    silent_val.extend(list_session(f'dataset/{s}'))
print(f'  silent val items (with audio cache): {len(silent_val)}')
sys.stdout.flush()

t0 = time.time()
freqs_v, P_voiced, nv = avg_psd(voiced_paths_train, 'voiced')
freqs_s, P_silent, ns = avg_psd(silent_paths_train, 'silent')
filter_response = np.sqrt(P_voiced / np.maximum(P_silent, 1e-12)).astype(np.float32)
print(f'  PSD build took {time.time()-t0:.1f}s')
mid = (freqs_v >= 30) & (freqs_v < 150)
print('  per-channel filter gain (mean over 30-150 Hz):')
for c in range(8):
    print(f'    ch{c}: {filter_response[c, mid].mean():.3f}')
sys.stdout.flush()

print('Loading encoder...'); sys.stdout.flush()
enc = EMGEncoder(hidden=HIDDEN, n_layers=N_LAYERS, n_heads=N_HEADS,
                 audio_embed_dim=1536, time_stride=TIME_STRIDE).to(DEVICE)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
enc.load_state_dict(state)
enc.train(False)
print(f'  loaded {CKPT}')

print('Computing norm stats from voiced training features...'); sys.stdout.flush()
from dataset import _load_session
voiced_train_feats = []
for s in TRAIN_PAIRED_SESSIONS:
    for feats, text, _ in _load_session(f'dataset/voiced_parallel_data/{s}'):
        voiced_train_feats.append(feats)
for sub in ['nonparallel_data']:
    base = f'dataset/{sub}'
    if os.path.isdir(base):
        for s in sorted(os.listdir(base)):
            sd = f'{base}/{s}'
            if os.path.isdir(sd):
                for feats, _, _ in _load_session(sd):
                    voiced_train_feats.append(feats)
print(f'  using {len(voiced_train_feats)} voiced train utterances for stats'); sys.stdout.flush()
all_feats = np.concatenate(voiced_train_feats)
mean = all_feats.mean(0); std = all_feats.std(0) + 1e-8
sys.stdout.flush()

def encode_and_score(emg_raw, audio_emb):
    with torch.no_grad():
        feats = extract_features(emg_raw)
        feats_n = (feats - mean) / std
        emg_t = torch.tensor(feats_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        lens = torch.tensor([feats_n.shape[0]])
        out, out_lens = enc(emg_t, lens)
        e = out_lens[0].item()
        if e == 0:
            return None
        emg_seq = out[0, :e]
        audio_seq = torch.tensor(audio_emb, dtype=torch.float32).to(DEVICE)
        a = audio_seq.shape[0]
        emg_i = F.interpolate(
            emg_seq.unsqueeze(0).transpose(1, 2),
            size=a, mode='linear', align_corners=False,
        ).transpose(1, 2).squeeze(0)
        return F.cosine_similarity(emg_i, audio_seq, dim=-1).mean().item()

print('\nEvaluating silent val set...'); sys.stdout.flush()
base_sims, eq_sims = [], []
t0 = time.time()
for i, (emg_path, ac_path, text) in enumerate(silent_val):
    emg = np.load(emg_path).astype(np.float32)
    audio = np.load(ac_path).astype(np.float32)
    if emg.ndim != 2 or emg.shape[1] != 8:
        continue
    s_base = encode_and_score(emg, audio)
    emg_eq = spectral_equalize(emg, freqs_v, filter_response)
    s_eq = encode_and_score(emg_eq, audio)
    if s_base is None or s_eq is None:
        continue
    base_sims.append(s_base); eq_sims.append(s_eq)
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{len(silent_val)}  base={np.mean(base_sims):.4f}  eq={np.mean(eq_sims):.4f}  ({time.time()-t0:.0f}s)')
        sys.stdout.flush()

base_sims = np.array(base_sims); eq_sims = np.array(eq_sims)
delta = eq_sims - base_sims
print('\n' + '='*60)
print(f'Silent val (n={len(base_sims)})')
print(f'  baseline cos_sim       : mean={base_sims.mean():.4f}  median={np.median(base_sims):.4f}')
print(f'  spectrally-eq cos_sim  : mean={eq_sims.mean():.4f}  median={np.median(eq_sims):.4f}')
print(f'  delta (eq - base)      : mean={delta.mean():+.4f}  median={np.median(delta):+.4f}')
print(f'  improved on            : {(delta > 0).mean()*100:.1f}% of utterances')
print('='*60)

os.makedirs('analysis/silent_vs_voiced', exist_ok=True)
np.savez('analysis/silent_vs_voiced/spectral_eq_results.npz',
         base=base_sims, eq=eq_sims, freqs=freqs_v,
         P_voiced=P_voiced, P_silent=P_silent, filter_response=filter_response)
print('saved analysis/silent_vs_voiced/spectral_eq_results.npz')
