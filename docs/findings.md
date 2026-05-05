# Findings

## Silent vs Voiced EMG

Comparison (1,584 paired utterances): silent is quieter on every channel (0.59–0.84× voiced std, ch3 lowest), 9% shorter, has 0.24–0.81× voiced PSD in 30–150 Hz (ch3 collapses to 0.10× at 150–300 Hz). Cross-channel correlation structure preserved (Δ=0.054). Paired raw waveforms uncorrelated (Pearson r ≈ 0 except ch3 = 0.21). → [`logs/silent_vs_voiced_comparison.json`](../logs/silent_vs_voiced_comparison.json)

Spectral equalization experiment: built a per-channel filter from voiced/silent average PSDs (boosts each frequency in silent up to voiced's level, e.g. ch3 mid-band ×3.8), applied it to silent raw EMG, ran the voiced-trained encoder. Silent val cos_sim dropped 0.344 → 0.253 (worse on 99.3% of utterances). → [`logs/spectral_equalization.json`](../logs/spectral_equalization.json)
