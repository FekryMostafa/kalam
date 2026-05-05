# Findings

## Silent vs Voiced EMG

Comparison on 1,584 paired utterances: silent/voiced amplitude ratio is 0.59-0.84 across channels, with `larynx` lowest at 0.59. Silent/voiced duration median is 0.914. Silent/voiced PSD ratio in 30-150 Hz is 0.24-0.81 across channels; in 150-300 Hz, `larynx` is 0.10. Mean absolute difference in cross-channel correlation matrices is 0.054. Median paired raw Pearson r is near 0 for all channels except `larynx` at 0.21. [`logs/silent_vs_voiced_comparison.json`](../logs/silent_vs_voiced_comparison.json)

Spectral equalization experiment: filter gain in 30-150 Hz is 1.17-3.82 across channels, with `larynx` at 3.82. On 438 silent validation utterances, cosine similarity changed from 0.3443 (baseline mean) to 0.2526 (equalized mean). Delta mean is -0.0917. Improvement rate is 0.7%. [`logs/spectral_equalization.json`](../logs/spectral_equalization.json)
