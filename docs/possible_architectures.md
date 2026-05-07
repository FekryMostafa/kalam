# Possible architectures

Each architecture is checked against the principles in [`first_principles.md`](first_principles.md):

- **P** — phoneme intermediate (claims 4–6, 19–20)
- **W** — encoder integrates context window across channels, sequence shorter than input (claim 11)
- **AR** — each decoded output conditions on previously decoded output (claim 18)
- **TXT** — optimizes text-level correctness in addition to phoneme accuracy (claim 21)

Notation: ✓ satisfies, ~ partial / configurable, ✗ violates.

## Custom encoder–decoder

### A1. AR attention encoder–decoder (LAS / Whisper)
Encoder + autoregressive decoder with cross-attention.
✓ W ✓ AR ~ TXT (text loss only) ✗ P (no explicit phoneme stage; output vocab is BPE/char)
Sources: [Chan 2015](https://arxiv.org/abs/1508.01211), [Whisper 2022](https://arxiv.org/abs/2212.04356).

### A2. Hybrid CTC + AR attention (ESPnet)
Shared encoder, two heads: CTC at frame level + autoregressive attention decoder. Loss `λ·CTC + (1-λ)·AED`.
✓ W ✓ AR ✓ TXT ~ P (output vocab configurable; phonemes possible but not separately decoded)
Sources: [Watanabe 2017](https://www.merl.com/publications/docs/TR2017-190.pdf).

### A3. Deliberation / two-pass
First pass produces hypothesis. Second-pass AR decoder cross-attends to both encoder features and full first-pass tokens.
✓ W ✓ AR ✓ TXT ~ P (first pass can emit phonemes)
Sources: [Hu 2020](https://arxiv.org/abs/2003.07962), [Transformer Deliberation 2021](https://arxiv.org/abs/2101.11577).

## Transducer family

### B1. RNN-T / Conformer-T / Transformer-T
Encoder + autoregressive predictor over output tokens + joiner.
✓ W ✓ AR ✓ TXT ~ P (vocab-agnostic; weak internal LM)
Sources: [Graves 2012](https://arxiv.org/abs/1211.3711), [Gulati 2020](https://arxiv.org/abs/2005.08100).

### B2. Factorized Neural Transducer (FNT)
Transducer with a dedicated AR LM head, separate from the blank head; adaptable on text-only data.
✓ W ✓ AR ✓ TXT ~ P
Sources: [FNT 2022](https://arxiv.org/abs/2110.01500).

### B3. Aligner-Encoders / Self-Transducers
Encoder self-aligns, no cross-attention; AR text-only predictor scans encoder frames in order.
✓ W ✓ AR ✓ TXT ~ P
Sources: [NeurIPS 2024](https://arxiv.org/abs/2502.05232).

## LLM as decoder

### C1. LLM with cross-attention adapter (Flamingo style)
Insert gated cross-attention layers into a frozen LLM that attend to encoder features. LLM is the AR decoder.
✓ W ✓ AR ✓ TXT (LLM language prior baked in) ~ P (phoneme can be auxiliary CTC head on encoder)
Sources: [Whisper-Flamingo 2024](https://arxiv.org/abs/2406.10082), [Audio Flamingo 3](https://arxiv.org/abs/2406.10082).

### C2. LLM with prefix / soft-prompt adapter
Encoder features projected as soft tokens, prepended to the LLM context. LLM decodes AR conditioned on prefix.
✓ W ✓ AR ~ TXT ✗ P (no phoneme intermediate; encoder feeds LLM directly)
Sources: [SLAM-ASR 2024](https://github.com/X-LANCE/SLAM-LLM), [SALMONN](https://proceedings.iclr.cc/paper_files/paper/2024/file/476ab8f369e489c04187ba84f68cfa68-Paper-Conference.pdf), [SilentSpeechLLM 2025](https://aclanthology.org/2025.acl-short.56.pdf).

### C3. End-to-end fine-tuned LLM decoder
Encoder + LLM trained jointly end-to-end as the AR decoder. Encoder may produce discrete tokens or continuous features.
✓ W ✓ AR ✓ TXT ~ P
Sources: [Qwen2-Audio 2024](https://arxiv.org/pdf/2407.10759), [SpeechGPT 2023](https://arxiv.org/abs/2305.11000).

## LLM in decoding loop

### D1. Shallow / deep / cold / density-ratio fusion
Custom encoder/decoder, LM scores added to beam-extension log-probs (or hidden states fused) at decode time.
✓ W ✓ AR ✓ TXT ~ P
Sources: [Toshniwal 2018](https://arxiv.org/pdf/1807.10857), [ILME 2020](https://arxiv.org/abs/2011.01991), [Density Ratio 2020](https://openreview.net/pdf?id=aWKX3wmhHnL).

### D2. Delayed Fusion
Large LLMs integrated into first-pass beam decoding with delay to amortize cost.
✓ W ✓ AR ✓ TXT ~ P
Sources: [arXiv Jan 2025](https://arxiv.org/html/2501.09258v1).

### D3. Speculative decoding inversions
Small ASR drafts tokens, large LLM verifies. Or LLM proposes, encoder verifies.
✓ W ✓ AR ✓ TXT ~ P
Sources: [SpecASR Jul 2025](https://arxiv.org/abs/2507.18181), [Mirror SpecDec Apple 2025](https://machinelearning.apple.com/research/mirror).

## Two-stage phoneme + AR LLM (most aligned with our principles)

### E1. LLM-P2G
Audio → CTC encoder → phonemes → fine-tuned LLM AR-generates text from phoneme sequence. Trained with noisy-phoneme augmentation.
✓ W ✓ AR ✓ TXT ✓ P
Sources: [Ma 2025 Interspeech](https://arxiv.org/abs/2506.04711).

### E2. VALLR (lipreading precedent)
Video Transformer + CTC → phoneme sequence → fine-tuned Llama-3.2-3B AR-decodes text. 22.1% WER on LRS3.
✓ W ✓ AR ✓ TXT ✓ P
Sources: [VALLR ICCV 2025](https://arxiv.org/abs/2503.21408).

### E3. DCoND-LIFT
Diphone CTC encoder + LLM ensemble fusion (LLM operates on N-best, not raw frames). Brain-to-Text 2024 winner, 5.77% WER.
✓ W ✓ AR ✓ TXT ✓ P
Sources: [B2T 2024 winner](https://github.com/CIBR-Okubo-Lab/speechBCI_2024), [Willett 2024 benchmark](https://arxiv.org/abs/2412.17227).

## Iterative refinement / non-AR with full output context

### F1. Mask-CTC / Mask-Predict / Imputer / Levenshtein
Initialize from CTC or template, mask low-confidence tokens, refine. Each refinement step conditions on full current output.
✓ W ~ AR (per refinement, not per token) ~ TXT ~ P
Sources: [Mask-CTC 2020](https://arxiv.org/abs/2005.08700).

### F2. Diffusion text decoders
Discrete diffusion LLM as decoder. Each denoising step conditions on entire current noisy output and audio.
✓ W ~ AR (per step, not per token) ✓ TXT (LLM prior) ~ P
Sources: [Whisfusion 2025](https://arxiv.org/html/2508.07048v1), [dLLM-ASR 2025](https://arxiv.org/html/2601.17902), [Audio-Conditioned Diffusion LLM 2025](https://arxiv.org/pdf/2509.16622).

## Retrieval / energy / MoE

### G1. Retrieval-augmented decoding
Token-level speech datastore queried at decode time; retrieved neighbors contribute to next-token distribution.
✓ W ~ AR (prefix-only conditioning) ✓ TXT ~ P
Sources: [LA-RAG 2024](https://arxiv.org/html/2409.08597v1), [kNN-CTC 2023](https://arxiv.org/html/2406.09618).

### G2. Energy-based / score-based rescorers
Sequence-level scorer trained with NCE or contrastive; used for rescoring rather than primary decoding.
✗ AR (rescoring only, not in-loop)
Sources: [Bakhtin 2020](https://openreview.net/forum?id=B1l4SgHKDH).

### G3. MoE decoders with output-conditional routing
Decoder routes tokens to experts based on current state. AR with conditional compute.
✓ W ✓ AR ✓ TXT ~ P
Sources: [LR-MoE 2023](https://ar5iv.labs.arxiv.org/html/2307.05956), [SC-MoE 2024](https://www.isca-archive.org/interspeech_2024/ye24_interspeech.pdf).

## Ruled out

These violate claim 18 (decoder must condition on previously decoded output during decoding):

- Pure CTC decoder + dictionary lookup
- Pure CTC decoder + LLM rescoring of completed N-best (MONA LISA shape; LLM sees committed beams, not in-loop)
- Pure non-autoregressive decoder with independent token outputs (Paraformer)
- Encoder-only systems with no decoder

## Strongest match to all four principles

E1, E2, E3 all satisfy all four constraints. E2 (VALLR) is the published precedent closest to our setup: a custom encoder + CTC → phoneme sequence → fine-tuned LLM autoregressively produces text. It has been demonstrated for lipreading in 2025 with 22.1% WER on LRS3. No equivalent exists for EMG.
