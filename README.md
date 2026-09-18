# Kalam

Silent speech interface: read EMG signals from face muscles and output text. Built so people who cannot produce audible speech can still communicate.

## Mission

Affects ~4M Americans (laryngectomy, ALS, dysarthria, post-stroke). Today's options are typing or AAC devices. A wearable that turns silent mouthing into text — no audio, no visible movement required — is a cleaner interface for people whose articulators still work but whose voice doesn't.

## Architecture

```
silent EMG (8 channels @ 1 kHz)
       │
       ▼
   encoder              Conformer + collapsed-inventory CTC
       │                outputs ~31-class phoneme posteriors
       │                with voicing-pair ambiguity baked in
       │
       ▼
phoneme posteriors      e.g. K AE (T:0.5/D:0.5) -- voicing pairs
with voicing            stay 50/50 because they are physically
alternatives            indistinguishable from silent EMG
       │
       ▼
  LLM decoder           frontier LLM (Gemini / Claude / GPT) via
                        prompt — resolves voicing ambiguity from
                        sentence context
       │
       ▼
   English text         lower-cased, no punctuation, single line
```

### Encoder

Trained on Gaddy 2020 EMG corpus (single speaker, 19.4h voiced + silent).

- **Architecture**: Conformer (planned ~10M params; the logged run used 3.48M), 8-channel EMG @ 1 kHz → stride-10 conv frontend → 4 Conformer blocks @ 100 Hz → linear head.
- **Output inventory**: ~31 collapsed-phoneme classes — 21 unambiguous phonemes + 8 voicing-pair classes (B/P, T/D, K/G, S/Z, F/V, SH/ZH, CH/JH, DH/TH) + 1 CTC blank. Voicing pairs are physically indistinguishable from silent EMG; the inventory encodes that constraint by construction.
- **Larynx (ch3) masked** during both train and inference. The throat electrode picks up vocal-fold activity that vanishes in silent speech (rms ratio 0.37×); training on it would teach the encoder a cue it loses at deployment.
- **Joint voiced + silent training**, balanced batches. Silent is the deployment target and gets higher loss weight.

Loss (single training run, multi-loss):

```
L = α · CTC(phoneme sequence)            primary, both modes
  + β · CE(per-frame phoneme labels)     auxiliary, voiced only
  + γ · contrastive(voiced[i], silent[i]) auxiliary, paired only
```

Frame labels for voiced come from forced alignment of the parallel audio (one-time, cached). Silent has no audio, so it gets sentence-level CTC only — its features are bridged toward voiced's via the contrastive term.

See [`docs/possible_architectures.md`](docs/possible_architectures.md) and [`docs/first_principles.md`](docs/first_principles.md).

### Decoder

Frontier LLM via prompt template. No training, no fine-tuning.

The encoder's phoneme posteriors are formatted as text with voicing alternatives at uncertain positions:

```
K AE (T:0.5/D:0.5) ...
```

The LLM resolves ambiguity from sentence context (same way it knows "_at sat on the mat" → "cat" not "pat"). On simulated encoder noise, Gemini hits 0% WER on a 10-sentence test set; no result on real encoder output exists yet (see Status).

See [`app/decoder_prompt.md`](app/decoder_prompt.md) and [`docs/llm_decoder_findings.md`](docs/llm_decoder_findings.md).

## Status (Sep 2026)

- Measurement, preprocessing, phoneme inventory, splits, CTC decode/PER, LLM decoding stage, and the
  8-channel acquisition rig (`docs/hardware.md`) are implemented and logged.
- The LLM stage reaches 0% WER on 10 Gaddy silent-test sentences with **simulated** encoder noise
  (`logs/llm_decoder_v4_results.json`). That bounds the decoder, not the system.
- Encoder training (Conformer-CTC, multi-loss, 3.48M params) started on 8,055 examples and stopped at
  the Apple MPS memory ceiling after ~70 steps (`logs/encoder/full_run.log`). **No WER on real silent
  EMG has been produced yet.**
- `app/encoder/model.py`, `train.py`, `data/dataset.py`, `data/loader.py`, `data/align.py` are referenced
  below but are not in this repository; they were never committed and are being recovered or rewritten.

## Repo layout

```
app/
  decoder_prompt.md           production prompt template for the LLM stage
  encoder/
    vocab.py                  ARPAbet + collapsed-inventory phoneme classes
    model.py                  ConformerCTC encoder   [not in repo; see Status]
    train.py                  multi-loss training loop + dry-run smoke test   [not in repo; see Status]
    eval.py                   CTC greedy decode + PER metric
    data/
      preprocess.py           EMG cleanup (60 Hz notch + harmonics, 2 Hz high-pass, per-channel z-score)
      dataset.py              GaddyEMGDataset + session discovery + GADDY_ROOT   [not in repo; see Status]
      loader.py               collate + DataLoader factory   [not in repo; see Status]
      align.py                forced alignment for frame labels (voiced only)   [not in repo; see Status]

docs/                          findings, architecture comparisons, prev research
experiments/                   per-experiment run scripts
logs/                          machine-readable JSON results
visuals/                       plots
cache/                         model weights, derived artifacts (gitignored)
dataset/                       Gaddy corpus (gitignored)
```

## References

- [`docs/vision.md`](docs/vision.md) — product mission
- [`docs/first_principles.md`](docs/first_principles.md) — 21 numbered claims grounding the architecture
- [`docs/possible_architectures.md`](docs/possible_architectures.md) — architecture comparison vs constraints
- [`docs/llm_decoder_findings.md`](docs/llm_decoder_findings.md) — LLM stage results
- [`docs/articulator_anatomy.md`](docs/articulator_anatomy.md) — electrode placement, silent-vs-voiced ratios
- [`docs/findings.md`](docs/findings.md) — paired-data measurements
- [`docs/prev_research/`](docs/prev_research/) — Gaddy 2020, MONA LISA 2024, AlterEgo 2018, Mohapatra 2025

## Benchmark target

Gaddy 2020 silent open-vocabulary WER. Published SOTA: 12.2% (MONA LISA, Benster 2024). Target: below 12.2%.
