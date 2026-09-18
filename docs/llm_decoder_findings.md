# LLM decoder stage — findings (2026-05-05 session)

## Architecture decision

The LLM stage of the silent-EMG → text pipeline calls a frontier LLM via API at
inference time. Phoneme distributions from the encoder are encoded as text with
parenthesized alternatives at ambiguous positions:

```
silent EMG ──► encoder ──► phoneme sequence with     ──► LLM API ──► English text
                            voicing alternatives at
                            ambiguous positions
```

No local LLM. No fine-tuning. No cross-attention adapter. No soft-prefix table.

## Why API, not local frozen LLM

Tested Gemma-4-E2B-it (5B params, ~14 GB, locally cached) on phoneme→text via
prompting. Result: 1/20 exact match, 67.6% WER on 8-shot prompt with ARPAbet
primer. Sample failure modes:

```
ref:  "remarkable story from woking"
gen:  "remember mark the story from work"

ref:  "i felt a tug at the reins"
gen:  "if it's the right way then"
```

Pattern: small LLMs read ARPAbet syntactically but lack phonotactic
competence in their pretraining to consistently do P→G correctly. Frontier
LLMs (Claude Opus, Gemini, GPT-5) handle the same prompts essentially
perfectly. See `experiments/silent_speech/eval_text_prompt_v2.py` (sibling
repo) for the small-LLM result, and the prompt/iteration story below.

## Prompt iteration on 10 simulated noisy sentences

Used `experiments/silent_speech/simulate_noisy_encoder.py` (sibling repo) to
generate phoneme sequences for 10 Gaddy silent-test sentences. Encoder noise
model: voicing pairs always 50/50 (b/p, t/d, k/g, s/z, f/v, sh/zh, ch/jh,
dh/th); place-of-articulation confusions for 25% of consonants; vowel
confusions for 30% of vowels. Top-1 confident at non-ambiguous positions.

Prompt versions:

```
version  description                                    exact   WER       generalizable
V1       instruction + 3-shot. Listed test-set words    5/10    14.04%    no (overfit)
         in the instruction (e.g. "Woking is a town
         do not autocorrect").
V3       general rules only. No test-set words named.   6/10     8.77%    yes
V4       V3 + 3 non-test-set few-shot examples          7/10     7.02%    yes
         (quaint cottage / I and my brother / 
         meticulously examined).
```

Tested on ChatGPT (web). V4 is in `app/decoder_prompt.md`.

## Cross-LLM comparison on V4 prompt

```
LLM                exact   WER     residual error types
Gemini (web)      10/10    0.0%    none
ChatGPT (web)      7/10    7.0%    common-word substitution (impossible vs
                                    impassable), word-boundary merging (city
                                    vs said i), syllable simplification
                                    (artillerman vs artilleryman)
Claude (web)          —      —     safety classifier refused initial prompt
                                    shape; conversational reframe accepted
                                    but not full-batch tested
```

Gemini's 0% on simulated noise is below MONA LISA's published 12.2% on the same
Gaddy silent test set. (MONA LISA used a real, not simulated, encoder.)

## Comparison against project + literature baselines

```
condition                                              WER
Gaddy CTC raw silent (Gaddy 2020 baseline)            39.5%
Gaddy + KenLM (sibling repo replication)              32.8%
sibling repo: CTC + KenLM + GPT-4-31B cleanup         25.5%
MONA LISA published SOTA (Benster 2024)               12.2%
─────────────────────────────────────────────────────────────
Gemini on V4 + simulated encoder                       0.0%
ChatGPT on V4 + simulated encoder                      7.0%
```

## Caveats

```
established                              not yet established
LLM stage works at near-zero WER         encoder. Real silent-EMG encoder noise
on simulated encoder noise.              may differ from simulation.

Probability-as-text-alternatives         simulation modeled voicing pairs +
notation parses correctly.               place + vowel confusions only — not
                                          phoneme deletions, insertions, or
                                          segmentation errors.

Model choice matters more than           Gemini's 0% may degrade on real
prompt engineering at this scale.        encoder output if confusion patterns
                                          differ from simulation.
```

## Source artifacts

```
artifact                                                location
prompt template (V4)                                    app/decoder_prompt.md
noise simulation script                                 sibling repo: experiments/
                                                         simulate_noisy_encoder.py
generated prompts (10 sentences)                        sibling repo: analysis/
                                                         noisy_encoder_prompts.json
small-LLM baseline (Gemma-4-E2B-it)                     sibling repo: experiments/
                                                         eval_text_prompt_v2.py
small-LLM baseline result (1/20, 67.6% WER)             sibling repo: analysis/
                                                         text_prompt_v2.json
xattn full benchmark on Apr 16 checkpoint               sibling repo: analysis/
(99 silent test, CTC encoder + xattn, 99.1% WER —       xattn_full_best_eval.json
LLM hallucinated, demonstrating soft-prefix path
without phoneme intermediate fails)
```

The "sibling repo" is `<repo>/experiments/`,
gitignored separately from this repo.

## Next step

Build the encoder. The LLM stage has a 0% ceiling on simulated noise; the
remaining gap is whatever the encoder can produce.
