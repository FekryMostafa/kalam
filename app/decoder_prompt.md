# Decoder prompt (Kalam, phoneme → text via frontier LLM)

Production prompt for the LLM stage of the Kalam silent-EMG → text pipeline.

## Pipeline position

```
silent EMG ──► encoder ──► phoneme sequence with     ──► LLM API ──► English text
                            voicing alternatives at
                            ambiguous positions
```

The encoder produces phoneme tokens at confident positions and `(X:p1 / Y:p2)`
alternatives at uncertain positions (typically voicing pairs). This file is the
prompt we send to the LLM.

## Prompt template (V4)

```
I'm working on a silent speech interface — a wearable that reads small face-muscle signals (EMG) and tries to figure out what someone is silently mouthing, so people who can't produce audible speech can still communicate. The system can detect most phonemes but physically cannot tell apart sounds that differ only by voicing (b vs p, t vs d, k vs g, s vs z, f vs v, sh vs zh, ch vs jh, dh vs th). Other ambiguities also occur: similar places of articulation (p/t/k), or similar vowels (AH/AA/EH). Where alternatives appear in (X:0.50 / Y:0.50) notation, those are positions the system was uncertain.

Rules for transcription:
1. Be faithful to the phonemes. At each ambiguous position, pick the option that yields a real English word consistent with the rest of the sentence. Do not pick options that don't match the phonemes.
2. Preserve the user's vocabulary and word order exactly. Do not paraphrase, modernize, or "improve" the sentence. Do not add words. Do not drop words. The user's phonemes ARE the intended sentence.
3. Uncommon words, proper nouns, place names, multi-syllable words, and slightly formal or non-modern phrasing are all valid outputs. Prefer the literal phoneme reading over a more common substitute.
4. Maintain word boundaries. Do not merge a final consonant of one word into the start of the next, and do not split a single word into two.
5. Output every word the phonemes encode — including short words like "i", "and", "the". Do not drop short words even when the resulting sentence sounds slightly old-fashioned.

Examples (these are NOT in the test set — they illustrate the rules):

Phonemes: (DH:0.50 / TH:0.50) AH (K:0.50 / G:0.50) W EY N (T:0.50 / D:0.50) L IH (T:0.50 / D:0.50) AH L (K:0.50 / G:0.50) AA (T:0.50 / D:0.50) IH (JH:0.50 / CH:0.50) (S:0.50 / Z:0.50) (T:0.50 / D:0.50) UH (D:0.50 / T:0.50) (B:0.50 / P:0.50) AY (DH:0.50 / TH:0.50) AH (B:0.50 / P:0.50) R UH (K:0.50 / G:0.50)
Output: the quaint little cottage stood by the brook

Phonemes: AY (AE:0.60 / EH:0.40) N (D:0.50 / T:0.50) M AY (B:0.50 / P:0.50) R AH (DH:0.50 / TH:0.50) ER W AO (K:0.50 / G:0.50) (T:0.50 / D:0.50) (TH:0.50 / DH:0.50) R UW (DH:0.50 / TH:0.50) AH (V:0.50 / F:0.50) IH L AH (JH:0.50 / CH:0.50)
Output: i and my brother walked through the village

Phonemes: (SH:0.50 / ZH:0.50) IY M AH (T:0.50 / D:0.50) IH (K:0.50 / G:0.50) Y AH L AH (S:0.50 / Z:0.50) L IY IH (G:0.50 / K:0.50) (Z:0.50 / S:0.50) AE M IH N (D:0.50 / T:0.50) (DH:0.50 / TH:0.50) AH AA R (T:0.50 / D:0.50) AH (F:0.50 / V:0.50) AE (K:0.50 / G:0.50) (T:0.50 / D:0.50)
Output: she meticulously examined the artifact

Now transcribe the sequence below. Output ONLY the English sentence, one line, lowercase, no punctuation, no explanation.

{PHONEME_SEQUENCE}
```

Replace `{PHONEME_SEQUENCE}` with the encoder's output for the utterance.

## Notation rules for `{PHONEME_SEQUENCE}`

- Confident positions: bare ARPAbet token (e.g. `K`, `AE`, `T`)
- Ambiguous positions: parenthesized list with probabilities, e.g.
  `(K:0.50 / G:0.50)` or `(P:0.50 / B:0.30 / M:0.20)`
- Whitespace separates phoneme positions
- Word boundaries are NOT explicitly marked; LLM infers from sequence

## Tested LLM performance

See [`docs/llm_decoder_findings.md`](../docs/llm_decoder_findings.md) for full
results. Summary on a 10-sentence simulation with realistic encoder noise:

```
LLM                                       exact   WER     notes
Gemini (web)                              10/10    0.0%   resolved every ambiguity
ChatGPT (web)                              7/10    7.0%   3 LLM-bias errors
Claude (web)                                  -       -   refused (safety classifier)
Gemma-4-E2B-it (5B local, prompted)        1/20  67.6%   small open-source — too weak
```

## Source / lineage

Prompt iterated across 4 versions during the 2026-05-05 session:

```
V1   instruction-only, 3-shot, included test-set words      14.04% WER  (overfit)
V3   instruction-only, no test-set words                     8.77% WER  (generalizable)
V4   V3 + 3 non-test-set few-shot examples                   7.02% WER  (ChatGPT)
                                                             0.00% WER  (Gemini)
```

V4 is in production use here.
