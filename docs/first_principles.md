# First principles

Each numbered claim follows from the previous. Measured-data claims are linked.

1. The problem is silent EMG → text.

2. Silent EMG measures electrical activity from speech-articulator muscles (face, jaw, lips, throat) at 1 kHz across 8 channels. Source: [`articulator_anatomy.md`](articulator_anatomy.md).

3. Muscles produce sounds, not words. The same `/p/` motor program generates the same muscle pattern in any word containing `/p/`.

4. From (3): the natural unit of mapping is sounds (phonemes), not words.

5. English has ~40 phonemes versus ~50,000 words. Phoneme vocabulary is bounded, compositional, and anatomically grounded; word vocabulary is open and arbitrary.

6. From (4) and (5): a phoneme-level mapping generalizes to any word, including names and neologisms. A word-level mapping cannot.

7. Phonemes are temporal events: each phoneme spans ~50–150 ms, much longer than a single EMG frame at 1 kHz.

8. Coarticulation: a phoneme's muscle pattern depends on its neighbors. The same phoneme has different muscle realizations in different contexts.

9. The phoneme sequence is much shorter than the EMG sequence. Typical ratio ~200× (a 1-second utterance is 1000 EMG frames and ~5 phonemes).

10. The 8 EMG channels carry distributed evidence: no single channel uniquely encodes a phoneme.

11. From (7)–(10): the muscle → phoneme map cannot be a frame-by-frame, single-channel classifier. It must integrate a window of context across all channels and produce a sequence shorter than its input.

12. Voiced and silent EMG differ at the larynx: silent has rms 0.37× and activity ratio 0.51× voiced. Source: [`logs/articulator_test.json`](../logs/articulator_test.json).

13. Voiced and silent EMG do not differ meaningfully at face/jaw/lip channels: rms 0.69–0.84×, activity ratio 0.98–1.24×. Source: same as (12).

14. From (12) and (13): silent muscle activity preserves articulator movement but loses voicing information.

15. Phoneme pairs that differ only in voicing (`b/p`, `d/t`, `g/k`, `s/z`, `f/v`, `sh/zh`) are physically indistinguishable from silent EMG alone.

16. From (15): voicing distinctions cannot be recovered from silent EMG. They must come from somewhere other than the muscle signal.

17. Language context resolves voicing ambiguity in nearly every case: `_at sat on the mat` resolves `_` from the rest of the sentence even though `cat` and `pat` are EMG-identical.

18. From (16) and (17): each decoded sound must condition on previously decoded output, not just on the current muscle signal.

19. Voiced training data has unambiguous phoneme labels (audio + forced alignment). Silent training data does not.

20. From (19): voiced is the supervised case. Whatever maps muscles to phonemes can be learned cleanly there.

21. The algorithm must be accurate on two independent measures: phoneme prediction from EMG, and grammatical correctness of the output text. Whether one model satisfies both or two stages cooperate is implementation.
