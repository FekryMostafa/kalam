# Articulator anatomy: silent vs voiced

## Electrode placement (Gaddy & Klein 2020, Table 3)

- `lips_L` (ch0): left cheek above mouth
- `chin_L` (ch1): left corner of chin
- `floor_of_mouth` (ch2): below chin, back 3 cm
- `larynx` (ch3): throat, 3 cm left of Adam's apple
- `jaw_R` (ch4): mid-jaw right
- `lips_R` (ch5): right cheek below mouth
- `upper_face` (ch6): right cheek 2 cm from nose
- `masseter` (ch7): right cheek 4 cm from ear

## Literature: what differs in silent speech

**Larynx:** reduced and variable across individuals; only the right cricothyroid shows consistent significant change (Stepp 2021).

**Lips:** hypo-articulated in most studies; sometimes hyper-articulated with increased aperture for visible consonants (Calliope; Lee 2021).

**Tongue:** mostly similar to vocalized; slight forward displacement for visible lingual consonants (Lee 2021).

**Duration:** shorter words; longer consonants — findings conflict across studies.

**Respiratory:** reduced effort, consistent across studies.

Cross-mode recognition (train vocalized to test silent) drops from 75% to 42% in ultrasound-based SSI (Hueber).

## Our measurement (1,584 paired utterances, 7 sessions, Gaddy preprocessed)

Silent / voiced ratios per anatomical group, plus voiced and silent spectral centroid (Hz):

```
group               rms_ratio  act_ratio   voiced_Hz   silent_Hz   shift_Hz
larynx                  0.368      0.507        66.1        44.0      -22.1
lips                    0.781      1.196        46.4        49.0       +2.6
chin_lower_lip          0.842      1.222        79.7        80.4       +0.8
floor_of_mouth          0.693      0.983        51.2        55.3       +4.2
jaw_masseter            0.749      1.092        58.4        62.9       +4.5
upper_face              0.779      1.243        51.1        54.5       +3.4
```

Duration silent/voiced median: 0.915.

Larynx activity ratio is 0.51 and centroid shift is -22.1 Hz. Lips, `upper_face`, and `chin_lower_lip` activity ratios are above 1.0.

[`logs/articulator_test.json`](../logs/articulator_test.json)

## Sources
- [Gaddy & Klein 2020 — Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445.pdf)
- [Stepp et al. 2021 — Intrinsic Laryngeal Muscle Activity During Subvocalization](https://pubmed.ncbi.nlm.nih.gov/33612369/)
- [Lee et al. 2021 — Visibility in Silent Speech Tongue Movements (JSLHR)](https://pubs.asha.org/doi/10.1044/2021_JSLHR-20-00266)
- [Calliope et al. — Articulatory strategies for lip and tongue movements (HAL-SHS)](https://shs.hal.science/halshs-00610870)
- [Hueber et al. — Silent vs Vocalized Articulation, Ultrasound SSI](https://www.academia.edu/1359056/Silent_vs_vocalized_articulation_for_a_portable_ultrasound-based_silent_speech_interface)
