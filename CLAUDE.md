# Project rules

## Writing
- State facts. No takeaways, implications, or framing prose ("universal", "not X-specific", "this means", etc.).
- Don't mix sources. Cited research stays separate from project measurements — never compare them in the same sentence.
- Don't repeat the same point in different words.
- Match length to content. Don't pad.
- For tables, use fenced code blocks with column-aligned monospace text, not pipe tables. Renders cleanly at any width.

## Repo layout
- `docs/` findings, `logs/` JSON results, `experiments/` scripts, `visuals/` plots, `cache/` precomputed artifacts (gitignored).
- Mirror structure across folders. Findings link to logs; logs link to scripts.

## EMG channel names
Always use the anatomical label, not the channel index, in plots / docs / conversation. Mapping (Gaddy 2020 Table 3):
- ch0 left cheek above mouth — `lips_L`
- ch1 left corner of chin — `chin_L`
- ch2 below chin back 3 cm — `floor_of_mouth`
- ch3 throat 3 cm L of Adam's apple — `larynx`
- ch4 mid-jaw right — `jaw_R`
- ch5 right cheek below mouth — `lips_R`
- ch6 right cheek near nose — `upper_face`
- ch7 right cheek near ear — `masseter`

## Workflow
- Never auto-commit. Stage and wait.
- Use relative paths or env-configured paths in scripts. Avoid machine-specific absolute paths.
