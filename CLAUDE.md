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

## Code style
- Docstrings: explain what each module / class / function does, briefly. Keep architectural notes (the *why*) when non-obvious.
- No project-journey commentary in docstrings or comments. Drop "v1", "first experiment", "from our session", "documented above", "for the planned X". Code outlives the conversation that produced it.
- Filenames and identifiers should describe what the thing is, not when it was written ("voiced_larynx_masked.py", not "v1_encoder.py").
- Single source of truth for paths and config. No `root=` / `path=` override parameters, no env-var defaults that shadow constants. Hardcode the constant in one place; if it ever needs to change, change it there.
- Modularize by purpose: when a module covers multiple distinct stages (e.g. data: cleanup, dataset, batching), split into a folder with one file per stage. Each file should hold a meaningful chunk — don't split a 10-line constant into its own file.

## Workflow
- Never auto-commit. Stage and wait.
- Never build before sign-off. Propose first; wait for an explicit go-ahead before writing or modifying code. "Auto mode" applies to low-risk plumbing only — design choices need approval.
- Use relative paths or env-configured paths in scripts. Avoid machine-specific absolute paths.

## Cloud compute discipline
- Don't waste paid GPU time. Smoke-test on a tiny subset before the real run; only kick off long training once the loop runs end-to-end (forward, backward, val, ckpt save).
- Measure batch size on the actual GPU with real utterance lengths. No guessing.
- Persistent-volume checkpoints only. Save best on val improvement, latest every N steps. Pull artifacts back to laptop periodically so a dead pod ≠ lost work.
- Stop the pod the moment training finishes. Idle GPU is wasted money.
- Don't run speculative experiments in parallel on rented compute. One thing at a time.
