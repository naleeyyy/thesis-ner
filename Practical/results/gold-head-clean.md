# Fine-tuned xlm-roberta-base

gold gold-no-ann11.jsonl (head spans), split 20260811 — train 435, dev 109, test 182.

**Test F1 0.7741 ± 0.0117** over 3 seeds [1, 2, 3].

Two uncertainties, deliberately separate: `±` is run-to-run variance from training randomness, the bracketed interval is a 95% bootstrap CI over test sentences for the best-dev seed.

| Seed | dev F1 | test F1 | seconds |
| --- | ---: | ---: | ---: |
| 1 | 0.8214 | 0.7855 | 324 |
| 2 | 0.8014 | 0.7621 | 321 |
| 3 | 0.8043 | 0.7746 | 319 |

Best seed (1): **0.7855** [0.7250, 0.8400] — PER 0.860, ORG 0.449, LOC 0.844

Hyperparameters: 8 epochs, batch 8, lr 3e-05, warmup 0.1, max length 160.

- Hardware: Apple M4 Pro (mps), 24.0 GB RAM
- Libraries: torch 2.12.0, transformers 5.10.2
- Total compute: 16.1 min
