# Fine-tuned xlm-roberta-base

gold gold.jsonl (full spans), split 20260811 — train 483, dev 121, test 201.

**Test F1 0.7092 ± 0.0282** over 3 seeds [1, 2, 3].

Two uncertainties, deliberately separate: `±` is run-to-run variance from training randomness, the bracketed interval is a 95% bootstrap CI over test sentences for the best-dev seed.

| Seed | dev F1 | test F1 | seconds |
| --- | ---: | ---: | ---: |
| 1 | 0.7740 | 0.7226 | 355 |
| 2 | 0.7471 | 0.7282 | 354 |
| 3 | 0.7701 | 0.6769 | 352 |

Best seed (1): **0.7226** [0.6599, 0.7818] — PER 0.855, ORG 0.530, LOC 0.722

Hyperparameters: 8 epochs, batch 8, lr 3e-05, warmup 0.1, max length 160.

- Hardware: Apple M4 Pro (mps), 24.0 GB RAM
- Libraries: torch 2.12.0, transformers 5.10.2
- Total compute: 17.7 min
