# Fine-tuned xlm-roberta-base

gold gold.jsonl (full spans), split 20260811 — train 483, dev 121, test 201.

**Test F1 0.7070 ± 0.0331** over 3 seeds [1, 2, 3].

Two uncertainties, deliberately separate: `±` is run-to-run variance from training randomness, the bracketed interval is a 95% bootstrap CI over test sentences for the best-dev seed.

| Seed | dev F1 | test F1 | seconds |
| --- | ---: | ---: | ---: |
| 1 | 0.7784 | 0.6998 | 353 |
| 2 | 0.7727 | 0.7431 | 352 |
| 3 | 0.7701 | 0.6780 | 351 |

Best seed (1): **0.6998** [0.6392, 0.7611] — PER 0.846, ORG 0.530, LOC 0.690

Hyperparameters: 8 epochs, batch 8, lr 3e-05, warmup 0.1, max length 160.

- Hardware: Apple M4 Pro (mps), 24.0 GB RAM
- Libraries: torch 2.12.0, transformers 5.10.2
- Total compute: 17.6 min
