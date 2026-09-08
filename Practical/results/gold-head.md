# Fine-tuned xlm-roberta-base

gold gold.jsonl (head spans), split 20260811 — train 483, dev 121, test 201.

**Test F1 0.7954 ± 0.0205** over 3 seeds [1, 2, 3].

Two uncertainties, deliberately separate: `±` is run-to-run variance from training randomness, the bracketed interval is a 95% bootstrap CI over test sentences for the best-dev seed.

| Seed | dev F1 | test F1 | seconds |
| --- | ---: | ---: | ---: |
| 1 | 0.7824 | 0.8055 | 363 |
| 2 | 0.7910 | 0.8090 | 356 |
| 3 | 0.7740 | 0.7719 | 359 |

Best seed (2): **0.8090** [0.7513, 0.8633] — PER 0.869, ORG 0.561, LOC 0.844

Hyperparameters: 8 epochs, batch 8, lr 3e-05, warmup 0.1, max length 160.

- Hardware: Apple M4 Pro (mps), 24.0 GB RAM
- Libraries: torch 2.12.0, transformers 5.10.2
- Total compute: 18.0 min
