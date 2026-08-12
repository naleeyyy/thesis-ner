# Active learning: random vs uncertainty

WikiANN-sq (train as pool) — pool 5000, dev 200, test 1000. xlm-roberta-base, seeds [1, 2, 3].

Seed set 250, then 4 rounds acquiring 250 sentences each. Simulated over the labelled pool: labels are hidden and revealed only as a strategy requests them.

F1 is mean ± standard deviation across seeds. The spread matters as much as the means — a gap smaller than the error bars is not a result.

| Labelled | random | uncertainty |
| ---: | ---: | ---: |
| 250 | 0.6847 ± 0.0003 | 0.6924 ± 0.0162 |
| 500 | 0.8630 ± 0.0165 | 0.8467 ± 0.0172 |
| 750 | 0.8815 ± 0.0067 | 0.8579 ± 0.0190 |
| 1000 | 0.8834 ± 0.0289 | 0.8881 ± 0.0247 |
| 1250 | 0.8983 ± 0.0042 | 0.9064 ± 0.0115 |

- Hardware: Apple M4 Pro (mps)
- Total compute: 95.9 min (30 trainings)
