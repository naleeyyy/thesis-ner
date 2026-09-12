# Does the gold set earn its keep?

All three conditions are evaluated on the **same** held-out gold test split (201 sentences, split seed fixed). Only the training data varies.

xlm-roberta-base, seeds [1, 2, 3], 8 epochs.

| Condition | Train size | Gold test F1 | WikiANN test F1 |
| --- | ---: | ---: | ---: |
| `gold_only` | 483 | 0.6874 ± 0.0207 | 0.5507 |
| `wikiann_only` | 5000 | 0.5434 ± 0.0311 | 0.9311 |
| `gold_plus_wikiann` | 5483 | 0.6924 ± 0.0064 | 0.9317 |

## Are the differences real?

Paired bootstrap on the gold test set, using each condition's best-by-dev seed. A gap whose interval includes zero is not a result.

| A | B | ΔF1 | 95% CI | verdict |
| --- | --- | ---: | ---: | :--- |
| `gold_only` | `wikiann_only` | +0.1315 | [+0.0718, +0.1906] | significant |
| `gold_only` | `gold_plus_wikiann` | +0.0146 | [-0.0253, +0.0543] | not distinguishable |
| `wikiann_only` | `gold_plus_wikiann` | -0.1170 | [-0.1734, -0.0594] | significant |

- Hardware: Apple M4 Pro (mps)
- Total compute: 419.5 min
