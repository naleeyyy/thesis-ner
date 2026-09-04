# Baselines on gold gold.jsonl (full spans), test split

Evaluated on 201 sentences from gold gold.jsonl (full spans), test split.

F1 is micro-averaged over entity spans. Brackets give a 95% bootstrap confidence interval from 2000 resamples of the test sentences (seed 12345); inference itself is deterministic, so this interval reflects test-set sampling variability, not run-to-run noise.

| Model | P | R | F1 [95% CI] | F1-PER | F1-ORG | F1-LOC | infer (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `davlan-xlmr-hrl` | 0.702 | 0.719 | **0.711** [0.648, 0.772] | 0.797 | 0.613 | 0.701 | 2 |
| `babelscape-wikineural` | 0.641 | 0.639 | **0.640** [0.574, 0.705] | 0.794 | 0.424 | 0.626 | 1 |
| `kushtrim-mbert-sq` | 0.542 | 0.611 | **0.574** [0.511, 0.636] | 0.689 | 0.476 | 0.570 | 2 |
| `akdeniz27-mbert-sq` | 0.531 | 0.579 | **0.554** [0.489, 0.615] | 0.701 | 0.433 | 0.546 | 1 |

## Pairwise significance

Paired bootstrap on the same resampled sentences for both models, so the shared difficulty of a resample cancels out of the difference. A gap is significant when the CI on the difference excludes zero.

| A | B | ΔF1 (A−B) | 95% CI | p | significant |
| --- | --- | ---: | ---: | ---: | :---: |
| `davlan-xlmr-hrl` | `babelscape-wikineural` | +0.071 | [+0.025, +0.115] | 0.001 | yes |
| `davlan-xlmr-hrl` | `kushtrim-mbert-sq` | +0.136 | [+0.080, +0.189] | < 0.001 | yes |
| `davlan-xlmr-hrl` | `akdeniz27-mbert-sq` | +0.157 | [+0.109, +0.204] | < 0.001 | yes |
| `babelscape-wikineural` | `kushtrim-mbert-sq` | +0.065 | [+0.007, +0.120] | 0.026 | yes |
| `babelscape-wikineural` | `akdeniz27-mbert-sq` | +0.086 | [+0.039, +0.134] | < 0.001 | yes |
| `kushtrim-mbert-sq` | `akdeniz27-mbert-sq` | +0.021 | [-0.022, +0.065] | 0.345 | no |

## Environment

- Hardware: Apple M4 Pro (mps), 12 CPU cores, 24.0 GB RAM
- Platform: macOS-26.2-arm64-arm-64bit, Python 3.11.14
- Libraries: torch 2.12.0, transformers 5.10.2, datasets 4.8.5
- Total compute: 0.2 min (0.1 min inference, 0.0 min model loading)
- Run date: 2026-09-04

> All baseline inference ran on Apple M4 Pro (mps), 24.0 GB RAM, for a total of 0.2 min of compute.
