# WikiANN-sq baselines

Evaluated on 1000 sentences from the `test` split of `unimelb-nlp/wikiann` (`sq`).

F1 is micro-averaged over entity spans. Brackets give a 95% bootstrap confidence interval from 2000 resamples of the test sentences (seed 12345); inference itself is deterministic, so this interval reflects test-set sampling variability, not run-to-run noise.

| Model | P | R | F1 [95% CI] | F1-PER | F1-ORG | F1-LOC | infer (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kushtrim-mbert-sq` | 0.918 | 0.931 | **0.925** [0.907, 0.942] | 0.937 | 0.902 | 0.938 | 7 |
| `akdeniz27-mbert-sq` | 0.813 | 0.750 | **0.780** [0.754, 0.806] | 0.891 | 0.686 | 0.764 | 7 |
| `babelscape-wikineural` | 0.725 | 0.696 | **0.710** [0.683, 0.737] | 0.843 | 0.578 | 0.682 | 7 |
| `davlan-xlmr-hrl` | 0.523 | 0.518 | **0.520** [0.489, 0.551] | 0.740 | 0.432 | 0.306 | 8 |

## Pairwise significance

Paired bootstrap on the same resampled sentences for both models, so the shared difficulty of a resample cancels out of the difference. A gap is significant when the CI on the difference excludes zero.

| A | B | ΔF1 (A−B) | 95% CI | p | significant |
| --- | --- | ---: | ---: | ---: | :---: |
| `kushtrim-mbert-sq` | `akdeniz27-mbert-sq` | +0.145 | [+0.115, +0.173] | < 0.001 | yes |
| `kushtrim-mbert-sq` | `babelscape-wikineural` | +0.214 | [+0.187, +0.242] | < 0.001 | yes |
| `kushtrim-mbert-sq` | `davlan-xlmr-hrl` | +0.404 | [+0.371, +0.440] | < 0.001 | yes |
| `akdeniz27-mbert-sq` | `babelscape-wikineural` | +0.070 | [+0.036, +0.105] | < 0.001 | yes |
| `akdeniz27-mbert-sq` | `davlan-xlmr-hrl` | +0.260 | [+0.222, +0.299] | < 0.001 | yes |
| `babelscape-wikineural` | `davlan-xlmr-hrl` | +0.190 | [+0.161, +0.220] | < 0.001 | yes |

## Environment

- Hardware: Apple M4 Pro (mps), 12 CPU cores, 24.0 GB RAM
- Platform: macOS-26.2-arm64-arm-64bit, Python 3.11.14
- Libraries: torch 2.12.0, transformers 5.10.2, datasets 4.8.5
- Total compute: 0.6 min (0.5 min inference, 0.1 min model loading)
- Run date: 2026-08-11

> All baseline inference ran on Apple M4 Pro (mps), 24.0 GB RAM, for a total of 0.6 min of compute.
