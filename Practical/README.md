# Albanian NER — Practical

Practical work for the bachelor thesis _"Albanian NER via LLM-assisted annotation and active learning"_ (JKU Linz).

## Setup

```bash
cd Practical
uv sync
```

Python 3.11 is pinned via `.python-version`; `uv` will fetch it if missing.

The Kushtrim model is gated on Hugging Face. Put your token in `Practical/.env` (gitignored):

```bash
echo 'HF_TOKEN=hf_...' > .env
set -a; source .env; set +a   # before running the baselines
```

## Step 1 — WikiANN-sq baselines

Evaluate existing Albanian and multilingual NER models on the WikiANN Albanian (`sq`) test split.

```bash
# smoke test (20 sentences) — separate prefix so it can't overwrite a real run
uv run python -m src.baselines.run_baselines --limit 20 --out-prefix _smoke

# full run; writes results/baselines.{md,json} and results/predictions/
uv run python -m src.baselines.run_baselines
```

Takes ~40 s total on an M4 Pro (MPS). Outputs:

| Path | Contents |
| --- | --- |
| `results/baselines.md` | leaderboard, pairwise significance table, environment |
| `results/baselines.json` | same numbers machine-readable, plus hardware/compute/library versions |
| `results/predictions/<model>.jsonl` | per-sentence gold + predicted tags |

Predictions are committed so the scoring and statistics can be re-derived without a GPU
or a token for the gated model.

Models benchmarked (defined in `src/baselines/models.py`):

| Model | Note |
| --- | --- |
| `Kushtrim/bert-base-multilingual-cased-finetuned-albanian-ner` | mBERT, trained on WikiANN — **gated on HF, request access** |
| `akdeniz27/mbert-base-albanian-cased-ner` | mBERT, trained on WikiANN |
| `Davlan/xlm-roberta-base-ner-hrl` | XLM-R, multilingual NER (not Albanian-specific) |
| `Babelscape/wikineural-multilingual-ner` | mBERT, multilingual NER (not Albanian-specific) |

The Albanian-specific checkpoints set the ceiling for the WikiANN distribution; the multilingual ones show what generic cross-lingual transfer gives you for free.

The runner uses word-level token-classification inference (not the `pipeline` aggregator) so per-word labels line up exactly with WikiANN's pre-tokenized gold. Each word gets the predicted label of its first subword. Cross-tagset entity types (`MISC`, `DATE`) are remapped to `O` before seqeval scoring; the mapping is declared per-model in `src/baselines/models.py`.

If a model fails to download (gated repo, network), the runner logs a `SKIP:` and continues with the rest.

### Uncertainty and significance

Inference over a frozen checkpoint is deterministic — `logits.argmax(-1)` gives byte-identical
predictions on every re-run — so seed variance is identically zero and would be a meaningless
error bar. The uncertainty that matters is that WikiANN-sq test is a 1000-sentence *sample*.
`src/baselines/stats.py` therefore reports a 95% percentile bootstrap CI over resampled test
sentences, and compares models with a **paired** bootstrap (same resampled indices for both
models, so shared resample difficulty cancels out of the difference). Control it with
`--bootstrap`, `--alpha` and `--seed`; the seed is recorded in `baselines.json` so intervals
reproduce exactly.

Seeds return as a genuine source of variance once there is training involved — fine-tuning and
the active learning loop will need multiple seeds *and* this bootstrap.

## Step 2 — 100-sentence hand-labeled gold set

Sample → segment → annotate.

```bash
# 1. sample ~30 Albanian Wikipedia articles
uv run python -m src.data.wiki_sample --n 30 --out data/raw/wiki_articles.jsonl

# 2. segment + filter into candidate sentences
uv run python -m src.data.segment --in data/raw/wiki_articles.jsonl \
  --out data/raw/wiki_segmented.jsonl --sample 2000 --rejects data/raw/wiki_rejects.jsonl

# 3. LLM pre-labeling (see below)

# 4. annotate in Jupyter
uv run jupyter lab notebooks/01_annotate.ipynb
```

### LLM pre-labeling

`src/annotate/` proposes entity spans for each segmented sentence so annotators verify
rather than label from scratch. Needs an Anthropic key alongside `HF_TOKEN`:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
set -a; source .env; set +a

# smoke test on 5 sentences, hard-capped at 50 cents
uv run python -m src.annotate.llm_label --limit 5 --max-usd 0.50 \
  --model claude-sonnet-5 --no-thinking \
  --in data/raw/wiki_segmented.jsonl --out data/interim/prelabeled.jsonl

# full run
uv run python -m src.annotate.llm_label --model claude-sonnet-5 --no-thinking \
  --in data/raw/wiki_segmented.jsonl --out data/interim/prelabeled.jsonl
```

**Model and cost.** Measured on this prompt: ≈ **$3.30–5.00 per 2000 sentences** on
`claude-sonnet-5` with `--no-thinking`. Thinking is on by default on Sonnet 5 and Opus 5
and bills at the output rate, which for a task this small dwarfs the answer — it is the
single largest cost lever. `--max-usd` caps a run and stops it mid-flight; everything
already labeled stays cached, so a resumed run picks up where it left off.

Haiku 4.5 is cheaper per token but its prompt-cache floor is 4096 tokens against a
1121-token prompt, so it **silently never caches** and pays full input price on every
call — which erases most of the saving. It also mislabeled a novel title (`Vargjet
satanike`) as LOC where Sonnet 5 correctly ignored it. The runner prints a warning when
the selected model can't cache this prompt.

**Spans, not per-token tags.** The model returns inclusive token-index spans and the code
converts to BIO. A parallel tag array invites length mismatches; a span list either
validates or doesn't. `validate_spans` range-, type- and overlap-checks everything before
conversion, so a malformed response can never yield a malformed BIO sequence. Rejections
are counted by reason in `<out>.stats.json` rather than silently dropped.

**Two boundary conventions, one annotation.** Albanian's `X i Y` construction
(`Stacioni i Bramit`, `qyteti i Tiranës`) is genuinely ambiguous about where a name ends,
and annotation standards differ. Rather than freezing that choice into the data, every
entity carries both a **full span** and a **head** (the proper-name core):

```json
{"type": "LOC", "surface": "Stacioni i Bramit", "head_surface": "Bramit",
 "start": 0, "end": 2, "head_start": 2, "head_end": 2}
```

Two flat BIO views are derived from it — `llm_tags_full` and `llm_tags_head` — so the
convention stays a reporting choice rather than a data commitment, and the thesis can
report F1 under both. Where there is no common noun to strip, the two are identical. A
missing or non-nested head falls back to the full span (counted as `head_fallbacks`)
rather than discarding the entity.

Field names are `llm_tags_*`, **not** `ner_tags` — these are suggestions until a human
signs off. The annotation tool renames on acceptance.

Two things the runs record for the report: responses are cached to
`data/interim/llm_cache.jsonl` keyed by (model, prompt version, tokens), so re-runs and
resumed jobs cost nothing for work already done; and `<out>.stats.json` carries token
usage, USD cost per sentence, wall-clock time, and the span-rejection breakdown.

Annotation conventions live in `src/annotate/prompt.py` and are versioned via
`PROMPT_VERSION` — bumping it invalidates the cache, so edits can never mix two
annotation standards in one output file. That constant is also the seed of the human
annotation guidelines; the two must stay in sync, or measured LLM-vs-human agreement
reflects the mismatch rather than anything real.

### Sentence quality filter

Wikipedia extraction and tokenization mangle a small share of sentences — split words
(`të ardhshë` + `m`, `Dommartin-au` + `x-Bois`), truncated fragments with unbalanced
brackets. `src/data/quality.py` drops them before they cost API spend and annotator time.
It runs *before* the sample is taken, so `--sample N` yields N usable sentences, and
`--rejects` writes what was dropped so the rate stays reportable.

Measured on the built pool — 900 articles, 3113 sentences, **2500 kept**:

| Dropped | Count |
| --- | ---: |
| `unbalanced-brackets` (fragment starting mid-parenthesis) | 69 |
| `template-repeat` (4th+ member of a stub family) | 22 |
| `orphan-letter` (split word: `Andre` + `w`) | 18 |
| `exact-duplicate` | 5 |
| `hyphen-fragment` (`Dommartin-au` + `x-Bois`) | 2 |
| **total** | **115 / 2688 (4.3%)** |

`deduplicate` also caps near-identical stub sentences: Albanian Wikipedia's generated
geography stubs (`Ka sipërfaqe prej # .# km² , dhe gjendet në lartësi mbidetare # m .`)
appeared 15 times with different numbers. They are genuine negatives and a model has to
learn that a sentence can contain no entities — but fifteen copies buy nothing over three,
and each costs an annotation slot and an API call. `--max-per-template` (default 3) keeps
the negatives diverse rather than repetitive.

Every rule was calibrated against that pool rather than guessed, and the calibration
changed the rules. A naive "a bare lowercase letter means a split word" would have deleted
`400 m` and `15 m lartësi`, where `m` is the metre unit, and `Diego Rodríguez de Silva y
Velázquez`, where `y` is a Spanish name particle — a sentence carrying a PER entity. So
the orphan-letter rule exempts letters after a number, letters between two capitalized
tokens, and uppercase letters (Roman numerals, initials, blood-type labels).

The filter is deliberately conservative, because the two error directions are not
symmetric: a false negative is cheap (annotators have a `junk` flag and catch the rest),
while a false positive is an invisible loss — a good sentence that silently never reaches
anyone. `tests/test_quality.py` keeps a regression test for each real sentence an earlier
version of the filter wrongly deleted.

Tagset: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC` — matches WikiANN, so the hand set stays directly comparable with the Step 1 baselines and with the upcoming fine-tuned model.

Output: `data/labeled/sq_hand100.jsonl`, one record per sentence:

```json
{"id": "sq-0001", "tokens": ["...", "..."], "ner_tags": ["O", "B-LOC"], "source_url": "https://sq.wikipedia.org/..."}
```

The first time `stanza` runs it downloads ~500 MB of Albanian models to `~/stanza_resources/` (gitignored).

## Tests

```bash
uv run pytest
```

Currently checks BIO validity of any committed `.jsonl` under `data/labeled/`.
