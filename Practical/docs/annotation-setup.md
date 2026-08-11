# Annotation platform — setup and campaign structure

Self-hosted Label Studio on Coolify, with an account per annotator. This document covers
deployment, the project layout that encodes the experiment, and the import/export cycle.

Files live in `deploy/label-studio/`.

---

## 1. Deploy on Coolify

New resource → **Docker Compose** → paste `deploy/label-studio/docker-compose.yml`.

Set these environment variables in Coolify:

| Variable | Value |
|---|---|
| `LABEL_STUDIO_HOST` | The public `https://…` URL Coolify assigns. Invite links are built from it, so a wrong value sends people to the wrong host. |
| `POSTGRES_PASSWORD` | A strong random password. |
| `LABEL_STUDIO_USERNAME` | Your admin email. |
| `LABEL_STUDIO_PASSWORD` | Admin password — change it after first login. |

Every variable is declared `:?` in the compose file, so a missing one fails the deploy
immediately rather than silently starting an instance with a blank password.

Then set the **domain** on the `label-studio` service, pointing at port **8080**.

**Two ordering traps:**

1. `LABEL_STUDIO_HOST` must be the final `https://…` URL, which you don't know until
   Coolify has assigned the domain. Deploy once, copy the domain, set the variable,
   **redeploy**. Get it wrong and invite links point at the wrong host — which you only
   discover when an annotator can't sign up.
2. First boot is slow: Postgres migrations run before Label Studio answers. The
   healthcheck allows 120 s, so don't panic at "unhealthy" in the first minute or two.

Three choices worth knowing about:

- **`expose`, not `ports`.** Coolify's reverse proxy reaches the container over the
  internal Docker network and terminates TLS itself, so publishing a host port buys
  nothing and collides with whatever already holds 8080 on the host:

  ```
  Bind for 0.0.0.0:8080 failed: port is already allocated
  ```

  If you see that, something is still publishing a host port.
- **Postgres, not SQLite.** SQLite locks under concurrent writes, which is exactly what
  5–10 simultaneous annotators produce. Losing an evening's work to a corrupt database
  costs more than the extra container.
- **`LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true`.** Registration is invite-only. The
  instance is on the public internet the moment Coolify assigns a domain, and an open
  signup form on a public URL gets found.

---

## 2. Project layout

Work is handed out in **batches of ~50**, not one big allocation each. A 50-sentence ask
(45–75 minutes) gets accepted and finished; a 200-sentence ask often gets accepted and
abandoned. People who finish a batch and enjoyed it can take another — the same person
doing three rounds over a week costs you one follow-up message and gets you the same data
as an annotator who never existed.

| Project | Sentences | Who | Predictions |
|---|---|---|---|
| `01-pilot` | 10 | **everyone** | assisted |
| `02-overlap-assisted` | 30 | group A | assisted |
| `03-overlap-scratch` | *the same* 30 | group B | **none** |
| `04-main-<name>` | 50 per round | one person each | assisted |

**One project per person for the main pass, reused across rounds.** Import 50 more tasks
into the same project when someone wants another batch — they keep one login and one
place to work, and you avoid a project explosion.

**Split annotators into two groups for projects 02 and 03.** Group A sees the model's
suggestions, group B labels the same sentences from scratch. Running the anchoring
comparison *between subjects* is cleaner than the same person doing both after a gap,
because nobody can half-remember their earlier answers. Projects 02 and 03 must hold
**the same sentences**, or the arms aren't comparable.

The pilot is deliberately separate from the overlap set and is **thrown away**. Its whole
job is to expose unclear rules, so its disagreements measure a guidelines problem, not
steady-state agreement — folding it into the headline κ would understate your agreement.

For each project: Settings → Labeling Interface → Code → paste
`deploy/label-studio/labeling-config.xml`.

---

## 3. Prepare batches

`scripts/prepare_batch.sh` does reserve → pre-label → task file in one go. Source your
credentials first:

```bash
set -a; source .env; set +a
```

**A whole round for five annotators:**

```bash
# 30 shared sentences for agreement, in both conditions
scripts/prepare_batch.sh overlap ov1 30 ana blerim drita edon fatos

# then 50 unique sentences each
for who in ana blerim drita edon fatos; do
  scripts/prepare_batch.sh unique "$who" r1 50
done
```

Roughly 3 minutes and 8 cents per 50-sentence batch, so a five-person round is about 20
minutes and 50 cents. Run it again with `r2`, `r3` … for later rounds; the ledger
guarantees nobody sees a sentence twice.

Overlap mode writes **two** task files from the same sentences —
`tasks_ov1_assisted.json` and `tasks_ov1_scratch.json`. Import the assisted one into
`02-overlap-assisted` (group A) and the scratch one into `03-overlap-scratch` (group B).
Same sentences, same interface, only the suggestions differ: that pair *is* the anchoring
experiment, and it collapses if the sentence sets diverge.

Check progress any time:

```bash
uv run python -m src.annotate.assign status --pool data/raw/wiki_segmented_v2.jsonl
```

### Doing it by hand

The script wraps three commands; run them directly if you want finer control. The ledger
is what stops a sentence being handed out twice (which inflates agreement) or never at
all (which silently shrinks the corpus) across the 20+ batches this campaign produces.

```bash
# 1. Reserve 50 sentences nobody has seen
uv run python -m src.annotate.assign reserve \
  --pool data/raw/wiki_segmented_v2.jsonl \
  --assignee ana --n 50 --batch r1 \
  --out data/interim/batch_ana_r1.jsonl

# 2. Pre-label just those 50 (~8 cents)
uv run python -m src.annotate.llm_label --model claude-sonnet-5 --no-thinking \
  --in data/interim/batch_ana_r1.jsonl \
  --out data/interim/prelabeled_ana_r1.jsonl --max-usd 0.50

# 3. Build the Label Studio task file
uv run python -m src.annotate.labelstudio tasks \
  --in data/interim/prelabeled_ana_r1.jsonl \
  --out data/interim/tasks_ana_r1.json --condition assisted
```

Then in that person's project: Import → drop the JSON.

**Pre-label per batch, never the whole pool.** You only pay for sentences somebody is
about to annotate, and you stay free to revise the prompt mid-campaign — a prompt edit
bumps `PROMPT_VERSION` and invalidates the response cache, so unlabelled sentences cost
nothing to re-plan.

For the overlap set, hand the *same* sentences to several people with `--overlap`:

```bash
# group A — assisted
uv run python -m src.annotate.assign reserve --pool data/raw/wiki_segmented_v2.jsonl \
  --overlap --assignee ana --assignee blerim --assignee drita \
  --n 30 --batch overlap-A --condition assisted --out data/interim/batch_ovA.jsonl

# group B — same sentences, no suggestions.
# Note --condition scratch on the tasks command; the reservation is bookkeeping only.
uv run python -m src.annotate.labelstudio tasks --condition scratch \
  --in data/interim/prelabeled_ovA.jsonl --out data/interim/tasks_ovB.json
```

Check progress at any time:

```bash
uv run python -m src.annotate.assign status --pool data/raw/wiki_segmented_v2.jsonl
```

```
pool 2500   assigned 310   remaining 2190

per annotator:
  ana                         80
  blerim                      80
...
```

---

## 4. Invite annotators

Organization → People → **Invite** → copy the link. Anyone with the link can create an
account; assign them to their project afterwards.

Send each person: the invite link, the URL of the
[annotation guidelines](./annotation-guidelines.md), and which project is theirs.

---

## 5. Collect results

Export → **JSON** (not JSON-MIN — the full format carries annotator identity and timing,
and time-per-sentence is one of the assisted-vs-scratch measurements).

```bash
uv run python -m src.annotate.labelstudio collect \
  --export ~/Downloads/project-2-at-2026-08-20.json \
  --source data/interim/prelabeled.jsonl \
  --out data/labeled/overlap_assisted.jsonl
```

Output is one record per *(sentence, annotator)* — so an overlap project with 4 annotators
over 150 sentences yields 600 records. That's the input the agreement code expects.

The command prints a breakdown of any problems it found. Two are worth acting on
immediately:

- **`span does not align to token boundaries`** — someone selected half a word. These are
  reported, never silently snapped to the nearest token, because snapping would invent
  a label nobody chose.
- **`overlapping entity spans`** — BIO can't represent them, so the second is dropped and
  logged.

---

## 6. What comes back

```json
{"id": "sq-0000", "tokens": ["Stacioni", "i", "Bramit", "…"],
 "spans": [{"start": 0, "end": 2, "head_start": 2, "head_end": 2, "type": "LOC",
            "surface": "Stacioni i Bramit", "head_surface": "Bramit"}],
 "flags": [], "junk": false,
 "annotator": "someone@example.com", "lead_time_s": 34.2}
```

`junk` sentences keep their record and are excluded when the gold set is built — see the
guidelines for why flagging beats deleting.

---

## Throughput and the size you actually need

Batches of 50 mean the corpus grows by rounds, not by headcount alone:

| | one round | two rounds | three rounds |
|---|---|---|---|
| 5 annotators | 250 | 500 | 750 |
| 8 annotators | 400 | 800 | 1200 |
| 10 annotators | 500 | 1000 | 1500 |

Plus whatever you annotate yourself — as the native speaker and adjudicator you are not
bound by the 50-sentence courtesy limit, and 300–400 of your own roughly doubles a
single-round campaign.

**Protect the test set.** Bootstrapped from the committed baseline predictions, the 95%
confidence interval on F1 is ±0.067 at 50 test sentences, ±0.050 at 100, ±0.036 at 250,
and ±0.028 at 400. Below ~150 a fine-tuned model at 0.85 and a baseline at 0.80 are
statistically indistinguishable, and the thesis result becomes "we couldn't tell."
Reserve **~250 for test** before allocating the rest to train and dev.

The adjudicated overlap sentences are the highest-quality data you will have — several
independent annotations reconciled by hand — so they are the natural core of the test
split rather than training filler.

The pool holds 2500 sentences, so it does not constrain any of these plans.
