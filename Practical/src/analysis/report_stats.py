"""Reproduce every report statistic that was not produced by another module.

Several headline numbers in the report were first computed in one-off scripts during
analysis: the anchoring permutation test, the annotation-time test, the relaxed-match
rescoring behind the delimitation claim, the entity-length comparison across corpora, and
the exact p-value on the four-model rank correlation. A number that only exists in a
terminal scrollback is not reproducible, so this module recomputes all of them from the
committed artefacts and writes them to results/report_stats.json.

    HF_HUB_OFFLINE=1 python -m src.analysis.report_stats
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from ..annotate.llm_label import spans_to_bio
from ..baselines import stats
from ..train.data import load_gold, load_wikiann

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTED = REPO_ROOT / "data" / "labeled" / "collected"
PRELABELED = REPO_ROOT / "data" / "interim"
PREDICTIONS = REPO_ROOT / "results" / "predictions"
GOLD = REPO_ROOT / "data" / "labeled" / "gold.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def exact_permutation(group_a: list[float], group_b: list[float]) -> dict:
    """Exact two-sample permutation test on the difference of means.

    Enumerates every way of splitting the pooled values into groups of the observed sizes,
    which is what makes the test exact rather than approximate; with five against five
    that is 252 splits and the smallest attainable one-tailed p is 1/252.
    """
    obs = st.mean(group_a) - st.mean(group_b)
    pool = group_a + group_b
    n_a = len(group_a)
    diffs = []
    for idx in combinations(range(len(pool)), n_a):
        chosen = set(idx)
        a = [pool[i] for i in chosen]
        b = [pool[i] for i in range(len(pool)) if i not in chosen]
        diffs.append(st.mean(a) - st.mean(b))
    one = sum(1 for d in diffs if d >= obs - 1e-12) / len(diffs)
    two = sum(1 for d in diffs if abs(d) >= abs(obs) - 1e-12) / len(diffs)
    return {
        "mean_a": round(st.mean(group_a), 4),
        "mean_b": round(st.mean(group_b), 4),
        "difference": round(obs, 4),
        "n_splits": len(diffs),
        "p_one_tailed": round(one, 5),
        "p_two_tailed": round(two, 5),
        "floor_one_tailed": round(1 / len(diffs), 5),
    }


# ------------------------------------------------------------------ anchoring


def condition_of(batch_stem: str) -> str:
    n = int(batch_stem.split("_")[1])
    return "assisted" if n % 2 else "scratch"


def anchoring_test() -> dict:
    """Each annotator's span-F1 against the model's pre-labels on the shared sentences,
    compared between the assisted and from-scratch groups."""
    model = {}
    for path in PRELABELED.glob("prelabeled_[0-9]*.jsonl"):
        for rec in read_jsonl(path):
            model[rec["id"]] = rec

    by_sentence: dict[str, dict[str, tuple[dict, str]]] = defaultdict(dict)
    tokens: dict[str, list[str]] = {}
    for path in sorted(COLLECTED.glob("batch_*.jsonl")):
        cond = condition_of(path.stem)
        for rec in read_jsonl(path):
            by_sentence[rec["id"]][rec["annotator"]] = (rec, cond)
            tokens[rec["id"]] = rec["tokens"]
    shared = {sid: by for sid, by in by_sentence.items() if len(by) > 1}

    out = {"n_shared": len(shared), "views": {}}
    for mode in ("full", "head"):
        per: dict[str, list] = defaultdict(list)
        cond_of: dict[str, str] = {}
        for sid, by in shared.items():
            mt = model[sid]["llm_tags_full" if mode == "full" else "llm_tags_head"]
            for who, (rec, cond) in by.items():
                at = spans_to_bio(rec["spans"], len(tokens[sid]), mode)
                per[who].append(stats.sentence_counts([at], [mt])[0])
                cond_of[who] = cond
        f1 = {
            who: stats.prf_from_counts(np.array(rows).sum(axis=0).reshape(1, -1))[2]
            for who, rows in per.items()
        }
        assisted = sorted(v for w, v in f1.items() if cond_of[w] == "assisted")
        scratch = sorted(v for w, v in f1.items() if cond_of[w] == "scratch")
        out["views"][mode] = {
            "per_annotator": {
                w: {"f1_vs_model": round(v, 4), "condition": cond_of[w]} for w, v in f1.items()
            },
            "assisted": [round(v, 3) for v in assisted],
            "scratch": [round(v, 3) for v in scratch],
            "groups_overlap": max(scratch) >= min(assisted),
            "test": exact_permutation(assisted, scratch),
        }
    return out


# ------------------------------------------------------------------ time


def time_test() -> dict:
    """Annotation time by condition, tested at the annotator level.

    Per-sentence times are not independent: one slow person contributes a hundred slow
    sentences. The unit is therefore the annotator's median, and the test is exact over
    those. The naive per-sentence figure is reported alongside to show what treating
    sentences as independent would have claimed.
    """
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_times: dict[str, list[float]] = defaultdict(list)
    for path in sorted(COLLECTED.glob("batch_*.jsonl")):
        cond = condition_of(path.stem)
        for rec in read_jsonl(path):
            t = rec.get("lead_time_s")
            if t:
                per[(rec["annotator"], cond)].append(t)
                all_times[cond].append(t)
    # An annotator with a single record in a condition has no median worth using.
    medians = {k: st.median(v) for k, v in per.items() if len(v) >= 2}
    assisted = sorted(m for (a, c), m in medians.items() if c == "assisted")
    scratch = sorted(m for (a, c), m in medians.items() if c == "scratch")
    naive_a, naive_s = st.median(all_times["assisted"]), st.median(all_times["scratch"])
    return {
        "per_annotator_medians": {f"{a} ({c})": round(m, 1) for (a, c), m in sorted(medians.items())},
        "assisted_medians": assisted,
        "scratch_medians": scratch,
        "test_on_annotator_medians": exact_permutation(scratch, assisted),
        "naive_per_sentence": {
            "median_assisted_s": round(naive_a, 1),
            "median_scratch_s": round(naive_s, 1),
            "apparent_speedup_pct": round(100 * (naive_s - naive_a) / naive_s, 1),
            "note": "treats 1000 sentences as independent; not the reported test",
        },
    }


# ------------------------------------------------------------------ delimitation


def relaxed_rescoring(pred_file: str) -> dict:
    """Exact-match F1 next to a relaxed rule where same-type overlap counts as a hit.

    The relaxed rule is diagnostic only. It prices how much of a model's error is
    delimitation rather than detection; it is not a metric used anywhere in the report.
    """
    tp = fp = fn = 0
    rtp = rfp = rfn = 0
    for rec in read_jsonl(PREDICTIONS / pred_file):
        gold = set(get_entities(rec["gold"]))
        pred = set(get_entities(rec["pred"]))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        gold_list, used, hits = list(gold), set(), 0
        for ptype, ps, pe in pred:
            for j, (gtype, gs, ge) in enumerate(gold_list):
                if j in used or gtype != ptype:
                    continue
                if ps <= ge and gs <= pe:
                    used.add(j)
                    hits += 1
                    break
        rtp += hits
        rfp += len(pred) - hits
        rfn += len(gold) - hits

    def f1(t, p, n):
        return round(2 * t / (2 * t + p + n), 4) if (2 * t + p + n) else 0.0

    return {
        "exact": {"tp": tp, "fp": fp, "fn": fn, "f1": f1(tp, fp, fn)},
        "relaxed": {"tp": rtp, "fp": rfp, "fn": rfn, "f1": f1(rtp, rfp, rfn)},
        "spans_overlapping_but_inexact": rtp - tp,
        "cost_of_exact_matching": round(f1(rtp, rfp, rfn) - f1(tp, fp, fn), 4),
    }


# ------------------------------------------------------------------ span lengths


def span_lengths(examples) -> list[int]:
    out = []
    for e in examples:
        for _, start, end in get_entities(e.tags):
            out.append(end - start + 1)
    return out


def entity_length_comparison() -> dict:
    wk = span_lengths(load_wikiann("test"))
    res = {"wikiann_test": {"n": len(wk), "mean": round(st.mean(wk), 2), "median": st.median(wk)}}
    for mode in ("full", "head"):
        g = span_lengths(load_gold(GOLD, mode))
        res[f"gold_{mode}"] = {"n": len(g), "mean": round(st.mean(g), 2), "median": st.median(g)}
    return res


# ------------------------------------------------------------------ rank correlation


def rank_correlation() -> dict:
    """Spearman rho between the two benchmark orderings, with its exact p-value.

    Four models give 24 possible orderings, so the most extreme rho attainable is -1 at
    one-tailed p = 1/24. The statistic is reported in the paper as description, not
    evidence, and this is why.
    """
    w = {
        r["model"]: r["overall_f1"]
        for r in json.load((REPO_ROOT / "results" / "baselines.json").open())["models"]
    }
    g = {
        r["model"]: r["overall_f1"]
        for r in json.load((REPO_ROOT / "results" / "gold-baselines.json").open())["models"]
    }
    models = sorted(set(w) & set(g))
    rank_w = {m: i for i, m in enumerate(sorted(models, key=lambda m: -w[m]))}
    rank_g = {m: i for i, m in enumerate(sorted(models, key=lambda m: -g[m]))}
    n = len(models)
    d2 = sum((rank_w[m] - rank_g[m]) ** 2 for m in models)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    dist = [
        1 - 6 * sum((i - p[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) for p in permutations(range(n))
    ]
    one = sum(1 for v in dist if v <= rho + 1e-9) / len(dist)
    two = sum(1 for v in dist if abs(v) >= abs(rho) - 1e-9) / len(dist)
    return {
        "n_models": n,
        "rho": round(rho, 3),
        "p_one_tailed": round(one, 4),
        "p_two_tailed": round(two, 4),
        "min_attainable_p_one_tailed": round(1 / len(dist), 4),
        "changes": {m: round(g[m] - w[m], 3) for m in models},
    }


# ------------------------------------------------------------------ head sensitivity


def _subset_f1(counts: np.ndarray, idx) -> float:
    """Micro F1 over a subset of per-sentence (tp, fp, fn) rows."""
    return round(stats.prf_from_counts(np.array([counts[i] for i in idx]).sum(axis=0).reshape(1, -1))[2], 4)


def head_sensitivity(annotator: str = "11") -> dict:
    """Score the head-convention predictions on subsets of the test split.

    Same model, same predictions; only the scoring subset varies. That isolates the
    effect of one annotator's sentences on evaluation. Training a separate model with
    those sentences removed would not answer this, because it changes the split, the
    training size and the test set at once.
    """
    from ..train.data import split_examples

    gold = {r["id"]: r for r in read_jsonl(GOLD)}
    _, _, test = split_examples(load_gold(GOLD, "head"), 0.15, 0.25, 20260811)
    ids = [e.source for e in test]
    flagged = [
        gold.get(s, {}).get("annotator") == annotator and gold.get(s, {}).get("n_annotators", 1) == 1
        for s in ids
    ]
    out = {"annotator": annotator, "n_test": len(ids), "n_flagged": sum(flagged), "seeds": []}
    for seed in (1, 2, 3):
        rows = read_jsonl(PREDICTIONS / f"gold-head-seed{seed}.jsonl")
        counts = stats.sentence_counts([r["pred"] for r in rows], [r["gold"] for r in rows])
        out["seeds"].append(
            {
                "seed": seed,
                "all": _subset_f1(counts, range(len(ids))),
                "excl_flagged": _subset_f1(counts, [i for i, f in enumerate(flagged) if not f]),
                "flagged_only": _subset_f1(counts, [i for i, f in enumerate(flagged) if f]),
            }
        )
    for k in ("all", "excl_flagged", "flagged_only"):
        out[f"{k}_mean"] = round(float(np.mean([s[k] for s in out["seeds"]])), 4)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "report_stats.json")
    args = p.parse_args()

    report = {
        "anchoring": anchoring_test(),
        "annotation_time": time_test(),
        "delimitation": {
            "gold-full-seed1": relaxed_rescoring("gold-full-seed1.jsonl"),
            "gold-head-seed1": relaxed_rescoring("gold-head-seed1.jsonl"),
        },
        "entity_lengths": entity_length_comparison(),
        "rank_correlation": rank_correlation(),
        "head_sensitivity": head_sensitivity(),
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    a = report["anchoring"]["views"]["full"]
    print(
        f"anchoring (full): assisted {a['test']['mean_a']} vs scratch {a['test']['mean_b']}, "
        f"p={a['test']['p_one_tailed']} one-tailed (floor {a['test']['floor_one_tailed']}), "
        f"overlap={a['groups_overlap']}"
    )
    t = report["annotation_time"]["test_on_annotator_medians"]
    print(
        f"time (annotator medians): scratch {t['mean_a']}s vs assisted {t['mean_b']}s, p={t['p_one_tailed']}"
    )
    for k, v in report["delimitation"].items():
        print(
            f"delimitation {k}: exact {v['exact']['f1']} relaxed {v['relaxed']['f1']} "
            f"(relaxed tp {v['relaxed']['tp']})"
        )
    e = report["entity_lengths"]
    print(
        f"entity length: wikiann {e['wikiann_test']['mean']}, gold full {e['gold_full']['mean']}, "
        f"gold head {e['gold_head']['mean']}"
    )
    r = report["rank_correlation"]
    print(f"rank correlation: rho {r['rho']}, p={r['p_one_tailed']} one-tailed")
    h = report["head_sensitivity"]
    print(
        f"head sensitivity: all {h['all_mean']} | excl. annotator {h['annotator']} {h['excl_flagged_mean']} "
        f"| theirs only {h['flagged_only_mean']} (n={h['n_flagged']})"
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
