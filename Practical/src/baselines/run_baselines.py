"""Run a fleet of HF NER models on the WikiANN `sq` test split.

Every reported F1 comes with a bootstrap confidence interval over test sentences, and
every pair of models is compared with a paired bootstrap test, so the leaderboard says
which gaps are real and which are sampling noise. See `stats.py` for why bootstrap
rather than seed variance.

Outputs:
    results/baselines.md              human-readable leaderboard + significance table
    results/baselines.json            raw metrics, CIs, comparisons, environment
    results/predictions/<model>.jsonl per-sentence gold/predicted tags
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from . import stats
from .envinfo import describe, hardware_info, software_info
from .metrics import resolve_device, wikiann_ids_to_bio
from .models import MODELS, ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

ENTITY_TYPES = ("PER", "ORG", "LOC")


def _jsonable(o):
    """Coerce numpy scalars to native Python for json.dumps."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _remap_tag(tag: str, label_map: dict[str, str]) -> str:
    """Apply the per-model entity-type rename + drop map to a BIO tag.

    Any tag that isn't `O` or `B-/I-X` (e.g. Kushtrim's stray `MISC` at id 0)
    is treated as O. Unknown entity types in `label_map` also become O.
    """
    if tag == "O":
        return "O"
    if "-" not in tag:
        return "O"
    prefix, etype = tag.split("-", 1)
    if prefix not in {"B", "I"}:
        return "O"
    mapped = label_map.get(etype, "O")
    if mapped == "O":
        return "O"
    return f"{prefix}-{mapped}"


def predict(spec: ModelSpec, dataset, label_names, device) -> tuple[list, list, float, float] | None:
    """Word-level tag predictions for every sentence.

    Returns (preds, refs, load_seconds, inference_seconds), or None if the checkpoint
    could not be fetched (gated repo, no network) so the rest of the fleet can proceed.
    """
    print(f"\n=== {spec.name} ({spec.hf_repo}) ===", flush=True)
    t_load = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo)
        model = AutoModelForTokenClassification.from_pretrained(spec.hf_repo)
    except Exception as e:
        msg = str(e).splitlines()[0]
        print(f"  SKIP: could not load {spec.hf_repo}: {msg}", file=sys.stderr)
        return None
    model = model.to(device).eval()
    load_s = time.perf_counter() - t_load

    id2label = model.config.id2label

    # Surface entity types this model predicts that the label_map silently drops.
    model_types = {
        tag.split("-", 1)[1]
        for tag in id2label.values()
        if "-" in tag and tag.split("-", 1)[0] in {"B", "I"}
    }
    unmapped = sorted(model_types - spec.label_map.keys())
    if unmapped:
        print(f"  note: unmapped entity types {unmapped} will be scored as O", flush=True)

    preds: list[list[str]] = []
    refs: list[list[str]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for row in tqdm(dataset, desc=spec.name, unit="sent"):
            tokens: list[str] = row["tokens"]
            gold = wikiann_ids_to_bio(label_names, row["ner_tags"])

            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            word_ids = enc.word_ids(batch_index=0)
            inputs = {k: v.to(device) for k, v in enc.items()}
            logits = model(**inputs).logits[0]
            sub_pred_ids = logits.argmax(dim=-1).tolist()

            # "first-subword" aggregation: each word gets the label of its first subtoken.
            tok_tags: list[str] = ["O"] * len(tokens)
            seen: set[int] = set()
            for sub_idx, wid in enumerate(word_ids):
                if wid is None or wid in seen or wid >= len(tokens):
                    continue
                seen.add(wid)
                tag = id2label[sub_pred_ids[sub_idx]]
                tok_tags[wid] = _remap_tag(tag, spec.label_map)
            # Truncation safety: any word missing a subword (rare for <40-token sents) stays "O".

            preds.append(tok_tags)
            refs.append(gold)
    inference_s = time.perf_counter() - t0
    return preds, refs, load_s, inference_s


def score(spec: ModelSpec, preds, refs, indices, alpha: float, seqeval) -> tuple[dict, dict]:
    """Point estimates + bootstrap CIs for one model.

    Returns (summary, counts_by_key) where counts_by_key holds the per-sentence count
    matrices ("overall" plus one per entity type) that the pairwise tests reuse.
    """
    counts = {"overall": stats.sentence_counts(preds, refs)}
    for ent in ENTITY_TYPES:
        counts[ent] = stats.sentence_counts(preds, refs, entity_type=ent)

    precision, recall, f1 = stats.prf_from_counts(counts["overall"])

    # Cross-check the count-based micro-F1 against seqeval's own aggregation. They should
    # agree exactly; a mismatch means the span extraction has drifted apart.
    reference = seqeval.compute(predictions=preds, references=refs, zero_division=0)
    if abs(reference["overall_f1"] - f1) > 1e-9:
        print(
            f"  WARNING: count-based F1 {f1:.6f} != seqeval {reference['overall_f1']:.6f}",
            file=sys.stderr,
        )

    lo, hi = stats.percentile_ci(stats.bootstrap_f1(counts["overall"], indices), alpha)
    summary = {
        "model": spec.name,
        "hf_repo": spec.hf_repo,
        "n_sentences": len(refs),
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "overall_f1_ci_low": lo,
        "overall_f1_ci_high": hi,
        "overall_accuracy": reference["overall_accuracy"],
    }
    for ent in ENTITY_TYPES:
        ent_p, ent_r, ent_f1 = stats.prf_from_counts(counts[ent])
        ent_lo, ent_hi = stats.percentile_ci(stats.bootstrap_f1(counts[ent], indices), alpha)
        summary[f"{ent}_precision"] = ent_p
        summary[f"{ent}_recall"] = ent_r
        summary[f"{ent}_f1"] = ent_f1
        summary[f"{ent}_f1_ci_low"] = ent_lo
        summary[f"{ent}_f1_ci_high"] = ent_hi
        # Gold span count: the denominator behind how much this class can move F1.
        summary[f"{ent}_support"] = int(counts[ent][:, [0, 2]].sum())
    return summary, counts


def write_markdown(rows: list[dict], comparisons: list, out_path: Path, meta: dict) -> None:
    alpha = meta["bootstrap"]["alpha"]
    conf = round((1 - alpha) * 100)
    lines = [
        "# WikiANN-sq baselines",
        "",
        f"Evaluated on {meta['n_sentences']} sentences from the `test` split of "
        "`unimelb-nlp/wikiann` (`sq`).",
        "",
        f"F1 is micro-averaged over entity spans. Brackets give a {conf}% bootstrap "
        f"confidence interval from {meta['bootstrap']['n_resamples']} resamples of the test "
        "sentences (seed "
        f"{meta['bootstrap']['seed']}); inference itself is deterministic, so this interval "
        "reflects test-set sampling variability, not run-to-run noise.",
        "",
        f"| Model | P | R | F1 [{conf}% CI] | F1-PER | F1-ORG | F1-LOC | infer (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model']}` "
            f"| {r['overall_precision']:.3f} "
            f"| {r['overall_recall']:.3f} "
            f"| **{r['overall_f1']:.3f}** [{r['overall_f1_ci_low']:.3f}, "
            f"{r['overall_f1_ci_high']:.3f}] "
            f"| {r['PER_f1']:.3f} "
            f"| {r['ORG_f1']:.3f} "
            f"| {r['LOC_f1']:.3f} "
            f"| {r['inference_s']:.0f} |"
        )

    lines += [
        "",
        "## Pairwise significance",
        "",
        "Paired bootstrap on the same resampled sentences for both models, so the shared "
        "difficulty of a resample cancels out of the difference. A gap is significant when "
        "the CI on the difference excludes zero.",
        "",
        f"| A | B | ΔF1 (A−B) | {conf}% CI | p | significant |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for c in comparisons:
        p_str = "< 0.001" if c.p_value < 0.001 else f"{c.p_value:.3f}"
        lines.append(
            f"| `{c.model_a}` | `{c.model_b}` | {c.delta:+.3f} "
            f"| [{c.ci_low:+.3f}, {c.ci_high:+.3f}] | {p_str} "
            f"| {'yes' if c.significant else 'no'} |"
        )

    hw = meta["hardware"]
    lines += [
        "",
        "## Environment",
        "",
        f"- Hardware: {hw['accelerator']} ({hw['device']}), {hw['cpu_count']} CPU cores, "
        f"{hw['total_ram_gb']} GB RAM",
        f"- Platform: {hw['platform']}, Python {hw['python']}",
        f"- Libraries: torch {meta['software']['torch']}, "
        f"transformers {meta['software']['transformers']}, "
        f"datasets {meta['software']['datasets']}",
        f"- Total compute: {meta['total_compute_s'] / 60:.1f} min "
        f"({meta['total_inference_s'] / 60:.1f} min inference, "
        f"{meta['total_load_s'] / 60:.1f} min model loading)",
        f"- Run date: {meta['date']}",
        "",
        f"> {describe(hw, meta['total_compute_s'])}",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test sentences (smoke testing).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of ModelSpec.name values to restrict the run to.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=stats.DEFAULT_RESAMPLES,
        help="Bootstrap resamples for confidence intervals and significance tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=stats.DEFAULT_ALPHA,
        help="Significance level; 0.05 gives 95%% intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=stats.DEFAULT_SEED,
        help="Seed for the bootstrap resampling, so intervals reproduce exactly.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="baselines",
        help="Basename for the result files, to avoid overwriting a full run with a smoke test.",
    )
    args = parser.parse_args()

    device = resolve_device()
    hardware = hardware_info(device)
    print(f"Device: {device} ({hardware['accelerator']})", flush=True)

    ds = load_dataset("unimelb-nlp/wikiann", "sq", split="test")
    label_names = ds.features["ner_tags"].feature.names
    print(f"WikiANN sq test: {len(ds)} sentences, labels={label_names}", flush=True)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limiting to {len(ds)} sentences", flush=True)

    # One shared resample matrix for every model — this is what makes comparisons paired.
    indices = stats.resample_indices(len(ds), args.bootstrap, args.seed)
    seqeval = evaluate.load("seqeval")

    only = set(args.only.split(",")) if args.only else None
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    counts_by_model: dict[str, dict] = {}
    total_load_s = 0.0
    total_inference_s = 0.0

    for spec in MODELS:
        if only and spec.name not in only:
            continue
        outcome = predict(spec, ds, label_names, device)
        if outcome is None:
            continue
        preds, refs, load_s, inference_s = outcome
        total_load_s += load_s
        total_inference_s += inference_s

        summary, counts = score(spec, preds, refs, indices, args.alpha, seqeval)
        summary["load_s"] = round(load_s, 1)
        summary["inference_s"] = round(inference_s, 1)
        summary["ms_per_sentence"] = round(1000 * inference_s / max(len(refs), 1), 1)
        rows.append(summary)
        counts_by_model[spec.name] = counts

        # Persist predictions so the scoring and stats can be re-derived without a GPU.
        with (PREDICTIONS_DIR / f"{spec.name}.jsonl").open("w") as fh:
            for i, (pred, ref) in enumerate(zip(preds, refs, strict=True)):
                fh.write(json.dumps({"id": i, "gold": ref, "pred": pred}) + "\n")

        print(
            f"  F1={summary['overall_f1']:.3f} "
            f"[{summary['overall_f1_ci_low']:.3f}, {summary['overall_f1_ci_high']:.3f}] "
            f"in {inference_s:.0f}s",
            flush=True,
        )

    rows.sort(key=lambda r: r["overall_f1"], reverse=True)
    comparisons = [
        stats.paired_bootstrap(
            a["model"],
            counts_by_model[a["model"]]["overall"],
            b["model"],
            counts_by_model[b["model"]]["overall"],
            indices,
            args.alpha,
        )
        for a, b in combinations(rows, 2)
    ]

    import datetime

    meta = {
        "date": datetime.date.today().isoformat(),
        "dataset": "unimelb-nlp/wikiann (sq, test)",
        "n_sentences": len(ds),
        "hardware": hardware,
        "software": software_info(),
        "bootstrap": {
            "n_resamples": args.bootstrap,
            "alpha": args.alpha,
            "seed": args.seed,
            "method": "percentile bootstrap over test sentences; paired for comparisons",
        },
        "total_load_s": round(total_load_s, 1),
        "total_inference_s": round(total_inference_s, 1),
        "total_compute_s": round(total_load_s + total_inference_s, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "models": rows,
        "comparisons": [c.to_dict() for c in comparisons],
    }
    (RESULTS_DIR / f"{args.out_prefix}.json").write_text(
        json.dumps(payload, indent=2, default=_jsonable)
    )
    write_markdown(rows, comparisons, RESULTS_DIR / f"{args.out_prefix}.md", meta)

    print("\nSummary:")
    for r in rows:
        print(
            f"  {r['model']:30s} F1={r['overall_f1']:.3f} "
            f"[{r['overall_f1_ci_low']:.3f}, {r['overall_f1_ci_high']:.3f}] "
            f"({r['inference_s']:.0f}s)"
        )
    print(f"\n{describe(hardware, meta['total_compute_s'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
