"""Does the hand-annotated gold set actually earn its keep?

Three training conditions, one held-out test set:

- `gold_only`          — train on the gold set alone
- `wikiann_only`       — train on WikiANN-sq alone, test on gold
- `gold_plus_wikiann`  — train on both

The comparison only means something because the **test set is identical across all
three**: the gold test split, fixed by `--split-seed` and never touched by training. What
varies is exclusively the training data.

Two questions this answers, both of which belong in the write-up:

1. *Is the annotation effort justified?* `gold_only` vs `wikiann_only` says whether a few
   hundred hand-verified sentences beat five thousand automatically-projected ones.
2. *Should WikiANN be used as extra training data?* `gold_plus_wikiann` vs `gold_only`.
   This is not free: WikiANN-sq sentences have a median length of 6 tokens against 20 in
   the gold pool, and carry markup artifacts and wiki directives, so more data also means
   a distribution shift. It could help or hurt, which is why it is measured rather than
   assumed.

Every condition is also scored on WikiANN's own test set, so the cross-domain direction
is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..baselines import stats
from ..baselines.envinfo import describe, hardware_info, software_info
from ..baselines.metrics import resolve_device
from .data import Example, load_gold, load_wikiann, split_examples
from .finetune import f1_of, free_cache, predict, train_one_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

CONDITIONS = ("gold_only", "wikiann_only", "gold_plus_wikiann")


def training_set(
    condition: str, gold_train: list[Example], wikiann_train: list[Example]
) -> list[Example]:
    if condition == "gold_only":
        return list(gold_train)
    if condition == "wikiann_only":
        return list(wikiann_train)
    if condition == "gold_plus_wikiann":
        return gold_train + wikiann_train
    raise ValueError(f"unknown condition {condition!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gold", type=Path, required=True, help="Adjudicated gold JSONL.")
    p.add_argument("--mode", choices=["full", "head"], default="full")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=20260811)
    p.add_argument("--wikiann-limit", type=int, default=None,
                   help="Cap the WikiANN training data (for a cheaper run).")
    p.add_argument("--out-prefix", default="ablation")
    args = p.parse_args()

    gold = load_gold(args.gold, args.mode)
    gold_train, dev, test = split_examples(gold, args.dev_frac, args.test_frac, args.split_seed)
    wikiann_train = load_wikiann("train")
    if args.wikiann_limit:
        wikiann_train = wikiann_train[: args.wikiann_limit]
    wikiann_test = load_wikiann("test")

    device = resolve_device()
    hardware = hardware_info(device)
    refs = [e.tags for e in test]
    wikiann_refs = [e.tags for e in wikiann_test]

    print(f"gold {args.gold.name} ({args.mode}): train {len(gold_train)}, dev {len(dev)}, "
          f"test {len(test)}   |   WikiANN train {len(wikiann_train)}")
    print(f"{len(args.conditions)} conditions x {len(args.seeds)} seeds on "
          f"{hardware['accelerator']}\n")

    results: dict[str, dict] = {}
    best_preds: dict[str, list[list[str]]] = {}
    t0 = time.perf_counter()

    for condition in args.conditions:
        train = training_set(condition, gold_train, wikiann_train)
        scores, wikiann_scores, per_seed = [], [], []
        best_dev, best_pred = -1.0, None

        for seed in args.seeds:
            t_seed = time.perf_counter()
            model, tokenizer, hist = train_one_seed(seed, train, dev, args, device)
            preds = predict(model, tokenizer, test, device, args.max_length)
            gold_f1 = f1_of(preds, refs)
            wikiann_f1 = f1_of(
                predict(model, tokenizer, wikiann_test, device, args.max_length), wikiann_refs
            )
            elapsed = time.perf_counter() - t_seed

            scores.append(gold_f1)
            wikiann_scores.append(wikiann_f1)
            per_seed.append({"seed": seed, "gold_test_f1": round(gold_f1, 4),
                             "wikiann_test_f1": round(wikiann_f1, 4),
                             "best_dev_f1": round(hist["best_dev_f1"], 4),
                             "n_train": len(train), "seconds": round(elapsed, 1)})
            # Keep the best-by-dev seed's predictions: selecting on test would leak.
            if hist["best_dev_f1"] > best_dev:
                best_dev, best_pred = hist["best_dev_f1"], preds
            print(f"  [{condition}] seed {seed}: gold {gold_f1:.4f}  "
                  f"wikiann {wikiann_f1:.4f}  ({elapsed:.0f}s)")

            del model
            free_cache(device)

        best_preds[condition] = best_pred
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        with (PREDICTIONS_DIR / f"{args.out_prefix}-{condition}.jsonl").open("w") as fh:
            for i, (pred, ref) in enumerate(zip(best_pred, refs, strict=True)):
                fh.write(json.dumps({"id": i, "gold": ref, "pred": pred}) + "\n")

        results[condition] = {
            "n_train": len(train),
            "gold_test_f1_mean": round(float(np.mean(scores)), 4),
            "gold_test_f1_std": round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 4),
            "wikiann_test_f1_mean": round(float(np.mean(wikiann_scores)), 4),
            "per_seed": per_seed,
        }
        print()

    # Paired bootstrap between every pair of conditions on the shared gold test set.
    indices = stats.resample_indices(len(refs), stats.DEFAULT_RESAMPLES, stats.DEFAULT_SEED)
    comparisons = []
    ordered = [c for c in args.conditions]
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            result = stats.paired_bootstrap(
                a, stats.sentence_counts(best_preds[a], refs),
                b, stats.sentence_counts(best_preds[b], refs), indices,
            )
            comparisons.append(result.to_dict())

    total_s = time.perf_counter() - t0
    report = {
        "gold": str(args.gold), "mode": args.mode, "model": args.model,
        "seeds": args.seeds, "n_gold_train": len(gold_train),
        "n_dev": len(dev), "n_test": len(test), "n_wikiann_train": len(wikiann_train),
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        "conditions": results, "comparisons": comparisons,
        "hardware": hardware, "software": software_info(),
        "total_compute_s": round(total_s, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.out_prefix}.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report, RESULTS_DIR / f"{args.out_prefix}.md")

    print("condition             n_train   gold test F1      wikiann test F1")
    for condition, r in results.items():
        print(f"  {condition:20s} {r['n_train']:6d}   "
              f"{r['gold_test_f1_mean']:.4f} ± {r['gold_test_f1_std']:.4f}   "
              f"{r['wikiann_test_f1_mean']:.4f}")
    print("\npaired comparisons on the gold test set:")
    for c in comparisons:
        verdict = "significant" if c["significant"] else "not distinguishable"
        print(f"  {c['model_a']:20s} vs {c['model_b']:20s} {c['delta']:+.4f}  "
              f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  {verdict}")
    print(f"\n{describe(hardware, total_s)}\n→ results/{args.out_prefix}.md")
    return 0


def _write_markdown(r: dict, path: Path) -> None:
    lines = [
        "# Does the gold set earn its keep?", "",
        f"All three conditions are evaluated on the **same** held-out gold test split "
        f"({r['n_test']} sentences, split seed fixed). Only the training data varies.", "",
        f"{r['model']}, seeds {r['seeds']}, {r['hyperparams']['epochs']} epochs.", "",
        "| Condition | Train size | Gold test F1 | WikiANN test F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition, c in r["conditions"].items():
        lines.append(
            f"| `{condition}` | {c['n_train']} | "
            f"{c['gold_test_f1_mean']:.4f} ± {c['gold_test_f1_std']:.4f} | "
            f"{c['wikiann_test_f1_mean']:.4f} |"
        )
    lines += [
        "",
        "## Are the differences real?",
        "",
        "Paired bootstrap on the gold test set, using each condition's best-by-dev seed. "
        "A gap whose interval includes zero is not a result.", "",
        "| A | B | ΔF1 | 95% CI | verdict |",
        "| --- | --- | ---: | ---: | :--- |",
    ]
    for c in r["comparisons"]:
        verdict = "significant" if c["significant"] else "not distinguishable"
        lines.append(
            f"| `{c['model_a']}` | `{c['model_b']}` | {c['delta']:+.4f} | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {verdict} |"
        )
    lines += [
        "",
        f"- Hardware: {r['hardware']['accelerator']} ({r['hardware']['device']})",
        f"- Total compute: {r['total_compute_s'] / 60:.1f} min", "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
