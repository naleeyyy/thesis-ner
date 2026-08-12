"""Active learning: does picking *which* sentences to label beat picking at random?

Run as a **simulation over the already-labelled pool** — labels are hidden and revealed
only as a strategy asks for them. That is standard practice for AL papers and it is the
only practical option here: a live loop would need annotators on standby between rounds,
and it could not be re-run with a different seed.

The result that matters is a learning curve with error bars: F1 against number of labelled
sentences, for each strategy, averaged over seeds. A single run of each strategy proves
nothing, because the gap between strategies is usually smaller than the run-to-run spread
of either one.

Compute warning: each point on the curve is a full retrain, so the cost is
`strategies x seeds x rounds` trainings. Defaults are deliberately modest.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..baselines.envinfo import describe, hardware_info, software_info
from ..baselines.metrics import resolve_device
from .data import Example, load_gold, load_wikiann, split_examples
from .finetune import f1_of, predict, set_seed, train_one_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

STRATEGIES = ("random", "uncertainty")


@torch.no_grad()
def sentence_uncertainty(model, tokenizer, examples: list[Example], device, max_length: int) -> list[float]:
    """Mean per-token predictive entropy, one score per sentence.

    Mean rather than sum, so the strategy doesn't simply prefer long sentences — that
    would confound "informative" with "more tokens to annotate" and quietly turn the
    comparison into a budget mismatch.
    """
    model.eval()
    scores: list[float] = []
    batch_size = 64
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        enc = tokenizer(
            [ex.tokens for ex in chunk], is_split_into_words=True, return_tensors="pt",
            truncation=True, max_length=max_length, padding=True,
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1).cpu()

        for i in range(len(chunk)):
            # One value per word (first subword only), matching how predictions are read.
            # Padding positions have word_id None and are skipped, so they never dilute
            # the mean.
            seen: set[int] = set()
            per_word = []
            for idx, wid in enumerate(enc.word_ids(batch_index=i)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                per_word.append(entropy[i][idx].item())
            scores.append(float(np.mean(per_word)) if per_word else 0.0)
    return scores


def select(
    strategy: str,
    unlabelled: list[int],
    k: int,
    rng: random.Random,
    scores: list[float] | None = None,
) -> list[int]:
    """Indices to reveal this round."""
    if k >= len(unlabelled):
        return list(unlabelled)
    if strategy == "random":
        return rng.sample(unlabelled, k)
    if strategy == "uncertainty":
        if scores is None:
            raise ValueError("uncertainty selection needs scores")
        ranked = sorted(unlabelled, key=lambda i: scores[i], reverse=True)
        return ranked[:k]
    raise ValueError(f"unknown strategy {strategy!r}")


@dataclass
class RoundResult:
    round_index: int
    n_labelled: int
    test_f1: float
    seconds: float


def run_loop(
    strategy: str,
    seed: int,
    pool: list[Example],
    dev: list[Example],
    test: list[Example],
    args,
    device,
) -> list[RoundResult]:
    """One strategy, one seed: seed set, then `rounds` acquire-and-retrain cycles."""
    rng = random.Random(seed)
    set_seed(seed)

    labelled = rng.sample(range(len(pool)), min(args.seed_size, len(pool)))
    unlabelled = [i for i in range(len(pool)) if i not in set(labelled)]
    refs = [e.tags for e in test]
    out: list[RoundResult] = []

    for round_index in range(args.rounds + 1):
        t0 = time.perf_counter()
        train_set = [pool[i] for i in labelled]
        model, tokenizer, _ = train_one_seed(seed, train_set, dev, args, device)
        test_f1 = f1_of(predict(model, tokenizer, test, device, args.max_length), refs)
        elapsed = time.perf_counter() - t0
        out.append(RoundResult(round_index, len(labelled), test_f1, round(elapsed, 1)))
        print(f"  [{strategy} seed {seed}] round {round_index}: "
              f"{len(labelled)} labelled → test F1 {test_f1:.4f} ({elapsed:.0f}s)")

        if round_index == args.rounds or not unlabelled:
            break

        scores = None
        if strategy == "uncertainty":
            # Score only the unlabelled remainder — scoring the whole pool would waste
            # a forward pass over sentences that cannot be selected.
            candidate_examples = [pool[i] for i in unlabelled]
            raw = sentence_uncertainty(model, tokenizer, candidate_examples, device, args.max_length)
            scores = [0.0] * len(pool)
            for idx, value in zip(unlabelled, raw, strict=True):
                scores[idx] = value

        picked = select(strategy, unlabelled, args.acquire, rng, scores)
        labelled.extend(picked)
        picked_set = set(picked)
        unlabelled = [i for i in unlabelled if i not in picked_set]

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gold", type=Path, default=None)
    p.add_argument("--wikiann", action="store_true",
                   help="Use WikiANN-sq as the pool. Proves the loop before gold exists.")
    p.add_argument("--mode", choices=["full", "head"], default="full")
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seed-size", type=int, default=100, help="Initial labelled set.")
    p.add_argument("--acquire", type=int, default=100, help="Sentences revealed per round.")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--pool-limit", type=int, default=None, help="Cap the unlabelled pool.")
    p.add_argument(
        "--dev-limit", type=int, default=None,
        help="Cap the dev set. Dev is scored once per epoch per training, so it "
        "dominates runtime across dozens of retrains. A smaller dev makes epoch "
        "selection noisier, but identically so for every strategy, so the comparison "
        "stays fair while the run gets several times cheaper.",
    )
    # Training hyperparameters, kept identical across strategies by construction.
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=20260811)
    p.add_argument("--out-prefix", default="active_learning")
    p.add_argument(
        "--resume", action="store_true",
        help="Skip (strategy, seed) pairs already present in the checkpoint file. "
        "A full sweep is dozens of trainings over hours; without this, one transient "
        "failure late in the run discards everything before it.",
    )
    args = p.parse_args()

    if not args.gold and not args.wikiann:
        raise SystemExit("pass --gold <file> or --wikiann")

    if args.wikiann:
        pool = load_wikiann("train")
        dev = load_wikiann("validation")
        test = load_wikiann("test")
        data_desc = "WikiANN-sq (train as pool)"
    else:
        gold = load_gold(args.gold, args.mode)
        pool, dev, test = split_examples(gold, args.dev_frac, args.test_frac, args.split_seed)
        data_desc = f"gold {args.gold.name} ({args.mode} spans)"
    if args.pool_limit:
        pool = pool[: args.pool_limit]
    if args.dev_limit:
        dev = dev[: args.dev_limit]

    device = resolve_device()
    hardware = hardware_info(device)
    n_trainings = len(args.strategies) * len(args.seeds) * (args.rounds + 1)
    print(f"{data_desc}: pool {len(pool)}, dev {len(dev)}, test {len(test)}")
    print(f"{n_trainings} trainings on {hardware['accelerator']} "
          f"({len(args.strategies)} strategies x {len(args.seeds)} seeds x {args.rounds + 1} rounds)\n")

    t0 = time.perf_counter()
    checkpoint = RESULTS_DIR / f"{args.out_prefix}.partial.json"
    done: dict[str, dict[str, list[dict]]] = {}
    if args.resume and checkpoint.exists():
        done = json.loads(checkpoint.read_text())
        n = sum(len(v) for v in done.values())
        print(f"resuming: {n} (strategy, seed) curves already complete\n")

    curves: dict[str, dict[int, list[RoundResult]]] = {}
    for strategy in args.strategies:
        curves[strategy] = {}
        for seed in args.seeds:
            cached = done.get(strategy, {}).get(str(seed))
            if cached is not None:
                curves[strategy][seed] = [RoundResult(**r) for r in cached]
                print(f"  [{strategy} seed {seed}] restored from checkpoint")
                continue
            curves[strategy][seed] = run_loop(strategy, seed, pool, dev, test, args, device)
            # Persist after every curve: a two-hour sweep should never lose more than
            # the one combination that was in flight.
            done.setdefault(strategy, {})[str(seed)] = [vars(r) for r in curves[strategy][seed]]
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(done, indent=2))
    total_s = time.perf_counter() - t0

    # Aggregate: at each budget, mean and std across seeds.
    summary: dict[str, list[dict]] = {}
    for strategy, per_seed in curves.items():
        n_points = len(next(iter(per_seed.values())))
        points = []
        for i in range(n_points):
            f1s = [per_seed[s][i].test_f1 for s in args.seeds]
            points.append({
                "round": i,
                "n_labelled": per_seed[args.seeds[0]][i].n_labelled,
                "mean_f1": round(float(np.mean(f1s)), 4),
                "std_f1": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 4),
                "per_seed_f1": [round(f, 4) for f in f1s],
            })
        summary[strategy] = points

    report = {
        "data": data_desc, "model": args.model,
        "strategies": args.strategies, "seeds": args.seeds,
        "seed_size": args.seed_size, "acquire": args.acquire, "rounds": args.rounds,
        "n_pool": len(pool), "n_dev": len(dev), "n_test": len(test),
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        "curves": summary,
        "raw": {s: {str(k): [vars(r) for r in v] for k, v in d.items()} for s, d in curves.items()},
        "hardware": hardware, "software": software_info(),
        "total_compute_s": round(total_s, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.out_prefix}.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report, RESULTS_DIR / f"{args.out_prefix}.md")

    print("\nlearning curve (mean F1 +/- std over seeds):")
    header = "  budget  " + "".join(f"{s:>22s}" for s in args.strategies)
    print(header)
    for i in range(len(summary[args.strategies[0]])):
        row = f"  {summary[args.strategies[0]][i]['n_labelled']:6d}  "
        for s in args.strategies:
            pt = summary[s][i]
            row += f"{pt['mean_f1']:>14.4f} ±{pt['std_f1']:.3f}"
        print(row)
    print(f"\n{describe(hardware, total_s)}")
    print(f"→ results/{args.out_prefix}.md")
    return 0


def _write_markdown(r: dict, path: Path) -> None:
    lines = [
        "# Active learning: random vs uncertainty", "",
        f"{r['data']} — pool {r['n_pool']}, dev {r['n_dev']}, test {r['n_test']}. "
        f"{r['model']}, seeds {r['seeds']}.", "",
        f"Seed set {r['seed_size']}, then {r['rounds']} rounds acquiring {r['acquire']} "
        "sentences each. Simulated over the labelled pool: labels are hidden and revealed "
        "only as a strategy requests them.", "",
        "F1 is mean ± standard deviation across seeds. The spread matters as much as the "
        "means — a gap smaller than the error bars is not a result.", "",
    ]
    strategies = r["strategies"]
    lines.append("| Labelled | " + " | ".join(strategies) + " |")
    lines.append("| ---: | " + " | ".join("---:" for _ in strategies) + " |")
    n_points = len(r["curves"][strategies[0]])
    for i in range(n_points):
        cells = []
        for s in strategies:
            pt = r["curves"][s][i]
            cells.append(f"{pt['mean_f1']:.4f} ± {pt['std_f1']:.4f}")
        lines.append(f"| {r['curves'][strategies[0]][i]['n_labelled']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"- Hardware: {r['hardware']['accelerator']} ({r['hardware']['device']})",
        f"- Total compute: {r['total_compute_s'] / 60:.1f} min "
        f"({len(strategies) * len(r['seeds']) * (r['rounds'] + 1)} trainings)", "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
