"""Fine-tune XLM-R for Albanian NER, over several seeds.

Unlike the frozen-checkpoint baselines — where inference is deterministic and the only
uncertainty is which test sentences you happened to draw — training has genuine run-to-run
variance from head initialisation and batch order. So this reports **both**: mean ± std
across seeds (model variance) and a bootstrap CI on the pooled predictions (test-set
variance). Reporting only one would understate the uncertainty.

Evaluation reuses `first_subword_tags` from the baselines, so a fine-tuned model and
Kushtrim's checkpoint are measured by the same code on the same split. Without that,
"my model beats 0.925" compares two measurements rather than two models.

**Seeding does not make training bit-reproducible on Apple silicon.** Two runs of this
script with identical seeds and hyperparameters produced test F1 0.9187 and 0.9279 — a
0.009 swing the seed does not control, because MPS kernels are not deterministic. Two
consequences worth stating in the write-up: never report a single run as *the* result,
and read the seed-to-seed standard deviation as an upper bound on model variance, since
it also absorbs this hardware noise. Reproducing a number exactly would need
`torch.use_deterministic_algorithms(True)` on a CUDA device, which costs speed.

Outputs:
    results/<prefix>.json   per-seed and aggregate metrics, environment, compute time
    results/<prefix>.md     human-readable summary
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..baselines import stats
from ..baselines.envinfo import describe, hardware_info, software_info
from ..baselines.metrics import first_subword_tags, resolve_device
from .data import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    Example,
    collate,
    encode,
    load_gold,
    load_wikiann,
    split_examples,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DEFAULT_MODEL = "xlm-roberta-base"


def free_cache(device) -> None:
    """Release cached allocator blocks between phases.

    XLM-R is ~278M parameters, so AdamW's two optimiser states plus gradients occupy
    several GB in fp32. On Apple silicon that shares one memory pool with everything
    else, and the allocator will refuse a request while holding freeable cached blocks —
    which surfaces as an out-of-memory error on a machine with plenty of free RAM.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(
    model, tokenizer, examples: list[Example], device, max_length: int, batch_size: int = 16
) -> list[list[str]]:
    """Word-level BIO predictions, one list per example.

    Batched, because this dominates the runtime: the dev set is evaluated once per epoch
    and the AL loop retrains dozens of times, so a one-sentence-at-a-time loop makes the
    experiment take hours rather than minutes. Padding is harmless — `word_ids` returns
    None for pad positions, so `first_subword_tags` never reads them.
    """
    model.eval()
    out: list[list[str]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        enc = tokenizer(
            [ex.tokens for ex in chunk],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        pred_ids = logits.argmax(-1).tolist()
        for i, ex in enumerate(chunk):
            out.append(first_subword_tags(enc.word_ids(batch_index=i), pred_ids[i], ID2LABEL, len(ex.tokens)))
    return out


def f1_of(preds: list[list[str]], refs: list[list[str]]) -> float:
    _, _, f1 = stats.prf_from_counts(stats.sentence_counts(preds, refs))
    return f1


def train_one_seed(
    seed: int,
    train: list[Example],
    dev: list[Example],
    args,
    device,
) -> tuple[object, object, dict]:
    """Train one model. Returns (model, tokenizer, history)."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    encoded = [encode(ex, tokenizer, args.max_length) for ex in train]
    loader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_frac), total_steps)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"seed {seed} epoch {epoch + 1}", unit="batch", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()

        free_cache(device)
        dev_f1 = f1_of(predict(model, tokenizer, dev, device, args.max_length), [e.tags for e in dev])
        history.append({"epoch": epoch + 1, "train_loss": running / len(loader), "dev_f1": dev_f1})
        print(f"  seed {seed} epoch {epoch + 1}: loss {running / len(loader):.4f}  dev F1 {dev_f1:.4f}")

        # Keep the best epoch by dev F1 rather than the last: with a small gold set the
        # final epoch is often past the peak, and selecting on dev keeps test honest.
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Drop optimiser state before returning: its two moments per parameter are several
    # GB that the caller no longer needs while evaluating.
    del optimizer, scheduler
    if best_state is not None:
        model.load_state_dict(best_state)
    free_cache(device)
    return model, tokenizer, {"best_dev_f1": best_f1, "epochs": history}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gold", type=Path, default=None, help="Adjudicated gold JSONL.")
    p.add_argument(
        "--wikiann",
        action="store_true",
        help="Train and evaluate on WikiANN-sq instead of gold. Used to prove the "
        "pipeline before gold data exists.",
    )
    p.add_argument(
        "--extra-wikiann-train",
        action="store_true",
        help="Append the WikiANN train split to the gold training data.",
    )
    p.add_argument(
        "--mode",
        choices=["full", "head"],
        default="full",
        help="Boundary convention when reading gold spans.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=20260811)
    p.add_argument("--limit", type=int, default=None, help="Truncate training data (smoke test).")
    p.add_argument("--out-prefix", default="finetune")
    p.add_argument(
        "--also-test-on",
        choices=["wikiann", "gold"],
        default=None,
        help="Additionally score each trained model on the other benchmark's test split. "
        "Training on one corpus and testing on both is what separates 'this model is "
        "weaker' from 'these benchmarks measure different things' — a claim the "
        "single-benchmark numbers cannot distinguish.",
    )
    args = p.parse_args()

    if not args.gold and not args.wikiann:
        raise SystemExit("pass --gold <file> or --wikiann")

    if args.wikiann:
        train = load_wikiann("train")
        dev = load_wikiann("validation")
        test = load_wikiann("test")
        data_desc = "WikiANN-sq official splits"
    else:
        gold = load_gold(args.gold, args.mode)
        train, dev, test = split_examples(gold, args.dev_frac, args.test_frac, args.split_seed)
        data_desc = f"gold {args.gold.name} ({args.mode} spans), split {args.split_seed}"
        if args.extra_wikiann_train:
            train = train + load_wikiann("train")
            data_desc += " + WikiANN train"

    if args.limit:
        train = train[: args.limit]

    cross, cross_desc = None, ""
    if args.also_test_on == "wikiann":
        cross, cross_desc = load_wikiann("test"), "WikiANN-sq test"
    elif args.also_test_on == "gold":
        if not args.gold:
            raise SystemExit("--also-test-on gold needs --gold to say which file")
        _, _, cross = split_examples(
            load_gold(args.gold, args.mode), args.dev_frac, args.test_frac, args.split_seed
        )
        cross_desc = f"gold {args.gold.name} test split"
    if cross is not None:
        # Guard against the silent disaster: if the cross set overlaps training data the
        # number is meaningless, and nothing else in the pipeline would notice.
        train_sents = {" ".join(e.tokens) for e in train}
        leaked = sum(1 for e in cross if " ".join(e.tokens) in train_sents)
        if leaked:
            raise SystemExit(
                f"{leaked} of {len(cross)} cross-eval sentences appear in training data; "
                "the cross-benchmark score would be contaminated."
            )

    device = resolve_device()
    hardware = hardware_info(device)
    print(f"{data_desc}\n  train {len(train)}  dev {len(dev)}  test {len(test)}")
    if cross is not None:
        print(f"  cross-eval on {cross_desc}: {len(cross)} sentences (no overlap with train)")
    print(f"  {args.model} on {device} ({hardware['accelerator']}), seeds {args.seeds}\n")

    refs = [e.tags for e in test]
    per_seed, all_preds = [], []
    t0 = time.perf_counter()

    for seed in args.seeds:
        t_seed = time.perf_counter()
        model, tokenizer, hist = train_one_seed(seed, train, dev, args, device)
        preds = predict(model, tokenizer, test, device, args.max_length)
        test_f1 = f1_of(preds, refs)
        cross_f1 = None
        if cross is not None:
            cross_preds = predict(model, tokenizer, cross, device, args.max_length)
            cross_refs = [e.tags for e in cross]
            cross_f1 = f1_of(cross_preds, cross_refs)
            # Save these too. A cross-benchmark score is exactly the kind of number
            # someone will want to attach a confidence interval to, and without the
            # per-sentence predictions the only available claim is the bare figure.
            cross_path = PREDICTIONS_DIR / f"{args.out_prefix}-cross-seed{seed}.jsonl"
            cross_path.parent.mkdir(parents=True, exist_ok=True)
            with cross_path.open("w", encoding="utf-8") as fh:
                for i, (pred, ref) in enumerate(zip(cross_preds, cross_refs, strict=True)):
                    fh.write(json.dumps({"id": i, "gold": ref, "pred": pred}) + "\n")
        elapsed = time.perf_counter() - t_seed
        del model
        free_cache(device)
        all_preds.append(preds)
        # Persist predictions so this model can be compared against a baseline with
        # the same paired bootstrap the baselines use on each other. Without them the
        # only possible claim is "0.919 vs 0.925", which says nothing about whether
        # the gap is real.
        pred_path = PREDICTIONS_DIR / f"{args.out_prefix}-seed{seed}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as fh:
            for i, (pred, ref) in enumerate(zip(preds, refs, strict=True)):
                fh.write(json.dumps({"id": i, "gold": ref, "pred": pred}) + "\n")
        row = {
            "seed": seed,
            "test_f1": test_f1,
            "best_dev_f1": hist["best_dev_f1"],
            "seconds": round(elapsed, 1),
            "epochs": hist["epochs"],
        }
        if cross_f1 is not None:
            row["cross_f1"] = cross_f1
        per_seed.append(row)
        extra = f", {cross_desc} F1 {cross_f1:.4f}" if cross_f1 is not None else ""
        print(f"  seed {seed}: test F1 {test_f1:.4f}{extra}  ({elapsed:.0f}s)\n")

    total_s = time.perf_counter() - t0
    scores = [r["test_f1"] for r in per_seed]

    # Seed variance and test-set variance are different uncertainties; report both.
    best_idx = int(np.argmax([r["best_dev_f1"] for r in per_seed]))
    counts = stats.sentence_counts(all_preds[best_idx], refs)
    indices = stats.resample_indices(len(refs), stats.DEFAULT_RESAMPLES, stats.DEFAULT_SEED)
    ci_low, ci_high = stats.percentile_ci(stats.bootstrap_f1(counts, indices))

    per_entity = {}
    for ent in ("PER", "ORG", "LOC"):
        ent_counts = stats.sentence_counts(all_preds[best_idx], refs, entity_type=ent)
        _, _, ent_f1 = stats.prf_from_counts(ent_counts)
        per_entity[ent] = round(ent_f1, 4)

    report = {
        "data": data_desc,
        "model": args.model,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_frac": args.warmup_frac,
            "max_length": args.max_length,
        },
        "seeds": args.seeds,
        "test_f1_mean": round(float(np.mean(scores)), 4),
        "test_f1_std": round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 4),
        "test_f1_per_seed": [round(s, 4) for s in scores],
        "best_seed": per_seed[best_idx]["seed"],
        "best_seed_f1": round(scores[best_idx], 4),
        "best_seed_f1_ci": [round(ci_low, 4), round(ci_high, 4)],
        "best_seed_per_entity_f1": per_entity,
        "per_seed": per_seed,
        "hardware": hardware,
        "software": software_info(),
        "total_compute_s": round(total_s, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.out_prefix}.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report, RESULTS_DIR / f"{args.out_prefix}.md")

    print(f"test F1 {report['test_f1_mean']:.4f} ± {report['test_f1_std']:.4f} over {len(scores)} seeds")
    print(
        f"best seed {report['best_seed']}: {report['best_seed_f1']:.4f} "
        f"[{ci_low:.4f}, {ci_high:.4f}]  PER {per_entity['PER']:.3f} "
        f"ORG {per_entity['ORG']:.3f} LOC {per_entity['LOC']:.3f}"
    )
    print(describe(hardware, total_s))
    print(f"→ results/{args.out_prefix}.md")
    return 0


def _write_markdown(r: dict, path: Path) -> None:
    hp = r["hyperparams"]
    lines = [
        f"# Fine-tuned {r['model']}",
        "",
        f"{r['data']} — train {r['n_train']}, dev {r['n_dev']}, test {r['n_test']}.",
        "",
        f"**Test F1 {r['test_f1_mean']:.4f} ± {r['test_f1_std']:.4f}** over "
        f"{len(r['seeds'])} seeds {r['seeds']}.",
        "",
        "Two uncertainties, deliberately separate: `±` is run-to-run variance from "
        "training randomness, the bracketed interval is a 95% bootstrap CI over test "
        "sentences for the best-dev seed.",
        "",
        "| Seed | dev F1 | test F1 | seconds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in r["per_seed"]:
        lines.append(f"| {s['seed']} | {s['best_dev_f1']:.4f} | {s['test_f1']:.4f} | {s['seconds']:.0f} |")
    ent = r["best_seed_per_entity_f1"]
    lines += [
        "",
        f"Best seed ({r['best_seed']}): **{r['best_seed_f1']:.4f}** "
        f"[{r['best_seed_f1_ci'][0]:.4f}, {r['best_seed_f1_ci'][1]:.4f}] — "
        f"PER {ent['PER']:.3f}, ORG {ent['ORG']:.3f}, LOC {ent['LOC']:.3f}",
        "",
        f"Hyperparameters: {hp['epochs']} epochs, batch {hp['batch_size']}, lr {hp['lr']}, "
        f"warmup {hp['warmup_frac']}, max length {hp['max_length']}.",
        "",
        f"- Hardware: {r['hardware']['accelerator']} ({r['hardware']['device']}), "
        f"{r['hardware']['total_ram_gb']} GB RAM",
        f"- Libraries: torch {r['software']['torch']}, transformers {r['software']['transformers']}",
        f"- Total compute: {r['total_compute_s'] / 60:.1f} min",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
