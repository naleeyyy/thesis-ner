"""Generate the report figures as vector PDFs.

Each figure is chosen for the job its data does, not for variety:

- baselines      magnitude with uncertainty  -> dot plot with error bars, not bars
                 (a bar's length implies a precision the confidence interval denies)
- learning curve change over a budget        -> lines with shaded seed spread
- errors         composition within a total  -> stacked horizontal bars
- corpus         two distributions compared  -> overlaid step histograms

Run: python -m src.figures.make_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import GRID, INK, SERIES, strip_frame, use_report_style

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT.parent / "report" / "images"

SHORT = {
    "kushtrim-mbert-sq": "Kushtrim mBERT",
    "akdeniz27-mbert-sq": "akdeniz27 mBERT",
    "babelscape-wikineural": "Babelscape WikiNeural",
    "davlan-xlmr-hrl": "Davlan XLM-R HRL",
    "xlmr_wikiann-seed1": "XLM-R (this work)",
}


def fig_baselines(path: Path) -> None:
    """Point estimate plus interval. A dot plot, because a bar drawn to 0.925 invites
    reading the length as exact when the interval spans 0.035."""
    data = json.loads((RESULTS / "baselines.json").read_text())
    rows = sorted(data["models"], key=lambda r: r["overall_f1"])

    fig, ax = plt.subplots(figsize=(5.4, 2.1))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi = r["overall_f1_ci_low"], r["overall_f1_ci_high"]
        ax.plot([lo, hi], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(r["overall_f1"], i, "o", color=SERIES[0], markersize=5, zorder=2)
        # Direct label: colour is never the only carrier of the value.
        ax.text(hi + 0.012, i, f"{r['overall_f1']:.3f}", va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y, [SHORT.get(r["model"], r["model"]) for r in rows])
    ax.set_xlabel("$F_1$ on WikiANN-sq test")
    ax.set_xlim(0.45, 1.02)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    strip_frame(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    fig.savefig(path)
    plt.close(fig)


def fig_learning_curve(path: Path) -> None:
    """Two strategies over a budget. The shaded band is the seed standard deviation —
    the point of the figure is that the bands overlap everywhere."""
    data = json.loads((RESULTS / "al_wikiann.json").read_text())
    fig, ax = plt.subplots(figsize=(5.4, 2.6))

    for slot, (name, label) in enumerate([("random", "Random"), ("uncertainty", "Uncertainty")]):
        pts = data["curves"][name]
        x = [p["n_labelled"] for p in pts]
        mean = np.array([p["mean_f1"] for p in pts])
        sd = np.array([p["std_f1"] for p in pts])
        ax.fill_between(x, mean - sd, mean + sd, color=SERIES[slot], alpha=0.16, lw=0)
        ax.plot(x, mean, "-o", color=SERIES[slot], label=label, markersize=4)
        # Direct label offset vertically so the two do not collide where the curves meet.
        ax.annotate(label, (x[-1], mean[-1]), xytext=(8, 6 if slot else -9),
                    textcoords="offset points", fontsize=7.5, color=SERIES[slot])

    ax.set_xlabel("Labelled sentences")
    ax.set_ylabel("$F_1$")
    ax.set_xticks([p["n_labelled"] for p in data["curves"]["random"]])
    ax.set_xlim(180, 1560)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), handlelength=1.2)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    strip_frame(ax)
    fig.savefig(path)
    plt.close(fig)


def fig_errors(path: Path) -> None:
    """Composition of each model's errors. Shares sum to 100%, so a stacked bar is
    honest here in a way it would not be for unrelated quantities."""
    reports = {r["model"]: r for r in json.loads((RESULTS / "error_analysis.json").read_text())}
    order = ["davlan-xlmr-hrl", "babelscape-wikineural", "akdeniz27-mbert-sq",
             "kushtrim-mbert-sq", "xlmr_wikiann-seed1"]
    kinds = ["missed", "spurious", "type", "boundary"]

    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for slot, kind in enumerate(kinds):
        vals = np.array([
            100 * reports[m]["by_kind"].get(kind, 0) / max(reports[m]["n_errors"], 1)
            for m in order
        ])
        ax.barh(y, vals, left=left, height=0.6, color=SERIES[slot], label=kind,
                edgecolor="white", linewidth=1.2)  # 2px-equivalent gap between segments
        for i, v in enumerate(vals):
            if v >= 9:  # only label segments wide enough to hold the text
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
        left += vals

    ax.set_yticks(y, [f"{SHORT[m]}  ({reports[m]['n_errors']})" for m in order])
    ax.set_xlabel("Share of that model's errors (\\%)")
    ax.set_xlim(0, 100)
    strip_frame(ax, keep=())
    ax.tick_params(axis="both", length=0)
    ax.set_xticks([])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1.1)
    fig.savefig(path)
    plt.close(fig)


def fig_corpus(path: Path) -> None:
    """Sentence-length distributions. Step outlines rather than filled bars, so the
    overlap region stays readable instead of one corpus hiding the other."""
    from datasets import load_dataset

    mine = [len(json.loads(line)["tokens"])
            for line in (REPO_ROOT / "data/raw/wiki_segmented_v2.jsonl").open()]
    wikiann = [len(r["tokens"]) for r in load_dataset("unimelb-nlp/wikiann", "sq", split="train")]

    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    bins = np.arange(0, 51, 2)
    medians = []
    for slot, (vals, label) in enumerate([(wikiann, "WikiANN-sq"), (mine, "This work")]):
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.6,
                color=SERIES[slot], label=label)
        medians.append((np.median(vals), label, SERIES[slot]))

    # Headroom first, then place the median callouts inside it so they never sit on the
    # curves. The WikiANN spike is tall enough to swallow anything drawn at data height.
    top = ax.get_ylim()[1] * 1.30
    ax.set_ylim(0, top)
    for i, (med, label, colour) in enumerate(medians):
        ax.axvline(med, color=colour, lw=0.8, ls=":", alpha=0.9)
        ax.annotate(f"{label}, median {int(med)}", xy=(med, top * 0.99),
                    xytext=(6, -10 - 11 * i), textcoords="offset points",
                    fontsize=7, color=colour, va="top")

    ax.set_xlabel("Sentence length (tokens)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 50)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    strip_frame(ax)
    fig.savefig(path)
    plt.close(fig)



def fig_transfer(path: Path) -> None:
    """Slope chart: the same four checkpoints on two benchmarks.

    A slope chart rather than grouped bars because the finding *is* the crossing --- the
    reader should see lines swapping order, which grouped bars force them to reconstruct
    by comparing heights across a gap.
    """
    wikiann = json.loads((RESULTS / "baselines.json").read_text())
    gold = json.loads((RESULTS / "gold-baselines.json").read_text())

    def scores(blob):
        return {row["model"]: row["overall_f1"] for row in blob["models"]}

    w, g = scores(wikiann), scores(gold)
    models = [m for m in SHORT if m in w and m in g]

    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    for i, m in enumerate(sorted(models, key=lambda x: -w[x])):
        colour = SERIES[i % len(SERIES)]
        ax.plot([0, 1], [w[m], g[m]], "-o", color=colour, lw=1.8, ms=5, zorder=3)
        ax.annotate(
            f"{SHORT[m]}  {w[m]:.3f}",
            (0, w[m]),
            textcoords="offset points",
            xytext=(-8, 0),
            ha="right",
            va="center",
            fontsize=7,
            color=INK,
        )
        ax.annotate(
            f"{g[m]:.3f}",
            (1, g[m]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontsize=7,
            color=INK,
        )

    ax.set_xlim(-0.72, 1.28)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WikiANN-sq test", "gold test (this work)"])
    ax.set_ylabel("$F_1$")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_frame(ax)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    use_report_style()
    OUT.mkdir(parents=True, exist_ok=True)
    figures = [
        ("baselines", fig_baselines),
        ("learning_curve", fig_learning_curve),
        ("errors", fig_errors),
        ("corpus_lengths", fig_corpus),
        ("transfer", fig_transfer),
    ]
    for name, fn in figures:
        path = OUT / f"fig_{name}.pdf"
        fn(path)
        print(f"  {path.relative_to(REPO_ROOT.parent)}  ({path.stat().st_size // 1024} KB)")
    print(f"\n{len(figures)} figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
