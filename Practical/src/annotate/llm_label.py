"""Pre-label segmented Albanian sentences with an LLM, for humans to verify.

This is the "LLM-assisted" half of the thesis pipeline: the model proposes entity
spans, annotators correct them, and the corrected result becomes gold. Nothing here
produces final labels — everything downstream treats these as suggestions.

Design notes that matter for the write-up:

- **Spans, not per-token tags.** The model returns inclusive token-index spans and we
  convert to BIO ourselves. Asking for a parallel tag array invites length mismatches
  that are tedious to repair; a span list is validated cheaply and fails loudly.
- **Every response is cached to disk**, keyed by (model, prompt version, tokens). Re-runs
  after a crash, or with a longer sentence list, cost nothing for work already done.
- **Usage and cost are recorded per sentence**, because the report has to state compute
  cost and the annotation experiment needs a per-sentence price.

Outputs:
    data/interim/llm_cache.jsonl   response cache, safe to delete (just re-costs money)
    <--out>                        pre-labeled JSONL, one record per sentence
    <--out>.stats.json             usage, cost, validation-failure rates
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .prompt import ENTITY_SCHEMA, ENTITY_TYPES, PROMPT_VERSION, SYSTEM_PROMPT, render_tokens

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "data" / "interim" / "llm_cache.jsonl"

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000

# USD per million (input, output) tokens, for the cost line in the methods section.
# Cache reads bill at 0.1x the input rate and cache writes at 1.25x (5-minute TTL).
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Sonnet 5 bills at a promotional rate through 2026-08-31. Reporting the list rate
# overstated spend by ~33% against the actual dashboard figure, which is the wrong
# direction of error for a budgeted project and for a methods section that has to state
# real compute cost. After the end date this falls back to the list rate automatically.
PROMOTIONAL = {
    "claude-sonnet-5": ((2.0, 10.0), datetime.date(2026, 8, 31)),
}


def rates_for(model: str, on: datetime.date | None = None) -> tuple[float, float] | None:
    """Effective (input, output) $/Mtok, or None when the model's price is unknown."""
    promo = PROMOTIONAL.get(model)
    if promo is not None:
        (rates, until) = promo
        if (on or datetime.date.today()) <= until:
            return rates
    return PRICING.get(model)

# Minimum cacheable prefix, per model. The system prompt is only ~900 tokens, so on
# models with a 4096-token floor it silently will not cache — no error, just full-price
# input on every call. Worth knowing before reading a cost report and blaming the model.
CACHE_MIN_TOKENS = {
    "claude-opus-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-haiku-4-5": 4096,
}


class BudgetExceeded(RuntimeError):
    """Raised when accumulated spend passes --max-usd, to stop a run mid-flight."""


class SpendTracker:
    """Thread-safe running total, so a concurrent run can stop at a hard dollar cap."""

    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd
        self.total = 0.0
        self._lock = threading.Lock()

    def add(self, amount: float) -> None:
        with self._lock:
            self.total += amount
            if self.limit is not None and self.total > self.limit:
                raise BudgetExceeded(
                    f"spend ${self.total:.4f} exceeded --max-usd ${self.limit:.2f}"
                )


# --------------------------------------------------------------------------- caching


def cache_key(model: str, tokens: list[str]) -> str:
    """Stable identity of one labeling request.

    Includes the prompt version so editing the conventions in `prompt.py` invalidates
    every cached label rather than silently mixing annotation standards.
    """
    payload = json.dumps([model, PROMPT_VERSION, tokens], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Append-only JSONL cache. Last write for a key wins on load."""

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._entries[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, value: dict) -> None:
        rec = {"key": key, **value}
        with self._lock:
            self._entries[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


# ------------------------------------------------------------------------ validation


@dataclass
class Rejection:
    """One dropped span, kept so the failure modes can be reported rather than guessed."""

    reason: str
    span: dict


@dataclass
class ValidationResult:
    spans: list[dict]
    rejections: list[Rejection] = field(default_factory=list)
    # Indices were in range but the model's own `text` field named different tokens.
    # Kept (indices win) and counted — a rising rate means the model is miscounting.
    surface_mismatches: int = 0
    # Head range was missing or not nested inside the full span, so it fell back to the
    # full span. The full span is the load-bearing annotation; a bad head degrades the
    # `head` view to the `full` view rather than discarding the entity.
    head_fallbacks: int = 0


def validate_spans(raw_spans: list[dict], tokens: list[str]) -> ValidationResult:
    """Drop spans that cannot be turned into a BIO tagging of `tokens`.

    Rejects out-of-range indices, inverted spans, unknown types, and any span
    overlapping one already accepted. Earlier spans win an overlap, so the result is
    deterministic given the model's ordering.
    """
    result = ValidationResult(spans=[])
    claimed: set[int] = set()

    for span in raw_spans:
        start, end, etype = span.get("start"), span.get("end"), span.get("type")

        if not isinstance(start, int) or not isinstance(end, int):
            result.rejections.append(Rejection("non-integer index", span))
            continue
        if etype not in ENTITY_TYPES:
            result.rejections.append(Rejection(f"unknown type {etype!r}", span))
            continue
        if start > end:
            result.rejections.append(Rejection("start after end", span))
            continue
        if start < 0 or end >= len(tokens):
            result.rejections.append(Rejection("index out of range", span))
            continue

        positions = set(range(start, end + 1))
        if positions & claimed:
            result.rejections.append(Rejection("overlaps an earlier span", span))
            continue

        surface = " ".join(tokens[start : end + 1])
        if span.get("text") and span["text"].strip() != surface:
            result.surface_mismatches += 1

        # Head must be a sub-range of the full span. Anything else falls back to the
        # full span so the entity survives with a degraded — never wrong — head.
        h_start, h_end = span.get("head_start"), span.get("head_end")
        if (
            not isinstance(h_start, int)
            or not isinstance(h_end, int)
            or h_start > h_end
            or h_start < start
            or h_end > end
        ):
            h_start, h_end = start, end
            result.head_fallbacks += 1

        claimed |= positions
        result.spans.append(
            {
                "start": start,
                "end": end,
                "head_start": h_start,
                "head_end": h_end,
                "type": etype,
                "surface": surface,
                "head_surface": " ".join(tokens[h_start : h_end + 1]),
            }
        )

    result.spans.sort(key=lambda s: s["start"])
    return result


def spans_to_bio(spans: list[dict], n_tokens: int, mode: str = "full") -> list[str]:
    """Convert validated, non-overlapping spans to a BIO tag per token.

    `mode="full"` tags the whole phrase (`Stacioni i Bramit`); `mode="head"` tags only
    the proper-name core (`Bramit`). Both views are derived from the same annotation, so
    the convention can be chosen — or reported both ways — after the data is collected.
    """
    if mode not in ("full", "head"):
        raise ValueError(f"mode must be 'full' or 'head', got {mode!r}")
    lo_key, hi_key = ("start", "end") if mode == "full" else ("head_start", "head_end")

    tags = ["O"] * n_tokens
    for span in spans:
        lo, hi = span[lo_key], span[hi_key]
        tags[lo] = f"B-{span['type']}"
        for i in range(lo + 1, hi + 1):
            tags[i] = f"I-{span['type']}"
    return tags


# ------------------------------------------------------------------------- labelling


def build_client():
    """Anthropic client, with a clear message when no credential is configured."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - dependency is declared in pyproject
        print("The `anthropic` package is missing. Run `uv sync`.", file=sys.stderr)
        raise

    try:
        return anthropic.Anthropic()
    except Exception as e:
        print(
            f"Could not construct the Anthropic client: {e}\n"
            "Add ANTHROPIC_API_KEY to Practical/.env (gitignored) and source it:\n"
            "  set -a; source .env; set +a",
            file=sys.stderr,
        )
        raise


def label_sentence(
    client,
    model: str,
    tokens: list[str],
    max_tokens: int,
    effort: str | None,
    thinking: bool = True,
) -> dict:
    """One API call. Returns the parsed entity list plus usage, or raises.

    `thinking=False` sends `{"type": "disabled"}`, which is the single biggest cost lever
    here: thinking is on by default on Opus 5 and Sonnet 5 and bills at the output rate,
    which for a task this small dwarfs the actual answer. Only pass it on models that
    accept it (4.6-and-later); older models like Haiku 4.5 simply don't think unless asked.
    """
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        # Cached: the conventions block is identical on every request and is the bulk
        # of the input, so this is where nearly all the token spend would otherwise go.
        "system": [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": render_tokens(tokens)}],
        "output_config": {"format": {"type": "json_schema", "schema": ENTITY_SCHEMA}},
    }
    if effort:
        kwargs["output_config"]["effort"] = effort
    if not thinking:
        kwargs["thinking"] = {"type": "disabled"}

    response = client.messages.create(**kwargs)

    if response.stop_reason == "refusal":
        raise RuntimeError("request refused by safety classifiers")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"hit max_tokens ({max_tokens}) before finishing")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError(f"no text block in response (stop_reason={response.stop_reason})")

    parsed = json.loads(text)
    usage = response.usage
    return {
        "entities": parsed.get("entities", []),
        "model": response.model,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        },
    }


def label_with_retry(
    client, model, tokens, max_tokens, effort, thinking=True, attempts=4
) -> dict:
    """Retry transient failures. The SDK already retries 429/5xx; this covers the rest
    (a malformed JSON parse, a truncated response) where a fresh sample usually works."""
    import anthropic

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return label_sentence(client, model, tokens, max_tokens, effort, thinking)
        except BudgetExceeded:
            raise  # a spend cap is not a transient failure — stop immediately
        except (json.JSONDecodeError, RuntimeError, anthropic.APIError) as e:
            last = e
            if attempt < attempts - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed after {attempts} attempts: {last}")


def usd_cost(model: str, usage: dict) -> float:
    """Cost of one call at the rate in force today. Unknown models cost 0 rather than
    crashing a run mid-campaign over a pricing table that is merely out of date."""
    rates = rates_for(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (
        usage["input_tokens"] * in_rate
        + usage["cache_read_input_tokens"] * in_rate * 0.1
        + usage["cache_creation_input_tokens"] * in_rate * 1.25
        + usage["output_tokens"] * out_rate
    ) / 1_000_000


# ------------------------------------------------------------------------------- CLI


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Segmented JSONL.")
    p.add_argument("--out", type=Path, required=True, help="Pre-labeled JSONL to write.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None, help="Label only the first N sentences.")
    p.add_argument("--workers", type=int, default=4, help="Concurrent requests.")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort. Omitted means the API default (high).",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking (Opus 5 / Sonnet 5 think by default and bill it at the "
        "output rate). The largest cost lever available; leave off for best quality.",
    )
    p.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="Hard spend cap. The run stops as soon as accumulated cost passes it; "
        "everything already labeled stays cached, so a resumed run picks up where it left off.",
    )
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument(
        "--no-cache", action="store_true", help="Ignore cached responses and re-request all."
    )
    args = p.parse_args()

    sentences = read_jsonl(args.inp)
    if args.limit:
        sentences = sentences[: args.limit]
    print(f"{len(sentences)} sentences from {args.inp}", flush=True)

    cache = ResponseCache(args.cache)
    print(f"cache: {len(cache)} entries at {args.cache}", flush=True)

    if args.model not in PRICING:
        print(f"warning: no pricing known for {args.model}; cost will report as $0", flush=True)
    elif CACHE_MIN_TOKENS.get(args.model, 0) > 2000:
        print(
            f"note: {args.model} needs a {CACHE_MIN_TOKENS[args.model]}-token prefix to cache; "
            "the system prompt is shorter, so every call pays full input price.",
            flush=True,
        )

    client = build_client()
    spend = SpendTracker(args.max_usd)
    t0 = time.perf_counter()

    def work(rec: dict) -> dict:
        tokens = rec["tokens"]
        key = cache_key(args.model, tokens)
        cached = None if args.no_cache else cache.get(key)
        if cached is not None:
            return {"rec": rec, "response": cached["response"], "cached": True}
        response = label_with_retry(
            client,
            args.model,
            tokens,
            args.max_tokens,
            args.effort,
            thinking=not args.no_thinking,
        )
        cache.put(key, {"response": response})
        # Charge before returning: a cap that only trips at the end is not a cap.
        spend.add(usd_cost(args.model, response["usage"]))
        return {"rec": rec, "response": response, "cached": False}

    results: list[dict] = []
    errors: list[tuple[str, str]] = []
    budget_hit = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, rec): rec for rec in sentences}
        for fut in tqdm(futures, total=len(futures), desc="labeling", unit="sent"):
            rec = futures[fut]
            try:
                results.append(fut.result())
            except BudgetExceeded as e:
                if not budget_hit:
                    budget_hit = True
                    print(f"\n{e} — cancelling remaining work", file=sys.stderr)
                    for pending in futures:
                        pending.cancel()
            except Exception as e:
                errors.append((rec.get("id", "?"), str(e)))

    elapsed = time.perf_counter() - t0

    # Preserve input order — the cache and thread pool both scramble it.
    order = {rec.get("id"): i for i, rec in enumerate(sentences)}
    results.sort(key=lambda r: order.get(r["rec"].get("id"), 0))

    n_cached = sum(1 for r in results if r["cached"])
    totals = dict.fromkeys(
        (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ),
        0,
    )
    cost = 0.0
    n_spans = n_rejected = n_mismatch = n_head_fallback = n_head_differs = 0
    rejection_reasons: dict[str, int] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in results:
            rec, response = r["rec"], r["response"]
            tokens = rec["tokens"]
            validated = validate_spans(response["entities"], tokens)

            n_spans += len(validated.spans)
            n_rejected += len(validated.rejections)
            n_mismatch += validated.surface_mismatches
            n_head_fallback += validated.head_fallbacks
            # How often the two conventions actually disagree — i.e. how much the
            # `X i Y` decision is worth. If this is near zero the choice is moot.
            n_head_differs += sum(
                1
                for s in validated.spans
                if (s["head_start"], s["head_end"]) != (s["start"], s["end"])
            )
            for rej in validated.rejections:
                rejection_reasons[rej.reason] = rejection_reasons.get(rej.reason, 0) + 1

            if not r["cached"]:
                for k in totals:
                    totals[k] += response["usage"][k]
                cost += usd_cost(args.model, response["usage"])

            fh.write(
                json.dumps(
                    {
                        "id": rec.get("id"),
                        "tokens": tokens,
                        # `llm_tags_*`, not `ner_tags` — these are suggestions. The
                        # annotation tool renames the field once a human has signed off.
                        # Two flat BIO views derived from one annotation, so the `X i Y`
                        # boundary convention stays a reporting choice, not a data commitment.
                        "llm_tags_full": spans_to_bio(validated.spans, len(tokens), "full"),
                        "llm_tags_head": spans_to_bio(validated.spans, len(tokens), "head"),
                        "llm_spans": validated.spans,
                        "source_url": rec.get("source_url"),
                        "n_rejected_spans": len(validated.rejections),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "effort": args.effort,
        "thinking": not args.no_thinking,
        "budget_stopped": budget_hit,
        "n_sentences": len(results),
        "n_from_cache": n_cached,
        "n_errors": len(errors),
        "elapsed_s": round(elapsed, 1),
        "usage": totals,
        "usd_cost_new_calls": round(cost, 4),
        "usd_per_sentence": round(cost / max(len(results) - n_cached, 1), 5),
        "spans_accepted": n_spans,
        "spans_rejected": n_rejected,
        "rejection_reasons": rejection_reasons,
        "surface_mismatches": n_mismatch,
        "head_fallbacks": n_head_fallback,
        "spans_where_head_differs_from_full": n_head_differs,
        "errors": errors[:20],
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"\nwrote {len(results)} → {args.out}")
    print(f"  {n_cached} served from cache, {len(errors)} failed")
    print(f"  {n_spans} spans accepted, {n_rejected} rejected, {n_mismatch} surface mismatches")
    print(f"  ${cost:.4f} for {len(results) - n_cached} new calls in {elapsed:.0f}s")
    print(f"  stats → {stats_path}")
    if errors:
        print("\nfirst failures:", file=sys.stderr)
        for sid, msg in errors[:5]:
            print(f"  {sid}: {msg}", file=sys.stderr)
    return 1 if len(errors) == len(sentences) and sentences else 0


if __name__ == "__main__":
    raise SystemExit(main())
