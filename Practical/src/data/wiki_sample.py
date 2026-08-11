"""Sample random Albanian Wikipedia article extracts.

Uses the public MediaWiki REST API (`/api/rest_v1/page/random/summary`). Drops
disambiguation pages and very short stubs. Output is JSONL, one record per
article:

    {"title": "...", "extract": "...", "source_url": "...", "page_id": 123}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://sq.wikipedia.org/api/rest_v1/page/random/summary"
USER_AGENT = (
    "albanian-ner-thesis/0.1 (https://github.com/krenarahmeti; "
    "k12337455@students.jku.at) requests"
)
MIN_EXTRACT_CHARS = 200


def fetch_one(session: requests.Session) -> dict | None:
    r = session.get(API, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    if data.get("type") == "disambiguation":
        return None
    extract = (data.get("extract") or "").strip()
    if len(extract) < MIN_EXTRACT_CHARS:
        return None
    return {
        "title": data.get("title"),
        "page_id": data.get("pageid"),
        "extract": extract,
        "source_url": (data.get("content_urls", {}).get("desktop") or {}).get("page"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="number of articles to keep")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sleep", type=float, default=0.2, help="seconds between API calls")
    p.add_argument("--max-attempts-per-article", type=int, default=8)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Accept"] = "application/json"

    seen_ids: set[int] = set()
    records: list[dict] = []
    with tqdm(total=args.n, desc="sq wiki random") as bar:
        attempts = 0
        while len(records) < args.n:
            attempts += 1
            if attempts > args.n * args.max_attempts_per_article:
                print(
                    f"Gave up after {attempts} attempts; collected {len(records)} articles",
                    file=sys.stderr,
                )
                break
            rec = fetch_one(session)
            time.sleep(args.sleep)  # pace every API call, including rejected ones
            if rec is None or rec["page_id"] in seen_ids:
                continue
            seen_ids.add(rec["page_id"])
            records.append(rec)
            bar.update(1)

    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} articles → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
