"""Back up every Label Studio project's annotations to a local snapshot.

Read-only against the server: it lists projects and pulls each one's export, so it
is always safe to run mid-campaign. Snapshots are timestamped and never overwritten,
because the point of a backup is to still have yesterday's copy after today's went
wrong.

Each run writes data/labeled/exports/<UTC timestamp>/ containing one raw JSON export
per project plus a manifest.json recording, per project, how many tasks exist and how
many carry annotations. The manifest is what you read to answer "how far along are
the annotators"; the raw exports are what src/annotate/agreement.py consumes.

    LABEL_STUDIO_ACCESS_TOKEN=... python -m scripts.export_annotations
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = REPO_ROOT / "data" / "labeled" / "exports"


def call(url: str, token: str, expect_json: bool = True):
    # Cloudflare in front of the instance rejects urllib's default agent with a 1010,
    # so send an ordinary one. Same reason as scripts/import_to_labelstudio.py.
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {token}",
            "User-Agent": "albanian-ner-thesis/1.0 (+https://github.com/naleeyyy/thesis-ner)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise SystemExit(f"GET {url} -> HTTP {e.code}\n{detail}") from e
    return json.loads(raw.decode()) if expect_json else raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="https://nerls.krenarahmeti.com")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Snapshot directory. Defaults to data/labeled/exports/<UTC timestamp>.",
    )
    p.add_argument(
        "--only-annotated",
        action="store_true",
        help="Skip projects where no task has been annotated yet.",
    )
    args = p.parse_args()

    token = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
    if not token:
        raise SystemExit("LABEL_STUDIO_ACCESS_TOKEN not set — run: set -a; source .env; set +a")

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (EXPORT_ROOT / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    projects = call(f"{args.host}/api/projects?page_size=200", token).get("results", [])
    if not projects:
        raise SystemExit(f"no projects found at {args.host}")

    manifest: list[dict] = []
    total_ann = 0
    for proj in sorted(projects, key=lambda x: x["title"]):
        tasks = proj.get("task_number") or 0
        done = proj.get("num_tasks_with_annotations") or 0
        total_ann += done

        if args.only_annotated and not done:
            print(f"  {proj['title']:20s} {done:4d}/{tasks:<4d} skipped (nothing annotated)")
            continue

        # exportType=JSON gives tasks with their annotations, which is what the
        # agreement code reads. The endpoint streams, so this can be slow on big
        # projects; the 300s timeout above is deliberate.
        query = urllib.parse.urlencode({"exportType": "JSON", "download_all_tasks": "true"})
        raw = call(f"{args.host}/api/projects/{proj['id']}/export?{query}", token, expect_json=False)

        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in proj["title"])
        path = out_dir / f"{safe}.json"
        path.write_bytes(raw)

        manifest.append(
            {
                "id": proj["id"],
                "title": proj["title"],
                "description": proj.get("description") or "",
                "tasks": tasks,
                "tasks_with_annotations": done,
                "file": path.name,
                "bytes": len(raw),
            }
        )
        print(f"  {proj['title']:20s} {done:4d}/{tasks:<4d} -> {path.name} ({len(raw):,} B)")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "exported_at": stamp,
                "host": args.host,
                "projects": manifest,
                "total_tasks": sum(m["tasks"] for m in manifest),
                "total_tasks_with_annotations": sum(m["tasks_with_annotations"] for m in manifest),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nsnapshot: {out_dir}")
    print(f"{len(manifest)} project(s) exported, {total_ann} annotated task(s) across the instance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
