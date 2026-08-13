"""Create one Label Studio project per batch and import its tasks.

Doing this through the API rather than the UI is not just faster: the labelling config
must be saved *before* the tasks are imported, or predictions arrive referencing labels
that do not exist yet and silently fail to attach. Scripting it makes that ordering
impossible to get wrong across a dozen projects.

Projects are identified by a `[batch:NN]` marker in their description rather than by
title, so you can rename them (to add an annotator's name, say) without a later run
creating duplicates. Tasks are only imported into an empty project, so re-running never
double-imports.

    LABEL_STUDIO_ACCESS_TOKEN=... python -m scripts.import_to_labelstudio --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "deploy" / "label-studio" / "labeling-config.xml"
TASK_DIR = REPO_ROOT / "data" / "interim" / "tasks"


def call(url: str, token: str, method: str = "GET", payload=None, raw: bytes | None = None):
    # Cloudflare in front of the instance rejects urllib's default agent with a 1010,
    # so send an ordinary browser-ish one. curl works out of the box, urllib does not.
    headers = {
        "Authorization": f"Token {token}",
        "User-Agent": "albanian-ner-thesis/1.0 (+https://github.com/naleeyyy/thesis-ner)",
        "Accept": "application/json",
    }
    if raw is not None:
        body, headers["Content-Type"] = raw, "application/json"
    elif payload is not None:
        body, headers["Content-Type"] = json.dumps(payload).encode(), "application/json"
    else:
        body = None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise SystemExit(f"{method} {url} -> HTTP {e.code}\n{detail}") from e


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="https://nerls.krenarahmeti.com")
    p.add_argument("--prefix", default="batch-", help="Project title prefix.")
    p.add_argument("--dry-run", action="store_true", help="Show the plan, change nothing.")
    p.add_argument(
        "--reset",
        action="store_true",
        help="DELETE every project whose title starts with --prefix, then re-import. "
        "Destroys any annotations in them, so only use this before people start work. "
        "Projects outside the prefix (e.g. the pilot) are never touched.",
    )
    args = p.parse_args()

    token = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
    if not token:
        raise SystemExit("LABEL_STUDIO_ACCESS_TOKEN not set — run: set -a; source .env; set +a")

    label_config = CONFIG.read_text(encoding="utf-8")
    task_files = sorted(TASK_DIR.glob("*.json"))
    if not task_files:
        raise SystemExit(f"no task files in {TASK_DIR}")

    # Projects are matched by a marker written into their description, not by title, so
    # you can rename them freely (e.g. "batch-01" -> "batch-01 ana") without a re-run
    # creating duplicates. Titles are for humans; the marker is the identity.
    def marker(stem: str) -> str:
        return f"[batch:{stem}]"

    all_projects = call(f"{args.host}/api/projects?page_size=200", token).get("results", [])
    stems = {path.stem for path in task_files}
    existing = {}
    for proj in all_projects:
        desc = proj.get("description") or ""
        for stem in stems:
            if marker(stem) in desc:
                existing[stem] = proj
        # Fall back to the title for projects created before markers existed. The stem
        # must name an actual task file, or a renamed project ("batch-01 — ana") would
        # register under a phantom key of its own.
        if proj["title"].startswith(args.prefix):
            stem = proj["title"][len(args.prefix) :]
            if stem in stems:
                existing.setdefault(stem, proj)
    print(f"{len(task_files)} task files, {len(existing)} existing projects\n")

    if args.reset:
        doomed = {
            proj["title"]: proj
            for proj in all_projects
            if proj["title"].startswith(args.prefix)
            or any(marker(f.stem) in (proj.get("description") or "") for f in task_files)
        }
        annotated = sum(p.get("num_tasks_with_annotations") or 0 for p in doomed.values())
        if annotated and not args.dry_run:
            raise SystemExit(
                f"refusing to reset: {annotated} tasks already have annotations across "
                f"{len(doomed)} projects. Deleting would destroy that work."
            )
        for title, proj in sorted(doomed.items()):
            if args.dry_run:
                print(f"  {title:12s} would DELETE (id={proj['id']})")
            else:
                call(f"{args.host}/api/projects/{proj['id']}", token, "DELETE")
                print(f"  {title:12s} deleted (id={proj['id']})")
        if not args.dry_run:
            existing = {}
            print()

    for path in task_files:
        title = f"{args.prefix}{path.stem}"
        tasks = json.loads(path.read_text(encoding="utf-8"))
        assisted = any("predictions" in t for t in tasks)
        kind = "assisted" if assisted else "scratch"

        if args.dry_run:
            state = "exists" if path.stem in existing else "create"
            print(f"  {title:12s} {kind:9s} {len(tasks):4d} tasks  [{state}]")
            continue

        project = existing.get(path.stem)
        if project is None:
            project = call(
                f"{args.host}/api/projects",
                token,
                "POST",
                {
                    "title": title,
                    "label_config": label_config,
                    "description": f"Albanian NER — batch {path.stem} ({kind}) {marker(path.stem)}",
                },
            )
            print(f"  {title:12s} created (id={project['id']})", end="", flush=True)
        else:
            print(f"  {title:12s} exists  (id={project['id']})", end="", flush=True)

        current = call(f"{args.host}/api/projects/{project['id']}", token).get("task_number") or 0
        if current:
            print(f"  — already holds {current} tasks, skipping import")
            continue

        result = call(
            f"{args.host}/api/projects/{project['id']}/import",
            token,
            "POST",
            raw=path.read_bytes(),
        )
        imported = result.get("task_count", result.get("total_tasks", "?"))
        preds = result.get("prediction_count", 0)
        print(f"  — imported {imported} tasks, {preds} predictions ({kind})")

    if not args.dry_run:
        print("\nfinal state:")
        for proj in call(f"{args.host}/api/projects?page_size=200", token).get("results", []):
            print(f"  id={proj['id']:<4} {proj['title']:14s} tasks={proj.get('task_number')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
