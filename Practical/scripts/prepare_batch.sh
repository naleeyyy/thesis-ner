#!/usr/bin/env bash
# Prepare one annotation batch end to end: reserve → pre-label → Label Studio tasks.
#
#   scripts/prepare_batch.sh unique  ana r1 50
#   scripts/prepare_batch.sh overlap ov1 30 ana blerim drita
#
# The output is a task JSON to import into that person's Label Studio project. Sentences
# are reserved in data/interim/assignments.jsonl, so no sentence is ever handed to two
# people by accident and none is silently skipped.
#
# Overlap mode writes TWO task files from the same sentences — one with the model's
# suggestions, one without. That pair is the anchoring experiment; the sentences must be
# identical or the two arms are not comparable.

set -euo pipefail
cd "$(dirname "$0")/.."

POOL="data/raw/wiki_segmented_v2.jsonl"
MODEL="claude-sonnet-5"
MAX_USD="1.00"          # per batch; a 50-sentence batch costs roughly $0.08
INTERIM="data/interim"

die() { echo "error: $*" >&2; exit 1; }

[ -f "$POOL" ] || die "pool not found at $POOL — run src.data.segment first"
[ -n "${ANTHROPIC_API_KEY:-}" ] || die "ANTHROPIC_API_KEY not set — run: set -a; source .env; set +a"

MODE="${1:-}"; shift || die "usage: $0 unique|overlap ..."

case "$MODE" in
  unique)
    WHO="${1:?annotator name}"; BATCH="${2:?batch label}"; N="${3:-50}"
    TAG="${BATCH}_${WHO}"
    ASSIGN_ARGS=(--assignee "$WHO")
    ;;
  overlap)
    BATCH="${1:?batch label}"; N="${2:?count}"; shift 2
    [ "$#" -ge 2 ] || die "overlap needs at least two annotators"
    TAG="$BATCH"
    ASSIGN_ARGS=(--overlap)
    for who in "$@"; do ASSIGN_ARGS+=(--assignee "$who"); done
    ;;
  *) die "first argument must be 'unique' or 'overlap'" ;;
esac

BATCH_FILE="$INTERIM/batch_${TAG}.jsonl"
PRELABEL_FILE="$INTERIM/prelabeled_${TAG}.jsonl"

echo "==> reserving $N sentences ($MODE)"
uv run python -m src.annotate.assign reserve \
  --pool "$POOL" --n "$N" --batch "$BATCH" --out "$BATCH_FILE" "${ASSIGN_ARGS[@]}"

echo
echo "==> pre-labeling (sequential: cheaper, since parallel calls miss the prompt cache)"
uv run python -m src.annotate.llm_label \
  --model "$MODEL" --no-thinking --workers 1 --max-usd "$MAX_USD" \
  --in "$BATCH_FILE" --out "$PRELABEL_FILE" 2>&1 | grep -vE "sent/s\]|it/s\]"

echo
echo "==> building task files"
PROMPT_VERSION=$(uv run python -c "from src.annotate.prompt import PROMPT_VERSION; print(PROMPT_VERSION)")
uv run python -m src.annotate.labelstudio tasks --condition assisted \
  --in "$PRELABEL_FILE" --out "$INTERIM/tasks_${TAG}_assisted.json" \
  --model-version "$MODEL/$PROMPT_VERSION"

if [ "$MODE" = "overlap" ]; then
  uv run python -m src.annotate.labelstudio tasks --condition scratch \
    --in "$PRELABEL_FILE" --out "$INTERIM/tasks_${TAG}_scratch.json"
fi

echo
echo "=========================================================="
echo "import into Label Studio:"
echo "  assisted → $INTERIM/tasks_${TAG}_assisted.json"
[ "$MODE" = "overlap" ] && echo "  scratch  → $INTERIM/tasks_${TAG}_scratch.json"
echo
uv run python -m src.annotate.assign status --pool "$POOL" | head -3
