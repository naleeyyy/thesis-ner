#!/usr/bin/env bash
# Turn generated batch files into Label Studio task JSONs.
#
#   scripts/build_batch_tasks.sh
#
# Assisted batches get pre-labeled first; scratch batches deliberately ship no
# suggestions, so they need no API call at all — which is also why this costs roughly
# half what pre-labeling every sentence would.
#
# The shared overlap sentences appear in every batch, but the response cache means they
# are paid for once.

set -euo pipefail
cd "$(dirname "$0")/.."

BATCH_DIR="data/interim/batches"
TASK_DIR="data/interim/tasks"
MODEL="claude-sonnet-5"
MAX_USD="3.00"

[ -d "$BATCH_DIR" ] || { echo "no batches at $BATCH_DIR — run 'assign batches' first" >&2; exit 1; }
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ANTHROPIC_API_KEY not set — run: set -a; source .env; set +a" >&2; exit 1; }

mkdir -p "$TASK_DIR"
PROMPT_VERSION=$(uv run python -c "from src.annotate.prompt import PROMPT_VERSION; print(PROMPT_VERSION)")

for batch in "$BATCH_DIR"/batch_*.jsonl; do
  name=$(basename "$batch" .jsonl)          # e.g. batch_01_assisted
  num=$(echo "$name" | cut -d_ -f2)
  condition=$(echo "$name" | cut -d_ -f3)

  if [ "$condition" = "assisted" ]; then
    echo "==> batch $num (assisted): pre-labeling"
    prelabeled="data/interim/prelabeled_${num}.jsonl"
    uv run python -m src.annotate.llm_label \
      --model "$MODEL" --no-thinking --workers 1 --max-usd "$MAX_USD" \
      --in "$batch" --out "$prelabeled" 2>&1 | grep -E "wrote|cache|\\\$" || true
    uv run python -m src.annotate.labelstudio tasks --condition assisted \
      --in "$prelabeled" --out "$TASK_DIR/${num}.json" \
      --model-version "$MODEL/$PROMPT_VERSION" | tail -2
  else
    echo "==> batch $num (scratch): no pre-labeling needed"
    uv run python -m src.annotate.labelstudio tasks --condition scratch \
      --in "$batch" --out "$TASK_DIR/${num}.json" | tail -1
  fi
  echo
done

echo "=========================================="
echo "task files in $TASK_DIR:"
ls -1 "$TASK_DIR"
