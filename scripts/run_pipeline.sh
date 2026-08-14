#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/transformer_verification.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
RUN_NAME="${RUN_NAME:-full-pipeline}"
SOURCE="${SOURCE:-auto}"
RUN_LLM="${RUN_LLM:-1}"
LLM_DRY_RUN="${LLM_DRY_RUN:-0}"
RUN_SNOWBALL="${RUN_SNOWBALL:-1}"
LIMIT_QUERIES="${LIMIT_QUERIES:-}"
SNOWBALL_LIMIT_SEEDS="${SNOWBALL_LIMIT_SEEDS:-}"
SNOWBALL_BACKWARD_LIMIT="${SNOWBALL_BACKWARD_LIMIT:-}"
SNOWBALL_FORWARD_LIMIT="${SNOWBALL_FORWARD_LIMIT:-}"
ENRICH_LIMIT="${ENRICH_LIMIT:-}"
LLM_LIMIT="${LLM_LIMIT:-}"
OVERWRITE_ABSTRACTS="${OVERWRITE_ABSTRACTS:-0}"
OVERWRITE_LLM="${OVERWRITE_LLM:-0}"
CORE_ONLINE="${CORE_ONLINE:-1}"

run() {
  printf "\n+ "
  printf "%q " "$@"
  printf "\n"
  "$@"
}

add_optional_limit() {
  local -n args_ref="$1"
  local option_name="$2"
  local option_value="$3"
  if [[ -n "$option_value" ]]; then
    args_ref+=("$option_name" "$option_value")
  fi
}

echo "Pipeline config: $CONFIG"
echo "Run name: $RUN_NAME"
echo "DBLP source: $SOURCE"
echo "Output root: $OUTPUT_DIR"
echo "Run snowballing: $RUN_SNOWBALL"
echo "Run LLM: $RUN_LLM"
echo "CORE online lookup: $CORE_ONLINE"

run uv sync

collect_args=(
  uv run vnn-survey collect-dblp
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --source "$SOURCE"
  --timestamped
  --run-name "$RUN_NAME"
)
add_optional_limit collect_args --limit-queries "$LIMIT_QUERIES"
run "${collect_args[@]}"

RUN_DIR="$OUTPUT_DIR/runs/latest"
PROCESSED_DIR="$RUN_DIR/processed"
SEARCH_CANDIDATES="$PROCESSED_DIR/candidate_papers.csv"
SNOWBALLED="$PROCESSED_DIR/candidate_papers_snowballed.csv"
CANDIDATES="$SEARCH_CANDIDATES"
VENUES="$PROCESSED_DIR/candidate_papers_venues.csv"
SCREENED="$PROCESSED_DIR/candidate_papers_screened.csv"
ENRICHED="$PROCESSED_DIR/candidate_papers_enriched.csv"
LLM_SCREENED="$PROCESSED_DIR/candidate_papers_llm_screened.csv"
TRACKED="$PROCESSED_DIR/candidate_papers_tracked.csv"

if [[ "$RUN_SNOWBALL" == "1" ]]; then
  snowball_args=(
    uv run vnn-survey snowball-candidates
    --config "$CONFIG"
    --input "$SEARCH_CANDIDATES"
    --output "$SNOWBALLED"
  )
  add_optional_limit snowball_args --limit-seeds "$SNOWBALL_LIMIT_SEEDS"
  add_optional_limit snowball_args --max-backward-per-seed "$SNOWBALL_BACKWARD_LIMIT"
  add_optional_limit snowball_args --max-forward-per-seed "$SNOWBALL_FORWARD_LIMIT"
  run "${snowball_args[@]}"
  CANDIDATES="$SNOWBALLED"
else
  echo "Skipping snowballing. Venue enrichment will use DBLP candidates only."
fi

venue_args=(
  uv run vnn-survey enrich-venues
  --config "$CONFIG"
  --input "$CANDIDATES"
  --output "$VENUES"
)
if [[ "$CORE_ONLINE" == "1" ]]; then
  venue_args+=(--core-online)
else
  venue_args+=(--no-core-online)
fi
run "${venue_args[@]}"

run uv run vnn-survey screen-candidates \
  --config "$CONFIG" \
  --input "$VENUES" \
  --output "$SCREENED"

enrich_args=(
  uv run vnn-survey enrich-abstracts
  --config "$CONFIG"
  --input "$SCREENED"
  --output "$ENRICHED"
)
add_optional_limit enrich_args --limit "$ENRICH_LIMIT"
if [[ "$OVERWRITE_ABSTRACTS" == "1" ]]; then
  enrich_args+=(--overwrite)
fi
run "${enrich_args[@]}"

if [[ "$RUN_LLM" == "1" ]]; then
  if [[ "$LLM_DRY_RUN" != "1" && -z "${OPENAI_API_KEY:-}" && ! -s ".secrets/openai_api_key" ]]; then
    echo "Warning: no OPENAI_API_KEY environment variable or .secrets/openai_api_key file found."
    echo "The LLM step may fail unless your config points to another valid key source."
  fi

  llm_args=(
    uv run vnn-survey llm-screen
    --config "$CONFIG"
    --input "$ENRICHED"
    --output "$LLM_SCREENED"
  )
  add_optional_limit llm_args --limit "$LLM_LIMIT"
  if [[ "$OVERWRITE_LLM" == "1" ]]; then
    llm_args+=(--overwrite)
  fi
  if [[ "$LLM_DRY_RUN" == "1" ]]; then
    llm_args+=(--dry-run)
  fi
  run "${llm_args[@]}"

  if [[ "$LLM_DRY_RUN" == "1" ]]; then
    echo "LLM dry run complete. Final summary is skipped because no LLM CSV was written."
  else
    run uv run vnn-survey classify-tracks \
      --input "$LLM_SCREENED" \
      --output "$TRACKED"

    run uv run vnn-survey summarize-llm-screening \
      --input "$TRACKED"
  fi
else
  echo "Skipping LLM screening. Run with RUN_LLM=1 when you are ready for API calls."
fi

echo
echo "Pipeline complete."
echo "Latest run directory: $RUN_DIR"
echo "Processed outputs: $PROCESSED_DIR"
