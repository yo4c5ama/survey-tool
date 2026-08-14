CONFIG ?= configs/transformer_verification.yaml
RUN_NAME ?= full-pipeline
SOURCE ?= auto
RUN_LLM ?= 1
RUN_SNOWBALL ?= 1
LIMIT_QUERIES ?=
SNOWBALL_LIMIT_SEEDS ?=
SNOWBALL_BACKWARD_LIMIT ?=
SNOWBALL_FORWARD_LIMIT ?=
ENRICH_LIMIT ?=
LLM_LIMIT ?=
OVERWRITE_ABSTRACTS ?= 0
OVERWRITE_LLM ?= 0
CORE_ONLINE ?= 1

.DEFAULT_GOAL := help

.PHONY: help setup app package queries pipeline pipeline-no-llm llm-dry-run smoke smoke-llm snowball tracks summarize strict-filter

help:
	@printf "Targets:\n"
	@printf "  make setup             Install/update the uv environment.\n"
	@printf "  make app               Start the local SurveyFlow graphical application.\n"
	@printf "  make package           Build the quick-start ZIP archive.\n"
	@printf "  make queries           Preview generated DBLP queries.\n"
	@printf "  make pipeline          Run the full pipeline, including OpenAI LLM screening.\n"
	@printf "  make pipeline-no-llm   Run collection, screening, and abstract enrichment only.\n"
	@printf "  make llm-dry-run       Run through enrichment, then count LLM-eligible rows only.\n"
	@printf "  make smoke             Tiny no-LLM smoke run: 1 DBLP query, 5 abstracts.\n"
	@printf "  make smoke-llm         Tiny LLM smoke run: 1 DBLP query, 5 abstracts, 5 LLM rows.\n"
	@printf "  make snowball          Add seed/backward/forward snowball rows to data/runs/latest.\n"
	@printf "  make tracks            Add research_track labels to data/runs/latest LLM output.\n"
	@printf "  make summarize         Regenerate final summary for data/runs/latest.\n"
	@printf "  make strict-filter     Create the strict non-arXiv/formal-only final candidate CSV.\n"
	@printf "\nVariables:\n"
	@printf "  CONFIG=%s\n" "$(CONFIG)"
	@printf "  RUN_NAME=%s\n" "$(RUN_NAME)"
	@printf "  SOURCE=%s\n" "$(SOURCE)"
	@printf "  CORE_ONLINE=%s\n" "$(CORE_ONLINE)"
	@printf "  RUN_SNOWBALL=%s\n" "$(RUN_SNOWBALL)"
	@printf "  LIMIT_QUERIES=%s ENRICH_LIMIT=%s LLM_LIMIT=%s\n" "$(LIMIT_QUERIES)" "$(ENRICH_LIMIT)" "$(LLM_LIMIT)"
	@printf "  SNOWBALL_LIMIT_SEEDS=%s SNOWBALL_BACKWARD_LIMIT=%s SNOWBALL_FORWARD_LIMIT=%s\n" "$(SNOWBALL_LIMIT_SEEDS)" "$(SNOWBALL_BACKWARD_LIMIT)" "$(SNOWBALL_FORWARD_LIMIT)"

setup:
	uv sync

app:
	uv run vnn-survey-app

package:
	scripts/build_release.sh

queries:
	uv run vnn-survey queries --config "$(CONFIG)"

pipeline:
	CONFIG="$(CONFIG)" RUN_NAME="$(RUN_NAME)" SOURCE="$(SOURCE)" RUN_LLM="$(RUN_LLM)" RUN_SNOWBALL="$(RUN_SNOWBALL)" LIMIT_QUERIES="$(LIMIT_QUERIES)" SNOWBALL_LIMIT_SEEDS="$(SNOWBALL_LIMIT_SEEDS)" SNOWBALL_BACKWARD_LIMIT="$(SNOWBALL_BACKWARD_LIMIT)" SNOWBALL_FORWARD_LIMIT="$(SNOWBALL_FORWARD_LIMIT)" ENRICH_LIMIT="$(ENRICH_LIMIT)" LLM_LIMIT="$(LLM_LIMIT)" OVERWRITE_ABSTRACTS="$(OVERWRITE_ABSTRACTS)" OVERWRITE_LLM="$(OVERWRITE_LLM)" CORE_ONLINE="$(CORE_ONLINE)" scripts/run_pipeline.sh

pipeline-no-llm:
	CONFIG="$(CONFIG)" RUN_NAME="$(RUN_NAME)" SOURCE="$(SOURCE)" RUN_LLM=0 RUN_SNOWBALL="$(RUN_SNOWBALL)" LIMIT_QUERIES="$(LIMIT_QUERIES)" SNOWBALL_LIMIT_SEEDS="$(SNOWBALL_LIMIT_SEEDS)" SNOWBALL_BACKWARD_LIMIT="$(SNOWBALL_BACKWARD_LIMIT)" SNOWBALL_FORWARD_LIMIT="$(SNOWBALL_FORWARD_LIMIT)" ENRICH_LIMIT="$(ENRICH_LIMIT)" CORE_ONLINE="$(CORE_ONLINE)" scripts/run_pipeline.sh

llm-dry-run:
	CONFIG="$(CONFIG)" RUN_NAME="$(RUN_NAME)" SOURCE="$(SOURCE)" RUN_LLM=1 LLM_DRY_RUN=1 RUN_SNOWBALL="$(RUN_SNOWBALL)" LIMIT_QUERIES="$(LIMIT_QUERIES)" SNOWBALL_LIMIT_SEEDS="$(SNOWBALL_LIMIT_SEEDS)" SNOWBALL_BACKWARD_LIMIT="$(SNOWBALL_BACKWARD_LIMIT)" SNOWBALL_FORWARD_LIMIT="$(SNOWBALL_FORWARD_LIMIT)" ENRICH_LIMIT="$(ENRICH_LIMIT)" LLM_LIMIT="$(LLM_LIMIT)" CORE_ONLINE="$(CORE_ONLINE)" scripts/run_pipeline.sh

smoke:
	CONFIG="$(CONFIG)" RUN_NAME=smoke SOURCE="$(SOURCE)" RUN_LLM=0 RUN_SNOWBALL=1 LIMIT_QUERIES=1 SNOWBALL_LIMIT_SEEDS=1 SNOWBALL_BACKWARD_LIMIT=1 SNOWBALL_FORWARD_LIMIT=1 ENRICH_LIMIT=5 CORE_ONLINE="$(CORE_ONLINE)" scripts/run_pipeline.sh

smoke-llm:
	CONFIG="$(CONFIG)" RUN_NAME=smoke-llm SOURCE="$(SOURCE)" RUN_LLM=1 RUN_SNOWBALL=1 LIMIT_QUERIES=1 SNOWBALL_LIMIT_SEEDS=1 SNOWBALL_BACKWARD_LIMIT=1 SNOWBALL_FORWARD_LIMIT=1 ENRICH_LIMIT=5 LLM_LIMIT=5 CORE_ONLINE="$(CORE_ONLINE)" scripts/run_pipeline.sh

snowball:
	uv run vnn-survey snowball-candidates --config "$(CONFIG)" --input data/runs/latest/processed/candidate_papers.csv --output data/runs/latest/processed/candidate_papers_snowballed.csv

tracks:
	uv run vnn-survey classify-tracks --input data/runs/latest/processed/candidate_papers_llm_screened.csv --output data/runs/latest/processed/candidate_papers_tracked.csv

summarize:
	uv run vnn-survey summarize-llm-screening --input data/runs/latest/processed/candidate_papers_tracked.csv

strict-filter:
	uv run vnn-survey filter-final-candidates --input data/runs/latest/processed/final_screening_recommendations.csv
