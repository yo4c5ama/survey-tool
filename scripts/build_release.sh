#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR="$ROOT_DIR/dist"
OUTPUT_FILE="$OUTPUT_DIR/SurveyFlow-quickstart.zip"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"
zip -q -FS -r "$OUTPUT_FILE" \
    README.md \
    pyproject.toml \
    uv.lock \
    start.sh \
    start.ps1 \
    start.bat \
    Dockerfile \
    compose.yaml \
    .dockerignore \
    src \
    configs/domain_profiles.yaml \
    configs/source_catalog.yaml \
    data/venue_quality \
    -x "*/__pycache__/*" "*.pyc"

printf 'Created %s\n' "$OUTPUT_FILE"
