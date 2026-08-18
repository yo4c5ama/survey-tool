#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR="$ROOT_DIR/dist"
OUTPUT_FILE="$OUTPUT_DIR/SurveyFlow-quickstart.zip"

mkdir -p "$OUTPUT_DIR"
STAGING_DIR=$(mktemp -d "$OUTPUT_DIR/.release-build.XXXXXX")
STAGING_ARCHIVE="$STAGING_DIR/SurveyFlow-quickstart.zip"
trap 'rm -rf "$STAGING_DIR"' 0 HUP INT TERM

set -- \
    README.md \
    pyproject.toml \
    uv.lock \
    start.sh \
    start.ps1 \
    start.bat \
    Dockerfile \
    compose.yaml \
    .dockerignore \
    .streamlit/config.toml \
    src \
    configs/domain_profiles.yaml \
    configs/model_pricing.yaml \
    configs/source_catalog.yaml \
    data/venue_quality

for path do
    parent=$(dirname "$path")
    mkdir -p "$STAGING_DIR/$parent"
    cp -Rp "$ROOT_DIR/$path" "$STAGING_DIR/$parent/"
done

find "$STAGING_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGING_DIR" -type f -name '*.pyc' -delete

(
    cd "$STAGING_DIR"
    zip -q -r "$STAGING_ARCHIVE" "$@"
)

# Refresh only release-controlled paths. Runtime projects, backups, secrets,
# caches, and the local virtual environment in dist/ remain untouched.
for path do
    parent=$(dirname "$path")
    mkdir -p "$OUTPUT_DIR/$parent"
    case "$path" in
        src|data/venue_quality)
            rm -rf "$OUTPUT_DIR/$path"
            cp -Rp "$STAGING_DIR/$path" "$OUTPUT_DIR/$parent/"
            ;;
        *)
            cp -p "$STAGING_DIR/$path" "$OUTPUT_DIR/$path"
            ;;
    esac
done

mv "$STAGING_ARCHIVE" "$OUTPUT_FILE"

printf 'Synced runnable workspace %s\n' "$OUTPUT_DIR"
printf 'Created %s\n' "$OUTPUT_FILE"
