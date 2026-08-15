#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TOOLS_DIR="$ROOT_DIR/.surveyflow-tools"
UV_BIN=""

if command -v uv >/dev/null 2>&1; then
    UV_BIN=$(command -v uv)
elif [ -x "$TOOLS_DIR/uv" ]; then
    UV_BIN="$TOOLS_DIR/uv"
else
    mkdir -p "$TOOLS_DIR"
    printf '%s\n' "uv was not found. Installing a private copy for SurveyFlow..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env \
            UV_INSTALL_DIR="$TOOLS_DIR" UV_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env \
            UV_INSTALL_DIR="$TOOLS_DIR" UV_NO_MODIFY_PATH=1 sh
    else
        printf '%s\n' "SurveyFlow needs curl or wget to install uv." >&2
        exit 1
    fi
    UV_BIN="$TOOLS_DIR/uv"
fi

cd "$ROOT_DIR"
printf '%s\n' "Preparing SurveyFlow. The first start may download Python and dependencies..."
"$UV_BIN" sync --frozen --no-dev --reinstall-package vnn-survey
printf '%s\n' "Opening SurveyFlow at http://localhost:${SURVEYFLOW_PORT:-8501}"
exec "$UV_BIN" run --frozen --no-dev --no-sync vnn-survey-app
