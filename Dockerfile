FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    SURVEYFLOW_ADDRESS=0.0.0.0 \
    SURVEYFLOW_PORT=8501 \
    VNN_SURVEY_APP_DATA=/app/data/app_projects \
    VNN_SURVEY_APP_SECRETS=/app/.secrets/app_projects

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY configs/domain_profiles.yaml configs/source_catalog.yaml ./configs/
COPY data/venue_quality ./data/venue_quality

RUN uv sync --frozen --no-dev \
    && mkdir -p /app/data/app_projects /app/.secrets/app_projects \
    && chown -R 1000:1000 /app

USER 1000:1000

EXPOSE 8501

CMD ["vnn-survey-app"]
