from pathlib import Path

import pytest

from vnn_survey.model_pricing import (
    get_model_price,
    is_openai_api_base_url,
    parse_openai_model_price,
    refresh_openai_model_price,
)

MODEL_PAGE = """# GPT-5.4 mini

Model ID: `gpt-5.4-mini`

## Pricing

### Text tokens

| Metric | Price | Unit |
| --- | ---: | --- |
| Input | $0.75 | 1M tokens |
| Cached input | $0.075 | 1M tokens |
| Output | $4.5 | 1M tokens |

## Endpoints
"""


def test_parse_openai_model_price_reads_only_the_pricing_section() -> None:
    price = parse_openai_model_price(
        MODEL_PAGE,
        requested_model="gpt-5.4-mini-2026-03-17",
        source_url="https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    )

    assert price.canonical_model == "gpt-5.4-mini"
    assert price.input_per_million == 0.75
    assert price.cached_input_per_million == 0.075
    assert price.output_per_million == 4.5
    assert price.source == "openai_official"


def test_local_catalog_resolves_the_default_snapshot() -> None:
    price = get_model_price("gpt-5.4-mini-2026-03-17")

    assert price is not None
    assert price.canonical_model == "gpt-5.4-mini"
    assert price.source == "local_catalog"


def test_refresh_caches_the_official_price(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Response:
        status_code = 200
        text = MODEL_PAGE

        @staticmethod
        def raise_for_status() -> None:
            return None

    requested_urls: list[str] = []

    def fake_get(url: str, **_kwargs: object) -> Response:
        requested_urls.append(url)
        return Response()

    monkeypatch.setattr("vnn_survey.model_pricing.requests.get", fake_get)
    cache_path = tmp_path / "model_pricing.json"

    refreshed = refresh_openai_model_price(
        "gpt-5.4-mini-2026-03-17",
        cache_path=cache_path,
    )
    cached = get_model_price("gpt-5.4-mini-2026-03-17", cache_path=cache_path)

    assert requested_urls == [
        "https://developers.openai.com/api/docs/models/gpt-5.4-mini.md"
    ]
    assert refreshed.source == "openai_official"
    assert cached == refreshed


def test_live_refresh_is_restricted_to_official_openai_hosts() -> None:
    assert is_openai_api_base_url("https://api.openai.com/v1")
    assert is_openai_api_base_url("https://kr.api.openai.com/v1")
    assert not is_openai_api_base_url("http://localhost:11434/v1")
