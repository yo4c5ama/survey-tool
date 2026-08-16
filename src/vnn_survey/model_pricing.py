from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = REPO_ROOT / "configs" / "model_pricing.yaml"
OFFICIAL_MODEL_URL = "https://developers.openai.com/api/docs/models/{model}"
_DATED_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class ModelPricingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    requested_model: str
    canonical_model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None
    source: str
    source_url: str
    updated_at: str


def get_model_price(
    model: str,
    *,
    cache_path: Path | None = None,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> ModelPrice | None:
    cached = _read_cached_price(cache_path, model) if cache_path else None
    return cached or load_catalog_price(model, catalog_path=catalog_path)


def refresh_openai_model_price(
    model: str,
    *,
    cache_path: Path,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    timeout_seconds: int = 15,
) -> ModelPrice:
    requested_model = model.strip()
    if not requested_model:
        raise ModelPricingError("A model name is required before refreshing its price.")

    catalog_price = load_catalog_price(requested_model, catalog_path=catalog_path)
    candidates = _price_page_candidates(requested_model, catalog_price)
    last_error = "No official model page was found."
    for candidate in candidates:
        url = OFFICIAL_MODEL_URL.format(model=quote(candidate, safe="-._"))
        try:
            response = requests.get(
                f"{url}.md",
                timeout=timeout_seconds,
                headers={"User-Agent": "SurveyFlow/1.0 model-pricing"},
            )
            if response.status_code == 404:
                last_error = f"The official model page does not list {candidate}."
                continue
            response.raise_for_status()
            price = parse_openai_model_price(
                response.text,
                requested_model=requested_model,
                source_url=url,
            )
            _write_cached_price(cache_path, price)
            return price
        except requests.RequestException as exc:
            last_error = str(exc)
            continue
    raise ModelPricingError(f"Could not refresh the OpenAI model price: {last_error}")


def parse_openai_model_price(
    markdown: str,
    *,
    requested_model: str,
    source_url: str,
) -> ModelPrice:
    pricing_section = _markdown_section(markdown, "## Pricing", "## Endpoints")
    input_price = _table_price(pricing_section, "Input")
    output_price = _table_price(pricing_section, "Output")
    cached_price = _table_price(pricing_section, "Cached input", required=False)
    model_match = re.search(r"^Model ID:\s*`([^`]+)`", markdown, flags=re.MULTILINE)
    canonical_model = model_match.group(1) if model_match else requested_model
    if input_price is None or output_price is None:
        raise ModelPricingError("The official model page did not contain text token prices.")
    return ModelPrice(
        requested_model=requested_model,
        canonical_model=canonical_model,
        input_per_million=input_price,
        output_per_million=output_price,
        cached_input_per_million=cached_price,
        source="openai_official",
        source_url=source_url,
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def load_catalog_price(
    model: str,
    *,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> ModelPrice | None:
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    except (OSError, TypeError, yaml.YAMLError):
        return None
    requested_model = model.strip()
    normalized = _normalize_snapshot(requested_model)
    models = payload.get("models", {})
    if not isinstance(models, dict):
        return None
    for canonical_model, raw_value in models.items():
        if not isinstance(raw_value, dict):
            continue
        aliases = {
            str(canonical_model),
            *(str(value) for value in raw_value.get("aliases", [])),
        }
        if requested_model not in aliases and normalized not in aliases:
            continue
        try:
            cached = raw_value.get("cached_input_per_million")
            return ModelPrice(
                requested_model=requested_model,
                canonical_model=str(canonical_model),
                input_per_million=float(raw_value["input_per_million"]),
                output_per_million=float(raw_value["output_per_million"]),
                cached_input_per_million=None if cached is None else float(cached),
                source="local_catalog",
                source_url=str(raw_value.get("source_url") or payload.get("source_url") or ""),
                updated_at=str(payload.get("last_updated") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def estimate_token_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    price: ModelPrice,
) -> float:
    return (
        max(int(input_tokens), 0) * price.input_per_million
        + max(int(output_tokens), 0) * price.output_per_million
    ) / 1_000_000


def output_tokens_per_paper(catalog_path: Path = DEFAULT_CATALOG_PATH) -> int:
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        return max(int(payload.get("screening_output_tokens_per_paper") or 220), 1)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return 220


def is_openai_api_base_url(base_url: str) -> bool:
    hostname = (urlparse(base_url.strip()).hostname or "").lower()
    return hostname == "api.openai.com" or hostname.endswith(".api.openai.com")


def _price_page_candidates(model: str, catalog_price: ModelPrice | None) -> list[str]:
    values = [
        catalog_price.canonical_model if catalog_price else "",
        _normalize_snapshot(model),
        model,
    ]
    return list(dict.fromkeys(value for value in values if value))


def _normalize_snapshot(model: str) -> str:
    return _DATED_SNAPSHOT.sub("", model.strip())


def _markdown_section(markdown: str, start: str, end: str) -> str:
    start_index = markdown.find(start)
    if start_index < 0:
        return ""
    end_index = markdown.find(end, start_index + len(start))
    return markdown[start_index:] if end_index < 0 else markdown[start_index:end_index]


def _table_price(section: str, metric: str, *, required: bool = True) -> float | None:
    match = re.search(
        rf"^\|\s*{re.escape(metric)}\s*\|\s*\$([0-9]+(?:\.[0-9]+)?)\s*\|",
        section,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    if required:
        return None
    return None


def _read_cached_price(cache_path: Path, model: str) -> ModelPrice | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        raw_value = payload.get("prices", {}).get(model)
        if not isinstance(raw_value, dict):
            return None
        return ModelPrice(**raw_value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_price(cache_path: Path, price: ModelPrice) -> None:
    payload: dict[str, Any] = {"prices": {}}
    try:
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and isinstance(existing.get("prices"), dict):
            payload = existing
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    payload.setdefault("prices", {})[price.requested_model] = asdict(price)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
