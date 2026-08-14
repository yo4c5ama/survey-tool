from __future__ import annotations

import hashlib
import json
from typing import Any

import requests

from vnn_survey.app.task_manager import cancellable_sleep, raise_if_cancelled
from vnn_survey.config import DblpConfig
from vnn_survey.models import PaperRecord

DBLP_API_URL = "https://dblp.org/search/publ/api"


class DblpClient:
    def __init__(self, config: DblpConfig, user_agent: str | None = None) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent
                or "vnn-survey/0.1 literature collection (https://dblp.org/search/publ/api)"
            }
        )

    def search(self, query: str) -> list[PaperRecord]:
        records: list[PaperRecord] = []
        for page in range(self.config.max_pages_per_query):
            offset = page * self.config.hits_per_page
            payload = self._request_page(query=query, offset=offset)
            hits = _extract_hits(payload)
            if not hits:
                break
            records.extend(_parse_hit(hit, query=query) for hit in hits)
            if len(hits) < self.config.hits_per_page:
                break
            cancellable_sleep(self.config.request_delay_seconds)
        return records

    def _request_page(self, query: str, offset: int) -> dict[str, Any]:
        params = {
            "q": query,
            "format": "json",
            "h": self.config.hits_per_page,
            "f": offset,
            "c": 0,
        }
        cache_path = self._cache_path(query, offset)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                raise_if_cancelled()
                response = self.session.get(
                    DBLP_API_URL,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                raise_if_cancelled()
                response.raise_for_status()
                payload = response.json()
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.config.retries:
                    cancellable_sleep(self.config.request_delay_seconds * attempt)
        raise RuntimeError(
            f"DBLP request failed for query={query!r}, offset={offset}"
        ) from last_error

    def _cache_path(self, query: str, offset: int):
        if not self.config.cache_dir:
            return None
        key = hashlib.sha256(f"{query}\0{offset}\0{self.config.hits_per_page}".encode()).hexdigest()
        return self.config.cache_dir / f"{key}.json"


def _extract_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits = payload.get("result", {}).get("hits", {}).get("hit", [])
    if isinstance(hits, dict):
        return [hits]
    if isinstance(hits, list):
        return hits
    return []


def _parse_hit(hit: dict[str, Any], query: str) -> PaperRecord:
    info = hit.get("info", {})
    return PaperRecord(
        title=_clean_title(str(info.get("title") or "")),
        source="dblp",
        query=query,
        year=_to_int(info.get("year")),
        authors=_parse_authors(info.get("authors")),
        venue=info.get("venue"),
        doi=info.get("doi"),
        url=info.get("url"),
        dblp_key=info.get("key"),
        publication_type=info.get("type"),
        provider_id=hit.get("@id"),
        raw=hit,
    )


def _parse_authors(value: Any) -> list[str]:
    if not value:
        return []
    authors = value.get("author") if isinstance(value, dict) else value
    if isinstance(authors, str):
        return [authors]
    if isinstance(authors, dict):
        return [str(authors.get("text") or authors.get("@pid") or "")]
    if isinstance(authors, list):
        parsed = []
        for author in authors:
            if isinstance(author, str):
                parsed.append(author)
            elif isinstance(author, dict):
                parsed.append(str(author.get("text") or author.get("@pid") or ""))
        return [author for author in parsed if author]
    return []


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_title(title: str) -> str:
    return title.removesuffix(".").strip()
