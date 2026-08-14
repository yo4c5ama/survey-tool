from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import requests

from vnn_survey.app.task_manager import cancellable_sleep, raise_if_cancelled
from vnn_survey.config import DblpConfig
from vnn_survey.models import PaperRecord

DBLP_SPARQL_URL = "https://sparql.dblp.org/sparql"


class DblpSparqlClient:
    """Small DBLP SPARQL fallback for publication-title keyword searches."""

    def __init__(self, config: DblpConfig, user_agent: str | None = None) -> None:
        self.config = config
        self.cache_dir = _sparql_cache_dir(config.cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent
                or "vnn-survey/0.1 literature collection (https://sparql.dblp.org/sparql)"
            }
        )

    def search(self, query: str) -> list[PaperRecord]:
        groups = _query_groups(query)
        if not groups:
            return []

        records: list[PaperRecord] = []
        for page in range(self.config.max_pages_per_query):
            offset = page * self.config.hits_per_page
            sparql_query = _build_title_query(
                groups=groups,
                limit=self.config.hits_per_page,
                offset=offset,
            )
            payload = self._request(query=query, sparql_query=sparql_query, offset=offset)
            bindings = _bindings(payload)
            if not bindings:
                break
            records.extend(_parse_binding(binding, query=query) for binding in bindings)
            if len(bindings) < self.config.hits_per_page:
                break
            cancellable_sleep(self.config.request_delay_seconds)
        return records

    def _request(self, query: str, sparql_query: str, offset: int) -> dict[str, Any]:
        cache_path = self._cache_path(query, offset)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                raise_if_cancelled()
                response = self.session.post(
                    DBLP_SPARQL_URL,
                    data=sparql_query.encode("utf-8"),
                    headers={
                        "Content-Type": "application/sparql-query",
                        "Accept": "application/sparql-results+json",
                    },
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
            f"DBLP SPARQL request failed for query={query!r}, offset={offset}"
        ) from last_error

    def _cache_path(self, query: str, offset: int) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(f"{query}\0{offset}\0{self.config.hits_per_page}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"


def _build_title_query(groups: list[list[list[str]]], limit: int, offset: int) -> str:
    filters = "\n  ".join(_group_filter(group) for group in groups)
    return f"""
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?publ ?title ?year ?venue ?doi ?type
       (GROUP_CONCAT(DISTINCT ?authorName; SEPARATOR="; ") AS ?authors)
WHERE {{
  ?publ dblp:title ?title .
  OPTIONAL {{ ?publ dblp:yearOfPublication ?year . }}
  OPTIONAL {{ ?publ dblp:publishedIn ?venue . }}
  OPTIONAL {{ ?publ dblp:doi ?doi . }}
  OPTIONAL {{ ?publ dblp:bibtexType ?type . }}
  OPTIONAL {{
    ?publ dblp:authoredBy ?author .
    ?author rdfs:label ?authorName .
  }}
  {filters}
}}
GROUP BY ?publ ?title ?year ?venue ?doi ?type
ORDER BY DESC(?year) ?title
LIMIT {int(limit)}
OFFSET {int(offset)}
""".strip()


def _group_filter(group: list[list[str]]) -> str:
    alternatives = []
    for alternative in group:
        contains_all_words = " && ".join(_contains_title_word(word) for word in alternative)
        alternatives.append(f"({contains_all_words})")
    return f"FILTER({' || '.join(alternatives)})"


def _contains_title_word(word: str) -> str:
    return f"CONTAINS(LCASE(STR(?title)), {json.dumps(word.lower())})"


def _bindings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bindings = payload.get("results", {}).get("bindings", [])
    return bindings if isinstance(bindings, list) else []


def _parse_binding(binding: dict[str, Any], query: str) -> PaperRecord:
    publ_uri = _value(binding, "publ")
    return PaperRecord(
        title=_clean_title(_value(binding, "title") or ""),
        source="dblp_sparql",
        query=query,
        year=_to_int(_value(binding, "year")),
        authors=_split_authors(_value(binding, "authors")),
        venue=_value(binding, "venue"),
        doi=_normalize_doi(_value(binding, "doi")),
        url=publ_uri,
        dblp_key=_dblp_key_from_uri(publ_uri),
        publication_type=_type_from_uri(_value(binding, "type")),
        provider_id=publ_uri,
        raw=binding,
    )


def _query_groups(query: str) -> list[list[list[str]]]:
    """Parse DBLP-style query logic into SPARQL title filters.

    DBLP uses whitespace as AND and the pipe symbol as OR. For example,
    ``transformer|bert verification|certification`` becomes:
    (transformer OR bert) AND (verification OR certification).
    Multiword terms still behave as AND, matching DBLP's current search behavior.
    """
    groups: list[list[list[str]]] = []
    for group_text in query.split():
        alternatives: list[list[str]] = []
        for alternative_text in group_text.split("|"):
            words = re.findall(r"[A-Za-z0-9]+", alternative_text.lower())
            words = [word for word in words if len(word) >= 2]
            if words:
                alternatives.append(words)
        if alternatives:
            groups.append(alternatives)
    return groups


def _value(binding: dict[str, Any], key: str) -> str | None:
    value = binding.get(key, {}).get("value")
    return str(value) if value is not None else None


def _split_authors(value: str | None) -> list[str]:
    if not value:
        return []
    return [author.strip() for author in value.split(";") if author.strip()]


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _clean_title(title: str) -> str:
    return title.removesuffix(".").strip()


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    return (
        value.removeprefix("https://doi.org/")
        .removeprefix("http://dx.doi.org/")
        .removeprefix("doi:")
        .strip()
    )


def _dblp_key_from_uri(value: str | None) -> str | None:
    if not value:
        return None
    prefix = "https://dblp.org/rec/"
    return value.removeprefix(prefix) if value.startswith(prefix) else value


def _type_from_uri(value: str | None) -> str | None:
    if not value:
        return None
    return value.rsplit("#", maxsplit=1)[-1].rsplit("/", maxsplit=1)[-1]


def _sparql_cache_dir(cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir.parent / "dblp_sparql"
