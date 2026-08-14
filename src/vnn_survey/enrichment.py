from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from vnn_survey.app.task_manager import cancellable_sleep, raise_if_cancelled
from vnn_survey.config import EnrichmentConfig
from vnn_survey.models import normalize_title

ItemProgressCallback = Callable[[int, int, str], None]


ENRICHMENT_FIELDS = [
    "abstract",
    "abstract_source",
    "abstract_status",
    "abstract_match_title",
    "abstract_match_score",
    "abstract_match_doi",
    "abstract_provider_id",
    "abstract_url",
    "abstract_citation_count",
    "abstract_checked_at",
    "abstract_error",
]

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

OPENALEX_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "abstract_inverted_index",
        "cited_by_count",
        "primary_location",
        "open_access",
    ]
)
SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "year",
        "abstract",
        "externalIds",
        "url",
        "citationCount",
        "openAccessPdf",
    ]
)


@dataclass(frozen=True, slots=True)
class AbstractMatch:
    source: str
    status: str
    abstract: str = ""
    match_title: str = ""
    match_score: float = 0.0
    match_doi: str = ""
    provider_id: str = ""
    url: str = ""
    citation_count: str = ""
    error: str = ""


@dataclass(frozen=True, slots=True)
class EnrichmentSummary:
    total: int
    eligible: int
    attempted: int
    with_abstract: int
    by_status: Counter[str]
    by_source: Counter[str]


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    rows: list[dict[str, str]]
    summary: EnrichmentSummary


def enrich_candidates(
    input_path: Path,
    output_path: Path,
    config: EnrichmentConfig,
    decisions: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    progress_callback: ItemProgressCallback | None = None,
) -> EnrichmentResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    clients = _build_clients(config)
    now = datetime.now().isoformat(timespec="seconds")
    attempted = 0
    enriched_rows: list[dict[str, str]] = []
    remaining_limit = limit
    attempt_total = _count_enrichment_attempts(rows, decisions, limit, overwrite)
    if progress_callback:
        progress_callback(0, attempt_total, "")

    for row in rows:
        enriched = dict(row)
        if _has_existing_abstract(enriched) and not overwrite:
            _mark_existing(enriched, checked_at=now)
            enriched_rows.append(enriched)
            continue

        if (
            decisions is not None
            and "auto_screening_decision" in row
            and row.get("auto_screening_decision") not in decisions
        ):
            _mark_skipped(enriched, status="skipped_decision", checked_at=now)
            enriched_rows.append(enriched)
            continue

        if remaining_limit is not None and remaining_limit <= 0:
            _mark_skipped(enriched, status="skipped_limit", checked_at=now)
            enriched_rows.append(enriched)
            continue

        attempted += 1
        if remaining_limit is not None:
            remaining_limit -= 1

        match = _find_abstract(row, clients)
        _apply_match(enriched, match=match, checked_at=now)
        enriched_rows.append(enriched)
        if progress_callback:
            progress_callback(attempted, attempt_total, row.get("title", ""))

    write_enriched_csv(enriched_rows, input_fields, output_path)
    return EnrichmentResult(
        rows=enriched_rows,
        summary=EnrichmentSummary(
            total=len(enriched_rows),
            eligible=sum(
                row.get("abstract_status", "") != "skipped_decision"
                for row in enriched_rows
            ),
            attempted=attempted,
            with_abstract=sum(bool((row.get("abstract") or "").strip()) for row in enriched_rows),
            by_status=Counter(row.get("abstract_status", "") for row in enriched_rows),
            by_source=Counter(row.get("abstract_source", "") for row in enriched_rows),
        ),
    )


def _count_enrichment_attempts(
    rows: list[dict[str, str]],
    decisions: set[str] | None,
    limit: int | None,
    overwrite: bool,
) -> int:
    eligible = 0
    for row in rows:
        if _has_existing_abstract(row) and not overwrite:
            continue
        if (
            decisions is not None
            and "auto_screening_decision" in row
            and row.get("auto_screening_decision") not in decisions
        ):
            continue
        eligible += 1
    return min(eligible, limit) if limit is not None else eligible


def write_enriched_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in ENRICHMENT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_enrichment_summary(summary: EnrichmentSummary, output_path: Path) -> None:
    payload = {
        "total": summary.total,
        "eligible": summary.eligible,
        "attempted": summary.attempted,
        "with_abstract": summary.with_abstract,
        "by_status": dict(summary.by_status),
        "by_source": dict(summary.by_source),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_clients(config: EnrichmentConfig) -> list[AbstractClient]:
    clients: list[AbstractClient] = []
    for provider in config.providers:
        normalized = provider.strip().lower()
        if normalized == "openalex":
            clients.append(OpenAlexClient(config))
        elif normalized in {"semantic_scholar", "semanticscholar", "s2"}:
            clients.append(SemanticScholarClient(config))
        else:
            raise ValueError(f"Unknown enrichment provider: {provider}")
    return clients


def _find_abstract(row: dict[str, str], clients: list[AbstractClient]) -> AbstractMatch:
    best_no_abstract: AbstractMatch | None = None
    errors: list[str] = []
    for client in clients:
        try:
            match = client.find(row)
        except Exception as exc:  # noqa: BLE001 - keep trying fallback providers.
            errors.append(f"{client.source}: {exc}")
            continue
        if match.status == "enriched":
            return match
        if match.status in {"found_no_abstract", "not_found"} and best_no_abstract is None:
            best_no_abstract = match

    if best_no_abstract:
        return best_no_abstract
    return AbstractMatch(source="", status="failed", error="; ".join(errors))


def _has_existing_abstract(row: dict[str, str]) -> bool:
    return bool((row.get("abstract") or "").strip())


def _mark_existing(row: dict[str, str], checked_at: str) -> None:
    row.setdefault("abstract_source", row.get("abstract_source", "existing") or "existing")
    row["abstract_status"] = row.get("abstract_status", "existing") or "existing"
    row["abstract_checked_at"] = checked_at


def _mark_skipped(row: dict[str, str], status: str, checked_at: str) -> None:
    row["abstract"] = row.get("abstract", "")
    row["abstract_source"] = ""
    row["abstract_status"] = status
    row["abstract_checked_at"] = checked_at
    for field in ENRICHMENT_FIELDS:
        row.setdefault(field, "")


def _apply_match(row: dict[str, str], match: AbstractMatch, checked_at: str) -> None:
    row.update(
        {
            "abstract": match.abstract,
            "abstract_source": match.source,
            "abstract_status": match.status,
            "abstract_match_title": match.match_title,
            "abstract_match_score": f"{match.match_score:.3f}" if match.match_score else "",
            "abstract_match_doi": match.match_doi,
            "abstract_provider_id": match.provider_id,
            "abstract_url": match.url,
            "abstract_citation_count": match.citation_count,
            "abstract_checked_at": checked_at,
            "abstract_error": match.error,
        }
    )


class AbstractClient:
    source: str

    def find(self, row: dict[str, str]) -> AbstractMatch:
        raise NotImplementedError


class OpenAlexClient(AbstractClient):
    source = "openalex"

    def __init__(self, config: EnrichmentConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(config)})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)
        self.api_key = os.environ.get(config.openalex_api_key_env, "").strip()
        self.email = os.environ.get(config.openalex_email_env, "").strip()

    def find(self, row: dict[str, str]) -> AbstractMatch:
        doi = _normalize_doi(row.get("doi", ""))
        title = row.get("title", "")
        if doi:
            payload = self._get_work_by_doi(doi)
            match = self._match_work(payload, source_title=title, trust_doi=True)
            if match.status != "not_found":
                return match
        if not title:
            return AbstractMatch(source=self.source, status="not_found")
        payload = self._search_by_title(title)
        return self._best_search_match(payload, source_title=title)

    def _get_work_by_doi(self, doi: str) -> dict[str, Any]:
        return self._request(
            cache_key=f"doi:{doi}",
            url=f"{OPENALEX_WORKS_URL}/doi:{quote(doi, safe='')}",
            params={"select": OPENALEX_SELECT_FIELDS},
        )

    def _search_by_title(self, title: str) -> dict[str, Any]:
        return self._request(
            cache_key=f"title:{title}",
            url=OPENALEX_WORKS_URL,
            params={
                "filter": f'title.search:"{title}"',
                "per-page": 5,
                "select": OPENALEX_SELECT_FIELDS,
            },
        )

    def _request(self, cache_key: str, url: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = dict(params)
        if self.api_key:
            request_params["api_key"] = self.api_key
        if self.email:
            request_params["mailto"] = self.email

        cache_path = _cache_path(self.cache_dir, cache_key, request_params)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = _request_json(
            session=self.session,
            method="GET",
            url=url,
            params=request_params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=self.config.request_delay_seconds,
        )
        _write_cache(cache_path, payload)
        return payload

    def _best_search_match(self, payload: dict[str, Any], source_title: str) -> AbstractMatch:
        candidates = payload.get("results", [])
        if not isinstance(candidates, list) or not candidates:
            return AbstractMatch(source=self.source, status="not_found")
        matches = [
            self._match_work(candidate, source_title=source_title, trust_doi=False)
            for candidate in candidates
        ]
        matches = [
            match
            for match in matches
            if match.match_score >= self.config.min_title_similarity
        ]
        if not matches:
            return AbstractMatch(source=self.source, status="not_found")
        return max(
            matches,
            key=lambda match: (
                match.status == "enriched",
                match.match_score,
                int(match.citation_count or 0),
            ),
        )

    def _match_work(
        self,
        payload: dict[str, Any],
        source_title: str,
        trust_doi: bool,
    ) -> AbstractMatch:
        if payload.get("error") == "not_found" or payload.get("_http_status") == 404:
            return AbstractMatch(source=self.source, status="not_found")

        match_title = str(payload.get("title") or payload.get("display_name") or "")
        score = 1.0 if trust_doi else title_similarity(source_title, match_title)
        if not trust_doi and score < self.config.min_title_similarity:
            return AbstractMatch(source=self.source, status="not_found")

        abstract = reconstruct_openalex_abstract(payload.get("abstract_inverted_index"))
        status = "enriched" if abstract else "found_no_abstract"
        primary_location = payload.get("primary_location") or {}
        open_access = payload.get("open_access") or {}
        url = (
            open_access.get("oa_url")
            or primary_location.get("landing_page_url")
            or primary_location.get("pdf_url")
            or payload.get("id")
            or ""
        )
        return AbstractMatch(
            source=self.source,
            status=status,
            abstract=abstract,
            match_title=match_title,
            match_score=score,
            match_doi=_normalize_doi(str(payload.get("doi") or "")),
            provider_id=str(payload.get("id") or ""),
            url=str(url),
            citation_count=str(payload.get("cited_by_count") or ""),
        )


class SemanticScholarClient(AbstractClient):
    source = "semantic_scholar"

    def __init__(self, config: EnrichmentConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(config)})
        self.api_key = os.environ.get(config.semantic_scholar_api_key_env, "").strip()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)

    def find(self, row: dict[str, str]) -> AbstractMatch:
        doi = _normalize_doi(row.get("doi", ""))
        title = row.get("title", "")
        if doi:
            payload = self._get_paper_by_id(f"DOI:{doi}")
            match = self._match_paper(payload, source_title=title, trust_doi=True)
            if match.status != "not_found":
                return match
        if not title:
            return AbstractMatch(source=self.source, status="not_found")
        payload = self._search_by_title(title)
        return self._best_search_match(payload, source_title=title)

    def _get_paper_by_id(self, paper_id: str) -> dict[str, Any]:
        return self._request(
            cache_key=f"paper:{paper_id}",
            url=f"{SEMANTIC_SCHOLAR_PAPER_URL}/{quote(paper_id, safe='')}",
            params={"fields": SEMANTIC_SCHOLAR_FIELDS},
        )

    def _search_by_title(self, title: str) -> dict[str, Any]:
        return self._request(
            cache_key=f"title:{title}",
            url=SEMANTIC_SCHOLAR_SEARCH_URL,
            params={"query": title, "limit": 5, "fields": SEMANTIC_SCHOLAR_FIELDS},
        )

    def _request(self, cache_key: str, url: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_path = _cache_path(self.cache_dir, cache_key, params)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = _request_json(
            session=self.session,
            method="GET",
            url=url,
            params=params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=max(self.config.request_delay_seconds, 1.0),
        )
        _write_cache(cache_path, payload)
        return payload

    def _best_search_match(self, payload: dict[str, Any], source_title: str) -> AbstractMatch:
        candidates = payload.get("data", [])
        if not isinstance(candidates, list) or not candidates:
            return AbstractMatch(source=self.source, status="not_found")
        matches = [
            self._match_paper(candidate, source_title=source_title, trust_doi=False)
            for candidate in candidates
        ]
        matches = [
            match
            for match in matches
            if match.match_score >= self.config.min_title_similarity
        ]
        if not matches:
            return AbstractMatch(source=self.source, status="not_found")
        return max(
            matches,
            key=lambda match: (
                match.status == "enriched",
                match.match_score,
                int(match.citation_count or 0),
            ),
        )

    def _match_paper(
        self,
        payload: dict[str, Any],
        source_title: str,
        trust_doi: bool,
    ) -> AbstractMatch:
        if payload.get("_http_status") == 404:
            return AbstractMatch(source=self.source, status="not_found")
        match_title = str(payload.get("title") or "")
        score = 1.0 if trust_doi else title_similarity(source_title, match_title)
        if not trust_doi and score < self.config.min_title_similarity:
            return AbstractMatch(source=self.source, status="not_found")

        external_ids = payload.get("externalIds") or {}
        open_access_pdf = payload.get("openAccessPdf") or {}
        abstract = str(payload.get("abstract") or "").strip()
        return AbstractMatch(
            source=self.source,
            status="enriched" if abstract else "found_no_abstract",
            abstract=abstract,
            match_title=match_title,
            match_score=score,
            match_doi=_normalize_doi(str(external_ids.get("DOI") or "")),
            provider_id=str(payload.get("paperId") or ""),
            url=str(open_access_pdf.get("url") or payload.get("url") or ""),
            citation_count=str(payload.get("citationCount") or ""),
        )


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    delay: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raise_if_cancelled()
            response = session.request(method, url, params=params, timeout=timeout)
            raise_if_cancelled()
            if response.status_code == 404:
                return {"_http_status": 404, "error": "not_found"}
            if response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else delay * attempt * 2
                cancellable_sleep(sleep_seconds)
                continue
            response.raise_for_status()
            cancellable_sleep(delay)
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                cancellable_sleep(delay * attempt)
    raise RuntimeError(f"request failed: {url}") from last_error


def reconstruct_openalex_abstract(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            try:
                words.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue
    return " ".join(word for _, word in sorted(words)).strip()


def title_similarity(left: str, right: str) -> float:
    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def _normalize_doi(value: str) -> str:
    return (
        value.strip()
        .removeprefix("https://doi.org/")
        .removeprefix("http://dx.doi.org/")
        .removeprefix("doi:")
        .strip()
        .lower()
    )


def _cache_path(cache_dir: Path | None, cache_key: str, params: dict[str, Any]) -> Path | None:
    if not cache_dir:
        return None
    normalized = json.dumps({"key": cache_key, "params": params}, sort_keys=True)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return cache_dir / f"{digest}.json"


def _write_cache(cache_path: Path | None, payload: dict[str, Any]) -> None:
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _provider_cache_dir(cache_dir: Path | None, provider: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / provider


def _user_agent(config: EnrichmentConfig) -> str:
    email = os.environ.get(config.openalex_email_env, "").strip()
    if email:
        return f"vnn-survey/0.1 abstract enrichment (mailto:{email})"
    return "vnn-survey/0.1 abstract enrichment"
