from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from vnn_survey.app.task_manager import TaskCancelled, cancellable_sleep, raise_if_cancelled
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
    "abstract_provider_chain",
]

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

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
    cache_hits: int = 0
    api_requests: int = 0
    rate_limit_retries: int = 0
    rate_limit_wait_seconds: float = 0.0
    rate_limit_remaining: str = ""
    rate_limit_limit: str = ""
    batch_requests: int = 0


@dataclass(slots=True)
class RequestDiagnostics:
    cache_hits: int = 0
    api_requests: int = 0
    rate_limit_retries: int = 0
    rate_limit_wait_seconds: float = 0.0
    rate_limit_remaining: str = ""
    rate_limit_limit: str = ""
    batch_requests: int = 0


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

    diagnostics = RequestDiagnostics()
    clients = _build_clients(config, diagnostics)
    now = datetime.now().isoformat(timespec="seconds")
    rows = _merge_existing_enrichment(rows, output_path)
    enriched_rows = [dict(row) for row in rows]
    provider_chain = _provider_chain(config.providers)
    attempt_total = _count_enrichment_attempts(
        rows,
        decisions,
        limit,
        overwrite,
        provider_chain=provider_chain,
    )
    if progress_callback:
        progress_callback(0, attempt_total, "")

    pending_indices = _prepare_enrichment_rows(
        enriched_rows,
        decisions=decisions,
        limit=limit,
        overwrite=overwrite,
        checked_at=now,
        provider_chain=provider_chain,
    )
    try:
        attempted = _cascade_enrichment(
            enriched_rows,
            pending_indices,
            clients,
            diagnostics=diagnostics,
            checked_at=now,
            input_fields=input_fields,
            output_path=output_path,
            progress_callback=progress_callback,
            attempt_total=attempt_total,
            provider_chain=provider_chain,
        )
    except TaskCancelled:
        write_enriched_csv(enriched_rows, input_fields, output_path)
        raise

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
            cache_hits=diagnostics.cache_hits,
            api_requests=diagnostics.api_requests,
            rate_limit_retries=diagnostics.rate_limit_retries,
            rate_limit_wait_seconds=diagnostics.rate_limit_wait_seconds,
            rate_limit_remaining=diagnostics.rate_limit_remaining,
            rate_limit_limit=diagnostics.rate_limit_limit,
            batch_requests=diagnostics.batch_requests,
        ),
    )


def _prepare_enrichment_rows(
    rows: list[dict[str, str]],
    decisions: set[str] | None,
    limit: int | None,
    overwrite: bool,
    checked_at: str,
    provider_chain: str,
) -> list[int]:
    pending: list[int] = []
    remaining_limit = limit
    for index, row in enumerate(rows):
        if _has_existing_enrichment(row, provider_chain=provider_chain) and not overwrite:
            _mark_existing(row, checked_at=checked_at)
            continue
        if (
            decisions is not None
            and "auto_screening_decision" in row
            and row.get("auto_screening_decision") not in decisions
        ):
            _mark_skipped(row, status="skipped_decision", checked_at=checked_at)
            continue
        if remaining_limit is not None and remaining_limit <= 0:
            _mark_skipped(row, status="skipped_limit", checked_at=checked_at)
            continue
        pending.append(index)
        if remaining_limit is not None:
            remaining_limit -= 1
    return pending


def _cascade_enrichment(
    rows: list[dict[str, str]],
    pending_indices: list[int],
    clients: list[AbstractClient],
    *,
    diagnostics: RequestDiagnostics,
    checked_at: str,
    input_fields: list[str],
    output_path: Path,
    progress_callback: ItemProgressCallback | None,
    attempt_total: int,
    provider_chain: str,
) -> int:
    unresolved = list(pending_indices)
    best_matches: dict[int, AbstractMatch] = {}
    errors: dict[int, list[str]] = {index: [] for index in pending_indices}
    completed = 0

    for client in clients:
        if not unresolved:
            break
        provider_rows = [rows[index] for index in unresolved]
        completed_before_provider = completed

        def prefetch_progress(
            done: int,
            total: int,
            current: str,
            source: str = client.source,
            base: int = completed_before_provider,
        ) -> None:
            if progress_callback:
                progress_callback(
                    base,
                    attempt_total,
                    f"{source}: {current} ({done}/{total})",
                )

        try:
            client.prefetch(
                provider_rows,
                progress_callback=prefetch_progress if progress_callback else None,
            )
        except TaskCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - continue with later providers.
            for index in unresolved:
                errors[index].append(f"{client.source}: {exc}")
            continue

        still_unresolved: list[int] = []
        for index in unresolved:
            row = rows[index]
            try:
                match = client.find(row)
            except TaskCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - continue with later providers.
                errors[index].append(f"{client.source}: {exc}")
                still_unresolved.append(index)
                continue
            if match.status == "enriched":
                _apply_match(
                    row,
                    match=match,
                    checked_at=checked_at,
                    provider_chain=provider_chain,
                )
                completed += 1
                _report_enrichment_progress(
                    row,
                    completed,
                    attempt_total,
                    diagnostics,
                    progress_callback,
                    provider=client.source,
                )
                if completed % 25 == 0:
                    write_enriched_csv(rows, input_fields, output_path)
                continue
            if match.status in {"found_no_abstract", "not_found"}:
                best_matches.setdefault(index, match)
            elif match.error:
                errors[index].append(f"{client.source}: {match.error}")
            still_unresolved.append(index)
        unresolved = still_unresolved

    for index in unresolved:
        match = best_matches.get(index)
        if match is None:
            match = AbstractMatch(source="", status="failed", error="; ".join(errors[index]))
        elif errors[index]:
            match = AbstractMatch(
                source=match.source,
                status=match.status,
                match_title=match.match_title,
                match_score=match.match_score,
                match_doi=match.match_doi,
                provider_id=match.provider_id,
                url=match.url,
                citation_count=match.citation_count,
                error="; ".join(errors[index]),
            )
        _apply_match(
            rows[index],
            match=match,
            checked_at=checked_at,
            provider_chain=provider_chain,
        )
        completed += 1
        _report_enrichment_progress(
            rows[index],
            completed,
            attempt_total,
            diagnostics,
            progress_callback,
        )
        if completed % 25 == 0:
            write_enriched_csv(rows, input_fields, output_path)
    return completed


def _report_enrichment_progress(
    row: dict[str, str],
    completed: int,
    total: int,
    diagnostics: RequestDiagnostics,
    progress_callback: ItemProgressCallback | None,
    provider: str = "",
) -> None:
    if not progress_callback:
        return
    title = row.get("title", "")
    current = f"{provider}: {title}" if provider else title
    progress_callback(completed, total, _diagnostic_progress(current, diagnostics))


def _count_enrichment_attempts(
    rows: list[dict[str, str]],
    decisions: set[str] | None,
    limit: int | None,
    overwrite: bool,
    provider_chain: str,
) -> int:
    eligible = 0
    for row in rows:
        if _has_existing_enrichment(row, provider_chain=provider_chain) and not overwrite:
            continue
        if (
            decisions is not None
            and "auto_screening_decision" in row
            and row.get("auto_screening_decision") not in decisions
        ):
            continue
        eligible += 1
    return min(eligible, limit) if limit is not None else eligible

def _merge_existing_enrichment(
    rows: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, str]]:
    if not output_path.exists():
        return rows
    try:
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            previous_rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return rows
    previous = {_enrichment_row_key(row): row for row in previous_rows}
    merged: list[dict[str, str]] = []
    for row in rows:
        restored = dict(row)
        old = previous.get(_enrichment_row_key(row))
        if old:
            for field in ENRICHMENT_FIELDS:
                if old.get(field, ""):
                    restored[field] = old[field]
        merged.append(restored)
    return merged


def _enrichment_row_key(row: dict[str, str]) -> str:
    doi = _normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"
    provider_id = (row.get("provider_id") or "").strip().lower()
    if provider_id:
        return f"provider:{provider_id}"
    return f"title:{normalize_title(row.get('title', ''))}:{row.get('year', '')}"


def _diagnostic_progress(title: str, diagnostics: RequestDiagnostics) -> str:
    if not (
        diagnostics.api_requests
        or diagnostics.cache_hits
        or diagnostics.rate_limit_retries
        or diagnostics.batch_requests
    ):
        return title
    return (
        f"{title} [API {diagnostics.api_requests}, batch {diagnostics.batch_requests}, "
        f"cache {diagnostics.cache_hits}, 429 {diagnostics.rate_limit_retries}]"
    )


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
        "cache_hits": summary.cache_hits,
        "api_requests": summary.api_requests,
        "rate_limit_retries": summary.rate_limit_retries,
        "rate_limit_wait_seconds": round(summary.rate_limit_wait_seconds, 3),
        "rate_limit_remaining": summary.rate_limit_remaining,
        "rate_limit_limit": summary.rate_limit_limit,
        "batch_requests": summary.batch_requests,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_clients(
    config: EnrichmentConfig,
    diagnostics: RequestDiagnostics,
) -> list[AbstractClient]:
    clients: list[AbstractClient] = []
    for provider in config.providers:
        normalized = provider.strip().lower()
        if normalized == "arxiv":
            clients.append(ArxivClient(config, diagnostics))
        elif normalized == "pubmed":
            clients.append(PubmedClient(config, diagnostics))
        elif normalized == "crossref":
            clients.append(CrossrefClient(config, diagnostics))
        elif normalized == "openalex":
            clients.append(OpenAlexClient(config, diagnostics))
        elif normalized in {"semantic_scholar", "semanticscholar", "s2"}:
            clients.append(SemanticScholarClient(config, diagnostics))
        else:
            raise ValueError(f"Unknown enrichment provider: {provider}")
    return clients


def _find_abstract(row: dict[str, str], clients: list[AbstractClient]) -> AbstractMatch:
    best_no_abstract: AbstractMatch | None = None
    errors: list[str] = []
    for client in clients:
        try:
            match = client.find(row)
        except TaskCancelled:
            raise
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


def _has_existing_enrichment(row: dict[str, str], provider_chain: str = "") -> bool:
    if _has_existing_abstract(row):
        return True
    status = row.get("abstract_status", "")
    if status in {"enriched", "existing"}:
        return True
    return bool(
        provider_chain
        and status in {"found_no_abstract", "not_found"}
        and row.get("abstract_provider_chain", "") == provider_chain
        and not row.get("abstract_error", "")
    )


def _mark_existing(row: dict[str, str], checked_at: str) -> None:
    if _has_existing_abstract(row) and not row.get("abstract_source"):
        row["abstract_source"] = "existing"
    row["abstract_status"] = row.get("abstract_status", "existing") or "existing"
    row["abstract_checked_at"] = checked_at


def _mark_skipped(row: dict[str, str], status: str, checked_at: str) -> None:
    row["abstract"] = row.get("abstract", "")
    row["abstract_source"] = ""
    row["abstract_status"] = status
    row["abstract_checked_at"] = checked_at
    for field in ENRICHMENT_FIELDS:
        row.setdefault(field, "")


def _apply_match(
    row: dict[str, str],
    match: AbstractMatch,
    checked_at: str,
    provider_chain: str = "",
) -> None:
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
            "abstract_provider_chain": provider_chain,
        }
    )


def _provider_chain(providers: list[str]) -> str:
    return ",".join(provider.strip().lower() for provider in providers if provider.strip())


class AbstractClient:
    source: str

    def prefetch(
        self,
        rows: list[dict[str, str]],
        progress_callback: ItemProgressCallback | None = None,
    ) -> None:
        return None

    def find(self, row: dict[str, str]) -> AbstractMatch:
        raise NotImplementedError


class ArxivClient(AbstractClient):
    source = "arxiv"

    def __init__(self, config: EnrichmentConfig, diagnostics: RequestDiagnostics) -> None:
        self.config = config
        self.diagnostics = diagnostics
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(config)})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)
        self._prefetched_ids: set[str] = set()
        self._prefetched_papers: dict[str, dict[str, str]] = {}

    def prefetch(
        self,
        rows: list[dict[str, str]],
        progress_callback: ItemProgressCallback | None = None,
    ) -> None:
        identifiers = list(
            dict.fromkeys(
                identifier
                for row in rows
                if (identifier := _arxiv_id(row))
            )
        )
        batch_size = min(self.config.batch_size, 100)
        for offset in range(0, len(identifiers), batch_size):
            batch = identifiers[offset : offset + batch_size]
            request_count = self.diagnostics.api_requests
            payload = _request_text_cached(
                session=self.session,
                method="GET",
                url=ARXIV_API_URL,
                params={"id_list": ",".join(batch), "max_results": len(batch)},
                cache_dir=self.cache_dir,
                cache_key=f"id-batch:{'|'.join(batch)}",
                timeout=self.config.timeout_seconds,
                retries=self.config.retries,
                delay=self.config.arxiv_request_delay_seconds,
                diagnostics=self.diagnostics,
            )
            if self.diagnostics.api_requests > request_count:
                self.diagnostics.batch_requests += 1
            self._prefetched_ids.update(batch)
            self._prefetched_papers.update(_parse_arxiv_abstracts(payload))
            if progress_callback:
                progress_callback(
                    min(offset + len(batch), len(identifiers)),
                    len(identifiers),
                    "Batching arXiv ID lookups",
                )

    def find(self, row: dict[str, str]) -> AbstractMatch:
        identifier = _arxiv_id(row)
        if not identifier:
            return AbstractMatch(source=self.source, status="not_found")
        if identifier not in self._prefetched_ids:
            self.prefetch([row])
        payload = self._prefetched_papers.get(identifier)
        if payload is None:
            return AbstractMatch(source=self.source, status="not_found")
        abstract = payload.get("abstract", "")
        return AbstractMatch(
            source=self.source,
            status="enriched" if abstract else "found_no_abstract",
            abstract=abstract,
            match_title=payload.get("title", ""),
            match_score=1.0,
            match_doi=payload.get("doi", ""),
            provider_id=identifier,
            url=payload.get("url", ""),
        )


class PubmedClient(AbstractClient):
    source = "pubmed"

    def __init__(self, config: EnrichmentConfig, diagnostics: RequestDiagnostics) -> None:
        self.config = config
        self.diagnostics = diagnostics
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(config)})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)
        self.api_key = os.environ.get(config.pubmed_api_key_env, "").strip()
        self.email = os.environ.get(config.pubmed_email_env, "").strip()
        self._prefetched_ids: set[str] = set()
        self._prefetched_papers: dict[str, dict[str, str]] = {}

    def prefetch(
        self,
        rows: list[dict[str, str]],
        progress_callback: ItemProgressCallback | None = None,
    ) -> None:
        identifiers = list(
            dict.fromkeys(
                identifier
                for row in rows
                if (identifier := _pubmed_id(row))
            )
        )
        batch_size = min(self.config.batch_size, 200)
        for offset in range(0, len(identifiers), batch_size):
            batch = identifiers[offset : offset + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "tool": "surveyflow",
            }
            if self.api_key:
                params["api_key"] = self.api_key
            if self.email:
                params["email"] = self.email
            request_count = self.diagnostics.api_requests
            payload = _request_text_cached(
                session=self.session,
                method="POST",
                url=PUBMED_EFETCH_URL,
                params=params,
                cache_dir=self.cache_dir,
                cache_key=f"pmid-batch:{'|'.join(batch)}",
                timeout=self.config.timeout_seconds,
                retries=self.config.retries,
                delay=max(self.config.request_delay_seconds, 0.1 if self.api_key else 0.34),
                diagnostics=self.diagnostics,
                form_body=True,
            )
            if self.diagnostics.api_requests > request_count:
                self.diagnostics.batch_requests += 1
            self._prefetched_ids.update(batch)
            self._prefetched_papers.update(_parse_pubmed_abstracts(payload))
            if progress_callback:
                progress_callback(
                    min(offset + len(batch), len(identifiers)),
                    len(identifiers),
                    "Batching PubMed ID lookups",
                )

    def find(self, row: dict[str, str]) -> AbstractMatch:
        identifier = _pubmed_id(row)
        if not identifier:
            return AbstractMatch(source=self.source, status="not_found")
        if identifier not in self._prefetched_ids:
            self.prefetch([row])
        payload = self._prefetched_papers.get(identifier)
        if payload is None:
            return AbstractMatch(source=self.source, status="not_found")
        abstract = payload.get("abstract", "")
        return AbstractMatch(
            source=self.source,
            status="enriched" if abstract else "found_no_abstract",
            abstract=abstract,
            match_title=payload.get("title", ""),
            match_score=1.0,
            match_doi=payload.get("doi", ""),
            provider_id=identifier,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/",
        )


class CrossrefClient(AbstractClient):
    source = "crossref"

    def __init__(self, config: EnrichmentConfig, diagnostics: RequestDiagnostics) -> None:
        self.config = config
        self.diagnostics = diagnostics
        self.session = requests.Session()
        self.email = os.environ.get(config.crossref_email_env, "").strip()
        self.session.headers.update({"User-Agent": _user_agent(config, email=self.email)})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)

    def find(self, row: dict[str, str]) -> AbstractMatch:
        doi = _normalize_doi(row.get("doi", ""))
        title = row.get("title", "")
        if doi:
            payload = self._request(
                cache_key=f"doi:{doi}",
                url=f"{CROSSREF_WORKS_URL}/{quote(doi, safe='')}",
                params=self._polite_params(),
            )
            match = self._match_item(
                payload.get("message", {}) if isinstance(payload, dict) else {},
                source_title=title,
                trust_doi=True,
            )
            if match.status != "not_found":
                return match
        if not title:
            return AbstractMatch(source=self.source, status="not_found")
        payload = self._request(
            cache_key=f"title:{title}",
            url=CROSSREF_WORKS_URL,
            params={**self._polite_params(), "query.bibliographic": title, "rows": 5},
        )
        items = payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []
        matches = [
            self._match_item(item, source_title=title, trust_doi=False)
            for item in items
            if isinstance(item, dict)
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

    def _polite_params(self) -> dict[str, str]:
        return {"mailto": self.email} if self.email else {}

    def _request(self, cache_key: str, url: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_path = _cache_path(self.cache_dir, cache_key, params)
        if cache_path and cache_path.exists():
            self.diagnostics.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))
        payload = _request_json(
            session=self.session,
            method="GET",
            url=url,
            params=params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=max(self.config.request_delay_seconds, 0.1),
            diagnostics=self.diagnostics,
        )
        _write_cache(cache_path, payload)
        return payload if isinstance(payload, dict) else {}

    def _match_item(
        self,
        payload: dict[str, Any],
        *,
        source_title: str,
        trust_doi: bool,
    ) -> AbstractMatch:
        if not payload or payload.get("_http_status") == 404:
            return AbstractMatch(source=self.source, status="not_found")
        match_title = _first_text(payload.get("title"))
        score = 1.0 if trust_doi else title_similarity(source_title, match_title)
        if not trust_doi and score < self.config.min_title_similarity:
            return AbstractMatch(source=self.source, status="not_found")
        abstract = _clean_markup(payload.get("abstract"))
        return AbstractMatch(
            source=self.source,
            status="enriched" if abstract else "found_no_abstract",
            abstract=abstract,
            match_title=match_title,
            match_score=score,
            match_doi=_normalize_doi(str(payload.get("DOI") or "")),
            provider_id=str(payload.get("DOI") or ""),
            url=str(payload.get("URL") or ""),
            citation_count=str(payload.get("is-referenced-by-count") or ""),
        )


class OpenAlexClient(AbstractClient):
    source = "openalex"

    def __init__(self, config: EnrichmentConfig, diagnostics: RequestDiagnostics) -> None:
        self.config = config
        self.diagnostics = diagnostics
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(config)})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)
        self.api_key = os.environ.get(config.openalex_api_key_env, "").strip()
        self.email = os.environ.get(config.openalex_email_env, "").strip()
        self._prefetched_dois: set[str] = set()
        self._prefetched_works: dict[str, dict[str, Any]] = {}

    def prefetch(
        self,
        rows: list[dict[str, str]],
        progress_callback: ItemProgressCallback | None = None,
    ) -> None:
        if rows and not self.api_key:
            raise RuntimeError(
                "OpenAlex requires an API key for sustained use. Add a free key on AI settings."
            )
        dois = list(
            dict.fromkeys(
                doi
                for row in rows
                if (doi := _normalize_doi(row.get("doi", "")))
            )
        )
        if not dois:
            return
        batch_size = min(self.config.batch_size, 100)
        for offset in range(0, len(dois), batch_size):
            batch = dois[offset : offset + batch_size]
            try:
                request_count = self.diagnostics.api_requests
                payload = self._request(
                    cache_key=f"doi-batch:{'|'.join(batch)}",
                    url=OPENALEX_WORKS_URL,
                    params={
                        "filter": "doi:"
                        + "|".join(f"https://doi.org/{doi}" for doi in batch),
                        "per-page": len(batch),
                        "select": OPENALEX_SELECT_FIELDS,
                    },
                )
            except TaskCancelled:
                raise
            except Exception:  # noqa: BLE001 - individual lookups remain available.
                continue
            if self.diagnostics.api_requests > request_count:
                self.diagnostics.batch_requests += 1
            self._prefetched_dois.update(batch)
            for item in payload.get("results", []):
                if not isinstance(item, dict):
                    continue
                doi = _normalize_doi(str(item.get("doi") or ""))
                if doi:
                    self._prefetched_works[doi] = item
            if progress_callback:
                progress_callback(
                    min(offset + len(batch), len(dois)),
                    len(dois),
                    "Batching OpenAlex DOI lookups",
                )

    def find(self, row: dict[str, str]) -> AbstractMatch:
        doi = _normalize_doi(row.get("doi", ""))
        title = row.get("title", "")
        if doi:
            if doi in self._prefetched_dois:
                payload = self._prefetched_works.get(
                    doi,
                    {"_http_status": 404, "error": "not_found"},
                )
            else:
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
            self.diagnostics.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if not self.api_key:
            raise RuntimeError(
                "OpenAlex requires an API key for sustained use. Add a free key on AI settings."
            )

        payload = _request_json(
            session=self.session,
            method="GET",
            url=url,
            params=request_params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=self.config.request_delay_seconds,
            diagnostics=self.diagnostics,
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

    def __init__(self, config: EnrichmentConfig, diagnostics: RequestDiagnostics) -> None:
        self.config = config
        self.diagnostics = diagnostics
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _user_agent(config)})
        self.api_key = os.environ.get(config.semantic_scholar_api_key_env, "").strip()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        self.cache_dir = _provider_cache_dir(config.cache_dir, self.source)
        self._prefetched_dois: set[str] = set()
        self._prefetched_papers: dict[str, dict[str, Any]] = {}

    def prefetch(
        self,
        rows: list[dict[str, str]],
        progress_callback: ItemProgressCallback | None = None,
    ) -> None:
        dois = list(
            dict.fromkeys(
                doi
                for row in rows
                if (doi := _normalize_doi(row.get("doi", "")))
            )
        )
        if not dois:
            return
        batch_size = min(self.config.batch_size, 500)
        for offset in range(0, len(dois), batch_size):
            batch = dois[offset : offset + batch_size]
            paper_ids = [f"DOI:{doi}" for doi in batch]
            try:
                payload = self._request_batch(paper_ids)
            except TaskCancelled:
                raise
            except Exception:  # noqa: BLE001 - individual lookups remain available.
                continue
            self._prefetched_dois.update(batch)
            for doi, item in zip(batch, payload, strict=False):
                if isinstance(item, dict):
                    self._prefetched_papers[doi] = item
            if progress_callback:
                progress_callback(
                    min(offset + len(batch), len(dois)),
                    len(dois),
                    "Batching Semantic Scholar DOI lookups",
                )

    def find(self, row: dict[str, str]) -> AbstractMatch:
        doi = _normalize_doi(row.get("doi", ""))
        title = row.get("title", "")
        if doi:
            if doi in self._prefetched_dois:
                payload = self._prefetched_papers.get(
                    doi,
                    {"_http_status": 404, "error": "not_found"},
                )
            else:
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

    def _request_batch(self, paper_ids: list[str]) -> list[Any]:
        params = {"fields": SEMANTIC_SCHOLAR_FIELDS}
        cache_path = _cache_path(
            self.cache_dir,
            f"paper-batch:{'|'.join(paper_ids)}",
            {**params, "ids": paper_ids},
        )
        if cache_path and cache_path.exists():
            self.diagnostics.cache_hits += 1
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else []

        request_count = self.diagnostics.api_requests
        payload = _request_json(
            session=self.session,
            method="POST",
            url=SEMANTIC_SCHOLAR_BATCH_URL,
            params=params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=max(self.config.request_delay_seconds, 1.0),
            diagnostics=self.diagnostics,
            json_body={"ids": paper_ids},
        )
        if self.diagnostics.api_requests > request_count:
            self.diagnostics.batch_requests += 1
        _write_cache(cache_path, payload)
        return payload if isinstance(payload, list) else []

    def _search_by_title(self, title: str) -> dict[str, Any]:
        return self._request(
            cache_key=f"title:{title}",
            url=SEMANTIC_SCHOLAR_SEARCH_URL,
            params={"query": title, "limit": 5, "fields": SEMANTIC_SCHOLAR_FIELDS},
        )

    def _request(self, cache_key: str, url: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_path = _cache_path(self.cache_dir, cache_key, params)
        if cache_path and cache_path.exists():
            self.diagnostics.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = _request_json(
            session=self.session,
            method="GET",
            url=url,
            params=params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=max(self.config.request_delay_seconds, 1.0),
            diagnostics=self.diagnostics,
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


def _arxiv_id(row: dict[str, str]) -> str:
    source = (row.get("source") or "").strip().lower()
    provider_id = (row.get("provider_id") or "").strip()
    if source == "arxiv" and provider_id:
        return _normalize_arxiv_id(provider_id)
    url = row.get("url", "")
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", url, flags=re.IGNORECASE)
    return _normalize_arxiv_id(match.group(1)) if match else ""


def _normalize_arxiv_id(value: str) -> str:
    identifier = value.strip().removeprefix("arXiv:").removesuffix(".pdf")
    return re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)


def _pubmed_id(row: dict[str, str]) -> str:
    source = (row.get("source") or "").strip().lower()
    provider_id = (row.get("provider_id") or "").strip()
    if source == "pubmed" and provider_id.isdigit():
        return provider_id
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", row.get("url", ""))
    return match.group(1) if match else ""


def _parse_arxiv_abstracts(payload: str) -> dict[str, dict[str, str]]:
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(payload)
    papers: dict[str, dict[str, str]] = {}
    for entry in root.findall("atom:entry", namespace):
        url = _element_text(entry.find("atom:id", namespace))
        identifier = _normalize_arxiv_id(url.rstrip("/").rsplit("/", 1)[-1])
        if not identifier:
            continue
        papers[identifier] = {
            "title": _collapse(_element_text(entry.find("atom:title", namespace))),
            "abstract": _collapse(_element_text(entry.find("atom:summary", namespace))),
            "doi": _normalize_doi(_element_text(entry.find("arxiv:doi", namespace))),
            "url": url,
        }
    return papers


def _parse_pubmed_abstracts(payload: str) -> dict[str, dict[str, str]]:
    root = ET.fromstring(payload)
    papers: dict[str, dict[str, str]] = {}
    for article_node in root.findall(".//PubmedArticle"):
        citation = article_node.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        if citation is None or article is None:
            continue
        pmid = _element_text(citation.find("PMID"))
        if not pmid:
            continue
        sections: list[str] = []
        for node in article.findall("Abstract/AbstractText"):
            text = _collapse(" ".join(node.itertext()))
            if not text:
                continue
            label = str(node.attrib.get("Label") or "").strip()
            sections.append(f"{label}: {text}" if label else text)
        doi = ""
        for article_id in article_node.findall("PubmedData/ArticleIdList/ArticleId"):
            if str(article_id.attrib.get("IdType") or "").lower() == "doi":
                doi = _normalize_doi(_element_text(article_id))
                break
        papers[pmid] = {
            "title": _collapse(" ".join(article.find("ArticleTitle").itertext()))
            if article.find("ArticleTitle") is not None
            else "",
            "abstract": " ".join(sections).strip(),
            "doi": doi,
        }
    return papers


def _element_text(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext()).strip()


def _collapse(value: str) -> str:
    return " ".join(str(value or "").split())


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0] if value else "").strip()
    return str(value or "").strip()


def _clean_markup(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        root = ET.fromstring(f"<root>{text}</root>")
        return _collapse(" ".join(root.itertext()))
    except ET.ParseError:
        return _collapse(html.unescape(re.sub(r"<[^>]+>", " ", text)))


def _request_text_cached(
    *,
    session: requests.Session,
    method: str,
    url: str,
    params: dict[str, Any],
    cache_dir: Path | None,
    cache_key: str,
    timeout: int,
    retries: int,
    delay: float,
    diagnostics: RequestDiagnostics,
    form_body: bool = False,
) -> str:
    cache_path = _cache_path(cache_dir, cache_key, params, suffix="xml")
    if cache_path and cache_path.exists():
        diagnostics.cache_hits += 1
        return cache_path.read_text(encoding="utf-8")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raise_if_cancelled()
            diagnostics.api_requests += 1
            response = session.request(
                method,
                url,
                params=None if form_body else params,
                data=params if form_body else None,
                timeout=timeout,
            )
            raise_if_cancelled()
            if response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else delay * attempt * 2
                diagnostics.rate_limit_retries += 1
                diagnostics.rate_limit_wait_seconds += sleep_seconds
                cancellable_sleep(sleep_seconds)
                continue
            response.raise_for_status()
            cancellable_sleep(delay)
            text = response.text
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
            return text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                cancellable_sleep(delay * attempt)
    raise RuntimeError(f"request failed: {url}") from last_error


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    delay: float,
    diagnostics: RequestDiagnostics | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raise_if_cancelled()
            if diagnostics is not None:
                diagnostics.api_requests += 1
            response = session.request(
                method,
                url,
                params=params,
                json=json_body,
                timeout=timeout,
            )
            raise_if_cancelled()
            if diagnostics is not None:
                diagnostics.rate_limit_remaining = response.headers.get(
                    "X-RateLimit-Remaining",
                    diagnostics.rate_limit_remaining,
                )
                diagnostics.rate_limit_limit = response.headers.get(
                    "X-RateLimit-Limit",
                    diagnostics.rate_limit_limit,
                )
            if response.status_code == 404:
                return {"_http_status": 404, "error": "not_found"}
            if response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else delay * attempt * 2
                if diagnostics is not None:
                    diagnostics.rate_limit_retries += 1
                    diagnostics.rate_limit_wait_seconds += sleep_seconds
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


def _cache_path(
    cache_dir: Path | None,
    cache_key: str,
    params: dict[str, Any],
    suffix: str = "json",
) -> Path | None:
    if not cache_dir:
        return None
    normalized = json.dumps({"key": cache_key, "params": params}, sort_keys=True)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return cache_dir / f"{digest}.{suffix}"


def _write_cache(cache_path: Path | None, payload: Any) -> None:
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _provider_cache_dir(cache_dir: Path | None, provider: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / provider


def _user_agent(config: EnrichmentConfig, email: str = "") -> str:
    contact = email or os.environ.get(config.openalex_email_env, "").strip()
    if contact:
        return f"vnn-survey/0.1 abstract enrichment (mailto:{contact})"
    return "vnn-survey/0.1 abstract enrichment"
