from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Protocol
from urllib.parse import quote

import requests
import yaml

from vnn_survey.app.task_manager import TaskCancelled, cancellable_sleep, raise_if_cancelled
from vnn_survey.config import SnowballingConfig, SurveyConfig
from vnn_survey.enrichment import title_similarity
from vnn_survey.manual_audit import is_manually_excluded
from vnn_survey.models import normalize_title

SnowballProgressCallback = Callable[[int, int, str, int], None]


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_GRAPH_URL = "https://api.semanticscholar.org/graph/v1"
OPENCITATIONS_INDEX_URL = "https://opencitations.net/index/api/v2"
OPENCITATIONS_META_URL = "https://api.opencitations.net/meta/v1"
SNOWBALL_PROVIDER_IDS = ("semantic_scholar", "opencitations", "openalex")
OPENALEX_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "authorships",
        "primary_location",
        "locations",
        "open_access",
        "type",
        "referenced_works",
        "referenced_works_count",
        "cited_by_count",
    ]
)

STANDARD_FIELDS = [
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "dblp_key",
    "publication_type",
    "source",
    "query",
    "provider_id",
]

SNOWBALL_FIELDS = [
    "discovery_sources",
    "snowball_relations",
    "snowball_seed_titles",
    "snowball_seed_ids",
    "snowball_depths",
    "snowball_provider",
    "snowball_citation_count",
    "snowball_reference_count",
    "snowball_coverage_status",
    "snowball_missing_providers",
    "snowball_coverage_notes",
]
PAPER_CSV_HIDDEN_FIELDS = frozenset(
    {
        "snowball_provider",
        "snowball_coverage_status",
        "snowball_missing_providers",
        "snowball_coverage_notes",
    }
)


@dataclass(frozen=True, slots=True)
class SeedPaper:
    title: str
    doi: str = ""
    semantic_scholar_id: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    source: str = "manual_seed"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SnowballingSummary:
    input_rows: int
    input_unique_rows: int
    output_rows: int
    seeds_loaded: int
    seeds_resolved: int
    added_rows: int
    merged_rows: int
    references_available: int
    references_fetched: int
    citations_available: int
    citations_fetched: int
    backward_truncated_seeds: int
    forward_truncated_seeds: int
    by_relation: Counter[str]
    by_source: Counter[str]
    by_provider: Counter[str]
    provider_order: tuple[str, ...]
    provider_strategy: str
    provider_successes: Counter[str]
    provider_failures: Counter[str]
    provider_errors: dict[str, list[str]]
    seed_diagnostics: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SnowballingResult:
    rows: list[dict[str, str]]
    summary: SnowballingSummary


@dataclass(frozen=True, slots=True)
class SeedExportResult:
    seeds: list[dict[str, str]]
    output_path: Path


@dataclass(frozen=True, slots=True)
class RetrievedWorks:
    works: list[dict[str, Any]]
    available: int
    truncated: bool


class SnowballClient(Protocol):
    provider_id: str

    def resolve_seed(self, seed: SeedPaper) -> dict[str, Any] | None: ...

    def referenced_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks: ...

    def citing_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks: ...


def snowball_candidates(
    input_path: Path,
    output_path: Path,
    config: SurveyConfig,
    seed_papers_path: Path | None = None,
    max_backward_per_seed: int | None = None,
    max_forward_per_seed: int | None = None,
    limit_seeds: int | None = None,
    include_seed_papers: bool | None = None,
    progress_callback: SnowballProgressCallback | None = None,
) -> SnowballingResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        input_rows = [dict(row) for row in reader]

    snowball_config = config.snowballing
    resolved_seed_path = seed_papers_path or snowball_config.seed_papers_path
    seeds = load_seed_papers(resolved_seed_path)
    if limit_seeds is not None:
        seeds = seeds[:limit_seeds]

    backward_limit = _effective_limit(
        max_backward_per_seed
        if max_backward_per_seed is not None
        else snowball_config.max_backward_per_seed
    )
    forward_limit = _effective_limit(
        max_forward_per_seed
        if max_forward_per_seed is not None
        else snowball_config.max_forward_per_seed
    )
    include_seeds = (
        include_seed_papers
        if include_seed_papers is not None
        else snowball_config.include_seed_papers
    )

    clients, initialization_errors = _create_snowball_clients(
        snowball_config,
        provider_order=snowball_config.providers,
        year_start=config.years.start,
        year_end=config.years.end,
    )
    provider_order = tuple(snowball_config.providers)
    if not clients:
        details = "; ".join(
            f"{provider}: {message}"
            for provider, messages in initialization_errors.items()
            for message in messages
        )
        raise RuntimeError(f"No citation provider is available. {details}".strip())
    merged_rows: list[dict[str, str]] = []
    row_index: dict[str, int] = {}
    by_relation: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_provider: Counter[str] = Counter()
    provider_successes: Counter[str] = Counter()
    provider_failures: Counter[str] = Counter(
        {provider: len(messages) for provider, messages in initialization_errors.items()}
    )
    provider_errors = {
        provider: list(messages) for provider, messages in initialization_errors.items()
    }
    seed_diagnostics: list[dict[str, Any]] = []
    merged_count = 0

    for row in input_rows:
        prepared = dict(row)
        prepared.setdefault("discovery_sources", "dblp_search")
        for field in (
            "snowball_coverage_status",
            "snowball_missing_providers",
            "snowball_coverage_notes",
        ):
            prepared[field] = ""
        _index_or_merge(prepared, merged_rows=merged_rows, row_index=row_index)
    input_unique_rows = len(merged_rows)

    seeds_resolved = 0
    if progress_callback:
        progress_callback(0, len(seeds), "", input_unique_rows)
    for seed_index, seed in enumerate(seeds, start=1):
        resolved_works: list[tuple[SnowballClient, dict[str, Any]]] = []
        seed_provider_errors = {
            provider: list(messages) for provider, messages in initialization_errors.items()
        }
        for client in clients:
            try:
                work = client.resolve_seed(seed)
            except TaskCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - provider failover is intentional.
                message = _record_provider_failure(
                    client.provider_id,
                    exc,
                    provider_failures,
                    provider_errors,
                )
                seed_provider_errors.setdefault(client.provider_id, []).append(message)
                continue
            if work:
                resolved_works.append((client, work))

        if not resolved_works:
            diagnostic = _seed_coverage_diagnostic(
                seed=seed,
                seed_id=_seed_fallback_id(seed),
                provider_order=provider_order,
                strategy=snowball_config.provider_strategy,
                resolved_providers=[],
                references=None,
                citations=None,
                provider_errors=seed_provider_errors,
            )
            seed_diagnostics.append(diagnostic)
            _annotate_seed_coverage(
                merged_rows,
                row_index,
                seed,
                diagnostic,
            )
            write_snowballed_csv(merged_rows, input_fields=input_fields, output_path=output_path)
            if progress_callback:
                progress_callback(seed_index, len(seeds), seed.title, len(merged_rows))
            continue
        seeds_resolved += 1
        seed_id = _provider_work_id(resolved_works[0][1])
        if include_seeds:
            seed_versions = (
                resolved_works
                if snowball_config.provider_strategy == "merge"
                else resolved_works[:1]
            )
            for client, work in seed_versions:
                merged_count += _add_work(
                    work=work,
                    relation="seed",
                    provider=client.provider_id,
                    seed=seed,
                    seed_id=seed_id,
                    depth=0,
                    merged_rows=merged_rows,
                    row_index=row_index,
                    config=config,
                )
                by_relation["seed"] += 1
                by_source[seed.source or "manual_seed"] += 1
                by_provider[client.provider_id] += 1

        references = _retrieve_from_providers(
            resolved_works,
            relation="backward",
            limit=backward_limit,
            strategy=snowball_config.provider_strategy,
            provider_successes=provider_successes,
            provider_failures=provider_failures,
            provider_errors=provider_errors,
        )
        for provider, referenced_work in references["works"]:
            merged_count += _add_work(
                work=referenced_work,
                relation="backward",
                provider=provider,
                seed=seed,
                seed_id=seed_id,
                depth=1,
                merged_rows=merged_rows,
                row_index=row_index,
                config=config,
            )
            by_relation["backward"] += 1
            by_source[seed.source or "manual_seed"] += 1
            by_provider[provider] += 1

        citations = _retrieve_from_providers(
            resolved_works,
            relation="forward",
            limit=forward_limit,
            strategy=snowball_config.provider_strategy,
            provider_successes=provider_successes,
            provider_failures=provider_failures,
            provider_errors=provider_errors,
        )
        for provider, citing_work in citations["works"]:
            merged_count += _add_work(
                work=citing_work,
                relation="forward",
                provider=provider,
                seed=seed,
                seed_id=seed_id,
                depth=1,
                merged_rows=merged_rows,
                row_index=row_index,
                config=config,
            )
            by_relation["forward"] += 1
            by_source[seed.source or "manual_seed"] += 1
            by_provider[provider] += 1
        for relation_result in (references, citations):
            for provider, messages in relation_result["errors"].items():
                target = seed_provider_errors.setdefault(provider, [])
                for message in messages:
                    if message not in target:
                        target.append(message)
        diagnostic = _seed_coverage_diagnostic(
            seed=seed,
            seed_id=seed_id,
            provider_order=provider_order,
            strategy=snowball_config.provider_strategy,
            resolved_providers=[client.provider_id for client, _work in resolved_works],
            references=references,
            citations=citations,
            provider_errors=seed_provider_errors,
        )
        seed_diagnostics.append(diagnostic)
        _annotate_seed_coverage(merged_rows, row_index, seed, diagnostic)
        write_snowballed_csv(merged_rows, input_fields=input_fields, output_path=output_path)
        if progress_callback:
            progress_callback(seed_index, len(seeds), seed.title, len(merged_rows))

    write_snowballed_csv(merged_rows, input_fields=input_fields, output_path=output_path)
    added_rows = len(merged_rows) - input_unique_rows
    summary = SnowballingSummary(
        input_rows=len(input_rows),
        input_unique_rows=input_unique_rows,
        output_rows=len(merged_rows),
        seeds_loaded=len(seeds),
        seeds_resolved=seeds_resolved,
        added_rows=added_rows,
        merged_rows=merged_count,
        references_available=sum(int(item["references_available"]) for item in seed_diagnostics),
        references_fetched=sum(int(item["references_fetched"]) for item in seed_diagnostics),
        citations_available=sum(int(item["citations_available"]) for item in seed_diagnostics),
        citations_fetched=sum(int(item["citations_fetched"]) for item in seed_diagnostics),
        backward_truncated_seeds=sum(bool(item["backward_truncated"]) for item in seed_diagnostics),
        forward_truncated_seeds=sum(bool(item["forward_truncated"]) for item in seed_diagnostics),
        by_relation=by_relation,
        by_source=by_source,
        by_provider=by_provider,
        provider_order=provider_order,
        provider_strategy=snowball_config.provider_strategy,
        provider_successes=provider_successes,
        provider_failures=provider_failures,
        provider_errors=provider_errors,
        seed_diagnostics=seed_diagnostics,
    )
    return SnowballingResult(rows=merged_rows, summary=summary)


def export_seed_papers_from_csv(
    input_path: Path,
    output_path: Path,
    source_label: str = "strict_include_arxiv",
) -> SeedExportResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    seeds: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("strict_filter_decision") and row.get("strict_filter_decision") != "keep":
            continue
        if is_manually_excluded(row):
            continue
        title = (row.get("title") or "").strip()
        if not title:
            continue
        doi = _normalize_doi(row.get("doi", ""))
        semantic_scholar_id = _semantic_scholar_id_from_row(row)
        arxiv_id = _arxiv_id_from_row(row)
        openalex_id = _openalex_id_from_row(row)
        key = (
            doi
            or semantic_scholar_id.lower()
            or arxiv_id.lower()
            or openalex_id.lower()
            or normalize_title(title)
        )
        if key in seen:
            continue
        seen.add(key)

        seed = {
            "title": title,
            "source": source_label,
            "notes": _seed_notes(row),
        }
        if doi:
            seed["doi"] = doi
        if semantic_scholar_id:
            seed["semantic_scholar_id"] = semantic_scholar_id
        if arxiv_id:
            seed["arxiv_id"] = arxiv_id
        if openalex_id:
            seed["openalex_id"] = openalex_id
        seeds.append(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            {"seed_papers": seeds},
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )
    return SeedExportResult(seeds=seeds, output_path=output_path)


def load_seed_papers(path: Path | None) -> list[SeedPaper]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw_items = raw.get("seed_papers", raw) if isinstance(raw, dict) else raw
    if not isinstance(raw_items, list):
        raise ValueError(f"Seed paper file must contain a list or seed_papers list: {path}")
    seeds: list[SeedPaper] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title and not any(
            item.get(field) for field in ["doi", "semantic_scholar_id", "arxiv_id", "openalex_id"]
        ):
            continue
        seeds.append(
            SeedPaper(
                title=title,
                doi=_normalize_doi(str(item.get("doi") or "")),
                semantic_scholar_id=str(item.get("semantic_scholar_id") or "").strip(),
                arxiv_id=_normalize_arxiv_id(str(item.get("arxiv_id") or "")),
                openalex_id=str(item.get("openalex_id") or "").strip(),
                source=str(item.get("source") or "manual_seed").strip(),
                notes=str(item.get("notes") or "").strip(),
            )
        )
    return seeds


def write_snowballed_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field for field in input_fields if field not in PAPER_CSV_HIDDEN_FIELDS]
    for field in [*STANDARD_FIELDS, *SNOWBALL_FIELDS]:
        if field not in PAPER_CSV_HIDDEN_FIELDS and field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def strip_per_paper_citation_diagnostics(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    fieldnames = [field for field in original_fields if field not in PAPER_CSV_HIDDEN_FIELDS]
    if fieldnames == original_fields:
        return False
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    os.replace(temporary_path, path)
    return True


def write_snowballing_summary(summary: SnowballingSummary, output_path: Path) -> None:
    payload = {
        "input_rows": summary.input_rows,
        "input_unique_rows": summary.input_unique_rows,
        "output_rows": summary.output_rows,
        "seeds_loaded": summary.seeds_loaded,
        "seeds_resolved": summary.seeds_resolved,
        "added_rows": summary.added_rows,
        "merged_rows": summary.merged_rows,
        "references_available": summary.references_available,
        "references_fetched": summary.references_fetched,
        "citations_available": summary.citations_available,
        "citations_fetched": summary.citations_fetched,
        "backward_truncated_seeds": summary.backward_truncated_seeds,
        "forward_truncated_seeds": summary.forward_truncated_seeds,
        "by_relation": dict(summary.by_relation),
        "by_source": dict(summary.by_source),
        "by_provider": dict(summary.by_provider),
        "provider_order": list(summary.provider_order),
        "provider_strategy": summary.provider_strategy,
        "provider_successes": dict(summary.provider_successes),
        "provider_failures": dict(summary.provider_failures),
        "provider_errors": summary.provider_errors,
        "seed_diagnostics": summary.seed_diagnostics,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_seed_coverage_report(summary: SnowballingSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed_title",
        "seed_id",
        "coverage_status",
        "missing_providers",
        "providers_resolved",
        "reference_providers",
        "citation_providers",
        "provider_errors",
        "references_fetched",
        "citations_fetched",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for diagnostic in summary.seed_diagnostics:
            writer.writerow(
                {
                    "seed_title": diagnostic.get("seed_title", ""),
                    "seed_id": diagnostic.get("seed_id", ""),
                    "coverage_status": diagnostic.get("coverage_status", ""),
                    "missing_providers": "; ".join(diagnostic.get("missing_providers", [])),
                    "providers_resolved": "; ".join(diagnostic.get("providers_resolved", [])),
                    "reference_providers": "; ".join(diagnostic.get("reference_providers", [])),
                    "citation_providers": "; ".join(diagnostic.get("citation_providers", [])),
                    "provider_errors": json.dumps(
                        diagnostic.get("provider_errors", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "references_fetched": diagnostic.get("references_fetched", 0),
                    "citations_fetched": diagnostic.get("citations_fetched", 0),
                }
            )


def _create_snowball_clients(
    config: SnowballingConfig,
    *,
    provider_order: list[str],
    year_start: int | None,
    year_end: int | None,
) -> tuple[list[SnowballClient], dict[str, list[str]]]:
    factories = {
        "semantic_scholar": SemanticScholarSnowballClient,
        "opencitations": OpenCitationsSnowballClient,
        "openalex": OpenAlexSnowballClient,
    }
    clients: list[SnowballClient] = []
    errors: dict[str, list[str]] = {}
    for provider in dict.fromkeys(provider_order):
        factory = factories.get(provider)
        if factory is None:
            errors.setdefault(provider, []).append("Unsupported citation provider.")
            continue
        try:
            clients.append(
                factory(
                    config,
                    year_start=year_start,
                    year_end=year_end,
                )
            )
        except TaskCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - unavailable fallbacks are reported.
            errors.setdefault(provider, []).append(str(exc))
    return clients, errors


def _retrieve_from_providers(
    resolved_works: list[tuple[SnowballClient, dict[str, Any]]],
    *,
    relation: str,
    limit: int | None,
    strategy: str,
    provider_successes: Counter[str],
    provider_failures: Counter[str],
    provider_errors: dict[str, list[str]],
) -> dict[str, Any]:
    works: list[tuple[str, dict[str, Any]]] = []
    providers: list[str] = []
    available_counts: list[int] = []
    truncated = False
    errors: dict[str, list[str]] = {}
    for client, seed_work in resolved_works:
        provider = client.provider_id
        try:
            retrieved = (
                client.referenced_works(seed_work, limit)
                if relation == "backward"
                else client.citing_works(seed_work, limit)
            )
        except TaskCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - provider failover is intentional.
            message = _record_provider_failure(
                provider,
                exc,
                provider_failures,
                provider_errors,
            )
            errors.setdefault(provider, []).append(message)
            continue
        provider_successes[provider] += 1
        providers.append(provider)
        available_counts.append(retrieved.available)
        truncated = truncated or retrieved.truncated
        works.extend((provider, work) for work in retrieved.works)
        if strategy == "failover":
            break
    return {
        "works": works,
        "providers": providers,
        "available": max(available_counts, default=0),
        "truncated": truncated,
        "errors": errors,
    }


def _record_provider_failure(
    provider: str,
    error: Exception,
    provider_failures: Counter[str],
    provider_errors: dict[str, list[str]],
) -> str:
    provider_failures[provider] += 1
    messages = provider_errors.setdefault(provider, [])
    message = str(error).strip() or type(error).__name__
    if message not in messages:
        messages.append(message[:1000])
    return message[:1000]


def _seed_coverage_diagnostic(
    *,
    seed: SeedPaper,
    seed_id: str,
    provider_order: tuple[str, ...],
    strategy: str,
    resolved_providers: list[str],
    references: dict[str, Any] | None,
    citations: dict[str, Any] | None,
    provider_errors: dict[str, list[str]],
) -> dict[str, Any]:
    reference_providers = list(dict.fromkeys((references or {}).get("providers", [])))
    citation_providers = list(dict.fromkeys((citations or {}).get("providers", [])))
    resolved = set(resolved_providers)
    complete_providers = set(reference_providers) & set(citation_providers)
    if strategy == "merge":
        missing_providers = [
            provider for provider in provider_order if provider not in complete_providers
        ]
    else:
        missing_providers = [
            provider
            for provider in provider_order
            if provider in provider_errors and provider not in complete_providers
        ]

    if not resolved:
        coverage_status = "failed"
        missing_providers = list(provider_order)
    elif not reference_providers and not citation_providers:
        coverage_status = "failed"
    elif not reference_providers or not citation_providers:
        coverage_status = "partial"
    elif missing_providers or provider_errors:
        coverage_status = "partial"
    else:
        coverage_status = "complete"

    normalized_errors = {
        provider: list(dict.fromkeys(str(message) for message in messages if message))
        for provider, messages in provider_errors.items()
        if messages
    }
    return {
        "seed_title": seed.title,
        "seed_id": seed_id,
        "resolved": bool(resolved),
        "coverage_status": coverage_status,
        "missing_providers": missing_providers,
        "providers_resolved": resolved_providers,
        "providers_used": list(dict.fromkeys([*reference_providers, *citation_providers])),
        "reference_providers": reference_providers,
        "citation_providers": citation_providers,
        "provider_errors": normalized_errors,
        "references_available": int((references or {}).get("available", 0)),
        "references_fetched": _unique_provider_work_count((references or {}).get("works", [])),
        "citations_available": int((citations or {}).get("available", 0)),
        "citations_fetched": _unique_provider_work_count((citations or {}).get("works", [])),
        "backward_truncated": bool((references or {}).get("truncated", False)),
        "forward_truncated": bool((citations or {}).get("truncated", False)),
    }


def _annotate_seed_coverage(
    merged_rows: list[dict[str, str]],
    row_index: dict[str, int],
    seed: SeedPaper,
    diagnostic: dict[str, Any],
) -> None:
    seed_title = seed.title.strip()
    seed_id = str(diagnostic.get("seed_id") or "").strip()
    target_indexes = {
        index
        for index, row in enumerate(merged_rows)
        if _semicolon_contains(row.get("snowball_seed_titles", ""), seed_title)
        or _semicolon_contains(row.get("snowball_seed_ids", ""), seed_id)
    }
    if not target_indexes:
        seed_keys = _dedupe_keys({"title": seed.title, "doi": seed.doi})
        target_indexes.update(row_index[key] for key in seed_keys if key in row_index)

    status = str(diagnostic.get("coverage_status") or "failed")
    missing = "; ".join(str(value) for value in diagnostic.get("missing_providers", []))
    notes = "; ".join(
        f"{provider}: {message}"
        for provider, messages in diagnostic.get("provider_errors", {}).items()
        for message in messages
    )
    for index in target_indexes:
        row = merged_rows[index]
        row["snowball_coverage_status"] = _worse_coverage_status(
            row.get("snowball_coverage_status", ""),
            status,
        )
        _merge_semicolon_field(row, "snowball_missing_providers", missing)
        _merge_semicolon_field(row, "snowball_coverage_notes", notes)


def _semicolon_contains(value: str, expected: str) -> bool:
    if not expected:
        return False
    normalized = expected.strip().lower()
    return any(item.strip().lower() == normalized for item in str(value or "").split(";"))


def _worse_coverage_status(current: str, candidate: str) -> str:
    severity = {"": -1, "complete": 0, "partial": 1, "failed": 2}
    current_value = str(current or "").strip().lower()
    candidate_value = str(candidate or "").strip().lower()
    return (
        candidate_value
        if severity.get(candidate_value, 2) > severity.get(current_value, -1)
        else current_value
    )


def _unique_provider_work_count(
    works: list[tuple[str, dict[str, Any]]],
) -> int:
    return len({_provider_work_key(work) for _provider, work in works})


class SemanticScholarSnowballClient:
    provider_id = "semantic_scholar"

    def __init__(
        self,
        config: SnowballingConfig,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SurveyFlow/0.1 Semantic Scholar snowballing"})
        api_key = os.environ.get(config.semantic_scholar_api_key_env, "").strip()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})
        self.cache_dir = config.cache_dir
        self.year_start = year_start
        self.year_end = year_end

    def resolve_seed(self, seed: SeedPaper) -> dict[str, Any] | None:
        identifiers = [
            seed.semantic_scholar_id,
            f"DOI:{seed.doi}" if seed.doi else "",
            f"ARXIV:{seed.arxiv_id}" if seed.arxiv_id else "",
        ]
        for identifier in identifiers:
            if identifier and (work := self.get_work(identifier)):
                return work
        return self.search_title(seed.title) if seed.title else None

    def get_work(self, identifier: str) -> dict[str, Any] | None:
        normalized = identifier.strip()
        if not normalized:
            return None
        payload = self._request(
            cache_key=f"work:{normalized}",
            url=(f"{SEMANTIC_SCHOLAR_GRAPH_URL}/paper/{quote(normalized, safe='')}"),
            params={"fields": _semantic_scholar_fields()},
        )
        if isinstance(payload, dict) and payload.get("_http_status") == 404:
            return None
        return payload if isinstance(payload, dict) else None

    def search_title(self, title: str) -> dict[str, Any] | None:
        payload = self._request(
            cache_key=f"title:{title}",
            url=f"{SEMANTIC_SCHOLAR_GRAPH_URL}/paper/search",
            params={
                "query": title,
                "limit": 5,
                "fields": _semantic_scholar_fields(),
            },
        )
        candidates = payload.get("data", []) if isinstance(payload, dict) else []
        best: dict[str, Any] | None = None
        best_score = 0.0
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            score = title_similarity(title, str(candidate.get("title") or ""))
            if score > best_score:
                best = candidate
                best_score = score
        return best if best_score >= 0.86 else None

    def referenced_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks:
        return self._related_works(work, "references", "citedPaper", limit)

    def citing_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks:
        return self._related_works(work, "citations", "citingPaper", limit)

    def _related_works(
        self,
        work: dict[str, Any],
        relation: str,
        payload_key: str,
        limit: int | None,
    ) -> RetrievedWorks:
        paper_id = str(work.get("paperId") or "").strip()
        if not paper_id:
            return RetrievedWorks([], 0, False)
        related: list[dict[str, Any]] = []
        offset = 0
        has_more = False
        while True:
            remaining = None if limit is None else max(limit - len(related), 0)
            if remaining == 0:
                has_more = True
                break
            page_size = min(remaining or 1000, 1000)
            params: dict[str, Any] = {
                "offset": offset,
                "limit": page_size,
                "fields": _semantic_scholar_fields(),
            }
            if year_filter := _semantic_scholar_year_filter(
                self.year_start,
                self.year_end,
            ):
                params["publicationDateOrYear"] = year_filter
            payload = self._request(
                cache_key=(
                    f"{relation}:{paper_id}:offset:{offset}:limit:{page_size}:"
                    f"years:{year_filter or 'all'}"
                ),
                url=(f"{SEMANTIC_SCHOLAR_GRAPH_URL}/paper/{quote(paper_id, safe='')}/{relation}"),
                params=params,
            )
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(data, list) or not data:
                break
            for item in data:
                candidate = item.get(payload_key) if isinstance(item, dict) else None
                if isinstance(candidate, dict) and candidate.get("title"):
                    related.append(candidate)
            next_offset = payload.get("next") if isinstance(payload, dict) else None
            has_more = next_offset not in (None, "")
            if not has_more:
                break
            offset = int(next_offset)
        return RetrievedWorks(
            works=related,
            available=len(related) + (1 if has_more else 0),
            truncated=limit is not None and has_more,
        )

    def _request(self, cache_key: str, url: str, params: dict[str, Any]) -> Any:
        return _cached_provider_request(
            provider=self.provider_id,
            session=self.session,
            cache_dir=self.cache_dir,
            cache_ttl_hours=self.config.cache_ttl_hours,
            cache_key=cache_key,
            url=url,
            params=params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=max(self.config.request_delay_seconds, 1.0),
        )


class OpenCitationsSnowballClient:
    provider_id = "opencitations"

    def __init__(
        self,
        config: SnowballingConfig,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SurveyFlow/0.1 OpenCitations snowballing"})
        token = os.environ.get(config.opencitations_access_token_env, "").strip()
        if token:
            self.session.headers.update({"authorization": token})
        self.cache_dir = config.cache_dir
        self.year_start = year_start
        self.year_end = year_end

    def resolve_seed(self, seed: SeedPaper) -> dict[str, Any] | None:
        if not seed.doi:
            return None
        return {
            "_provider_id": f"doi:{seed.doi}",
            "title": seed.title,
            "doi": seed.doi,
            "url": f"https://doi.org/{seed.doi}",
        }

    def referenced_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks:
        return self._related_works(work, relation="references", target_field="cited", limit=limit)

    def citing_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks:
        return self._related_works(work, relation="citations", target_field="citing", limit=limit)

    def _related_works(
        self,
        work: dict[str, Any],
        *,
        relation: str,
        target_field: str,
        limit: int | None,
    ) -> RetrievedWorks:
        doi = _normalize_doi(str(work.get("doi") or ""))
        if not doi:
            return RetrievedWorks([], 0, False)
        identifier = f"doi:{doi}"
        payload = self._request(
            cache_key=f"{relation}:{identifier}",
            url=(f"{OPENCITATIONS_INDEX_URL}/{relation}/{quote(identifier, safe=':._-/')}"),
            params={},
        )
        edges = payload if isinstance(payload, list) else []
        target_ids: list[str] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            if relation == "citations" and not _date_in_year_range(
                str(edge.get("creation") or ""),
                self.year_start,
                self.year_end,
            ):
                continue
            target = _preferred_opencitations_id(str(edge.get(target_field) or ""))
            if target and target not in target_ids:
                target_ids.append(target)
        works = self._metadata(target_ids)
        works = [
            item
            for item in works
            if _year_in_range(_work_year(item), self.year_start, self.year_end)
        ]
        available = len(works)
        selected = works if limit is None else works[:limit]
        return RetrievedWorks(
            works=selected,
            available=available,
            truncated=limit is not None and available > len(selected),
        )

    def _metadata(self, identifiers: list[str]) -> list[dict[str, Any]]:
        works: list[dict[str, Any]] = []
        for offset in range(0, len(identifiers), 20):
            chunk = identifiers[offset : offset + 20]
            joined = "__".join(chunk)
            payload = self._request(
                cache_key=f"metadata:{joined}",
                url=(f"{OPENCITATIONS_META_URL}/metadata/{quote(joined, safe=':._-/')}"),
                params={},
                metadata=True,
            )
            if isinstance(payload, list):
                works.extend(
                    _normalize_opencitations_work(item)
                    for item in payload
                    if isinstance(item, dict)
                )
        return works

    def _request(
        self,
        cache_key: str,
        url: str,
        params: dict[str, Any],
        *,
        metadata: bool = False,
    ) -> Any:
        return _cached_provider_request(
            provider=self.provider_id,
            session=self.session,
            cache_dir=self.cache_dir,
            cache_ttl_hours=self.config.cache_ttl_hours,
            cache_key=cache_key,
            url=url,
            params=params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=max(self.config.request_delay_seconds, 0.34 if metadata else 0.2),
        )


class OpenAlexSnowballClient:
    provider_id = "openalex"

    def __init__(
        self,
        config: SnowballingConfig,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "vnn-survey/0.1 snowballing"})
        self.cache_dir = config.cache_dir
        self.api_key = os.environ.get(config.openalex_api_key_env, "").strip()
        self.email = os.environ.get(config.openalex_email_env, "").strip()
        self.year_start = year_start
        self.year_end = year_end
        if not self.api_key:
            raise RuntimeError(
                "OpenAlex requires an API key for sustained use. Add a free key on AI settings."
            )

    def resolve_seed(self, seed: SeedPaper) -> dict[str, Any] | None:
        if seed.openalex_id:
            work = self.get_work(seed.openalex_id)
            if work:
                return work
        if seed.doi:
            work = self.get_work(f"doi:{seed.doi}")
            if work:
                return work
        if seed.title:
            return self.search_title(seed.title)
        return None

    def referenced_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks:
        raw_references = work.get("referenced_works") or []
        if not isinstance(raw_references, list):
            raw_references = []
        reference_ids = list(
            dict.fromkeys(
                work_id for item in raw_references if (work_id := _openalex_short_id(str(item)))
            )
        )
        reported_available = _nonnegative_int(work.get("referenced_works_count"))
        available = max(len(reference_ids), reported_available)
        selected_ids = reference_ids if limit is None else reference_ids[:limit]
        works_by_id: dict[str, dict[str, Any]] = {}
        for chunk_index in range(0, len(selected_ids), 100):
            chunk = selected_ids[chunk_index : chunk_index + 100]
            payload = self._request(
                cache_key=f"references:{'|'.join(chunk)}",
                url=OPENALEX_WORKS_URL,
                params={
                    "filter": self._with_year_filter(f"openalex:{'|'.join(chunk)}"),
                    "per_page": len(chunk),
                    "select": OPENALEX_SELECT_FIELDS,
                },
            )
            results = payload.get("results", [])
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                result_id = _openalex_short_id(str(result.get("id") or ""))
                if result_id:
                    works_by_id[result_id] = result
        works = [works_by_id[item] for item in selected_ids if item in works_by_id]
        return RetrievedWorks(
            works=works,
            available=available,
            truncated=limit is not None and available > limit,
        )

    def citing_works(
        self,
        work: dict[str, Any],
        limit: int | None,
    ) -> RetrievedWorks:
        work_id = _openalex_short_id(str(work.get("id") or ""))
        if not work_id:
            return RetrievedWorks(works=[], available=0, truncated=False)

        works: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        available = 0 if self._has_year_filter else _nonnegative_int(work.get("cited_by_count"))
        cursor = "*"
        seen_cursors: set[str] = set()
        while cursor and cursor not in seen_cursors:
            seen_cursors.add(cursor)
            remaining = None if limit is None else max(limit - len(works), 0)
            if remaining == 0:
                break
            page_size = min(remaining or 100, 100)
            payload = self._request(
                cache_key=(f"cites:{work_id}:cursor:{cursor}:page_size:{page_size}:newest"),
                url=OPENALEX_WORKS_URL,
                params={
                    "filter": self._with_year_filter(f"cites:{work_id}"),
                    "per_page": page_size,
                    "cursor": cursor,
                    "sort": "publication_date:desc",
                    "select": OPENALEX_SELECT_FIELDS,
                },
            )
            meta = payload.get("meta") or {}
            if isinstance(meta, dict):
                available = max(available, _nonnegative_int(meta.get("count")))
            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                break
            for result in results:
                if not isinstance(result, dict):
                    continue
                result_id = _openalex_short_id(str(result.get("id") or ""))
                dedupe_key = result_id or _normalize_doi(str(result.get("doi") or ""))
                if dedupe_key and dedupe_key in seen_ids:
                    continue
                if dedupe_key:
                    seen_ids.add(dedupe_key)
                works.append(result)
                if limit is not None and len(works) >= limit:
                    break
            next_cursor = meta.get("next_cursor") if isinstance(meta, dict) else None
            cursor = str(next_cursor or "")
        return RetrievedWorks(
            works=works,
            available=max(available, len(works)),
            truncated=limit is not None and available > len(works),
        )

    def get_work(self, work_id: str) -> dict[str, Any] | None:
        normalized = _normalize_openalex_work_id(work_id)
        if not normalized:
            return None
        payload = self._request(
            cache_key=f"work:{normalized}",
            url=f"{OPENALEX_WORKS_URL}/{quote(normalized, safe=':')}",
            params={"select": OPENALEX_SELECT_FIELDS},
        )
        if payload.get("_http_status") == 404 or payload.get("error") == "not_found":
            return None
        return payload

    def search_title(self, title: str) -> dict[str, Any] | None:
        payload = self._request(
            cache_key=f"title:{title}",
            url=OPENALEX_WORKS_URL,
            params={
                "filter": f'title.search:"{title}"',
                "per_page": 5,
                "select": OPENALEX_SELECT_FIELDS,
            },
        )
        candidates = payload.get("results", [])
        if not isinstance(candidates, list):
            return None
        best: dict[str, Any] | None = None
        best_score = 0.0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_title = str(candidate.get("title") or candidate.get("display_name") or "")
            score = title_similarity(title, candidate_title)
            if score > best_score:
                best = candidate
                best_score = score
        return best if best_score >= 0.86 else None

    def _request(self, cache_key: str, url: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = dict(params)
        if self.api_key:
            request_params["api_key"] = self.api_key
        if self.email:
            request_params["mailto"] = self.email

        cache_path = self._cache_path(cache_key, request_params)
        if cache_path and cache_path.exists() and self._cache_is_fresh(cache_path):
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = _request_json(
            session=self.session,
            url=url,
            params=request_params,
            timeout=self.config.timeout_seconds,
            retries=self.config.retries,
            delay=self.config.request_delay_seconds,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def _cache_is_fresh(self, path: Path) -> bool:
        ttl_seconds = self.config.cache_ttl_hours * 60 * 60
        return ttl_seconds > 0 and time() - path.stat().st_mtime <= ttl_seconds

    @property
    def _has_year_filter(self) -> bool:
        return self.year_start is not None or self.year_end is not None

    def _with_year_filter(self, base_filter: str) -> str:
        filters = [base_filter]
        if self.year_start is not None:
            filters.append(f"from_publication_date:{self.year_start}-01-01")
        if self.year_end is not None:
            filters.append(f"to_publication_date:{self.year_end}-12-31")
        return ",".join(filters)

    def _cache_path(self, cache_key: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps({"key": cache_key, "params": params}, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.json"


def _add_work(
    work: dict[str, Any],
    relation: str,
    provider: str,
    seed: SeedPaper,
    seed_id: str,
    depth: int,
    merged_rows: list[dict[str, str]],
    row_index: dict[str, int],
    config: SurveyConfig,
) -> int:
    year = _work_year(work)
    if not config.years.contains(year):
        return 0
    row = _work_to_row(work, provider)
    if not row.get("title"):
        return 0
    row.update(
        {
            "discovery_sources": f"snowball_{relation}",
            "snowball_relations": relation,
            "snowball_seed_titles": seed.title,
            "snowball_seed_ids": seed_id,
            "snowball_depths": str(depth),
            "snowball_provider": provider,
            "snowball_citation_count": str(
                work.get("cited_by_count") or work.get("citationCount") or ""
            ),
            "snowball_reference_count": str(
                work.get("referenced_works_count")
                or work.get("referenceCount")
                or len(work.get("referenced_works") or [])
                or ""
            ),
        }
    )
    return _index_or_merge(row, merged_rows=merged_rows, row_index=row_index)


def _index_or_merge(
    row: dict[str, str],
    merged_rows: list[dict[str, str]],
    row_index: dict[str, int],
) -> int:
    keys = _dedupe_keys(row)
    existing_index = next((row_index[key] for key in keys if key in row_index), None)
    if existing_index is None:
        row_index.update({key: len(merged_rows) for key in keys})
        merged_rows.append(row)
        return 0

    existing = merged_rows[existing_index]
    for field in SNOWBALL_FIELDS:
        _merge_semicolon_field(existing, field, row.get(field, ""))
    if _is_richer(row, existing):
        for field in STANDARD_FIELDS:
            if row.get(field) and not existing.get(field):
                existing[field] = row[field]
    for key in keys:
        row_index.setdefault(key, existing_index)
    return 1


def _work_to_row(work: dict[str, Any], provider: str) -> dict[str, str]:
    title = str(work.get("title") or work.get("display_name") or "").strip()
    venue = _work_venue(work)
    external_ids = work.get("externalIds") or {}
    doi = _normalize_doi(
        str(
            work.get("doi")
            or (external_ids.get("DOI") if isinstance(external_ids, dict) else "")
            or ""
        )
    )
    open_access = work.get("open_access") or {}
    primary_location = work.get("primary_location") or {}
    open_access_pdf = work.get("openAccessPdf") or {}
    url = (
        (open_access.get("oa_url") if isinstance(open_access, dict) else "")
        or (primary_location.get("landing_page_url") if isinstance(primary_location, dict) else "")
        or (primary_location.get("pdf_url") if isinstance(primary_location, dict) else "")
        or (open_access_pdf.get("url") if isinstance(open_access_pdf, dict) else "")
        or work.get("url")
        or work.get("id")
        or ""
    )
    return {
        "title": title,
        "authors": "; ".join(_work_authors(work)),
        "year": str(_work_year(work) or ""),
        "venue": venue,
        "doi": doi,
        "url": str(url),
        "dblp_key": "",
        "publication_type": _publication_type(work),
        "source": f"{provider}_snowball",
        "query": "snowball",
        "provider_id": _provider_work_id(work),
    }


def _openalex_id_from_row(row: dict[str, str]) -> str:
    for field in ["provider_id", "abstract_provider_id"]:
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        if "openalex.org/" in value or value.startswith("W"):
            return _openalex_short_id(value)
    return ""


def _semantic_scholar_id_from_row(row: dict[str, str]) -> str:
    for field in ["provider_id", "abstract_provider_id"]:
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        if re.fullmatch(r"[0-9a-fA-F]{40}", value):
            return value
        if "semanticscholar.org/paper/" in value:
            candidate = value.rstrip("/").rsplit("/", maxsplit=1)[-1]
            if re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
                return candidate
    return ""


def _arxiv_id_from_row(row: dict[str, str]) -> str:
    for field in ["url", "provider_id", "abstract_provider_id"]:
        value = str(row.get(field) or "").strip()
        if "arxiv.org/" not in value:
            continue
        candidate = value.split("arxiv.org/", maxsplit=1)[-1]
        candidate = candidate.removeprefix("abs/").removeprefix("pdf/")
        return _normalize_arxiv_id(candidate)
    return ""


def _seed_notes(row: dict[str, str]) -> str:
    parts = []
    for field, label in [
        ("year", "year"),
        ("venue", "venue"),
        ("research_track", "track"),
        ("llm_scope", "llm_scope"),
        ("final_recommendation", "final_recommendation"),
        ("manual_decision", "manual_decision"),
        ("manual_notes", "manual_notes"),
    ]:
        value = str(row.get(field) or "").strip()
        if value:
            parts.append(f"{label}={value}")
    return "; ".join(parts)


def _work_authors(work: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    raw_authors = work.get("authors") or []
    if isinstance(raw_authors, str):
        authors.extend(item.strip() for item in raw_authors.split(";") if item.strip())
    elif isinstance(raw_authors, list):
        for author in raw_authors:
            if isinstance(author, dict):
                name = str(author.get("name") or author.get("display_name") or "").strip()
            else:
                name = str(author or "").strip()
            if name:
                authors.append(name)
    for authorship in work.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name:
            authors.append(name)
    return authors


def _work_venue(work: dict[str, Any]) -> str:
    direct_venue = work.get("venue")
    if isinstance(direct_venue, str) and direct_venue.strip():
        return direct_venue.strip()
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if isinstance(source, dict) and source.get("display_name"):
        return str(source.get("display_name") or "")
    for location in work.get("locations") or []:
        if not isinstance(location, dict):
            continue
        source = location.get("source") or {}
        if isinstance(source, dict) and source.get("display_name"):
            return str(source.get("display_name") or "")
    return ""


def _publication_type(work: dict[str, Any]) -> str:
    publication_types = work.get("publicationTypes") or []
    if isinstance(publication_types, list):
        normalized_types = {str(item).lower() for item in publication_types}
        if "conference" in normalized_types:
            return "Inproceedings"
        if "journalarticle" in normalized_types:
            return "Article"
    crossref_type = str(work.get("type_crossref") or "").lower()
    work_type = str(work.get("type") or "").lower()
    if "proceedings" in crossref_type or work_type in {"proceedings-article", "book-chapter"}:
        return "Inproceedings"
    if "journal" in crossref_type or work_type in {"article", "journal-article"}:
        return "Article"
    if work_type == "preprint":
        return "Article"
    return str(work.get("type_crossref") or work.get("type") or "")


def _work_year(work: dict[str, Any]) -> int | None:
    direct = _to_int(work.get("publication_year") or work.get("year"))
    if direct is not None:
        return direct
    date_value = str(work.get("pub_date") or work.get("publicationDate") or "")
    match = re.search(r"\b(18|19|20|21)\d{2}\b", date_value)
    return int(match.group(0)) if match else None


def _provider_work_id(work: dict[str, Any]) -> str:
    return str(work.get("_provider_id") or work.get("paperId") or work.get("id") or "").strip()


def _provider_work_key(work: dict[str, Any]) -> str:
    external_ids = work.get("externalIds") or {}
    doi = _normalize_doi(
        str(
            work.get("doi")
            or (external_ids.get("DOI") if isinstance(external_ids, dict) else "")
            or ""
        )
    )
    if doi:
        return f"doi:{doi}"
    title = normalize_title(str(work.get("title") or work.get("display_name") or ""))
    if title:
        return f"title:{title}:{_work_year(work) or ''}"
    provider_id = _provider_work_id(work).lower()
    if provider_id:
        return f"provider:{provider_id}"
    return hashlib.sha256(json.dumps(work, sort_keys=True, default=str).encode()).hexdigest()


def _seed_fallback_id(seed: SeedPaper) -> str:
    return (
        seed.semantic_scholar_id
        or (f"doi:{seed.doi}" if seed.doi else "")
        or (f"arxiv:{seed.arxiv_id}" if seed.arxiv_id else "")
        or seed.openalex_id
    )


def _semantic_scholar_fields() -> str:
    return ",".join(
        [
            "externalIds",
            "title",
            "year",
            "authors",
            "venue",
            "publicationTypes",
            "url",
            "openAccessPdf",
            "citationCount",
            "referenceCount",
        ]
    )


def _semantic_scholar_year_filter(
    year_start: int | None,
    year_end: int | None,
) -> str:
    if year_start is not None and year_end is not None:
        return f"{year_start}:{year_end}"
    if year_start is not None:
        return f"{year_start}:"
    if year_end is not None:
        return f":{year_end}"
    return ""


def _preferred_opencitations_id(value: str) -> str:
    identifiers = value.strip().split()
    for prefix in ["doi:", "omid:", "pmid:"]:
        if identifier := next(
            (item for item in identifiers if item.lower().startswith(prefix)),
            "",
        ):
            return identifier
    return ""


def _normalize_opencitations_work(item: dict[str, Any]) -> dict[str, Any]:
    raw_ids = str(item.get("id") or "")
    preferred_id = _preferred_opencitations_id(raw_ids)
    doi = next(
        (
            _normalize_doi(identifier)
            for identifier in raw_ids.split()
            if identifier.lower().startswith("doi:")
        ),
        "",
    )
    venue = re.sub(r"\s*\[[^\]]+\]\s*$", "", str(item.get("venue") or "")).strip()
    authors = [
        re.sub(r"\s*\[[^\]]+\]\s*$", "", author).strip()
        for author in str(item.get("author") or "").split(";")
        if author.strip()
    ]
    return {
        "_provider_id": preferred_id or raw_ids,
        "title": str(item.get("title") or "").strip(),
        "authors": authors,
        "publication_year": _work_year(item),
        "pub_date": str(item.get("pub_date") or ""),
        "venue": venue,
        "doi": doi,
        "url": f"https://doi.org/{doi}" if doi else "",
        "type": str(item.get("type") or ""),
    }


def _date_in_year_range(
    value: str,
    year_start: int | None,
    year_end: int | None,
) -> bool:
    match = re.search(r"\b(18|19|20|21)\d{2}\b", value)
    return not match or _year_in_range(int(match.group(0)), year_start, year_end)


def _year_in_range(
    year: int | None,
    year_start: int | None,
    year_end: int | None,
) -> bool:
    if year is None:
        return True
    if year_start is not None and year < year_start:
        return False
    return year_end is None or year <= year_end


def _dedupe_keys(row: dict[str, str]) -> list[str]:
    keys = []
    doi = _normalize_doi(row.get("doi", ""))
    if doi:
        keys.append(f"doi:{doi.lower()}")
    title = normalize_title(row.get("title", ""))
    if title:
        keys.append(f"title:{title}")
    provider_id = row.get("provider_id", "").strip().lower()
    if provider_id:
        keys.append(f"provider:{provider_id}")
    dblp_key = row.get("dblp_key", "").strip().lower()
    if dblp_key:
        keys.append(f"dblp:{dblp_key}")
    return keys


def _is_richer(left: dict[str, str], right: dict[str, str]) -> bool:
    return _richness_score(left) > _richness_score(right)


def _richness_score(row: dict[str, str]) -> int:
    return sum(
        [
            3 if row.get("doi") else 0,
            3 if row.get("venue") else 0,
            2 if row.get("authors") else 0,
            2 if row.get("url") else 0,
            1 if row.get("publication_type") else 0,
            1 if row.get("year") else 0,
        ]
    )


def _merge_semicolon_field(row: dict[str, str], field: str, value: str) -> None:
    values = [item.strip() for item in str(row.get(field) or "").split(";") if item.strip()]
    for item in str(value or "").split(";"):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    row[field] = "; ".join(values)


def _cached_provider_request(
    *,
    provider: str,
    session: requests.Session,
    cache_dir: Path | None,
    cache_ttl_hours: float,
    cache_key: str,
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    delay: float,
) -> Any:
    cache_path = _provider_cache_path(cache_dir, provider, cache_key, url, params)
    if (
        cache_path
        and cache_path.exists()
        and _cache_is_fresh(
            cache_path,
            cache_ttl_hours,
        )
    ):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = _provider_request_json(
        provider=provider,
        session=session,
        url=url,
        params=params,
        timeout=timeout,
        retries=retries,
        delay=delay,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _provider_request_json(
    *,
    provider: str,
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    delay: float,
) -> Any:
    label = {
        "semantic_scholar": "Semantic Scholar",
        "opencitations": "OpenCitations",
        "openalex": "OpenAlex",
    }.get(provider, provider)
    last_error = "unknown error"
    for attempt in range(1, max(retries, 1) + 1):
        try:
            raise_if_cancelled()
            response = session.get(url, params=params, timeout=timeout)
            raise_if_cancelled()
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            if attempt < max(retries, 1):
                cancellable_sleep(_backoff_seconds(delay, attempt))
                continue
            break
        if response.status_code == 404:
            return {"_http_status": 404, "error": "not_found"}
        if 200 <= response.status_code < 300:
            try:
                payload = response.json()
            except ValueError:
                last_error = "invalid JSON response"
                if attempt < max(retries, 1):
                    cancellable_sleep(_backoff_seconds(delay, attempt))
                    continue
                break
            cancellable_sleep(delay)
            return payload
        message = _openalex_error_message(response)
        retryable = response.status_code == 429 or response.status_code >= 500
        last_error = f"{label} snowball request failed (HTTP {response.status_code})" + (
            f": {message}" if message else ""
        )
        if retryable and attempt < max(retries, 1):
            cancellable_sleep(
                _response_retry_seconds(
                    response,
                    delay,
                    attempt,
                    max_wait=10.0,
                )
            )
            continue
        raise RuntimeError(last_error) from None
    raise RuntimeError(
        f"{label} snowball request failed after {max(retries, 1)} attempts: {last_error}."
    ) from None


def _provider_cache_path(
    cache_dir: Path | None,
    provider: str,
    cache_key: str,
    url: str,
    params: dict[str, Any],
) -> Path | None:
    if cache_dir is None:
        return None
    key = hashlib.sha256(
        json.dumps(
            {
                "provider": provider,
                "key": cache_key,
                "url": url,
                "params": params,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return cache_dir / provider / f"{key}.json"


def _cache_is_fresh(path: Path, ttl_hours: float) -> bool:
    ttl_seconds = ttl_hours * 60 * 60
    return ttl_seconds > 0 and time() - path.stat().st_mtime <= ttl_seconds


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    delay: float,
) -> dict[str, Any]:
    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            raise_if_cancelled()
            response = session.get(url, params=params, timeout=timeout)
            raise_if_cancelled()
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            if attempt < retries:
                cancellable_sleep(_backoff_seconds(delay, attempt))
                continue
            break

        if response.status_code == 404:
            return {"_http_status": 404, "error": "not_found"}
        if 200 <= response.status_code < 300:
            try:
                payload = response.json()
            except ValueError:
                last_error = "invalid JSON response"
                if attempt < retries:
                    cancellable_sleep(_backoff_seconds(delay, attempt))
                    continue
                break
            cancellable_sleep(delay)
            return payload

        last_error = _openalex_http_error(response, attempt, retries)
        retryable = response.status_code in {403, 429} or response.status_code >= 500
        allowance_exhausted = _rate_limit_remaining(response) == 0
        if retryable and not allowance_exhausted and attempt < retries:
            cancellable_sleep(_response_retry_seconds(response, delay, attempt))
            continue
        raise RuntimeError(last_error) from None
    raise RuntimeError(
        f"OpenAlex snowball request failed after {retries} attempts: {last_error}."
    ) from None


def _openalex_http_error(
    response: requests.Response,
    attempt: int,
    retries: int,
) -> str:
    status = int(response.status_code)
    message = _openalex_error_message(response)
    rate_detail = _rate_limit_detail(response)
    hint = {
        400: "The OpenAlex query was rejected; update SurveyFlow or check the request filters.",
        401: "The OpenAlex API key was not accepted. Save a valid key in AI settings.",
        403: "OpenAlex denied the request. Check the API key and current usage allowance.",
        429: (
            "The OpenAlex allowance is exhausted or temporarily throttled. "
            "Check usage and retry after the reported reset."
        ),
    }.get(status)
    if status >= 500:
        hint = "OpenAlex reported a temporary server error; retry this snowball round."
    parts = [f"OpenAlex snowball request failed (HTTP {status})"]
    if message:
        parts.append(message)
    if rate_detail:
        parts.append(rate_detail)
    parts.append(f"attempt {attempt}/{retries}")
    if hint:
        parts.append(hint)
    return ". ".join(part.rstrip(".") for part in parts if part) + "."


def _openalex_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = str(payload.get("error") or "").strip()
        message = str(payload.get("message") or "").strip()
        if error and message and message.lower() != error.lower():
            return f"{error}: {message}"[:500]
        return (message or error)[:500]
    return str(getattr(response, "text", "") or "").strip()[:500]


def _rate_limit_remaining(response: requests.Response) -> float | None:
    value = response.headers.get("X-RateLimit-Remaining")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _rate_limit_detail(response: requests.Response) -> str:
    remaining = response.headers.get("X-RateLimit-Remaining", "").strip()
    limit = response.headers.get("X-RateLimit-Limit", "").strip()
    reset = response.headers.get("X-RateLimit-Reset", "").strip()
    details = []
    if remaining:
        details.append(f"remaining {remaining}" + (f" of {limit}" if limit else ""))
    if reset:
        details.append(f"reset in {reset} seconds")
    return "OpenAlex rate limit: " + ", ".join(details) if details else ""


def _response_retry_seconds(
    response: requests.Response,
    delay: float,
    attempt: int,
    *,
    max_wait: float = 300.0,
) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), max_wait)
        except ValueError:
            pass
    return _backoff_seconds(delay, attempt)


def _backoff_seconds(delay: float, attempt: int) -> float:
    return min(max(float(delay), 0.5) * (2 ** (attempt - 1)), 60.0)


def _normalize_openalex_work_id(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith("doi:"):
        return stripped
    if stripped.startswith("https://openalex.org/"):
        return stripped.removeprefix("https://openalex.org/")
    if stripped.startswith("http://openalex.org/"):
        return stripped.removeprefix("http://openalex.org/")
    return stripped


def _openalex_short_id(value: str) -> str:
    normalized = _normalize_openalex_work_id(value)
    return normalized.rsplit("/", maxsplit=1)[-1]


def _normalize_doi(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .removeprefix("https://doi.org/")
        .removeprefix("http://dx.doi.org/")
        .removeprefix("doi:")
    )


def _normalize_arxiv_id(value: str) -> str:
    normalized = str(value or "").strip().removesuffix(".pdf").split("?", maxsplit=1)[0]
    return re.sub(r"v\d+$", "", normalized, flags=re.IGNORECASE)


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: Any) -> int:
    parsed = _to_int(value)
    return max(parsed or 0, 0)


def _effective_limit(value: int) -> int | None:
    return value if value > 0 else None
