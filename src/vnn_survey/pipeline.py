from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from vnn_survey.config import SurveyConfig, expand_query_alternatives
from vnn_survey.dblp import DblpClient
from vnn_survey.dblp_sparql import DblpSparqlClient
from vnn_survey.export import write_csv, write_jsonl
from vnn_survey.models import PaperRecord
from vnn_survey.sources import create_provider

CollectionProgressCallback = Callable[[int, int, str, int], None]


@dataclass(frozen=True, slots=True)
class CollectionResult:
    raw_records: list[PaperRecord]
    filtered_records: list[PaperRecord]
    deduped_records: list[PaperRecord]
    failed_queries: dict[str, str]
    fallback_queries: dict[str, str]


def collect_from_dblp(
    config: SurveyConfig,
    console: Console,
    limit_queries: int | None = None,
    source: str = "auto",
    progress_callback: CollectionProgressCallback | None = None,
) -> CollectionResult:
    api_client = DblpClient(config.dblp)
    sparql_client = DblpSparqlClient(config.dblp)
    queries = config.build_queries()
    if limit_queries is not None:
        queries = queries[:limit_queries]
    raw_records: list[PaperRecord] = []
    failed_queries: dict[str, str] = {}
    fallback_queries: dict[str, str] = {}
    if progress_callback:
        progress_callback(0, len(queries), "", 0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching DBLP", total=len(queries))
        for query_index, query in enumerate(queries, start=1):
            progress.update(task, description=f"DBLP: {query}")
            try:
                raw_records.extend(
                    _search_query(
                        query=query,
                        source=source,
                        api_client=api_client,
                        sparql_client=sparql_client,
                        fallback_queries=fallback_queries,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep collecting other queries.
                failed_queries[query] = str(exc)
            progress.advance(task)
            if progress_callback:
                candidate_count = len(dedupe_records(apply_filters(raw_records, config)))
                progress_callback(query_index, len(queries), query, candidate_count)

    filtered = apply_filters(raw_records, config)
    deduped = dedupe_records(filtered)
    return CollectionResult(
        raw_records=raw_records,
        filtered_records=filtered,
        deduped_records=deduped,
        failed_queries=failed_queries,
        fallback_queries=fallback_queries,
    )


def collect_from_sources(
    config: SurveyConfig,
    console: Console,
    *,
    source_ids: list[str] | None = None,
    limit_queries: int | None = None,
    dblp_mode: str = "auto",
    additional_records: list[PaperRecord] | None = None,
    progress_callback: CollectionProgressCallback | None = None,
) -> CollectionResult:
    selected_sources = source_ids or config.discovery.sources or ["dblp"]
    selected_sources = list(dict.fromkeys(selected_sources))
    queries = config.build_queries()
    if limit_queries is not None:
        queries = queries[:limit_queries]
    request_plan = [
        (source_id, source_query)
        for source_id in selected_sources
        for query in queries
        for source_query in (
            [query] if source_id == "dblp" else expand_query_alternatives(query)
        )
    ]
    total_requests = len(request_plan)
    completed_requests = 0
    raw_records: list[PaperRecord] = []
    failed_queries: dict[str, str] = {}
    fallback_queries: dict[str, str] = {}
    if progress_callback:
        progress_callback(0, total_requests, "", 0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching literature sources", total=total_requests)
        providers = {
            source_id: create_provider(source_id, config, dblp_mode=dblp_mode)
            for source_id in selected_sources
        }
        for source_id, query in request_plan:
            current = f"{source_id}: {query}"
            progress.update(task, description=current)
            try:
                result = providers[source_id].search(query)
                raw_records.extend(result.records)
                if result.fallback_error:
                    fallback_queries[current] = result.fallback_error
            except Exception as exc:  # noqa: BLE001 - preserve partial multi-source results.
                failed_queries[current] = str(exc)
            completed_requests += 1
            progress.advance(task)
            if progress_callback:
                candidate_count = len(dedupe_records(apply_filters(raw_records, config)))
                progress_callback(
                    completed_requests,
                    total_requests,
                    current,
                    candidate_count,
                )

    raw_records.extend(additional_records or [])
    filtered = apply_filters(raw_records, config)
    deduped = dedupe_records(filtered)
    return CollectionResult(
        raw_records=raw_records,
        filtered_records=filtered,
        deduped_records=deduped,
        failed_queries=failed_queries,
        fallback_queries=fallback_queries,
    )


def _search_query(
    query: str,
    source: str,
    api_client: DblpClient,
    sparql_client: DblpSparqlClient,
    fallback_queries: dict[str, str],
) -> list[PaperRecord]:
    if source == "api":
        return api_client.search(query)
    if source == "sparql":
        return sparql_client.search(query)
    if source != "auto":
        raise ValueError(f"Unknown DBLP source mode: {source}")

    try:
        return api_client.search(query)
    except Exception as exc:  # noqa: BLE001 - fallback is intentional.
        fallback_queries[query] = str(exc)
        return sparql_client.search(query)


def apply_filters(records: list[PaperRecord], config: SurveyConfig) -> list[PaperRecord]:
    filtered = []
    for record in records:
        if not config.years.contains(record.year):
            continue
        if not config.filters.include_corr and _is_corr(record):
            continue
        if not config.filters.include_informal and _is_informal(record):
            continue
        if not record.title:
            continue
        filtered.append(record)
    return filtered


def dedupe_records(records: list[PaperRecord]) -> list[PaperRecord]:
    deduped: dict[int, PaperRecord] = {}
    index: dict[str, int] = {}
    for record in records:
        keys = _dedupe_keys(record)
        group_ids = {index[key] for key in keys if key in index}
        if not group_ids:
            group_id = len(deduped)
            while group_id in deduped:
                group_id += 1
            deduped[group_id] = record
            for key in keys:
                index[key] = group_id
            continue
        group_id = min(group_ids)
        merged = record
        for matched_id in sorted(group_ids):
            merged = _merge_records(deduped.pop(matched_id), merged)
        deduped[group_id] = merged
        for key, indexed_group in list(index.items()):
            if indexed_group in group_ids:
                index[key] = group_id
        for key in _dedupe_keys(merged):
            index[key] = group_id
    return sorted(deduped.values(), key=lambda item: ((item.year or 9999), item.title.lower()))


def _dedupe_keys(record: PaperRecord) -> list[str]:
    keys = []
    if record.dblp_key:
        keys.append(f"dblp:{record.dblp_key.lower().strip()}")
    if record.doi:
        keys.append(f"doi:{record.doi.lower().strip()}")
    normalized_title = _normalized_title(record.title)
    if normalized_title:
        keys.append(f"title:{normalized_title}")
    if record.provider_id:
        keys.append(f"{record.source}:{record.provider_id.lower().strip()}")
    return keys


def save_collection(result: CollectionResult, output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    write_jsonl(result.raw_records, raw_dir / "discovery_raw.jsonl", include_raw=True)
    write_jsonl(result.filtered_records, processed_dir / "discovery_filtered.jsonl")
    if all(record.source == "dblp" for record in result.raw_records):
        write_jsonl(result.raw_records, raw_dir / "dblp_raw.jsonl", include_raw=True)
        write_jsonl(result.filtered_records, processed_dir / "dblp_filtered.jsonl")
    write_jsonl(result.deduped_records, processed_dir / "candidate_papers.jsonl")
    write_csv(result.deduped_records, processed_dir / "candidate_papers.csv")

    _write_query_log(processed_dir / "failed_queries.txt", result.failed_queries)
    _write_query_log(processed_dir / "fallback_queries.txt", result.fallback_queries)


def _write_query_log(path: Path, entries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        path.write_text("", encoding="utf-8")
        return
    path.write_text(
        "\n".join(f"{query}\t{error}" for query, error in entries.items()) + "\n",
        encoding="utf-8",
    )


def summarize(result: CollectionResult) -> dict[str, object]:
    by_year = Counter(record.year for record in result.deduped_records if record.year is not None)
    by_venue = Counter(record.venue or "unknown" for record in result.deduped_records)
    by_source = Counter(
        source
        for record in result.deduped_records
        for source in (record.discovery_sources or [record.source])
    )
    return {
        "raw_records": len(result.raw_records),
        "filtered_records": len(result.filtered_records),
        "deduped_records": len(result.deduped_records),
        "failed_queries": len(result.failed_queries),
        "fallback_queries": len(result.fallback_queries),
        "top_years": by_year.most_common(10),
        "top_venues": by_venue.most_common(10),
        "by_source": dict(by_source),
    }


def _merge_records(left: PaperRecord, right: PaperRecord) -> PaperRecord:
    left_score = _richness_score(left)
    right_score = _richness_score(right)
    primary = right if right_score > left_score else left
    return replace(
        primary,
        abstract=primary.abstract or left.abstract or right.abstract,
        abstract_source=(
            primary.abstract_source
            if primary.abstract
            else left.abstract_source
            if left.abstract
            else right.abstract_source
        ),
        discovery_sources=_merge_values(
            left.discovery_sources,
            right.discovery_sources,
            [left.source, right.source],
        ),
        discovery_queries=_merge_values(
            left.discovery_queries,
            right.discovery_queries,
            [left.query, right.query],
        ),
        manual_added=left.manual_added or right.manual_added,
        manual_note=left.manual_note or right.manual_note,
    )


def _richness_score(record: PaperRecord) -> int:
    return sum(
        [
            6 if record.venue and record.venue.lower() != "corr" else 0,
            4 if not _is_informal(record) else 0,
            3 if record.source == "dblp" else 0,
            2 if record.doi else 0,
            2 if record.abstract else 0,
            1 if record.url else 0,
            1 if record.venue else 0,
            1 if record.year else 0,
            1 if record.authors else 0,
            1 if record.dblp_key else 0,
            1 if record.provider_id else 0,
        ]
    )


def _merge_values(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in (item for group in groups for item in group):
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def _normalized_title(title: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in title).split())


def _is_corr(record: PaperRecord) -> bool:
    venue = (record.venue or "").lower()
    key = (record.dblp_key or "").lower()
    return (
        record.source == "arxiv"
        or "arxiv" in record.discovery_sources
        or "corr" in venue
        or key.startswith("journals/corr")
    )


def _is_informal(record: PaperRecord) -> bool:
    publication_type = (record.publication_type or "").lower()
    return publication_type in {"informal", "withdrawn"}
