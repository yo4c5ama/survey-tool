from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import yaml

from vnn_survey.app.task_manager import cancellable_sleep, raise_if_cancelled
from vnn_survey.config import SnowballingConfig, SurveyConfig
from vnn_survey.enrichment import title_similarity
from vnn_survey.manual_audit import is_manually_excluded
from vnn_survey.models import normalize_title

SnowballProgressCallback = Callable[[int, int, str, int], None]


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
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
        "type_crossref",
        "referenced_works",
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
]


@dataclass(frozen=True, slots=True)
class SeedPaper:
    title: str
    doi: str = ""
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
    by_relation: Counter[str]
    by_source: Counter[str]


@dataclass(frozen=True, slots=True)
class SnowballingResult:
    rows: list[dict[str, str]]
    summary: SnowballingSummary


@dataclass(frozen=True, slots=True)
class SeedExportResult:
    seeds: list[dict[str, str]]
    output_path: Path


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

    backward_limit = (
        max_backward_per_seed
        if max_backward_per_seed is not None
        else snowball_config.max_backward_per_seed
    )
    forward_limit = (
        max_forward_per_seed
        if max_forward_per_seed is not None
        else snowball_config.max_forward_per_seed
    )
    include_seeds = (
        include_seed_papers
        if include_seed_papers is not None
        else snowball_config.include_seed_papers
    )

    client = OpenAlexSnowballClient(snowball_config)
    merged_rows: list[dict[str, str]] = []
    row_index: dict[str, int] = {}
    by_relation: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    merged_count = 0

    for row in input_rows:
        prepared = dict(row)
        prepared.setdefault("discovery_sources", "dblp_search")
        _index_or_merge(prepared, merged_rows=merged_rows, row_index=row_index)
    input_unique_rows = len(merged_rows)

    seeds_resolved = 0
    if progress_callback:
        progress_callback(0, len(seeds), "", input_unique_rows)
    for seed_index, seed in enumerate(seeds, start=1):
        work = client.resolve_seed(seed)
        if not work:
            if progress_callback:
                progress_callback(seed_index, len(seeds), seed.title, len(merged_rows))
            continue
        seeds_resolved += 1
        seed_id = _openalex_short_id(work.get("id", ""))
        if include_seeds:
            merged_count += _add_work(
                work=work,
                relation="seed",
                seed=seed,
                seed_id=seed_id,
                depth=0,
                merged_rows=merged_rows,
                row_index=row_index,
                config=config,
            )
            by_relation["seed"] += 1
            by_source[seed.source or "manual_seed"] += 1

        for referenced_work in client.referenced_works(work, limit=backward_limit):
            merged_count += _add_work(
                work=referenced_work,
                relation="backward",
                seed=seed,
                seed_id=seed_id,
                depth=1,
                merged_rows=merged_rows,
                row_index=row_index,
                config=config,
            )
            by_relation["backward"] += 1
            by_source[seed.source or "manual_seed"] += 1

        for citing_work in client.citing_works(work, limit=forward_limit):
            merged_count += _add_work(
                work=citing_work,
                relation="forward",
                seed=seed,
                seed_id=seed_id,
                depth=1,
                merged_rows=merged_rows,
                row_index=row_index,
                config=config,
            )
            by_relation["forward"] += 1
            by_source[seed.source or "manual_seed"] += 1
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
        by_relation=by_relation,
        by_source=by_source,
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
        openalex_id = _openalex_id_from_row(row)
        key = doi or openalex_id.lower() or normalize_title(title)
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
        if not title and not item.get("doi") and not item.get("openalex_id"):
            continue
        seeds.append(
            SeedPaper(
                title=title,
                doi=_normalize_doi(str(item.get("doi") or "")),
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
    fieldnames = list(input_fields)
    for field in [*STANDARD_FIELDS, *SNOWBALL_FIELDS]:
        if field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_snowballing_summary(summary: SnowballingSummary, output_path: Path) -> None:
    payload = {
        "input_rows": summary.input_rows,
        "input_unique_rows": summary.input_unique_rows,
        "output_rows": summary.output_rows,
        "seeds_loaded": summary.seeds_loaded,
        "seeds_resolved": summary.seeds_resolved,
        "added_rows": summary.added_rows,
        "merged_rows": summary.merged_rows,
        "by_relation": dict(summary.by_relation),
        "by_source": dict(summary.by_source),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenAlexSnowballClient:
    def __init__(self, config: SnowballingConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "vnn-survey/0.1 snowballing"})
        self.cache_dir = config.cache_dir
        self.api_key = os.environ.get(config.openalex_api_key_env, "").strip()
        self.email = os.environ.get(config.openalex_email_env, "").strip()

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

    def referenced_works(self, work: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        references = work.get("referenced_works") or []
        if not isinstance(references, list):
            return []
        works: list[dict[str, Any]] = []
        for reference in references[:limit]:
            referenced_work = self.get_work(str(reference))
            if referenced_work:
                works.append(referenced_work)
        return works

    def citing_works(self, work: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        work_id = _openalex_short_id(str(work.get("id") or ""))
        if not work_id:
            return []
        payload = self._request(
            cache_key=f"cites:{work_id}:limit:{limit}",
            url=OPENALEX_WORKS_URL,
            params={
                "filter": f"cites:{work_id}",
                "per-page": min(limit, 200),
                "sort": "cited_by_count:desc",
                "select": OPENALEX_SELECT_FIELDS,
            },
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []
        return [result for result in results[:limit] if isinstance(result, dict)]

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
                "per-page": 5,
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
        if cache_path and cache_path.exists():
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
    seed: SeedPaper,
    seed_id: str,
    depth: int,
    merged_rows: list[dict[str, str]],
    row_index: dict[str, int],
    config: SurveyConfig,
) -> int:
    year = _to_int(work.get("publication_year"))
    if not config.years.contains(year):
        return 0
    row = _work_to_row(work)
    if not row.get("title"):
        return 0
    row.update(
        {
            "discovery_sources": f"snowball_{relation}",
            "snowball_relations": relation,
            "snowball_seed_titles": seed.title,
            "snowball_seed_ids": seed_id,
            "snowball_depths": str(depth),
            "snowball_provider": "openalex",
            "snowball_citation_count": str(work.get("cited_by_count") or ""),
            "snowball_reference_count": str(len(work.get("referenced_works") or [])),
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


def _work_to_row(work: dict[str, Any]) -> dict[str, str]:
    title = str(work.get("title") or work.get("display_name") or "").strip()
    venue = _work_venue(work)
    doi = _normalize_doi(str(work.get("doi") or ""))
    open_access = work.get("open_access") or {}
    primary_location = work.get("primary_location") or {}
    url = (
        open_access.get("oa_url")
        or primary_location.get("landing_page_url")
        or primary_location.get("pdf_url")
        or work.get("id")
        or ""
    )
    return {
        "title": title,
        "authors": "; ".join(_work_authors(work)),
        "year": str(work.get("publication_year") or ""),
        "venue": venue,
        "doi": doi,
        "url": str(url),
        "dblp_key": "",
        "publication_type": _publication_type(work),
        "source": "openalex_snowball",
        "query": "snowball",
        "provider_id": str(work.get("id") or ""),
    }


def _openalex_id_from_row(row: dict[str, str]) -> str:
    for field in ["provider_id", "abstract_provider_id"]:
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        if "openalex.org/" in value or value.startswith("W"):
            return _openalex_short_id(value)
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
    for authorship in work.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if name:
            authors.append(name)
    return authors


def _work_venue(work: dict[str, Any]) -> str:
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
    crossref_type = str(work.get("type_crossref") or "").lower()
    work_type = str(work.get("type") or "").lower()
    if "proceedings" in crossref_type or work_type in {"proceedings-article", "book-chapter"}:
        return "Inproceedings"
    if "journal" in crossref_type or work_type in {"article", "journal-article"}:
        return "Article"
    if work_type == "preprint":
        return "Article"
    return str(work.get("type_crossref") or work.get("type") or "")


def _dedupe_keys(row: dict[str, str]) -> list[str]:
    keys = []
    doi = _normalize_doi(row.get("doi", ""))
    if doi:
        keys.append(f"doi:{doi.lower()}")
    title = normalize_title(row.get("title", ""))
    if title:
        keys.append(f"title:{title}")
    provider_id = _openalex_short_id(row.get("provider_id", ""))
    if provider_id:
        keys.append(f"openalex:{provider_id.lower()}")
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


def _request_json(
    session: requests.Session,
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
            response = session.get(url, params=params, timeout=timeout)
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
    raise RuntimeError(f"OpenAlex snowball request failed: {url}") from last_error


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


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
