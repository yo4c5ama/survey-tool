from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

import requests

from vnn_survey.config import VenueQualityConfig

ItemProgressCallback = Callable[[int, int, str], None]


VENUE_QUALITY_FIELDS = [
    "venue_type",
    "core_rank",
    "impact_factor",
]
LEGACY_VENUE_QUALITY_FIELDS = [
    "venue_key",
    "venue_display",
    "venue_quality_source",
    "core_rank_year",
    "core_source_url",
    "journal_impact_factor",
    "journal_metric_year",
    "journal_metric_source",
    "venue_quality_notes",
]
CORE_CONF_RANKS_URL = "https://portal.core.edu.au/conf-ranks/"


@dataclass(frozen=True, slots=True)
class VenueQualitySummary:
    total: int
    by_venue_type: Counter[str]
    by_core_rank: Counter[str]
    by_journal_impact_factor_band: Counter[str]
    conferences: int
    conferences_with_core_rank: int
    journals: int
    journals_with_impact_factor: int
    arxiv: int


@dataclass(frozen=True, slots=True)
class VenueQualityResult:
    rows: list[dict[str, str]]
    summary: VenueQualitySummary


class VenueLookup:
    def __init__(self, config: VenueQualityConfig) -> None:
        self.core_rows = _load_lookup_rows(config.core_rankings_path)
        self.journal_rows = _load_lookup_rows(config.journal_impact_factors_path)
        self.core_client = CorePortalClient(config) if config.core_online_enabled else None
        self.core_index = _build_index(
            self.core_rows,
            fields=["venue_key", "venue", "acronym", "full_name"],
        )
        self.journal_index = _build_index(
            self.journal_rows,
            fields=["venue_key", "venue", "journal_name", "issn"],
        )

    def find_core(self, row: dict[str, str]) -> dict[str, str] | None:
        local_match = _find_match(row=row, index=self.core_index)
        if local_match:
            return local_match
        if self.core_client:
            return self.core_client.find(row)
        return None

    def find_journal(self, row: dict[str, str]) -> dict[str, str] | None:
        return _find_match(row=row, index=self.journal_index)


def enrich_venue_quality(
    input_path: Path,
    output_path: Path,
    config: VenueQualityConfig,
    progress_callback: ItemProgressCallback | None = None,
) -> VenueQualityResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    lookup = VenueLookup(config)
    enriched_rows: list[dict[str, str]] = []
    if progress_callback:
        progress_callback(0, len(rows), "")
    for index, row in enumerate(rows, start=1):
        enriched_rows.append(_annotate_row(row, lookup=lookup))
        if progress_callback:
            progress_callback(index, len(rows), row.get("title", ""))
    write_venue_quality_csv(enriched_rows, input_fields, output_path)
    return VenueQualityResult(rows=enriched_rows, summary=summarize_venue_quality(enriched_rows))


def write_venue_quality_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in VENUE_QUALITY_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    fieldnames = [
        field
        for field in fieldnames
        if field in VENUE_QUALITY_FIELDS or field not in LEGACY_VENUE_QUALITY_FIELDS
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_venue_quality_summary(summary: VenueQualitySummary, output_path: Path) -> None:
    payload = {
        "total": summary.total,
        "by_venue_type": dict(summary.by_venue_type),
        "by_core_rank": dict(summary.by_core_rank),
        "by_journal_impact_factor_band": dict(summary.by_journal_impact_factor_band),
        "conferences": summary.conferences,
        "conferences_with_core_rank": summary.conferences_with_core_rank,
        "journals": summary.journals,
        "journals_with_impact_factor": summary.journals_with_impact_factor,
        "arxiv": summary.arxiv,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_venue_quality(rows: list[dict[str, str]]) -> VenueQualitySummary:
    conference_rows = [row for row in rows if row.get("venue_type") == "conference"]
    journal_rows = [row for row in rows if row.get("venue_type") == "journal"]
    return VenueQualitySummary(
        total=len(rows),
        by_venue_type=Counter(row.get("venue_type", "") or "unknown" for row in rows),
        by_core_rank=Counter(row.get("core_rank") or "missing" for row in conference_rows),
        by_journal_impact_factor_band=Counter(_impact_factor_band(row) for row in journal_rows),
        conferences=len(conference_rows),
        conferences_with_core_rank=sum(bool(row.get("core_rank")) for row in conference_rows),
        journals=len(journal_rows),
        journals_with_impact_factor=sum(
            bool(row.get("impact_factor")) for row in journal_rows
        ),
        arxiv=sum(row.get("venue_type") == "arxiv" for row in rows),
    )


def _annotate_row(row: dict[str, str], lookup: VenueLookup) -> dict[str, str]:
    annotated = dict(row)
    venue_type = _infer_venue_type(row)

    annotated.update(
        {
            "venue_type": venue_type,
            "core_rank": "",
            "impact_factor": "",
        }
    )

    if venue_type == "conference":
        match = lookup.find_core(row)
        if match:
            annotated["core_rank"] = match.get("core_rank", "") or match.get("rank", "")
    elif venue_type == "journal":
        match = lookup.find_journal(row)
        if match:
            annotated["impact_factor"] = (
                match.get("impact_factor", "") or match.get("journal_impact_factor", "")
            )

    return annotated


def _infer_venue_type(row: dict[str, str]) -> str:
    key = (row.get("dblp_key") or "").lower()
    venue = (row.get("venue") or "").lower()
    publication_type = (row.get("publication_type") or "").lower()

    if key.startswith("journals/corr") or venue in {"corr", "arxiv"} or "arxiv" in venue:
        return "arxiv"
    if key.startswith("conf/") or publication_type in {
        "inproceedings",
        "conference and workshop papers",
    }:
        return "conference"
    if key.startswith("journals/") or publication_type in {"article", "journal articles"}:
        return "journal"
    if key.startswith("books/") or "book" in publication_type:
        return "book"
    return "unknown"


def _venue_key(row: dict[str, str]) -> str:
    key = (row.get("dblp_key") or "").strip()
    parts = key.split("/")
    if len(parts) >= 2 and parts[0] in {"conf", "journals"}:
        return "/".join(parts[:2])
    return ""


def _load_lookup_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]


def _build_index(
    rows: list[dict[str, str]],
    fields: list[str],
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        for field in fields:
            for key in _lookup_keys(row.get(field, "")):
                index.setdefault(key, row)
    return index


def _find_match(
    row: dict[str, str],
    index: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    for key in _row_lookup_keys(row):
        if key in index:
            return index[key]
    return None


def _row_lookup_keys(row: dict[str, str]) -> list[str]:
    keys: list[str] = []
    venue_key = _venue_key(row)
    if venue_key:
        keys.extend(_lookup_keys(venue_key))
        keys.extend(_lookup_keys(venue_key.split("/", maxsplit=1)[-1]))
    keys.extend(_lookup_keys(row.get("venue", "")))
    return _dedupe(keys)


def _lookup_keys(value: str | None) -> list[str]:
    if not value:
        return []
    stripped = str(value).strip()
    keys = [stripped.lower(), _normalize_label(stripped)]
    if "/" in stripped:
        keys.append(stripped.split("/", maxsplit=1)[-1].lower())
    return [key for key in _dedupe(keys) if key]


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _impact_factor_band(row: dict[str, str]) -> str:
    impact_factor = _to_float(row.get("impact_factor"))
    if impact_factor is None:
        return "missing"
    if impact_factor >= 10:
        return ">=10"
    if impact_factor >= 5:
        return "5-9.99"
    if impact_factor >= 2:
        return "2-4.99"
    return "<2"


def _to_float(value: str | None) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


def _to_int(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return -1


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


class CorePortalClient:
    def __init__(self, config: VenueQualityConfig) -> None:
        self.cache_dir = config.core_cache_dir
        self.timeout = config.core_timeout_seconds
        self.request_delay_seconds = config.core_request_delay_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "vnn-survey/0.1 CORE ranking lookup"})
        self._last_request_at = 0.0
        self._memory_cache: dict[str, list[dict[str, str]]] = {}

    def find(self, row: dict[str, str]) -> dict[str, str] | None:
        venue_key = _venue_key(row)
        for query in _core_search_queries(row):
            matches = self.search(query)
            best = _choose_core_match(row=row, matches=matches, venue_key=venue_key)
            if best:
                return best
        return None

    def search(self, query: str) -> list[dict[str, str]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        cache_key = normalized_query.lower()
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        cache_path = self._cache_path(normalized_query)
        if cache_path and cache_path.exists():
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
            self._memory_cache[cache_key] = rows
            return rows

        self._sleep_if_needed()
        params = {
            "by": "all",
            "page": "1",
            "search": normalized_query,
            "sort": "atitle",
            "source": "all",
        }
        response = self.session.get(
            f"{CORE_CONF_RANKS_URL}?{urlencode(params)}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        self._last_request_at = time.monotonic()
        rows = _parse_core_rows(response.text)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        self._memory_cache[cache_key] = rows
        return rows

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay_seconds:
            time.sleep(self.request_delay_seconds - elapsed)

    def _cache_path(self, query: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(query.lower().encode()).hexdigest()
        return self.cache_dir / f"{key}.json"


class CoreRowsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, str]] = []
        self._in_data_row = False
        self._in_cell = False
        self._cells: list[str] = []
        self._cell_parts: list[str] = []
        self._dblp_href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        if tag == "tr" and "navigate('/conf-ranks/" in attrs_dict.get("onclick", ""):
            self._in_data_row = True
            self._cells = []
            self._dblp_href = ""
        elif self._in_data_row and tag == "td":
            self._in_cell = True
            self._cell_parts = []
        elif self._in_data_row and tag == "a":
            href = attrs_dict.get("href", "")
            if "/db/conf/" in href:
                self._dblp_href = href

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            text = " ".join(data.split())
            if text:
                self._cell_parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        if self._in_data_row and tag == "td":
            self._cells.append(" ".join(self._cell_parts).strip())
            self._in_cell = False
        elif self._in_data_row and tag == "tr":
            self._append_current_row()
            self._in_data_row = False

    def _append_current_row(self) -> None:
        if len(self._cells) < 4:
            return
        self.rows.append(
            {
                "venue": self._cells[0],
                "acronym": self._cells[1],
                "source": self._cells[2],
                "core_rank": self._cells[3],
                "dblp_key": _dblp_conf_key_from_href(self._dblp_href),
            }
        )


def _parse_core_rows(html: str) -> list[dict[str, str]]:
    parser = CoreRowsParser()
    parser.feed(html)
    return parser.rows


def _core_search_queries(row: dict[str, str]) -> list[str]:
    queries: list[str] = []
    venue_key = _venue_key(row)
    if venue_key:
        queries.append(venue_key.split("/", maxsplit=1)[-1])
    venue = row.get("venue", "")
    if venue:
        queries.append(venue)
    return _dedupe([query for query in queries if query])


def _choose_core_match(
    row: dict[str, str],
    matches: list[dict[str, str]],
    venue_key: str,
) -> dict[str, str] | None:
    if not matches:
        return None
    if venue_key:
        for match in matches:
            if match.get("dblp_key") == venue_key:
                return match

    expected = _normalize_label(row.get("venue", ""))
    expected_key = venue_key.split("/", maxsplit=1)[-1] if venue_key else ""
    for match in matches:
        if expected and _normalize_label(match.get("acronym", "")) == expected:
            return match
        if expected_key and match.get("acronym", "").lower() == expected_key.lower():
            return match
    return None


def _dblp_conf_key_from_href(value: str) -> str:
    match = re.search(r"/db/conf/([^/?#]+)", value)
    if not match:
        return ""
    return f"conf/{match.group(1).lower()}"
