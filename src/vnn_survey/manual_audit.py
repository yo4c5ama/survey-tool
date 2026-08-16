from __future__ import annotations

import csv
from pathlib import Path

from vnn_survey.models import normalize_title

MANUAL_INCLUDE_DECISIONS = {"include", "include_related", "keep"}
MANUAL_EXCLUDE_DECISIONS = {"exclude", "remove", "reject"}
MANUAL_FIELDS = ["manual_decision", "manual_notes"]


def filter_manual_includes(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Write rows explicitly accepted during manual audit."""
    input_fields, rows = _read_csv(input_path)
    kept = [
        row
        for row in rows
        if _normalize_decision(row.get("manual_decision", "")) in MANUAL_INCLUDE_DECISIONS
    ]
    _write_csv(output_path, input_fields, kept)
    return len(rows), len(kept)


def merge_manual_audits(input_paths: list[Path], output_path: Path) -> tuple[int, int]:
    """Merge audit rounds while deduplicating papers across provider identifiers."""
    fieldnames: list[str] = []
    merged_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    total = 0

    for path in input_paths:
        input_fields, rows = _read_csv(path)
        for field in input_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        total += len(rows)
        for row in rows:
            keys = _paper_keys(row)
            if keys & seen:
                continue
            merged_rows.append(row)
            seen.update(keys)

    _write_csv(output_path, fieldnames, merged_rows)
    return total, len(merged_rows)


def prepare_audit_round(
    input_path: Path,
    output_path: Path,
    previous_audit_paths: list[Path],
) -> tuple[int, int]:
    """Create a review sheet containing only papers not seen in earlier audit rounds."""
    seen: set[str] = set()
    for path in previous_audit_paths:
        _, previous_rows = _read_csv(path)
        for row in previous_rows:
            seen.update(_paper_keys(row))

    input_fields, rows = _read_csv(input_path)
    output_rows: list[dict[str, str]] = []
    emitted: set[str] = set()
    for row in rows:
        keys = _paper_keys(row)
        if keys & seen or keys & emitted:
            continue
        prepared = dict(row)
        prepared["manual_decision"] = ""
        prepared["manual_notes"] = ""
        output_rows.append(prepared)
        emitted.update(keys)

    _write_csv(output_path, input_fields, output_rows)
    return len(rows), len(output_rows)


def is_manually_excluded(row: dict[str, str]) -> bool:
    return _normalize_decision(row.get("manual_decision", "")) in MANUAL_EXCLUDE_DECISIONS


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(output_path: Path, input_fields: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in MANUAL_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _paper_keys(row: dict[str, str]) -> set[str]:
    keys: set[str] = set()
    doi = (row.get("doi") or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    if doi:
        keys.add(f"doi:{doi}")
    dblp_key = (row.get("dblp_key") or "").strip().lower()
    if dblp_key:
        keys.add(f"dblp:{dblp_key}")
    provider_id = (row.get("provider_id") or "").strip().lower()
    if provider_id:
        keys.add(f"provider:{provider_id}")
    title = normalize_title(row.get("title", ""))
    year = (row.get("year") or "").strip()
    if title:
        keys.add(f"title:{title}")
        keys.add(f"title:{title}:{year}")
    return keys


def _normalize_decision(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
