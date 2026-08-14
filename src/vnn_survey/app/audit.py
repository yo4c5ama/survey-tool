from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vnn_survey.manual_audit import filter_manual_includes, merge_manual_audits
from vnn_survey.models import normalize_title

AUDIT_DECISIONS = {"include", "include_related", "exclude", "later"}
REVIEW_RECOMMENDATIONS = {
    "manual_include_review",
    "manual_review",
    "conflict_review",
    "retry_llm",
    "needs_llm_screening",
}


@dataclass(frozen=True, slots=True)
class AuditSummary:
    total: int
    reviewed: int
    unreviewed: int
    by_decision: Counter[str]


def create_manual_recommendations(input_path: Path, output_path: Path) -> int:
    fieldnames, rows = read_csv(input_path)
    for row in rows:
        row["final_recommendation"] = "manual_review"
        row["final_priority"] = "3"
        row["final_reason"] = "No LLM screening was requested; human review is required."
    write_csv(
        output_path,
        rows,
        [*fieldnames, "final_recommendation", "final_priority", "final_reason"],
    )
    return len(rows)


def create_audit_queue(
    input_path: Path,
    output_path: Path,
    previous_audit_paths: list[Path] | None = None,
) -> tuple[int, int]:
    fieldnames, rows = read_csv(input_path)
    previous_audit_paths = previous_audit_paths or []
    seen: set[str] = set()
    for path in previous_audit_paths:
        _, previous_rows = read_csv(path)
        seen.update(paper_key(row) for row in previous_rows)

    queue: list[dict[str, str]] = []
    emitted: set[str] = set()
    for row in rows:
        recommendation = row.get("final_recommendation", "")
        if recommendation and recommendation not in REVIEW_RECOMMENDATIONS:
            continue
        if not recommendation and row.get("auto_screening_decision") == "exclude":
            continue
        key = paper_key(row)
        if key in seen or key in emitted:
            continue
        prepared = dict(row)
        prepared["manual_decision"] = ""
        prepared["manual_notes"] = ""
        queue.append(prepared)
        emitted.add(key)

    write_csv(output_path, queue, [*fieldnames, "manual_decision", "manual_notes"])
    return len(rows), len(queue)


def update_audit_rows(path: Path, updates: list[dict[str, str]]) -> AuditSummary:
    fieldnames, rows = read_csv(path)
    update_map = {paper_key(row): row for row in updates}
    for row in rows:
        update = update_map.get(paper_key(row))
        if not update:
            continue
        decision = (update.get("manual_decision") or "").strip().lower()
        if decision and decision not in AUDIT_DECISIONS:
            raise ValueError(f"Unsupported manual decision: {decision}")
        row["manual_decision"] = decision
        row["manual_notes"] = (update.get("manual_notes") or "").strip()
    write_csv(path, rows, [*fieldnames, "manual_decision", "manual_notes"])
    return summarize_audit(rows)


def load_audit(path: Path) -> tuple[list[str], list[dict[str, str]], AuditSummary]:
    fieldnames, rows = read_csv(path)
    return fieldnames, rows, summarize_audit(rows)


def summarize_audit(rows: list[dict[str, str]]) -> AuditSummary:
    decisions = Counter((row.get("manual_decision") or "").strip() for row in rows)
    reviewed = sum(decision not in {"", "later"} for decision in decisions.elements())
    return AuditSummary(
        total=len(rows),
        reviewed=reviewed,
        unreviewed=len(rows) - reviewed,
        by_decision=decisions,
    )


def build_cumulative_audit(
    audit_paths: list[Path],
    cumulative_path: Path,
    included_path: Path,
) -> tuple[int, int, int]:
    total, unique = merge_manual_audits(audit_paths, cumulative_path)
    _, included = filter_manual_includes(cumulative_path, included_path)
    return total, unique, included


def paper_key(row: dict[str, str]) -> str:
    for field in ["doi", "dblp_key", "provider_id"]:
        value = (row.get(field) or "").strip().lower()
        if value:
            return f"{field}:{value}"
    title = normalize_title(row.get("title", ""))
    year = (row.get("year") or "").strip()
    return f"title:{title}:{year}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fields = list(dict.fromkeys(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ordered_fields})
