from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


FINAL_FIELDS = [
    "final_recommendation",
    "final_priority",
    "final_reason",
]


@dataclass(frozen=True, slots=True)
class LlmReportSummary:
    total: int
    screened: int
    auto_eligible: int
    auto_excluded: int
    unscreened_eligible: int
    by_llm_status: Counter[str]
    by_llm_decision: Counter[str]
    by_llm_scope: Counter[str]
    by_final_recommendation: Counter[str]
    by_research_track: Counter[str]
    by_venue_type: Counter[str]
    by_core_rank: Counter[str]
    by_journal_impact_factor_band: Counter[str]


@dataclass(frozen=True, slots=True)
class LlmReportResult:
    rows: list[dict[str, str]]
    summary: LlmReportSummary
    report_path: Path
    recommendations_path: Path
    summary_path: Path


def summarize_llm_screening(
    input_path: Path,
    report_path: Path,
    recommendations_path: Path,
    summary_path: Path,
    high_confidence: float = 0.8,
    max_examples: int = 40,
) -> LlmReportResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    annotated_rows = [
        _annotate_final_recommendation(row, high_confidence=high_confidence) for row in rows
    ]
    annotated_rows.sort(key=_recommendation_sort_key)
    summary = _summarize(annotated_rows)

    write_recommendations_csv(annotated_rows, input_fields, recommendations_path)
    write_report(
        rows=annotated_rows,
        summary=summary,
        input_path=input_path,
        report_path=report_path,
        recommendations_path=recommendations_path,
        high_confidence=high_confidence,
        max_examples=max_examples,
    )
    write_report_summary(summary, summary_path)

    return LlmReportResult(
        rows=annotated_rows,
        summary=summary,
        report_path=report_path,
        recommendations_path=recommendations_path,
        summary_path=summary_path,
    )


def write_recommendations_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in FINAL_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(
    rows: list[dict[str, str]],
    summary: LlmReportSummary,
    input_path: Path,
    report_path: Path,
    recommendations_path: Path,
    high_confidence: float,
    max_examples: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# LLM Screening Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Recommendation CSV: `{recommendations_path}`")
    lines.append(f"- High-confidence threshold: `{high_confidence:.2f}`")
    lines.append("")
    lines.extend(_summary_section(summary))
    lines.extend(_counter_section("LLM Status", summary.by_llm_status))
    lines.extend(_counter_section("LLM Decisions", summary.by_llm_decision))
    lines.extend(_counter_section("LLM Scopes", summary.by_llm_scope))
    lines.extend(_counter_section("Final Recommendations", summary.by_final_recommendation))
    lines.extend(_counter_section("Research Tracks", summary.by_research_track))
    lines.extend(_counter_section("Venue Types", summary.by_venue_type))
    lines.extend(_counter_section("Conference CORE Rankings", summary.by_core_rank))
    lines.extend(
        _counter_section(
            "Journal Impact Factor Bands",
            summary.by_journal_impact_factor_band,
        )
    )
    lines.extend(_crosstab_section(rows))
    lines.extend(_venue_crosstab_section(rows))
    lines.extend(_queue_section(rows, max_examples=max_examples))
    lines.extend(
        _examples_section(
            title="LLM Include Candidates",
            rows=[row for row in rows if row.get("llm_decision") == "include"],
            max_examples=max_examples,
        )
    )
    lines.extend(
        _examples_section(
            title="LLM Maybes",
            rows=[row for row in rows if row.get("llm_decision") == "maybe"],
            max_examples=max_examples,
        )
    )
    lines.extend(
        _examples_section(
            title="LLM-As-Tool Exclusions",
            rows=[
                row
                for row in rows
                if row.get("final_recommendation") == "likely_exclude_llm_as_tool"
            ],
            max_examples=max_examples,
        )
    )
    lines.extend(
        _examples_section(
            title="High-Confidence LLM Exclusions",
            rows=[
                row
                for row in rows
                if row.get("llm_decision") == "exclude"
                and _float(row.get("llm_confidence")) >= high_confidence
            ],
            max_examples=max_examples,
        )
    )
    lines.extend(
        _examples_section(
            title="Auto Include But LLM Exclude",
            rows=[
                row
                for row in rows
                if row.get("auto_screening_decision") == "include_candidate"
                and row.get("llm_decision") == "exclude"
            ],
            max_examples=max_examples,
        )
    )
    lines.extend(
        _examples_section(
            title="Eligible Rows Not Yet Screened By LLM",
            rows=[
                row
                for row in rows
                if _is_auto_eligible(row) and not row.get("llm_decision")
            ],
            max_examples=max_examples,
        )
    )
    lines.append("## Suggested Next Step")
    lines.append("")
    lines.append(
        "Review `final_screening_recommendations.csv` from top to bottom. "
        "Start with `manual_include_review`, then `manual_review`, then conflicts. "
        "Treat high-confidence LLM excludes as low priority, but sample-check them "
        "before reporting final exclusion counts."
    )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_summary(summary: LlmReportSummary, output_path: Path) -> None:
    payload = {
        "total": summary.total,
        "screened": summary.screened,
        "auto_eligible": summary.auto_eligible,
        "auto_excluded": summary.auto_excluded,
        "unscreened_eligible": summary.unscreened_eligible,
        "by_llm_status": dict(summary.by_llm_status),
        "by_llm_decision": dict(summary.by_llm_decision),
        "by_llm_scope": dict(summary.by_llm_scope),
        "by_final_recommendation": dict(summary.by_final_recommendation),
        "by_research_track": dict(summary.by_research_track),
        "by_venue_type": dict(summary.by_venue_type),
        "by_core_rank": dict(summary.by_core_rank),
        "by_journal_impact_factor_band": dict(summary.by_journal_impact_factor_band),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _annotate_final_recommendation(
    row: dict[str, str],
    high_confidence: float,
) -> dict[str, str]:
    annotated = dict(row)
    auto_decision = row.get("auto_screening_decision", "")
    llm_decision = row.get("llm_decision", "")
    llm_scope = row.get("llm_scope", "")
    llm_status = row.get("llm_status", "")
    confidence = _float(row.get("llm_confidence"))

    if llm_status == "failed":
        recommendation = "retry_llm"
        priority = "2"
        reason = row.get("llm_error", "") or "LLM request failed."
    elif not llm_decision:
        if auto_decision == "exclude":
            recommendation = "auto_exclude"
            priority = "8"
            reason = "High-confidence automatic exclusion; no LLM review requested."
        else:
            recommendation = "needs_llm_screening"
            priority = "3"
            reason = "Auto-eligible row has not yet received an LLM decision."
    elif llm_decision == "include":
        recommendation = "manual_include_review"
        priority = "1" if confidence >= high_confidence else "2"
        reason = "LLM judged the paper in scope; manually validate before final inclusion."
    elif llm_decision == "maybe":
        recommendation = "manual_review"
        priority = "2"
        reason = "LLM found the paper ambiguous or under-specified."
    elif llm_decision == "exclude":
        if auto_decision == "include_candidate":
            recommendation = "conflict_review"
            priority = "2"
            reason = "Automatic screening included this paper but LLM excluded it."
        elif llm_scope == "llm_as_tool":
            recommendation = "likely_exclude_llm_as_tool"
            priority = "6"
            reason = "LLM judged the model as a tool rather than the verification target."
        elif confidence >= high_confidence:
            recommendation = "likely_exclude"
            priority = "7"
            reason = "High-confidence LLM exclusion."
        else:
            recommendation = "manual_review"
            priority = "4"
            reason = "Low-confidence LLM exclusion."
    else:
        recommendation = "manual_review"
        priority = "4"
        reason = "Unrecognized LLM decision."

    annotated["final_recommendation"] = recommendation
    annotated["final_priority"] = priority
    annotated["final_reason"] = reason
    return annotated


def _summarize(rows: list[dict[str, str]]) -> LlmReportSummary:
    return LlmReportSummary(
        total=len(rows),
        screened=sum(bool(row.get("llm_decision")) for row in rows),
        auto_eligible=sum(_is_auto_eligible(row) for row in rows),
        auto_excluded=sum(row.get("auto_screening_decision") == "exclude" for row in rows),
        unscreened_eligible=sum(
            _is_auto_eligible(row) and not row.get("llm_decision") for row in rows
        ),
        by_llm_status=Counter(row.get("llm_status", "") for row in rows),
        by_llm_decision=Counter(
            row.get("llm_decision", "") for row in rows if row.get("llm_decision")
        ),
        by_llm_scope=Counter(row.get("llm_scope", "") for row in rows if row.get("llm_scope")),
        by_final_recommendation=Counter(row.get("final_recommendation", "") for row in rows),
        by_research_track=Counter(row.get("research_track", "") or "unclassified" for row in rows),
        by_venue_type=Counter(row.get("venue_type", "") or "unknown" for row in rows),
        by_core_rank=Counter(
            row.get("core_rank") or "missing"
            for row in rows
            if row.get("venue_type") == "conference"
        ),
        by_journal_impact_factor_band=Counter(
            _impact_factor_band(row)
            for row in rows
            if row.get("venue_type") == "journal"
        ),
    )


def _summary_section(summary: LlmReportSummary) -> list[str]:
    screened_rate = _percent(summary.screened, summary.auto_eligible)
    return [
        "## Overview",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Total rows | {summary.total} |",
        f"| Auto-eligible rows | {summary.auto_eligible} |",
        f"| Auto-excluded rows | {summary.auto_excluded} |",
        f"| Rows with LLM decision | {summary.screened} |",
        f"| LLM coverage of auto-eligible rows | {screened_rate} |",
        f"| Auto-eligible rows not yet LLM-screened | {summary.unscreened_eligible} |",
        "",
    ]


def _counter_section(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    if not counter:
        lines.append("| none | 0 |")
    for value, count in counter.most_common():
        lines.append(f"| {_md(value or 'empty')} | {count} |")
    lines.append("")
    return lines


def _crosstab_section(rows: list[dict[str, str]]) -> list[str]:
    decisions = sorted({row.get("llm_decision", "") for row in rows if row.get("llm_decision")})
    auto_values = sorted({row.get("auto_screening_decision", "") for row in rows})
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        auto = row.get("auto_screening_decision", "") or "empty"
        decision = row.get("llm_decision", "") or "no_llm_decision"
        matrix[auto][decision] += 1
    columns = [*decisions, "no_llm_decision"]
    lines = ["## Auto Screening vs LLM Decision", ""]
    lines.append("| Auto decision | " + " | ".join(columns) + " |")
    lines.append("|---" + "|---:" * len(columns) + "|")
    for auto in auto_values:
        cells = [str(matrix[auto or "empty"][column]) for column in columns]
        lines.append(f"| {_md(auto or 'empty')} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _venue_crosstab_section(rows: list[dict[str, str]]) -> list[str]:
    recommendations = sorted(
        {row.get("final_recommendation", "") for row in rows if row.get("final_recommendation")}
    )
    venue_types = sorted({row.get("venue_type", "") or "unknown" for row in rows})
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        venue_type = row.get("venue_type", "") or "unknown"
        recommendation = row.get("final_recommendation", "") or "none"
        matrix[venue_type][recommendation] += 1
    lines = ["## Venue Type vs Final Recommendation", ""]
    if not recommendations:
        lines.append("No final recommendations.")
        lines.append("")
        return lines
    lines.append("| Venue type | " + " | ".join(recommendations) + " |")
    lines.append("|---" + "|---:" * len(recommendations) + "|")
    for venue_type in venue_types:
        cells = [str(matrix[venue_type][recommendation]) for recommendation in recommendations]
        lines.append(f"| {_md(venue_type)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _queue_section(rows: list[dict[str, str]], max_examples: int) -> list[str]:
    review_rows = [
        row
        for row in rows
        if row.get("final_recommendation")
        in {"manual_include_review", "manual_review", "conflict_review", "retry_llm"}
    ]
    lines = ["## Priority Review Queue", ""]
    if not review_rows:
        lines.append("No priority review rows.")
        lines.append("")
        return lines
    lines.append(
        "| Priority | Recommendation | Year | Venue | Quality | LLM decision | Scope | Confidence | Title | Reason |"
    )
    lines.append("|---:|---|---:|---|---|---|---|---:|---|---|")
    for row in review_rows[:max_examples]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("final_priority", "")),
                    _md(row.get("final_recommendation", "")),
                    _md(row.get("year", "")),
                    _md(_venue_label(row)),
                    _md(_quality_label(row)),
                    _md(row.get("llm_decision", "")),
                    _md(row.get("llm_scope", "")),
                    _md(row.get("llm_confidence", "")),
                    _md(row.get("title", "")),
                    _md(_truncate(row.get("final_reason", ""), 120)),
                ]
            )
            + " |"
        )
    if len(review_rows) > max_examples:
        lines.append("")
        lines.append(f"Showing {max_examples} of {len(review_rows)} rows.")
    lines.append("")
    return lines


def _examples_section(title: str, rows: list[dict[str, str]], max_examples: int) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("No rows.")
        lines.append("")
        return lines
    lines.append("| Year | Venue | Quality | Decision | Scope | Confidence | Title | Evidence |")
    lines.append("|---:|---|---|---|---|---:|---|---|")
    for row in rows[:max_examples]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("year", "")),
                    _md(_venue_label(row)),
                    _md(_quality_label(row)),
                    _md(row.get("llm_decision", "")),
                    _md(row.get("llm_scope", "")),
                    _md(row.get("llm_confidence", "")),
                    _md(row.get("title", "")),
                    _md(_truncate(row.get("llm_evidence", ""), 180)),
                ]
            )
            + " |"
        )
    if len(rows) > max_examples:
        lines.append("")
        lines.append(f"Showing {max_examples} of {len(rows)} rows.")
    lines.append("")
    return lines


def _recommendation_sort_key(row: dict[str, str]) -> tuple[int, int, float, str]:
    priority = int(row.get("final_priority") or 9)
    year = int(row.get("year") or 9999)
    confidence = _float(row.get("llm_confidence"))
    return (priority, year, -confidence, row.get("title", "").lower())


def _is_auto_eligible(row: dict[str, str]) -> bool:
    return row.get("auto_screening_decision") in {"include_candidate", "needs_review"}


def _float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def _venue_label(row: dict[str, str]) -> str:
    venue_type = row.get("venue_type", "") or "unknown"
    venue = row.get("venue_display") or row.get("venue") or ""
    if venue:
        return f"{venue_type}: {venue}"
    return venue_type


def _quality_label(row: dict[str, str]) -> str:
    if row.get("core_rank"):
        year = row.get("core_rank_year")
        return f"CORE {row.get('core_rank')}" + (f" ({year})" if year else "")
    if row.get("impact_factor"):
        return f"IF {row.get('impact_factor')}"
    return ""


def _impact_factor_band(row: dict[str, str]) -> str:
    impact_factor = _optional_float(row.get("impact_factor"))
    if impact_factor is None:
        return "missing"
    if impact_factor >= 10:
        return ">=10"
    if impact_factor >= 5:
        return "5-9.99"
    if impact_factor >= 2:
        return "2-4.99"
    return "<2"


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value or "")
    except ValueError:
        return None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
