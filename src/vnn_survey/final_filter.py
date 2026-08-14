from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vnn_survey.models import normalize_title


STRICT_FIELDS = [
    "strict_filter_decision",
    "strict_filter_reason",
]

REVIEW_RECOMMENDATIONS = {
    "manual_include_review",
    "manual_review",
    "conflict_review",
}
LLM_SCOPE = "llm_target_verification"

FORMAL_LLM_PATTERNS = [
    r"\bformal verification\b",
    r"\bformal method(s)?\b",
    r"\bformal guarantee(s)?\b",
    r"\bformal proof(s)?\b",
    r"\bformal mathematical language\b",
    r"\bformal specification(s)?\b",
    r"\brobustness certificate\b",
    r"\bcertified robustness\b",
    r"\brobustness certification\b",
    r"\badversarial robustness certification\b",
    r"\brisk certification\b",
    r"\bdomain certification\b",
    r"\bformal domain certificate\b",
    r"\bcertification bounds?\b",
    r"\bfairness certification\b",
    r"\bperformance certification\b",
    r"\bfailure-rate certification\b",
    r"\breachability analysis\b",
    r"\bruntime verification\b",
    r"\bstatistical runtime verification\b",
    r"\babstract interpretation\b",
    r"\bPAC-Bayes\b",
    r"\bnon-interference guarantee(s)?\b",
    r"\bsafety guarantee(s)?\b",
    r"\bprovable (security|safety|robustness|guarantee|guarantees)\b",
    r"\bprovably\b",
    r"\bLean 4\b",
    r"\btheorem proving\b",
    r"\bproof assistant\b",
    r"\bconstraint satisfaction\b",
    r"\bcontrol barrier function\b",
    r"\bCBF\b",
]

NON_FORMAL_LLM_TOPICS = [
    r"\bself[- ]verification\b",
    r"\bhallucination(s)?\b",
    r"\bfactual(ity)?\b",
    r"\bwatermark(?:ing)?\b",
    r"\bownership verification\b",
    r"\bcopyright verification\b",
    r"\bmembership verification\b",
    r"\bfingerprint(?:ing)?\b",
    r"\bhuman verification\b",
    r"\buser trust\b",
    r"\bover-refusal\b",
    r"\berasure\b",
    r"\bsurvey\b",
    r"\boverview\b",
    r"\bbenchmark\b",
    r"\bevaluating\b",
]

EXPLICIT_SCOPE_PATTERNS = [
    r"\btransformer(s)?\b",
    r"\bvision transformer(s)?\b",
    r"\bViT\b",
    r"\bBERT\b",
    r"\bself[- ]attention\b",
    r"\bself[- ]attentive\b",
    r"\battention[- ]based\b",
    r"\battention network(s)?\b",
    r"\bLLM(s)?\b",
    r"\blarge language model(s)?\b",
    r"\blanguage model(s)?\b",
    r"\bNLP verification\b",
    r"\bNLP\b.*\bverification\b",
    r"\bverification\b.*\bNLP\b",
    r"\bprovable repair\b",
    r"\bprovable model editing\b",
    r"\bprovable editing\b",
    r"\bformal XAI\b",
    r"\bformal explainability\b",
    r"\bformal interpretability\b",
    r"\bmechanistic interpretability\b",
    r"\bcircuit discovery\b",
    r"\bprovable explanation(s)?\b",
    r"\bformally approximate minimal explanation(s)?\b",
    r"\bformally explaining\b",
]

FORMAL_TOPIC_OVERRIDES = [
    r"\bformal verification\b",
    r"\bretrospective, step-aware formal verification\b",
    r"\bLean 4\b",
    r"\bformal proof(s)?\b",
    r"\bdomain certification\b",
    r"\bformal domain certificate\b",
    r"\bruntime verification\b",
    r"\bstatistical runtime verification\b",
    r"\breachability analysis\b",
    r"\bcontrol barrier function\b",
    r"\bCBF\b",
    r"\brisk certification\b",
    r"\bperformance certification\b",
    r"\bfailure-rate certification\b",
    r"\bnon-interference guarantee(s)?\b",
]


@dataclass(frozen=True, slots=True)
class StrictFilterSummary:
    total: int
    kept: int
    removed: int
    by_decision: Counter[str]
    by_reason: Counter[str]
    kept_by_scope: Counter[str]
    kept_by_venue_type: Counter[str]
    kept_by_research_track: Counter[str]


@dataclass(frozen=True, slots=True)
class StrictFilterResult:
    kept_rows: list[dict[str, str]]
    all_rows: list[dict[str, str]]
    summary: StrictFilterSummary


def filter_final_candidates(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    removed_path: Path | None = None,
    include_recommendations: set[str] | None = None,
    exclude_arxiv: bool = True,
    strict_llm_formal_only: bool = True,
) -> StrictFilterResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    include_recommendations = include_recommendations or REVIEW_RECOMMENDATIONS
    seen: set[str] = set()
    annotated_rows: list[dict[str, str]] = []
    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []

    for row in rows:
        annotated = dict(row)
        decision, reason = _strict_decision(
            row=row,
            seen=seen,
            include_recommendations=include_recommendations,
            exclude_arxiv=exclude_arxiv,
            strict_llm_formal_only=strict_llm_formal_only,
        )
        annotated["strict_filter_decision"] = decision
        annotated["strict_filter_reason"] = reason
        annotated_rows.append(annotated)
        if decision == "keep":
            kept_rows.append(annotated)
        else:
            removed_rows.append(annotated)

    write_strict_filter_csv(kept_rows, input_fields, output_path)
    if removed_path:
        write_strict_filter_csv(removed_rows, input_fields, removed_path)
    summary = summarize_strict_filter(annotated_rows, kept_rows)
    write_strict_filter_summary(summary, summary_path)
    return StrictFilterResult(kept_rows=kept_rows, all_rows=annotated_rows, summary=summary)


def write_strict_filter_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in STRICT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_strict_filter_summary(summary: StrictFilterSummary, output_path: Path) -> None:
    payload = {
        "total": summary.total,
        "kept": summary.kept,
        "removed": summary.removed,
        "by_decision": dict(summary.by_decision),
        "by_reason": dict(summary.by_reason),
        "kept_by_scope": dict(summary.kept_by_scope),
        "kept_by_venue_type": dict(summary.kept_by_venue_type),
        "kept_by_research_track": dict(summary.kept_by_research_track),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_strict_filter(
    all_rows: list[dict[str, str]],
    kept_rows: list[dict[str, str]],
) -> StrictFilterSummary:
    return StrictFilterSummary(
        total=len(all_rows),
        kept=len(kept_rows),
        removed=len(all_rows) - len(kept_rows),
        by_decision=Counter(row.get("strict_filter_decision", "") for row in all_rows),
        by_reason=Counter(row.get("strict_filter_reason", "") for row in all_rows),
        kept_by_scope=Counter(row.get("llm_scope", "") for row in kept_rows),
        kept_by_venue_type=Counter(row.get("venue_type", "") for row in kept_rows),
        kept_by_research_track=Counter(
            row.get("research_track", "") or "unclassified" for row in kept_rows
        ),
    )


def _strict_decision(
    row: dict[str, str],
    seen: set[str],
    include_recommendations: set[str],
    exclude_arxiv: bool,
    strict_llm_formal_only: bool,
) -> tuple[str, str]:
    if row.get("final_recommendation") not in include_recommendations:
        return "remove", "not_in_review_queue"

    if exclude_arxiv and row.get("venue_type") == "arxiv":
        return "remove", "arxiv"

    dedupe_key = _dedupe_key(row)
    if dedupe_key in seen:
        return "remove", "duplicate"
    seen.add(dedupe_key)

    if not _has_explicit_scope_signal(row):
        return "remove", "generic_background"

    llm_scope = row.get("llm_scope", "")
    if llm_scope == LLM_SCOPE:
        if strict_llm_formal_only and _is_formal_llm_verification(row):
            return "keep", "llm_formal_verification"
        return "remove", "llm_not_formal_verification"

    if llm_scope in {"llm_as_tool", "unrelated"}:
        return "remove", llm_scope

    if llm_scope == "insufficient_information":
        return "remove", "insufficient_information"

    return "keep", llm_scope or "kept"


def _dedupe_key(row: dict[str, str]) -> str:
    doi = (row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    dblp_key = (row.get("dblp_key") or "").strip().lower()
    if dblp_key:
        return f"dblp:{dblp_key}"
    title = normalize_title(row.get("title", ""))
    year = (row.get("year") or "").strip()
    return f"title:{title}:{year}"


def _is_formal_llm_verification(row: dict[str, str]) -> bool:
    text = _combined_text(row)
    if row.get("llm_decision") == "exclude":
        return False
    if not _has_any(text, FORMAL_LLM_PATTERNS):
        return False
    if _has_any(text, NON_FORMAL_LLM_TOPICS) and not _has_any(text, FORMAL_TOPIC_OVERRIDES):
        return False
    return True


def _combined_text(row: dict[str, str]) -> str:
    return " ".join(
        [
            row.get("title", ""),
            row.get("abstract", ""),
            row.get("llm_evidence", ""),
        ]
    )


def _has_explicit_scope_signal(row: dict[str, str]) -> bool:
    text = row.get("title", "")
    return _has_any(text, EXPLICIT_SCOPE_PATTERNS)


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
