from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TRACK_FIELDS = [
    "research_track",
    "research_track_reason",
]


@dataclass(frozen=True, slots=True)
class TrackSummary:
    total: int
    by_track: Counter[str]


@dataclass(frozen=True, slots=True)
class TrackResult:
    rows: list[dict[str, str]]
    summary: TrackSummary


def classify_research_tracks(input_path: Path, output_path: Path) -> TrackResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    annotated_rows = [annotate_track(row) for row in rows]
    write_track_csv(annotated_rows, input_fields=input_fields, output_path=output_path)
    return TrackResult(rows=annotated_rows, summary=summarize_tracks(annotated_rows))


def annotate_track(row: dict[str, str]) -> dict[str, str]:
    text = _combined_text(row)
    llm_scope = row.get("llm_scope", "")

    if _has(text, r"\bNLP verification\b|\bembedding gap\b|\bsemantic similarity\b"):
        return _with_track(row, "nlp_verification_methodology", "NLP methodology signal")

    if _has(
        text,
        r"\bprovable repair\b|\bneural network repair\b|\bmodel repair\b|"
        r"\bnetwork repair\b|\bprovable editing\b|\bmodel editing\b|\bPRoViT\b",
    ):
        return _with_track(row, "provable_repair", "repair/editing signal")

    if _has(
        text,
        r"\bformal XAI\b|\bformal explainability\b|\bformal interpretability\b|"
        r"\bprovable explanation\b|\bprovably correct explanation\b|"
        r"\bminimal explanation\b|\bmechanistic interpretability\b|"
        r"\bcircuit discovery\b|\brobust patching\b",
    ):
        return _with_track(row, "formal_xai_interpretability", "formal XAI signal")

    if llm_scope == "llm_target_verification" or (
        not llm_scope
        and
        _has(text, r"\bLLM\b|\blarge language model(s)?\b|\blanguage model(s)?\b")
        and _has(
            text,
            r"\bformal verification\b|\bcertification\b|\bformal proof\b|"
            r"\bruntime verification\b|\breachability\b|\bprovable\b",
        )
    ):
        return _with_track(row, "llm_formal_verification", "LLM formal-verification signal")

    if llm_scope == "transformer_verification" or (
        _has(text, r"\btransformer(s)?\b|\bViT\b|\bBERT\b|\battention\b")
        and _has(
            text,
            r"\bverification\b|\bcertification\b|\bcertified robustness\b|"
            r"\breachability\b|\babstract interpretation\b|\bbound propagation\b|"
            r"\blinear relaxation\b|\bSMT\b|\bMILP\b",
        )
    ):
        return _with_track(row, "core_transformer_verification", "core verification signal")

    return _with_track(row, "other_or_unclear", "no configured track signal")


def write_track_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in TRACK_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_track_summary(summary: TrackSummary, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(
            {
                "total": summary.total,
                "by_track": dict(summary.by_track),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize_tracks(rows: list[dict[str, str]]) -> TrackSummary:
    return TrackSummary(
        total=len(rows),
        by_track=Counter(row.get("research_track", "") for row in rows),
    )


def _with_track(row: dict[str, str], track: str, reason: str) -> dict[str, str]:
    annotated = dict(row)
    annotated["research_track"] = track
    annotated["research_track_reason"] = reason
    return annotated


def _combined_text(row: dict[str, str]) -> str:
    fields = [
        "title",
        "abstract",
        "query",
        "llm_scope",
        "llm_reason",
        "llm_evidence",
        "snowball_seed_titles",
    ]
    return "\n".join(str(row.get(field) or "") for field in fields)


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))
