from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vnn_survey.config import ScreeningConfig


SCREENING_FIELDS = [
    "auto_screening_decision",
    "auto_screening_bucket",
    "auto_screening_reason",
    "exclusion_code",
    "manual_decision",
    "manual_notes",
]


@dataclass(frozen=True, slots=True)
class ScreeningSummary:
    total: int
    by_decision: Counter[str]
    by_bucket: Counter[str]
    by_exclusion_code: Counter[str]


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    rows: list[dict[str, str]]
    summary: ScreeningSummary


def screen_candidates(
    input_path: Path,
    output_path: Path,
    config: ScreeningConfig,
) -> ScreeningResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    screened_rows = [annotate_row(row, config) for row in rows]
    write_screened_csv(screened_rows, input_fields, output_path)
    return ScreeningResult(rows=screened_rows, summary=summarize_screening(screened_rows))


def annotate_row(row: dict[str, str], config: ScreeningConfig) -> dict[str, str]:
    title = row.get("title", "")
    haystack = normalize_text(title)

    if config.profile == "generic":
        return _annotate_generic(row=row, haystack=haystack, config=config)

    decision = "needs_review"
    bucket = "needs_review"
    reason: list[str] = []
    exclusion_code = ""

    noise = _matched_noise(haystack)
    llm_tool = _is_llm_as_verification_tool(haystack)
    core = _is_core_transformer_verification(haystack)
    llm_target = _is_llm_target_verification(haystack)

    if noise:
        decision = "exclude"
        bucket = "out_of_scope_noise"
        exclusion_code = noise
        reason.append(f"matched high-confidence noise pattern: {noise}")
    elif config.exclude_llm_as_verification_tool and llm_tool:
        decision = "exclude"
        bucket = "llm_as_verification_tool"
        exclusion_code = "llm_as_verification_tool"
        reason.append("LLM appears to be used as a tool for verifying another artifact")
    elif core:
        decision = "include_candidate"
        bucket = "transformer_verification"
        reason.append("title/query suggests Transformer/attention model verification or certification")
    elif llm_target:
        decision = "include_candidate"
        bucket = "llm_target_verification"
        reason.append("title suggests the LLM itself is the verification/certification target")
    else:
        reason.append("no high-confidence automatic rule matched")

    annotated = dict(row)
    annotated.update(
        {
            "auto_screening_decision": decision,
            "auto_screening_bucket": bucket,
            "auto_screening_reason": "; ".join(reason),
            "exclusion_code": exclusion_code,
            "manual_decision": row.get("manual_decision", ""),
            "manual_notes": row.get("manual_notes", ""),
        }
    )
    return annotated


def _annotate_generic(
    row: dict[str, str],
    haystack: str,
    config: ScreeningConfig,
) -> dict[str, str]:
    matched_exclusion = next(
        (
            term
            for term in config.exclude_terms
            if normalize_text(term) and normalize_text(term) in haystack
        ),
        "",
    )
    if matched_exclusion:
        decision = "exclude"
        bucket = "user_exclusion_term"
        reason = f"title matched user exclusion term: {matched_exclusion}"
        exclusion_code = "user_exclusion_term"
    else:
        decision = "include_candidate"
        bucket = "project_query_match"
        reason = "candidate matched the project's configured database query"
        exclusion_code = ""

    annotated = dict(row)
    annotated.update(
        {
            "auto_screening_decision": decision,
            "auto_screening_bucket": bucket,
            "auto_screening_reason": reason,
            "exclusion_code": exclusion_code,
            "manual_decision": row.get("manual_decision", ""),
            "manual_notes": row.get("manual_notes", ""),
        }
    )
    return annotated


def write_screened_csv(rows: list[dict[str, str]], input_fields: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in SCREENING_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_screening(rows: list[dict[str, str]]) -> ScreeningSummary:
    return ScreeningSummary(
        total=len(rows),
        by_decision=Counter(row["auto_screening_decision"] for row in rows),
        by_bucket=Counter(row["auto_screening_bucket"] for row in rows),
        by_exclusion_code=Counter(row["exclusion_code"] for row in rows if row["exclusion_code"]),
    )


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _matched_noise(text: str) -> str:
    patterns = [
        ("electrical_transformer", r"\b(power|voltage|current|currents|electric|electrical|dc dc|thermal|oil|winding|converter|core loss|metrological|mva|kv|substation|instrument|distribution)\b.*\btransformer"),
        ("electrical_transformer", r"\btransformer(s)?\b.*\b(power|voltage|current|currents|electric|electrical|dc dc|thermal|oil|winding|converter|core loss|metrological|mva|kv|substation|instrument|distribution|protection)\b"),
        ("program_transformer", r"\b(predicate transformer|state transformer(s)?|transformer program(s)?)\b"),
        ("task_fact_verification", r"\b(fact|claim|rumou?r|stance|evidence|table|statement)\b.*\bverification\b"),
        ("task_fact_verification", r"\bcross candidate verification\b.*\bsemantic parsing\b"),
        ("task_speaker_verification", r"\b(speaker|voice|utterance|speech|audio|cry|gait|face|fingerprint|fingervein|vein|kinship|author|authorship|signature|identity|person|biometric|bank check|ear|scribe|canine)\b.*\bverification\b"),
        ("task_document_verification", r"\b(promise|document|financial document|structured data extraction|news headline|account|social media)\b.*\bverification\b"),
        ("task_tool_verification", r"\bsoftware verification\b.*\bgraph attention networks?\b"),
        ("task_tool_verification", r"\bgraph attention networks?\b.*\bfunctional safety verification\b"),
        ("task_tool_verification", r"\b(transformer based|transformer accelerated|transformer enhanced)\b.*\b(data driven )?(output )?reachability\b"),
        ("task_tool_verification", r"\b(video sequence|hyperlink|semantic financial document)\b.*\bverification\b"),
        ("task_tool_verification", r"\btext to sql\b.*\bbert\b.*\bverification\b"),
        ("model_transformation", r"\bmodel transformation(s)?\b"),
        ("network_packet_transformer", r"\bpacket transformer(s)?\b"),
        ("unrelated_certification", r"\b(leed|professional|exam|medical certification|animal production certification|construction management certification)\b"),
    ]
    for code, pattern in patterns:
        if re.search(pattern, text):
            return code
    return ""


def _is_llm_as_verification_tool(text: str) -> bool:
    if not _has_llm_term(text):
        return False
    if re.search(r"\bformal verification of llm generated code\b", text):
        return True
    if re.search(r"\bverification of llm generated (scientific )?code\b", text):
        return True
    if re.search(r"\bllm generated spatiotemporal knowledge\b", text):
        return True
    if re.search(r"\bllm powered structured data extraction\b", text):
        return True
    artifact_terms = (
        r"rtl|vlsi|systemverilog|verilog|hardware|circuit|fpga|smart contract|"
        r"software verification|program verification|automated verification|formal software|"
        r"code verification|backend systems|protocol|protocols|cryptographic|assertion|"
        r"property generation|propertygpt|loop invariant|proof assistant|theorem proving|"
        r"smt solver|solver fuzzing|verification assertion|design verification|"
        r"natural language requirements|ctl specification|dafny|openmp|firewall polic(?:y|ies)|"
        r"llm generated code|generated code|scientific code|spatiotemporal knowledge|"
        r"structured data extraction|data extraction|static analysis|compliance verification"
    )
    llm_as_tool_phrases = (
        r"(using|with|via|leveraging|integrating|combining|supporting|facilitat(?:e|ing)|"
        r"guided|driven|powered|generated|synthesizing|translating|fine tuning)"
    )
    return bool(
        re.search(rf"\b{llm_as_tool_phrases}\b.*\b(llm|llms|large language model|language models)\b.*\b({artifact_terms})\b", text)
        or re.search(rf"\b(llm|llms|large language model|language models)\b.*\b{llm_as_tool_phrases}\b.*\b({artifact_terms})\b", text)
        or re.search(rf"\b({artifact_terms})\b.*\b(llm|llms|large language model|language models)\b", text)
    )


def _is_core_transformer_verification(text: str) -> bool:
    transformer_terms = r"transformer|transformers|vision transformer|vision transformers|vit|bert|self attention|attention network|attention networks"
    verification_terms = (
        r"verification|certification|certified robustness|robustness verification|"
        r"certified|exact robustness|abstract interpretation|bound propagation|linear relaxation|"
        r"mixed integer|reachability"
    )
    target_terms = (
        r"vision transformer|vision transformers|vit|bert|transformer|"
        r"self attention|attention network|attention networks|transformers|bert"
    )
    return bool(
        re.search(rf"\b({transformer_terms})\b.*\b({verification_terms})\b", text)
        or re.search(rf"\b({verification_terms})\b.*\b({target_terms})\b", text)
    )


def _is_llm_target_verification(text: str) -> bool:
    if not _has_llm_term(text):
        return False
    target_property_terms = (
        r"domain certification|certified robustness|robustness certification|"
        r"verification of llm|verification of large language|verify llm|verifying llm|"
        r"llm output|llm outputs|ownership verification|watermark|replicability|"
        r"misalignment|safety verification|contextual integrity|risk certification|"
        r"self verification limitations|verification dynamics|asymmetric verification"
    )
    return bool(re.search(target_property_terms, text))


def _has_llm_term(text: str) -> bool:
    return bool(re.search(r"\b(llm|llms|large language model|large language models|language model|language models)\b", text))
