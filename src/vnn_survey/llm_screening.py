from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from vnn_survey.app.task_manager import cancellable_sleep, raise_if_cancelled
from vnn_survey.config import LlmScreeningConfig
from vnn_survey.models import normalize_title

ItemProgressCallback = Callable[[int, int, str], None]


LLM_SCREENING_FIELDS = [
    "llm_decision",
    "llm_scope",
    "llm_confidence",
    "llm_reason",
    "llm_evidence",
    "llm_model",
    "llm_status",
    "llm_prompt_version",
    "llm_response_id",
    "llm_checked_at",
    "llm_error",
]

ALLOWED_DECISIONS = ["include", "exclude", "maybe"]
ALLOWED_SCOPES = [
    "in_scope",
    "related",
    "out_of_scope",
    "transformer_verification",
    "llm_target_verification",
    "llm_as_tool",
    "unrelated",
    "insufficient_information",
]

DEFAULT_SYSTEM_PROMPT = """You are screening papers for a survey on Transformer verification.

Screen based only on the provided title, abstract, and metadata. Do not invent
facts not supported by the title or abstract.

Survey scope:
- Include formal verification, certification, certified robustness, reachability
  analysis, abstract interpretation, bound propagation, linear relaxation,
  SMT/MILP-based verification, or comparable rigorous analysis of Transformers,
  Vision Transformers, attention-based neural networks, or LLMs themselves.
- Include downstream techniques derived from verification when the verified or
  guaranteed target is a Transformer, Vision Transformer, attention model, LLM,
  or LLM output. This includes provable repair, provable model editing, formal
  XAI, formal/provable explanations, mechanistic interpretability with formal
  guarantees, and circuit discovery with certified robustness/minimality/patching
  guarantees.
- Include work that verifies/certifies LLM behavior, outputs, ownership,
  watermarking, robustness, safety, misalignment, domain certification, or
  reasoning, when the LLM itself or its output is the verification target.
- Exclude work where LLMs, Transformers, BERT, or attention networks are merely
  tools/features/classifiers used to verify another artifact or solve another
  task, such as software, hardware, code, protocols, smart contracts, documents,
  claims, facts, speakers, signatures, identity, or electrical transformers.
- Exclude generic neural-network verification papers unless the title or abstract
  clearly targets Transformers, attention models, ViTs, LLMs, or directly
  supports a methodology paper required for NLP/Transformer verification.

Scope labels:
- Use "transformer_verification" for Transformer/ViT/attention verification,
  certification, provable repair, or formal explainability.
- Use "llm_target_verification" for formal verification/certification/repair or
  formal explainability of LLMs or LLM outputs.
- Use "llm_as_tool" when an LLM is only a tool for verifying another artifact.
- Use "unrelated" for fact checking, speaker verification, document verification,
  software/hardware verification, or other tasks where the model is not the
  verification target.
- Use "insufficient_information" when the title/abstract do not support a stable
  scope judgment.

Decision policy:
- Use "include" only when the paper is clearly in scope.
- Use "exclude" when it is clearly out of scope.
- Use "maybe" when title/abstract are ambiguous or the abstract is missing.

Return concise evidence copied or paraphrased from the title/abstract."""


DEFAULT_USER_PROMPT_TEMPLATE = """title: {title}
year: {year}
venue: {venue}
doi: {doi}
auto_screening_decision: {auto_screening_decision}
auto_screening_bucket: {auto_screening_bucket}
abstract: {abstract}"""


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "transformer_verification_screening",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ALLOWED_DECISIONS},
            "scope": {"type": "string", "enum": ALLOWED_SCOPES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["decision", "scope", "confidence", "reason", "evidence"],
    },
}


@dataclass(frozen=True, slots=True)
class LlmScreeningSummary:
    total: int
    eligible: int
    attempted: int
    by_status: Counter[str]
    by_decision: Counter[str]
    by_scope: Counter[str]


@dataclass(frozen=True, slots=True)
class LlmScreeningResult:
    rows: list[dict[str, str]]
    summary: LlmScreeningSummary


def llm_screen_candidates(
    input_path: Path,
    output_path: Path,
    config: LlmScreeningConfig,
    decisions: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    progress_callback: ItemProgressCallback | None = None,
) -> LlmScreeningResult:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    effective_decisions = decisions if decisions is not None else set(config.include_decisions)
    eligible_indexes = [
        index
        for index, row in enumerate(rows)
        if _is_eligible(row=row, decisions=effective_decisions)
    ]
    if limit is not None:
        eligible_indexes = eligible_indexes[:limit]

    if dry_run:
        output_rows = [dict(row) for row in rows]
        for row in output_rows:
            _mark_skipped(row, status="dry_run", checked_at="")
        eligible_rows = [rows[index] for index in eligible_indexes]
        return LlmScreeningResult(
            rows=output_rows,
            summary=LlmScreeningSummary(
                total=len(rows),
                eligible=len(eligible_indexes),
                attempted=0,
                by_status=Counter(
                    {
                        "would_screen": len(eligible_indexes),
                        "skipped_decision": len(rows) - len(eligible_indexes),
                    }
                ),
                by_decision=Counter(
                    row.get("auto_screening_decision", "") for row in eligible_rows
                ),
                by_scope=Counter(row.get("auto_screening_bucket", "") for row in eligible_rows),
            ),
        )

    client = OpenAIResponsesClient(config)
    now = datetime.now().isoformat(timespec="seconds")
    eligible_index_set = set(eligible_indexes)
    attempted = 0
    completed = 0
    output_rows: list[dict[str, str]] = []
    if progress_callback:
        progress_callback(0, len(eligible_indexes), "")

    for index, row in enumerate(rows):
        output_row = dict(row)
        if index not in eligible_index_set:
            _mark_skipped(output_row, status="skipped_decision", checked_at=now)
            output_rows.append(output_row)
            continue
        if _has_existing_llm_result(output_row) and not overwrite:
            output_row["llm_status"] = output_row.get("llm_status") or "existing"
            output_row["llm_checked_at"] = output_row.get("llm_checked_at") or now
            output_rows.append(output_row)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(eligible_indexes), row.get("title", ""))
            continue

        attempted += 1
        try:
            result = client.screen(row)
            _apply_llm_result(output_row, result=result, config=config, checked_at=now)
        except Exception as exc:  # noqa: BLE001 - keep processing other rows.
            _mark_failed(output_row, error=str(exc), config=config, checked_at=now)
        output_rows.append(output_row)
        completed += 1
        if progress_callback:
            progress_callback(completed, len(eligible_indexes), row.get("title", ""))

    write_llm_screened_csv(output_rows, input_fields, output_path)
    return LlmScreeningResult(
        rows=output_rows,
        summary=_summarize(
            output_rows,
            total=len(rows),
            eligible=len(eligible_index_set),
            attempted=attempted,
        ),
    )


def write_llm_screened_csv(
    rows: list[dict[str, str]],
    input_fields: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_fields)
    for field in LLM_SCREENING_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_llm_screening_summary(summary: LlmScreeningSummary, output_path: Path) -> None:
    payload = {
        "total": summary.total,
        "eligible": summary.eligible,
        "attempted": summary.attempted,
        "by_status": dict(summary.by_status),
        "by_decision": dict(summary.by_decision),
        "by_scope": dict(summary.by_scope),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenAIResponsesClient:
    def __init__(self, config: LlmScreeningConfig) -> None:
        self.config = config
        self.api_key = _load_api_key(config)
        if not self.api_key:
            raise RuntimeError(
                "Missing OpenAI API key. Set "
                f"{config.api_key_env} or write the key to {config.api_key_file}."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "vnn-survey/0.1 llm screening",
            }
        )
        self.cache_dir = config.cache_dir
        self.system_prompt = _load_prompt_text(
            config.system_prompt_path,
            default=DEFAULT_SYSTEM_PROMPT,
            label="system prompt",
        )
        self.user_prompt_template = _load_prompt_text(
            config.user_prompt_template_path,
            default=DEFAULT_USER_PROMPT_TEMPLATE,
            label="user prompt template",
        )
        self.system_prompt_hash = _text_hash(self.system_prompt)
        self.user_prompt_template_hash = _text_hash(self.user_prompt_template)

    def screen(self, row: dict[str, str]) -> dict[str, Any]:
        cache_path = self._cache_path(row)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = {
            "model": self.config.model,
            "instructions": self.system_prompt,
            "input": _build_user_prompt(row, template=self.user_prompt_template),
            "text": {"format": RESPONSE_SCHEMA},
            "max_output_tokens": self.config.max_output_tokens,
        }
        response_payload = self._request(payload)
        parsed = _parse_response_payload(response_payload)
        parsed["_response_id"] = str(response_payload.get("id") or "")
        _write_cache(cache_path, parsed)
        return parsed

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.config.base_url.rstrip('/')}/responses"
        for attempt in range(1, self.config.retries + 1):
            try:
                raise_if_cancelled()
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                raise_if_cancelled()
                if response.status_code == 429 and attempt < self.config.retries:
                    retry_after = response.headers.get("Retry-After")
                    sleep_seconds = (
                        float(retry_after)
                        if retry_after
                        else self.config.request_delay_seconds * attempt * 4
                    )
                    cancellable_sleep(sleep_seconds)
                    continue
                response.raise_for_status()
                cancellable_sleep(self.config.request_delay_seconds)
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.config.retries:
                    cancellable_sleep(self.config.request_delay_seconds * attempt)
        raise RuntimeError("OpenAI Responses API request failed") from last_error

    def _cache_path(self, row: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = {
            "prompt_version": self.config.prompt_version,
            "system_prompt_hash": self.system_prompt_hash,
            "user_prompt_template_hash": self.user_prompt_template_hash,
            "model": self.config.model,
            "paper_key": _paper_key(row),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"


def _parse_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    output_text = _extract_output_text(payload)
    if not output_text:
        raise RuntimeError("OpenAI response did not contain output text")
    parsed = json.loads(output_text)
    _validate_llm_result(parsed)
    return parsed


def _load_api_key(config: LlmScreeningConfig) -> str:
    env_key = os.environ.get(config.api_key_env, "").strip() if config.api_key_env else ""
    if env_key:
        return env_key
    if config.api_key_file:
        try:
            return config.api_key_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
    return ""


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    for output_item in payload.get("output", []) or []:
        for content_item in output_item.get("content", []) or []:
            if isinstance(content_item.get("text"), str):
                return str(content_item["text"])
    return ""


def _validate_llm_result(value: dict[str, Any]) -> None:
    if value.get("decision") not in ALLOWED_DECISIONS:
        raise RuntimeError(f"invalid LLM decision: {value.get('decision')!r}")
    if value.get("scope") not in ALLOWED_SCOPES:
        raise RuntimeError(f"invalid LLM scope: {value.get('scope')!r}")
    confidence = value.get("confidence")
    if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
        raise RuntimeError(f"invalid LLM confidence: {confidence!r}")


def _build_user_prompt(row: dict[str, str], template: str) -> str:
    abstract = (row.get("abstract") or "").strip()
    if not abstract:
        abstract = "[No abstract available.]"
    fields = {str(key): str(value or "") for key, value in row.items()}
    fields.update(
        {
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "venue": row.get("venue", ""),
            "doi": row.get("doi", ""),
            "auto_screening_decision": row.get("auto_screening_decision", ""),
            "auto_screening_bucket": row.get("auto_screening_bucket", ""),
            "abstract": abstract[:5000],
            "abstract_truncated": abstract[:5000],
        }
    )
    try:
        return template.format_map(fields)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise RuntimeError(
            f"Unknown LLM user prompt placeholder {{{missing}}}. "
            "Use CSV field names such as {title}, {year}, {venue}, {doi}, "
            "{abstract}, {auto_screening_decision}, or {auto_screening_bucket}."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "Invalid LLM user prompt template. If you need literal braces, "
            "escape them as '{{' and '}}'."
        ) from exc


def _load_prompt_text(path: Path | None, default: str, label: str) -> str:
    if path is None:
        return default
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Configured LLM {label} file was not found: {path}") from exc
    if not text:
        raise RuntimeError(f"Configured LLM {label} file is empty: {path}")
    return text


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_eligible(row: dict[str, str], decisions: set[str] | None) -> bool:
    if decisions is None:
        return True
    if "auto_screening_decision" not in row:
        return True
    return row.get("auto_screening_decision") in decisions


def _has_existing_llm_result(row: dict[str, str]) -> bool:
    return bool((row.get("llm_decision") or "").strip())


def _apply_llm_result(
    row: dict[str, str],
    result: dict[str, Any],
    config: LlmScreeningConfig,
    checked_at: str,
) -> None:
    row.update(
        {
            "llm_decision": str(result.get("decision", "")),
            "llm_scope": str(result.get("scope", "")),
            "llm_confidence": f"{float(result.get('confidence', 0.0)):.3f}",
            "llm_reason": str(result.get("reason", "")),
            "llm_evidence": str(result.get("evidence", "")),
            "llm_model": config.model,
            "llm_status": "screened",
            "llm_prompt_version": config.prompt_version,
            "llm_response_id": str(result.get("_response_id", "")),
            "llm_checked_at": checked_at,
            "llm_error": "",
        }
    )


def _mark_skipped(row: dict[str, str], status: str, checked_at: str) -> None:
    row["llm_status"] = status
    row["llm_checked_at"] = checked_at
    for field in LLM_SCREENING_FIELDS:
        row.setdefault(field, "")


def _mark_failed(
    row: dict[str, str],
    error: str,
    config: LlmScreeningConfig,
    checked_at: str,
) -> None:
    row.update(
        {
            "llm_model": config.model,
            "llm_status": "failed",
            "llm_prompt_version": config.prompt_version,
            "llm_checked_at": checked_at,
            "llm_error": error,
        }
    )
    for field in LLM_SCREENING_FIELDS:
        row.setdefault(field, "")


def _summarize(
    rows: list[dict[str, str]],
    total: int,
    eligible: int,
    attempted: int,
) -> LlmScreeningSummary:
    return LlmScreeningSummary(
        total=total,
        eligible=eligible,
        attempted=attempted,
        by_status=Counter(row.get("llm_status", "") for row in rows),
        by_decision=Counter(row.get("llm_decision", "") for row in rows if row.get("llm_decision")),
        by_scope=Counter(row.get("llm_scope", "") for row in rows if row.get("llm_scope")),
    )


def _paper_key(row: dict[str, str]) -> str:
    for field in ["doi", "dblp_key", "provider_id"]:
        value = (row.get(field) or "").strip().lower()
        if value:
            return f"{field}:{value}"
    return f"title:{normalize_title(row.get('title', ''))}"


def _write_cache(cache_path: Path | None, payload: dict[str, Any]) -> None:
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
