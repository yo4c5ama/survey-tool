from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import get_ident
from typing import Any

import requests

from vnn_survey.models import normalize_title

CorpusProgressCallback = Callable[[int, int, str], None]
CorpusStageCallback = Callable[[str, str], None]

TAXONOMY_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "survey_taxonomy",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "overview": {"type": "string"},
            "categories": {
                "type": "array",
                "minItems": 2,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "inclusion_signals": {"type": "string"},
                    },
                    "required": ["id", "label", "description", "inclusion_signals"],
                },
            },
        },
        "required": ["title", "overview", "categories"],
    },
}


@dataclass(frozen=True, slots=True)
class CorpusAnalysisResult:
    taxonomy: dict[str, Any]
    classifications: list[dict[str, str]]
    taxonomy_path: Path
    classifications_path: Path
    report_path: Path


class PaperWorkspace:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.papers_dir = project_dir / "papers"
        self.conversations_dir = project_dir / "conversations"
        self.index_path = self.papers_dir / "index.json"
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def paper_id(self, paper: dict[str, str]) -> str:
        identity = paper.get("doi") or normalize_title(paper.get("title", ""))
        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    def save_pdf(self, paper: dict[str, str], content: bytes) -> Path:
        if not content.startswith(b"%PDF"):
            raise ValueError("The uploaded file is not a valid PDF.")
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("The PDF must not exceed 50 MB.")
        paper_id = self.paper_id(paper)
        path = self.papers_dir / f"{paper_id}.pdf"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        index = self._load_index()
        previous = index.get(paper_id, {})
        index[paper_id] = {
            "title": paper.get("title", ""),
            "doi": paper.get("doi", ""),
            "pdf_path": str(path),
            "pdf_sha256": hashlib.sha256(content).hexdigest(),
            "openai_file_id": "",
            "saved_at": _now(),
            **{
                key: value
                for key, value in previous.items()
                if key
                not in {
                    "pdf_path",
                    "pdf_sha256",
                    "openai_file_id",
                    "file_uploaded_at",
                    "saved_at",
                }
            },
        }
        self._write_index(index)
        return path

    def pdf_path(self, paper: dict[str, str]) -> Path | None:
        paper_id = self.paper_id(paper)
        indexed = self._load_index().get(paper_id, {})
        value = indexed.get("pdf_path")
        path = Path(value) if value else self.papers_dir / f"{paper_id}.pdf"
        return path if path.exists() else None

    def file_id(self, paper: dict[str, str]) -> str:
        return str(
            self._load_index().get(self.paper_id(paper), {}).get("openai_file_id") or ""
        )

    def save_file_id(self, paper: dict[str, str], file_id: str) -> None:
        paper_id = self.paper_id(paper)
        index = self._load_index()
        entry = index.setdefault(
            paper_id,
            {"title": paper.get("title", ""), "doi": paper.get("doi", "")},
        )
        entry["openai_file_id"] = file_id
        entry["file_uploaded_at"] = _now()
        self._write_index(index)

    def load_conversation(self, paper: dict[str, str]) -> list[dict[str, str]]:
        path = self._conversation_path(paper)
        if not path.exists():
            return []
        value = json.loads(path.read_text(encoding="utf-8"))
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def append_exchange(
        self,
        paper: dict[str, str],
        *,
        question: str,
        answer: str,
        model: str,
        response_id: str,
    ) -> list[dict[str, str]]:
        messages = self.load_conversation(paper)
        now = _now()
        messages.extend(
            [
                {"role": "user", "content": question, "created_at": now},
                {
                    "role": "assistant",
                    "content": answer,
                    "created_at": now,
                    "model": model,
                    "response_id": response_id,
                },
            ]
        )
        _write_json(self._conversation_path(paper), messages)
        return messages

    def clear_conversation(self, paper: dict[str, str]) -> None:
        self._conversation_path(paper).unlink(missing_ok=True)

    def _conversation_path(self, paper: dict[str, str]) -> Path:
        return self.conversations_dir / f"{self.paper_id(paper)}.json"

    def _load_index(self) -> dict[str, dict[str, str]]:
        if not self.index_path.exists():
            return {}
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def _write_index(self, value: dict[str, dict[str, str]]) -> None:
        _write_json(self.index_path, value)


class OpenAIResearchClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        retries: int = 3,
    ) -> None:
        if not api_key.strip():
            raise RuntimeError("Save or provide an OpenAI API key before using AI research.")
        if not model.strip():
            raise ValueError("Select an AI model before continuing.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "SurveyFlow/0.1 AI research",
            }
        )

    def ask_pdf(
        self,
        *,
        pdf_path: Path,
        paper: dict[str, str],
        conversation: list[dict[str, str]],
        question: str,
        file_id: str = "",
    ) -> tuple[str, str, str]:
        active_file_id = file_id or self.upload_pdf(pdf_path)
        try:
            payload = self._pdf_payload(
                active_file_id,
                paper,
                conversation,
                question,
            )
            response = self._post_response(payload)
        except requests.HTTPError as exc:
            if not file_id or exc.response is None or exc.response.status_code != 400:
                raise
            active_file_id = self.upload_pdf(pdf_path)
            response = self._post_response(
                self._pdf_payload(
                    active_file_id,
                    paper,
                    conversation,
                    question,
                )
            )
        answer = _extract_output_text(response)
        if not answer:
            raise RuntimeError("The model returned no text answer.")
        return answer, str(response.get("id") or ""), active_file_id

    def upload_pdf(self, pdf_path: Path) -> str:
        with pdf_path.open("rb") as handle:
            response = self.session.post(
                f"{self.base_url}/files",
                data={"purpose": "user_data"},
                files={"file": (pdf_path.name, handle, "application/pdf")},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        file_id = str(response.json().get("id") or "")
        if not file_id:
            raise RuntimeError("The PDF upload returned no file ID.")
        return file_id

    def json_response(
        self,
        *,
        instructions: str,
        input_text: str,
        schema: dict[str, Any],
        max_output_tokens: int = 6000,
    ) -> dict[str, Any]:
        response = self._post_response(
            {
                "model": self.model,
                "instructions": instructions,
                "input": input_text,
                "text": {"format": schema},
                "max_output_tokens": max_output_tokens,
            }
        )
        output = _extract_output_text(response)
        if not output:
            raise RuntimeError("The model returned no structured output.")
        value = json.loads(output)
        if not isinstance(value, dict):
            raise RuntimeError("The model returned an invalid JSON object.")
        return value

    def _pdf_payload(
        self,
        file_id: str,
        paper: dict[str, str],
        conversation: list[dict[str, str]],
        question: str,
    ) -> dict[str, Any]:
        context = (
            f"Paper title: {paper.get('title', '')}\n"
            f"Authors: {paper.get('authors', '')}\n"
            f"Year: {paper.get('year', '')}\n"
            f"Venue: {paper.get('venue', '')}"
        )
        inputs: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {
                        "type": "input_text",
                        "text": f"Use this PDF as the primary source.\n{context}",
                    },
                ],
            }
        ]
        for message in conversation[-12:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                inputs.append({"role": role, "content": str(content)})
        inputs.append({"role": "user", "content": question})
        return {
            "model": self.model,
            "instructions": (
                "Answer questions about the attached academic paper. Ground every factual claim "
                "in the PDF, distinguish the authors' claims from your interpretation, cite a "
                "page or section when possible, and say clearly when the PDF does not support an "
                "answer. Respond in the language used by the user."
            ),
            "input": inputs,
            "max_output_tokens": 3000,
        }

    def _post_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/responses",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 and attempt < self.retries:
                    time.sleep(float(response.headers.get("Retry-After") or attempt * 2))
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    if exc.response.status_code == 400:
                        raise
                if attempt < self.retries:
                    time.sleep(attempt)
        raise RuntimeError("OpenAI Responses API request failed") from last_error


class CorpusAnalyzer:
    def __init__(self, client: OpenAIResearchClient) -> None:
        self.client = client

    def analyze(
        self,
        *,
        rows: list[dict[str, str]],
        research_question: str,
        scope_description: str,
        criteria: str,
        output_dir: Path,
        progress_callback: CorpusProgressCallback | None = None,
        stage_callback: CorpusStageCallback | None = None,
    ) -> CorpusAnalysisResult:
        if not rows:
            raise ValueError("The final corpus is empty.")
        prepared = [_with_paper_id(row) for row in rows]
        if stage_callback:
            stage_callback("Taxonomy design", "Designing a stable taxonomy for the corpus.")
        taxonomy = self.design_taxonomy(
            prepared,
            research_question=research_question,
            scope_description=scope_description,
            criteria=criteria,
        )
        categories = taxonomy.get("categories", [])
        category_ids = [str(item.get("id") or "") for item in categories if item.get("id")]
        if len(category_ids) < 2:
            raise RuntimeError("The model did not produce a usable taxonomy.")

        if stage_callback:
            stage_callback(
                "Paper classification",
                "Classifying every paper with the fixed taxonomy.",
            )
        classifications: list[dict[str, str]] = []
        chunk_size = 20
        total = len(prepared)
        if progress_callback:
            progress_callback(0, total, "")
        for start in range(0, total, chunk_size):
            chunk = prepared[start : start + chunk_size]
            batch = self.classify_batch(chunk, taxonomy)
            classifications.extend(batch)
            completed = min(start + len(chunk), total)
            if progress_callback:
                progress_callback(completed, total, chunk[-1].get("title", ""))

        classifications = _complete_classifications(prepared, classifications)
        if stage_callback:
            stage_callback("Analysis report", "Writing the taxonomy and classification report.")
        output_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_path = output_dir / "taxonomy.json"
        classifications_path = output_dir / "classifications.csv"
        report_path = output_dir / "report.md"
        _write_json(taxonomy_path, taxonomy)
        _write_classifications(classifications_path, classifications)
        report_path.write_text(
            _build_analysis_report(
                taxonomy=taxonomy,
                classifications=classifications,
                criteria=criteria,
                model=self.client.model,
            ),
            encoding="utf-8",
        )
        return CorpusAnalysisResult(
            taxonomy=taxonomy,
            classifications=classifications,
            taxonomy_path=taxonomy_path,
            classifications_path=classifications_path,
            report_path=report_path,
        )

    def design_taxonomy(
        self,
        rows: list[dict[str, str]],
        *,
        research_question: str,
        scope_description: str,
        criteria: str,
    ) -> dict[str, Any]:
        guidance = criteria.strip() or (
            "No classification criteria were supplied. Infer a useful, academically meaningful "
            "taxonomy from the corpus. Prefer categories based on research objective, method, or "
            "guarantee type rather than superficial title words."
        )
        return self.client.json_response(
            instructions=(
                "Design a stable taxonomy for an academic survey corpus. Category IDs must be "
                "short lowercase snake_case strings. Categories should be distinct, collectively "
                "useful, and reusable for every paper."
            ),
            input_text=(
                f"Research question:\n{research_question}\n\n"
                f"Scope:\n{scope_description}\n\n"
                f"User classification guidance:\n{guidance}\n\n"
                f"Corpus:\n{_corpus_context(rows)}"
            ),
            schema=TAXONOMY_SCHEMA,
            max_output_tokens=4000,
        )

    def classify_batch(
        self,
        rows: list[dict[str, str]],
        taxonomy: dict[str, Any],
    ) -> list[dict[str, str]]:
        category_ids = [
            str(item["id"])
            for item in taxonomy.get("categories", [])
            if isinstance(item, dict) and item.get("id")
        ]
        schema = _classification_schema(category_ids, len(rows))
        response = self.client.json_response(
            instructions=(
                "Classify every supplied paper using the fixed taxonomy. Use one primary category, "
                "at most two genuinely useful secondary categories, and a concise evidence-based "
                "rationale. Do not invent categories or paper IDs."
            ),
            input_text=(
                f"Fixed taxonomy:\n{json.dumps(taxonomy, ensure_ascii=False)}\n\n"
                f"Papers:\n{_corpus_context(rows, max_abstract_chars=2200, max_chars=70000)}"
            ),
            schema=schema,
            max_output_tokens=max(3000, len(rows) * 300),
        )
        values = response.get("classifications", [])
        return [
            {
                "paper_id": str(item.get("paper_id") or ""),
                "primary_category": str(item.get("primary_category") or ""),
                "secondary_categories": "; ".join(
                    str(value) for value in item.get("secondary_categories", [])
                ),
                "rationale": str(item.get("rationale") or ""),
            }
            for item in values
            if isinstance(item, dict)
        ]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def estimate_corpus_requests(paper_count: int, chunk_size: int = 20) -> int:
    return 1 + math.ceil(max(paper_count, 0) / chunk_size)


def _classification_schema(category_ids: list[str], count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "survey_paper_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "classifications": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "paper_id": {"type": "string"},
                            "primary_category": {
                                "type": "string",
                                "enum": category_ids,
                            },
                            "secondary_categories": {
                                "type": "array",
                                "maxItems": 2,
                                "items": {"type": "string", "enum": category_ids},
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "paper_id",
                            "primary_category",
                            "secondary_categories",
                            "rationale",
                        ],
                    },
                }
            },
            "required": ["classifications"],
        },
    }


def _corpus_context(
    rows: list[dict[str, str]],
    *,
    max_abstract_chars: int = 1200,
    max_chars: int = 80000,
) -> str:
    blocks: list[str] = []
    size = 0
    for row in rows:
        abstract = str(row.get("abstract") or "")[:max_abstract_chars]
        block = (
            f"paper_id: {row.get('paper_id', '')}\n"
            f"title: {row.get('title', '')}\n"
            f"year: {row.get('year', '')}\n"
            f"venue: {row.get('venue', '')}\n"
            f"abstract: {abstract}\n"
        )
        if blocks and size + len(block) > max_chars:
            break
        blocks.append(block)
        size += len(block)
    return "\n".join(blocks)


def _with_paper_id(row: dict[str, str]) -> dict[str, str]:
    value = dict(row)
    identity = row.get("doi") or normalize_title(row.get("title", ""))
    value["paper_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return value


def _complete_classifications(
    rows: list[dict[str, str]],
    classifications: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_id = {item.get("paper_id", ""): item for item in classifications}
    output = []
    for row in rows:
        classification = by_id.get(row["paper_id"], {})
        output.append(
            {
                "paper_id": row["paper_id"],
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "year": row.get("year", ""),
                "venue": row.get("venue", ""),
                "doi": row.get("doi", ""),
                "primary_category": classification.get(
                    "primary_category", "unclassified"
                ),
                "secondary_categories": classification.get(
                    "secondary_categories", ""
                ),
                "rationale": classification.get(
                    "rationale", "No classification was returned."
                ),
            }
        )
    return output


def _write_classifications(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "paper_id",
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "primary_category",
        "secondary_categories",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_analysis_report(
    *,
    taxonomy: dict[str, Any],
    classifications: list[dict[str, str]],
    criteria: str,
    model: str,
) -> str:
    counts = Counter(item["primary_category"] for item in classifications)
    labels = {
        str(item.get("id")): str(item.get("label") or item.get("id"))
        for item in taxonomy.get("categories", [])
        if isinstance(item, dict)
    }
    lines = [
        f"# {taxonomy.get('title') or 'Corpus Classification'}",
        "",
        str(taxonomy.get("overview") or ""),
        "",
        f"- Model: `{model}`",
        f"- Papers: {len(classifications)}",
        f"- User criteria: {criteria.strip() or 'Model-proposed taxonomy'}",
        "",
        "## Categories",
        "",
    ]
    for category in taxonomy.get("categories", []):
        category_id = str(category.get("id") or "")
        lines.extend(
            [
                f"### {category.get('label') or category_id} ({counts.get(category_id, 0)})",
                "",
                str(category.get("description") or ""),
                "",
            ]
        )
    lines.extend(["## Papers", "", "| Category | Year | Title | Rationale |", "|---|---:|---|---|"])
    for row in classifications:
        title = row["title"].replace("|", "\\|")
        rationale = row["rationale"].replace("|", "\\|")
        category = labels.get(row["primary_category"], row["primary_category"])
        lines.append(f"| {category} | {row['year']} | {title} | {rationale} |")
    lines.append("")
    return "\n".join(lines)


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for output in payload.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
