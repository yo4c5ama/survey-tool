from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from threading import get_ident

from vnn_survey.models import PaperRecord
from vnn_survey.pipeline import dedupe_records


class ManualPaperStore:
    def __init__(self, project_dir: Path) -> None:
        self.path = project_dir / "manual" / "papers.jsonl"

    def load(self) -> list[PaperRecord]:
        if not self.path.exists():
            return []
        records: list[PaperRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(PaperRecord.from_dict(value))
        return records

    def add(self, record: PaperRecord, note: str = "") -> tuple[PaperRecord, bool]:
        existing = self.load()
        manual_record = replace(
            record,
            manual_added=True,
            manual_note=note.strip() or record.manual_note,
            discovery_sources=_merge_values(record.discovery_sources, ["manual"]),
        )
        merged = dedupe_records([*existing, manual_record])
        added = len(merged) > len(existing)
        self._write(merged)
        saved = next(
            item for item in merged if len(dedupe_records([item, manual_record])) == 1
        )
        return saved, added

    def remove(self, dedupe_key: str) -> bool:
        existing = self.load()
        retained = [record for record in existing if record.dedupe_key() != dedupe_key]
        if len(retained) == len(existing):
            return False
        self._write(retained)
        return True

    def _write(self, records: list[PaperRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{get_ident()}.tmp"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.to_dict(include_raw=True), ensure_ascii=False) + "\n"
                )
        temporary.replace(self.path)


def create_manual_record(
    *,
    title: str,
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str = "",
    doi: str = "",
    url: str = "",
    publication_type: str = "",
    note: str = "",
) -> PaperRecord:
    normalized_title = title.strip()
    if not normalized_title:
        raise ValueError("A paper title is required.")
    normalized_doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized_doi.lower().startswith(prefix):
            normalized_doi = normalized_doi[len(prefix) :]
            break
    return PaperRecord(
        title=normalized_title,
        source="manual",
        query="manual addition",
        year=year,
        authors=authors or [],
        venue=venue.strip() or None,
        doi=normalized_doi or None,
        url=url.strip() or None,
        publication_type=publication_type.strip() or None,
        discovery_sources=["manual"],
        discovery_queries=["manual addition"],
        manual_added=True,
        manual_note=note.strip() or None,
    )


def _merge_values(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(value for group in groups for value in group if value))
