from __future__ import annotations

import csv
import json
from pathlib import Path

from vnn_survey.models import PaperRecord


FIELDNAMES = [
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "dblp_key",
    "publication_type",
    "source",
    "query",
    "provider_id",
]


def write_jsonl(records: list[PaperRecord], path: Path, include_raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_row()
            if include_raw:
                payload["raw"] = record.raw
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(records: list[PaperRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            row = record.to_row()
            writer.writerow({field: row.get(field) for field in FIELDNAMES})

