from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PaperRecord:
    """A normalized bibliographic record collected from a search provider."""

    title: str
    source: str
    query: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    dblp_key: str | None = None
    publication_type: str | None = None
    provider_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def dedupe_key(self) -> str:
        if self.dblp_key:
            return f"dblp:{self.dblp_key.lower().strip()}"
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        if self.provider_id:
            return f"{self.source}:{self.provider_id.lower().strip()}"
        return f"title:{normalize_title(self.title)}"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["authors"] = "; ".join(self.authors)
        row.pop("raw", None)
        return row


def normalize_title(title: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in title).split()
    ).strip()
