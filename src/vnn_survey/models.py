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
    abstract: str | None = None
    abstract_source: str | None = None
    discovery_sources: list[str] = field(default_factory=list)
    discovery_queries: list[str] = field(default_factory=list)
    manual_added: bool = False
    manual_note: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.discovery_sources = _dedupe_values(
            [
                *self.discovery_sources,
                self.source,
                *(["manual"] if self.manual_added else []),
            ]
        )
        self.discovery_queries = _dedupe_values(
            [*self.discovery_queries, *([self.query] if self.query else [])]
        )

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
        row["discovery_sources"] = "; ".join(self.discovery_sources)
        row["discovery_queries"] = "; ".join(self.discovery_queries)
        row.pop("raw", None)
        return row

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_raw:
            value.pop("raw", None)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PaperRecord:
        authors = value.get("authors", [])
        if isinstance(authors, str):
            authors = [item.strip() for item in authors.split(";") if item.strip()]
        sources = value.get("discovery_sources", [])
        if isinstance(sources, str):
            sources = [item.strip() for item in sources.split(";") if item.strip()]
        queries = value.get("discovery_queries", [])
        if isinstance(queries, str):
            queries = [item.strip() for item in queries.split(";") if item.strip()]
        return cls(
            title=str(value.get("title") or "").strip(),
            source=str(value.get("source") or "manual").strip(),
            query=str(value.get("query") or "manual addition").strip(),
            year=_optional_int(value.get("year")),
            authors=[str(item) for item in authors],
            venue=_optional_text(value.get("venue")),
            doi=_optional_text(value.get("doi")),
            url=_optional_text(value.get("url")),
            dblp_key=_optional_text(value.get("dblp_key")),
            publication_type=_optional_text(value.get("publication_type")),
            provider_id=_optional_text(value.get("provider_id")),
            abstract=_optional_text(value.get("abstract")),
            abstract_source=_optional_text(value.get("abstract_source")),
            discovery_sources=[str(item) for item in sources],
            discovery_queries=[str(item) for item in queries],
            manual_added=bool(value.get("manual_added", False)),
            manual_note=_optional_text(value.get("manual_note")),
            raw=value.get("raw", {}) if isinstance(value.get("raw"), dict) else {},
        )


def normalize_title(title: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in title).split()
    ).strip()


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
