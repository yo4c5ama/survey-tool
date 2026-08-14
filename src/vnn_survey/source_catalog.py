from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CATALOG = REPO_ROOT / "configs" / "source_catalog.yaml"
DEFAULT_DOMAIN_PROFILES = REPO_ROOT / "configs" / "domain_profiles.yaml"


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    id: str
    label: str
    status: str
    record_types: tuple[str, ...]
    capabilities: tuple[str, ...]
    requires_key: bool
    scope: dict[str, str]
    limitation: dict[str, str]

    @property
    def available(self) -> bool:
        return self.status == "available"

    def localized_scope(self, language: str) -> str:
        return _localized(self.scope, language)

    def localized_limitation(self, language: str) -> str:
        return _localized(self.limitation, language)


@dataclass(frozen=True, slots=True)
class DomainProfile:
    id: str
    label: dict[str, str]
    description: dict[str, str]
    recommended_sources: tuple[str, ...]
    optional_sources: tuple[str, ...]

    def localized_label(self, language: str) -> str:
        return _localized(self.label, language)

    def localized_description(self, language: str) -> str:
        return _localized(self.description, language)


@dataclass(frozen=True, slots=True)
class SourceCatalog:
    sources: dict[str, SourceDefinition]
    profiles: dict[str, DomainProfile]

    def available_source_ids(self) -> list[str]:
        return [source_id for source_id, source in self.sources.items() if source.available]

    def recommended_sources(self, profile_id: str) -> list[str]:
        profile = self.profiles.get(profile_id) or self.profiles["general"]
        return [
            source_id
            for source_id in profile.recommended_sources
            if self.sources.get(source_id) and self.sources[source_id].available
        ]


def load_source_catalog(
    source_path: Path = DEFAULT_SOURCE_CATALOG,
    profile_path: Path = DEFAULT_DOMAIN_PROFILES,
) -> SourceCatalog:
    source_raw = _load_yaml(source_path).get("sources", {})
    profile_raw = _load_yaml(profile_path).get("profiles", {})
    sources = {
        str(source_id): SourceDefinition(
            id=str(source_id),
            label=str(value.get("label") or source_id),
            status=str(value.get("status") or "planned"),
            record_types=tuple(str(item) for item in value.get("record_types", [])),
            capabilities=tuple(str(item) for item in value.get("capabilities", [])),
            requires_key=bool(value.get("requires_key", False)),
            scope=_localized_map(value.get("scope")),
            limitation=_localized_map(value.get("limitation")),
        )
        for source_id, value in source_raw.items()
        if isinstance(value, dict)
    }
    profiles = {
        str(profile_id): DomainProfile(
            id=str(profile_id),
            label=_localized_map(value.get("label")),
            description=_localized_map(value.get("description")),
            recommended_sources=tuple(
                str(item) for item in value.get("recommended_sources", [])
            ),
            optional_sources=tuple(str(item) for item in value.get("optional_sources", [])),
        )
        for profile_id, value in profile_raw.items()
        if isinstance(value, dict)
    }
    if not sources or not profiles:
        raise ValueError("The source catalog and domain profiles must not be empty.")
    unknown = {
        source_id
        for profile in profiles.values()
        for source_id in (*profile.recommended_sources, *profile.optional_sources)
        if source_id not in sources
    }
    if unknown:
        raise ValueError(f"Domain profiles reference unknown sources: {sorted(unknown)}")
    return SourceCatalog(sources=sources, profiles=profiles)


def _localized(values: dict[str, str], language: str) -> str:
    return values.get(language) or values.get("en") or next(iter(values.values()), "")


def _localized_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {"en": str(value or "")}
    return {str(language): str(text) for language, text in value.items()}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value
