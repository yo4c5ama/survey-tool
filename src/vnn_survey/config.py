from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class YearRange:
    start: int | None = None
    end: int | None = None

    def contains(self, year: int | None) -> bool:
        if year is None:
            return True
        if self.start is not None and year < self.start:
            return False
        return not (self.end is not None and year > self.end)


@dataclass(frozen=True, slots=True)
class DblpConfig:
    cache_dir: Path | None = None
    hits_per_page: int = 100
    max_pages_per_query: int = 10
    request_delay_seconds: float = 1.0
    timeout_seconds: int = 30
    retries: int = 3


@dataclass(frozen=True, slots=True)
class FilterConfig:
    include_corr: bool = True
    include_informal: bool = True


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    profile: str = "transformer_verification"
    exclude_llm_as_verification_tool: bool = True
    exclude_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EnrichmentConfig:
    providers: list[str]
    cache_dir: Path | None = None
    request_delay_seconds: float = 0.2
    timeout_seconds: int = 30
    retries: int = 3
    min_title_similarity: float = 0.86
    semantic_scholar_api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    openalex_api_key_env: str = "OPENALEX_API_KEY"
    openalex_email_env: str = "OPENALEX_EMAIL"


@dataclass(frozen=True, slots=True)
class VenueQualityConfig:
    core_rankings_path: Path | None = None
    journal_impact_factors_path: Path | None = None
    core_online_enabled: bool = True
    core_cache_dir: Path | None = None
    core_request_delay_seconds: float = 0.5
    core_timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class SnowballingConfig:
    seed_papers_path: Path | None = None
    cache_dir: Path | None = None
    request_delay_seconds: float = 0.2
    timeout_seconds: int = 30
    retries: int = 3
    max_backward_per_seed: int = 30
    max_forward_per_seed: int = 30
    include_seed_papers: bool = True
    openalex_api_key_env: str = "OPENALEX_API_KEY"
    openalex_email_env: str = "OPENALEX_EMAIL"


@dataclass(frozen=True, slots=True)
class LlmScreeningConfig:
    model: str = "gpt-5.5"
    api_key_env: str = "OPENAI_API_KEY"
    api_key_file: Path | None = None
    base_url: str = "https://api.openai.com/v1"
    cache_dir: Path | None = None
    request_delay_seconds: float = 0.2
    timeout_seconds: int = 60
    retries: int = 3
    max_output_tokens: int = 800
    prompt_version: str = "transformer-verification-screen-v1"
    system_prompt_path: Path | None = None
    user_prompt_template_path: Path | None = None
    include_decisions: list[str] = field(
        default_factory=lambda: ["include_candidate", "needs_review"]
    )


@dataclass(frozen=True, slots=True)
class LogicalQueryBlock:
    name: str
    groups: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    years: YearRange
    query_strategy: str
    term_sets: dict[str, list[str]]
    model_terms: list[str]
    verification_terms: list[str]
    logical_groups: list[LogicalQueryBlock]
    logical_queries: list[str]
    extra_queries: list[str]
    dblp: DblpConfig
    filters: FilterConfig
    screening: ScreeningConfig
    enrichment: EnrichmentConfig
    venue_quality: VenueQualityConfig
    snowballing: SnowballingConfig
    llm_screening: LlmScreeningConfig

    def build_queries(self) -> list[str]:
        if self.query_strategy == "grouped":
            queries = (
                self._grouped_queries()
                or list(self.logical_queries)
                or self._cartesian_queries()
            )
        elif self.query_strategy == "logical":
            queries = (
                list(self.logical_queries)
                or self._grouped_queries()
                or self._cartesian_queries()
            )
        elif self.query_strategy == "cartesian":
            queries = self._cartesian_queries()
        elif self.query_strategy == "all":
            queries = [
                *self._grouped_queries(),
                *self.logical_queries,
                *self._cartesian_queries(),
            ]
        else:
            raise ValueError(
                "query.strategy must be one of: grouped, logical, cartesian, all "
                f"(got {self.query_strategy!r})"
            )
        queries.extend(self.extra_queries)
        return dedupe_preserving_order(queries)

    def _cartesian_queries(self) -> list[str]:
        return [
            f"{model} {verification}"
            for model in self.model_terms
            for verification in self.verification_terms
        ]

    def _grouped_queries(self) -> list[str]:
        queries: list[str] = []
        for block in self.logical_groups:
            group_alternatives = [
                alternatives
                for alternatives in (_compile_group_terms(terms) for terms in block.groups.values())
                if alternatives
            ]
            if not group_alternatives:
                continue
            queries.extend(" ".join(parts) for parts in product(*group_alternatives))
        return queries


def load_config(path: Path) -> SurveyConfig:
    raw = _load_config_dict(path)

    query = raw.get("query", {})
    years = raw.get("years", {})
    dblp = raw.get("dblp", {})
    filters = raw.get("filters", {})
    screening = raw.get("screening", {})
    enrichment = raw.get("enrichment", {})
    venue_quality = raw.get("venue_quality", {})
    snowballing = raw.get("snowballing", {})
    llm_screening = raw.get("llm_screening", {})
    term_sets = _parse_term_sets(query.get("term_sets", {}))

    return SurveyConfig(
        years=YearRange(start=years.get("start"), end=years.get("end")),
        query_strategy=str(query.get("strategy", "cartesian")).strip().lower(),
        term_sets=term_sets,
        model_terms=list(query.get("model_terms", [])),
        verification_terms=list(query.get("verification_terms", [])),
        logical_groups=_parse_logical_groups(
            raw_blocks=query.get("logical_groups", []),
            term_sets=term_sets,
        ),
        logical_queries=list(query.get("logical_queries", [])),
        extra_queries=list(query.get("extra_queries", [])),
        dblp=DblpConfig(
            cache_dir=Path(dblp["cache_dir"]) if dblp.get("cache_dir") else None,
            hits_per_page=int(dblp.get("hits_per_page", 100)),
            max_pages_per_query=int(dblp.get("max_pages_per_query", 10)),
            request_delay_seconds=float(dblp.get("request_delay_seconds", 1.0)),
            timeout_seconds=int(dblp.get("timeout_seconds", 30)),
            retries=int(dblp.get("retries", 3)),
        ),
        filters=FilterConfig(
            include_corr=bool(filters.get("include_corr", True)),
            include_informal=bool(filters.get("include_informal", True)),
        ),
        screening=ScreeningConfig(
            profile=str(screening.get("profile", "transformer_verification")).strip().lower(),
            exclude_llm_as_verification_tool=bool(
                screening.get("exclude_llm_as_verification_tool", True)
            ),
            exclude_terms=[str(term) for term in screening.get("exclude_terms", [])],
        ),
        enrichment=EnrichmentConfig(
            providers=list(enrichment.get("providers", ["openalex"])),
            cache_dir=Path(enrichment["cache_dir"]) if enrichment.get("cache_dir") else None,
            request_delay_seconds=float(enrichment.get("request_delay_seconds", 0.2)),
            timeout_seconds=int(enrichment.get("timeout_seconds", 30)),
            retries=int(enrichment.get("retries", 3)),
            min_title_similarity=float(enrichment.get("min_title_similarity", 0.86)),
            semantic_scholar_api_key_env=str(
                enrichment.get(
                    "semantic_scholar_api_key_env",
                    "SEMANTIC_SCHOLAR_API_KEY",
                )
            ),
            openalex_api_key_env=str(enrichment.get("openalex_api_key_env", "OPENALEX_API_KEY")),
            openalex_email_env=str(enrichment.get("openalex_email_env", "OPENALEX_EMAIL")),
        ),
        venue_quality=VenueQualityConfig(
            core_rankings_path=Path(venue_quality["core_rankings_path"])
            if venue_quality.get("core_rankings_path")
            else None,
            journal_impact_factors_path=Path(venue_quality["journal_impact_factors_path"])
            if venue_quality.get("journal_impact_factors_path")
            else None,
            core_online_enabled=bool(venue_quality.get("core_online_enabled", True)),
            core_cache_dir=Path(venue_quality["core_cache_dir"])
            if venue_quality.get("core_cache_dir")
            else None,
            core_request_delay_seconds=float(
                venue_quality.get("core_request_delay_seconds", 0.5)
            ),
            core_timeout_seconds=int(venue_quality.get("core_timeout_seconds", 30)),
        ),
        snowballing=SnowballingConfig(
            seed_papers_path=Path(snowballing["seed_papers_path"])
            if snowballing.get("seed_papers_path")
            else None,
            cache_dir=Path(snowballing["cache_dir"]) if snowballing.get("cache_dir") else None,
            request_delay_seconds=float(snowballing.get("request_delay_seconds", 0.2)),
            timeout_seconds=int(snowballing.get("timeout_seconds", 30)),
            retries=int(snowballing.get("retries", 3)),
            max_backward_per_seed=int(snowballing.get("max_backward_per_seed", 30)),
            max_forward_per_seed=int(snowballing.get("max_forward_per_seed", 30)),
            include_seed_papers=bool(snowballing.get("include_seed_papers", True)),
            openalex_api_key_env=str(
                snowballing.get("openalex_api_key_env", "OPENALEX_API_KEY")
            ),
            openalex_email_env=str(snowballing.get("openalex_email_env", "OPENALEX_EMAIL")),
        ),
        llm_screening=LlmScreeningConfig(
            model=str(llm_screening.get("model", "gpt-5.5")),
            api_key_env=str(llm_screening.get("api_key_env", "OPENAI_API_KEY")),
            api_key_file=Path(llm_screening["api_key_file"])
            if llm_screening.get("api_key_file")
            else None,
            base_url=str(llm_screening.get("base_url", "https://api.openai.com/v1")),
            cache_dir=Path(llm_screening["cache_dir"])
            if llm_screening.get("cache_dir")
            else None,
            request_delay_seconds=float(llm_screening.get("request_delay_seconds", 0.2)),
            timeout_seconds=int(llm_screening.get("timeout_seconds", 60)),
            retries=int(llm_screening.get("retries", 3)),
            max_output_tokens=int(llm_screening.get("max_output_tokens", 800)),
            prompt_version=str(
                llm_screening.get(
                    "prompt_version",
                    "transformer-verification-screen-v1",
                )
            ),
            system_prompt_path=Path(llm_screening["system_prompt_path"])
            if llm_screening.get("system_prompt_path")
            else None,
            user_prompt_template_path=Path(llm_screening["user_prompt_template_path"])
            if llm_screening.get("user_prompt_template_path")
            else None,
            include_decisions=list(
                llm_screening.get(
                    "include_decisions",
                    ["include_candidate", "needs_review"],
                )
            ),
        ),
    )


def _parse_term_sets(raw_term_sets: Any) -> dict[str, list[str]]:
    if not isinstance(raw_term_sets, dict):
        return {}
    return {
        str(name): [str(term) for term in terms]
        for name, terms in raw_term_sets.items()
        if isinstance(terms, list)
    }


def _parse_logical_groups(
    raw_blocks: Any,
    term_sets: dict[str, list[str]],
) -> list[LogicalQueryBlock]:
    if not isinstance(raw_blocks, list):
        return []

    blocks: list[LogicalQueryBlock] = []
    for index, raw_block in enumerate(raw_blocks, start=1):
        if not isinstance(raw_block, dict):
            continue
        raw_groups = raw_block.get("groups", {})
        if not isinstance(raw_groups, dict):
            continue
        groups = _parse_group_terms(raw_groups=raw_groups, term_sets=term_sets)
        if groups:
            blocks.append(
                LogicalQueryBlock(
                    name=str(raw_block.get("name") or f"group_{index}"),
                    groups=groups,
                )
            )
    return blocks


def _parse_group_terms(
    raw_groups: dict[Any, Any],
    term_sets: dict[str, list[str]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for group_name, raw_terms in raw_groups.items():
        terms = _resolve_group_terms(raw_terms=raw_terms, term_sets=term_sets)
        if terms:
            groups[str(group_name)] = terms
    return groups


def _resolve_group_terms(raw_terms: Any, term_sets: dict[str, list[str]]) -> list[str]:
    if isinstance(raw_terms, str):
        return list(term_sets.get(raw_terms, [raw_terms]))
    if isinstance(raw_terms, list):
        return [str(term) for term in raw_terms]
    return []


def _compile_group_terms(terms: list[str]) -> list[str]:
    single_word_terms: list[str] = []
    multi_word_terms: list[str] = []

    for term in terms:
        normalized = _normalize_query_term(term)
        if not normalized:
            continue
        if _query_word_count(normalized) == 1:
            single_word_terms.append(normalized)
        else:
            multi_word_terms.append(normalized)

    alternatives: list[str] = []
    if single_word_terms:
        alternatives.append("|".join(dedupe_preserving_order(single_word_terms)))
    alternatives.extend(multi_word_terms)
    return dedupe_preserving_order(alternatives)


def _normalize_query_term(value: str) -> str:
    return " ".join(str(value).split())


def _query_word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", value))


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


def _load_config_dict(path: Path) -> dict[str, Any]:
    raw = _load_yaml_dict(path)
    user_config = raw.pop("user_config", None)
    if not user_config:
        return raw

    user_paths = user_config if isinstance(user_config, list) else [user_config]
    merged = raw
    for user_path in user_paths:
        if not isinstance(user_path, str):
            raise ValueError("user_config must be a path string or a list of path strings.")
        merged = _deep_merge(
            merged,
            _load_yaml_dict(_resolve_config_reference(path.parent, user_path)),
        )
    merged.pop("user_config", None)
    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return raw


def _resolve_config_reference(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged
