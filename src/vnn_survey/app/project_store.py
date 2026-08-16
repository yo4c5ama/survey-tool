from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECTS_ROOT = REPO_ROOT / "data" / "app_projects"
DEFAULT_SECRETS_ROOT = REPO_ROOT / ".secrets" / "app_projects"


@dataclass(slots=True)
class KeywordGroup:
    name: str
    terms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KeywordGroup:
        return cls(
            name=str(value.get("name") or "group"),
            terms=[str(term) for term in value.get("terms", []) if str(term).strip()],
        )


@dataclass(slots=True)
class ProjectSettings:
    name: str
    slug: str
    research_question: str
    scope_description: str
    year_start: int
    year_end: int
    keyword_groups: list[KeywordGroup]
    research_domain: str = "computer_science"
    discovery_sources: list[str] = field(default_factory=lambda: ["dblp"])
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    title_exclude_terms: list[str] = field(default_factory=list)
    include_arxiv: bool = True
    include_informal: bool = True
    llm_model: str = "gpt-5.4-mini"
    title_screening_model: str = "gpt-5.4-mini"
    prompt_refinement_model: str = "gpt-5.4-mini"
    prompt_replay_model: str = "gpt-5.4-mini"
    paper_qa_model: str = "gpt-5.4"
    corpus_analysis_model: str = "gpt-5.4"
    llm_base_url: str = "https://api.openai.com/v1"
    abstract_providers: list[str] = field(
        default_factory=lambda: [
            "arxiv",
            "pubmed",
            "crossref",
            "semantic_scholar",
            "openalex",
        ]
    )
    abstract_batch_size: int = 100
    llm_screen_batch_size: int = 20
    scholarly_api_email: str = ""
    created_at: str = ""
    updated_at: str = ""
    current_run_id: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProjectSettings:
        return cls(
            name=str(value.get("name") or "Untitled survey"),
            slug=str(value.get("slug") or "untitled-survey"),
            research_question=str(value.get("research_question") or ""),
            scope_description=str(value.get("scope_description") or ""),
            year_start=int(value.get("year_start") or 2000),
            year_end=int(value.get("year_end") or datetime.now().year),
            keyword_groups=[
                KeywordGroup.from_dict(item)
                for item in value.get("keyword_groups", [])
                if isinstance(item, dict)
            ],
            research_domain=str(value.get("research_domain") or "computer_science"),
            discovery_sources=[str(item) for item in value.get("discovery_sources", ["dblp"])]
            or ["dblp"],
            inclusion_criteria=[str(item) for item in value.get("inclusion_criteria", [])],
            exclusion_criteria=[str(item) for item in value.get("exclusion_criteria", [])],
            title_exclude_terms=[str(item) for item in value.get("title_exclude_terms", [])],
            include_arxiv=bool(value.get("include_arxiv", True)),
            include_informal=bool(value.get("include_informal", True)),
            llm_model=str(value.get("llm_model") or "gpt-5.4-mini"),
            title_screening_model=str(
                value.get("title_screening_model") or value.get("llm_model") or "gpt-5.4-mini"
            ),
            prompt_refinement_model=str(
                value.get("prompt_refinement_model") or value.get("llm_model") or "gpt-5.4-mini"
            ),
            prompt_replay_model=str(
                value.get("prompt_replay_model") or value.get("llm_model") or "gpt-5.4-mini"
            ),
            paper_qa_model=str(value.get("paper_qa_model") or value.get("llm_model") or "gpt-5.4"),
            corpus_analysis_model=str(
                value.get("corpus_analysis_model") or value.get("llm_model") or "gpt-5.4"
            ),
            llm_base_url=str(value.get("llm_base_url") or "https://api.openai.com/v1"),
            abstract_providers=[
                str(item)
                for item in value.get(
                    "abstract_providers",
                    ["arxiv", "pubmed", "crossref", "semantic_scholar", "openalex"],
                )
            ],
            abstract_batch_size=max(1, int(value.get("abstract_batch_size") or 100)),
            llm_screen_batch_size=max(
                1,
                min(50, int(value.get("llm_screen_batch_size") or 20)),
            ),
            scholarly_api_email=str(value.get("scholarly_api_email") or ""),
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
            current_run_id=str(value.get("current_run_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectStore:
    def __init__(
        self,
        root: Path | None = None,
        secrets_root: Path | None = None,
    ) -> None:
        self.root = Path(root or os.environ.get("VNN_SURVEY_APP_DATA", DEFAULT_PROJECTS_ROOT))
        self.secrets_root = Path(
            secrets_root or os.environ.get("VNN_SURVEY_APP_SECRETS", DEFAULT_SECRETS_ROOT)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.secrets_root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[ProjectSettings]:
        projects: list[ProjectSettings] = []
        for metadata_path in self.root.glob("*/project.json"):
            try:
                projects.append(ProjectSettings.from_dict(_read_json(metadata_path)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def create_project(
        self,
        *,
        name: str,
        research_question: str,
        scope_description: str,
        year_start: int,
        year_end: int,
        keyword_groups: list[KeywordGroup],
        research_domain: str = "computer_science",
        discovery_sources: list[str] | None = None,
        inclusion_criteria: list[str] | None = None,
        exclusion_criteria: list[str] | None = None,
        title_exclude_terms: list[str] | None = None,
        include_arxiv: bool = True,
        include_informal: bool = True,
        llm_model: str = "gpt-5.4-mini",
        title_screening_model: str = "gpt-5.4-mini",
        prompt_refinement_model: str = "gpt-5.4-mini",
        prompt_replay_model: str = "gpt-5.4-mini",
        paper_qa_model: str = "gpt-5.4",
        corpus_analysis_model: str = "gpt-5.4",
        llm_base_url: str = "https://api.openai.com/v1",
    ) -> ProjectSettings:
        _validate_project_input(
            name,
            year_start,
            year_end,
            keyword_groups,
            discovery_sources or ["dblp"],
        )
        slug = self._available_slug(_slugify(name))
        now = datetime.now().isoformat(timespec="seconds")
        settings = ProjectSettings(
            name=name.strip(),
            slug=slug,
            research_question=research_question.strip(),
            scope_description=scope_description.strip(),
            year_start=year_start,
            year_end=year_end,
            keyword_groups=keyword_groups,
            research_domain=research_domain.strip() or "custom",
            discovery_sources=discovery_sources or ["dblp"],
            inclusion_criteria=inclusion_criteria or [],
            exclusion_criteria=exclusion_criteria or [],
            title_exclude_terms=title_exclude_terms or [],
            include_arxiv=include_arxiv,
            include_informal=include_informal,
            llm_model=llm_model.strip(),
            title_screening_model=title_screening_model.strip(),
            prompt_refinement_model=prompt_refinement_model.strip(),
            prompt_replay_model=prompt_replay_model.strip(),
            paper_qa_model=paper_qa_model.strip(),
            corpus_analysis_model=corpus_analysis_model.strip(),
            llm_base_url=llm_base_url.strip(),
            created_at=now,
            updated_at=now,
        )
        project_dir = self.project_dir(slug)
        for directory in [
            "configs/prompts",
            "runs",
            "audits",
            "exports",
            "cache",
            "seeds",
            "manual",
            "papers",
            "conversations",
            "analysis",
        ]:
            (project_dir / directory).mkdir(parents=True, exist_ok=True)
        self.save_project(settings)
        return settings

    def load_project(self, slug: str) -> ProjectSettings:
        path = self.project_dir(slug) / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"Survey project does not exist: {slug}")
        value = _read_json(path)
        settings = ProjectSettings.from_dict(value)
        if any(
            field not in value
            for field in (
                "abstract_providers",
                "abstract_batch_size",
                "llm_screen_batch_size",
                "scholarly_api_email",
                "title_screening_model",
                "prompt_refinement_model",
                "prompt_replay_model",
            )
        ):
            _write_json(path, settings.to_dict())
            self._write_project_configs(settings)
        return settings

    def save_project(self, settings: ProjectSettings) -> None:
        _validate_project_input(
            settings.name,
            settings.year_start,
            settings.year_end,
            settings.keyword_groups,
            settings.discovery_sources,
        )
        settings.updated_at = datetime.now().isoformat(timespec="seconds")
        project_dir = self.project_dir(settings.slug)
        project_dir.mkdir(parents=True, exist_ok=True)
        _write_json(project_dir / "project.json", settings.to_dict())
        self._write_project_configs(settings)

    def set_current_run(self, slug: str, run_id: str) -> ProjectSettings:
        settings = self.load_project(slug)
        settings.current_run_id = run_id
        self.save_project(settings)
        return settings

    def project_dir(self, slug: str) -> Path:
        return self.root / slug

    def config_path(self, slug: str) -> Path:
        return self.project_dir(slug) / "configs" / "pipeline.yaml"

    def system_prompt_path(self, slug: str) -> Path:
        return self.project_dir(slug) / "configs" / "prompts" / "llm_screening_system.txt"

    def user_prompt_path(self, slug: str) -> Path:
        return self.project_dir(slug) / "configs" / "prompts" / "llm_screening_user.txt"

    def api_key_path(self, slug: str) -> Path:
        return self.secrets_root / slug / "openai_api_key"

    def openalex_api_key_path(self, slug: str) -> Path:
        return self.secrets_root / slug / "openalex_api_key"

    def semantic_scholar_api_key_path(self, slug: str) -> Path:
        return self.secrets_root / slug / "semantic_scholar_api_key"

    def ncbi_api_key_path(self, slug: str) -> Path:
        return self.secrets_root / slug / "ncbi_api_key"

    def save_api_key(self, slug: str, api_key: str) -> Path:
        value = api_key.strip()
        if not value:
            raise ValueError("The API key cannot be empty.")
        path = self.api_key_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        return path

    def has_api_key(self, slug: str) -> bool:
        path = self.api_key_path(slug)
        return path.exists() and bool(path.read_text(encoding="utf-8").strip())

    def read_api_key(self, slug: str) -> str:
        try:
            return self.api_key_path(slug).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def save_openalex_api_key(self, slug: str, api_key: str) -> Path:
        value = api_key.strip()
        if not value:
            raise ValueError("The OpenAlex API key cannot be empty.")
        path = self.openalex_api_key_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        return path

    def read_openalex_api_key(self, slug: str) -> str:
        try:
            return self.openalex_api_key_path(slug).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def save_semantic_scholar_api_key(self, slug: str, api_key: str) -> Path:
        return self._save_provider_key(
            self.semantic_scholar_api_key_path(slug),
            api_key,
            "Semantic Scholar",
        )

    def read_semantic_scholar_api_key(self, slug: str) -> str:
        return self._read_provider_key(self.semantic_scholar_api_key_path(slug))

    def save_ncbi_api_key(self, slug: str, api_key: str) -> Path:
        return self._save_provider_key(self.ncbi_api_key_path(slug), api_key, "NCBI")

    def read_ncbi_api_key(self, slug: str) -> str:
        return self._read_provider_key(self.ncbi_api_key_path(slug))

    @staticmethod
    def _save_provider_key(path: Path, api_key: str, provider: str) -> Path:
        value = api_key.strip()
        if not value:
            raise ValueError(f"The {provider} API key cannot be empty.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        return path

    @staticmethod
    def _read_provider_key(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def save_system_prompt(self, slug: str, value: str) -> None:
        prompt = value.strip()
        if not prompt:
            raise ValueError("The system prompt cannot be empty.")
        self.system_prompt_path(slug).write_text(prompt + "\n", encoding="utf-8")

    def reset_system_prompt(self, slug: str) -> str:
        settings = self.load_project(slug)
        prompt = _build_system_prompt(settings)
        self.system_prompt_path(slug).write_text(prompt + "\n", encoding="utf-8")
        return prompt

    def delete_project(self, slug: str) -> None:
        shutil.rmtree(self.project_dir(slug), ignore_errors=True)
        shutil.rmtree(self.secrets_root / slug, ignore_errors=True)

    def refresh_project_configs(self, slug: str) -> ProjectSettings:
        """Regenerate machine-local paths without changing project metadata."""

        settings = self.load_project(slug)
        self._write_project_configs(settings)
        return settings

    def _available_slug(self, base: str) -> str:
        candidate = base
        counter = 2
        while self.project_dir(candidate).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _write_project_configs(self, settings: ProjectSettings) -> None:
        project_dir = self.project_dir(settings.slug).resolve()
        configs_dir = project_dir / "configs"
        prompts_dir = configs_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        term_sets: dict[str, list[str]] = {}
        block_groups: dict[str, str] = {}
        for index, group in enumerate(settings.keyword_groups, start=1):
            key = _config_key(group.name, fallback=f"group_{index}")
            while key in term_sets:
                key = f"{key}_{index}"
            term_sets[key] = _dedupe_terms(group.terms)
            block_groups[key] = key

        survey_inputs = {
            "years": {"start": settings.year_start, "end": settings.year_end},
            "discovery": {"sources": _dedupe_terms(settings.discovery_sources)},
            "query": {
                "strategy": "grouped",
                "term_sets": term_sets,
                "logical_groups": [{"name": "project_scope", "groups": block_groups}],
            },
            "filters": {
                "include_corr": settings.include_arxiv,
                "include_informal": settings.include_informal,
            },
            "screening": {
                "profile": "generic",
                "exclude_llm_as_verification_tool": False,
                "exclude_terms": _dedupe_terms(settings.title_exclude_terms),
            },
        }
        _write_yaml(configs_dir / "survey_inputs.yaml", survey_inputs)

        key_path = self.api_key_path(settings.slug).resolve()
        pipeline_config = {
            "user_config": "survey_inputs.yaml",
            "discovery": {
                "sources": _dedupe_terms(settings.discovery_sources),
                "cache_dir": str(project_dir / "cache" / "discovery"),
                "results_per_page": 100,
                "max_pages_per_query": 2,
                "request_delay_seconds": 0.25,
                "arxiv_request_delay_seconds": 3.0,
                "timeout_seconds": 30,
                "retries": 3,
                "crossref_email_env": "CROSSREF_EMAIL",
                "openalex_api_key_env": "OPENALEX_API_KEY",
                "openalex_email_env": "OPENALEX_EMAIL",
                "pubmed_api_key_env": "NCBI_API_KEY",
                "pubmed_email_env": "NCBI_EMAIL",
            },
            "dblp": {
                "cache_dir": str(project_dir / "cache" / "dblp"),
                "hits_per_page": 100,
                "max_pages_per_query": 10,
                "request_delay_seconds": 1.0,
                "timeout_seconds": 30,
                "retries": 3,
            },
            "enrichment": {
                "providers": settings.abstract_providers,
                "cache_dir": str(project_dir / "cache" / "abstracts"),
                "batch_size": settings.abstract_batch_size,
                "request_delay_seconds": 0.2,
                "arxiv_request_delay_seconds": 3.0,
                "timeout_seconds": 30,
                "retries": 3,
                "min_title_similarity": 0.86,
                "crossref_email_env": "CROSSREF_EMAIL",
                "semantic_scholar_api_key_env": "SEMANTIC_SCHOLAR_API_KEY",
                "openalex_api_key_env": "OPENALEX_API_KEY",
                "openalex_email_env": "OPENALEX_EMAIL",
                "pubmed_api_key_env": "NCBI_API_KEY",
                "pubmed_email_env": "NCBI_EMAIL",
            },
            "venue_quality": {
                "core_rankings_path": str(
                    (REPO_ROOT / "data/venue_quality/core_rankings.csv").resolve()
                ),
                "journal_impact_factors_path": str(
                    (REPO_ROOT / "data/venue_quality/journal_impact_factors.csv").resolve()
                ),
                "core_online_enabled": True,
                "core_cache_dir": str(project_dir / "cache" / "core_rankings"),
                "core_request_delay_seconds": 0.5,
                "core_timeout_seconds": 30,
            },
            "snowballing": {
                "seed_papers_path": str(project_dir / "seeds" / "current.yaml"),
                "cache_dir": str(project_dir / "cache" / "snowballing"),
                "cache_ttl_hours": 24,
                "request_delay_seconds": 0.2,
                "timeout_seconds": 30,
                "retries": 3,
                "max_backward_per_seed": 0,
                "max_forward_per_seed": 0,
                "include_seed_papers": True,
                "providers": ["semantic_scholar", "opencitations"],
                "provider_strategy": "merge",
                "semantic_scholar_api_key_env": "SEMANTIC_SCHOLAR_API_KEY",
                "opencitations_access_token_env": "OPENCITATIONS_ACCESS_TOKEN",
                "openalex_api_key_env": "OPENALEX_API_KEY",
                "openalex_email_env": "OPENALEX_EMAIL",
            },
            "llm_screening": {
                "model": settings.llm_model,
                "api_key_env": "OPENAI_API_KEY",
                "api_key_file": str(key_path),
                "base_url": settings.llm_base_url,
                "cache_dir": str(project_dir / "cache" / "llm_screening"),
                "request_delay_seconds": 0.2,
                "timeout_seconds": 60,
                "retries": 3,
                "batch_size": settings.llm_screen_batch_size,
                "max_output_tokens": 800,
                "prompt_version": f"surveyflow-{settings.slug}-v1",
                "system_prompt_path": str(self.system_prompt_path(settings.slug).resolve()),
                "user_prompt_template_path": str(self.user_prompt_path(settings.slug).resolve()),
                "include_decisions": ["include_candidate", "needs_review"],
            },
        }
        _write_yaml(configs_dir / "pipeline.yaml", pipeline_config)

        if not self.system_prompt_path(settings.slug).exists():
            self.system_prompt_path(settings.slug).write_text(
                _build_system_prompt(settings) + "\n",
                encoding="utf-8",
            )
        if not self.user_prompt_path(settings.slug).exists():
            self.user_prompt_path(settings.slug).write_text(
                "\n".join(
                    [
                        "title: {title}",
                        "year: {year}",
                        "venue: {venue}",
                        "doi: {doi}",
                        "abstract: {abstract}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )


def _build_system_prompt(settings: ProjectSettings) -> str:
    inclusion = _criteria_text(settings.inclusion_criteria, "No additional criteria supplied.")
    exclusion = _criteria_text(settings.exclusion_criteria, "No additional criteria supplied.")
    return f"""You are screening academic papers for a systematic literature survey.

Use only the supplied title, abstract, and metadata. Do not invent evidence.

Research question:
{settings.research_question or "Not specified."}

Survey scope:
{settings.scope_description or "Use the research question and criteria below."}

Inclusion criteria:
{inclusion}

Exclusion criteria:
{exclusion}

Scope labels:
- Use \"in_scope\" when the paper clearly satisfies the survey scope.
- Use \"related\" when it is useful context but not a direct answer to the research question.
- Use \"out_of_scope\" when it clearly fails the scope or meets an exclusion criterion.
- Use \"insufficient_information\" when the title and abstract are not enough for a stable judgment.

Decision policy:
- Return \"include\" for clearly in-scope papers.
- Return \"maybe\" for related, ambiguous, or insufficiently described papers.
- Return \"exclude\" for clearly out-of-scope papers.
- Give a confidence from 0 to 1.
- Give a concise reason and concrete evidence from the supplied text.

Every final inclusion or exclusion will be checked by a human reviewer."""


def _validate_project_input(
    name: str,
    year_start: int,
    year_end: int,
    keyword_groups: list[KeywordGroup],
    discovery_sources: list[str] | None = None,
) -> None:
    if not name.strip():
        raise ValueError("Project name is required.")
    if year_start > year_end:
        raise ValueError("The start year must not be later than the end year.")
    valid_groups = [group for group in keyword_groups if group.name.strip() and group.terms]
    if not valid_groups:
        raise ValueError("Add at least one non-empty keyword group.")
    if not discovery_sources:
        raise ValueError("Select at least one available literature source.")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "survey-project"


def _config_key(value: str, fallback: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return key or fallback


def _dedupe_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _criteria_text(values: list[str], empty: str) -> str:
    items = [item.strip() for item in values if item.strip()]
    return "\n".join(f"- {item}" for item in items) if items else empty


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
