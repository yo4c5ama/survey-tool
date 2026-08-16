from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from shutil import copy2
from threading import get_ident
from typing import Any

import requests
from rich.console import Console

from vnn_survey.ai_research import CorpusAnalyzer, OpenAIResearchClient, load_csv_rows
from vnn_survey.app.audit import (
    AuditSummary,
    build_cumulative_audit,
    create_audit_queue,
    create_manual_recommendations,
    load_audit,
    paper_key,
    paper_keys,
    read_csv,
    summarize_audit,
    update_audit_rows,
)
from vnn_survey.app.audit import write_csv as write_audit_csv
from vnn_survey.app.manual_papers import ManualPaperStore
from vnn_survey.app.project_store import ProjectStore
from vnn_survey.app.run_flow import build_flow_svg, flow_summary_payload, record_flow_stage
from vnn_survey.app.task_manager import TaskCancelled, raise_if_cancelled
from vnn_survey.config import SurveyConfig, load_config
from vnn_survey.enrichment import enrich_candidates, write_enrichment_summary
from vnn_survey.export import write_csv, write_jsonl
from vnn_survey.llm_screening import llm_screen_candidates, write_llm_screening_summary
from vnn_survey.llm_summary import summarize_llm_screening
from vnn_survey.manual_audit import filter_manual_includes
from vnn_survey.model_pricing import (
    estimate_token_cost,
    get_model_price,
    is_openai_api_base_url,
    output_tokens_per_paper,
    refresh_openai_model_price,
)
from vnn_survey.models import PaperRecord
from vnn_survey.pipeline import (
    collect_from_sources,
    dedupe_records,
    save_collection,
    summarize,
)
from vnn_survey.prompt_refinement import (
    generate_prompt_refinement as build_prompt_refinement,
)
from vnn_survey.prompt_refinement import load_prompt_refinement
from vnn_survey.screening import screen_candidates
from vnn_survey.snowballing import (
    SnowballingResult,
    export_seed_papers_from_csv,
    snowball_candidates,
    strip_per_paper_citation_diagnostics,
    write_seed_coverage_report,
    write_snowballing_summary,
)
from vnn_survey.title_screening import (
    TitleScreeningResult,
    screen_titles_with_llm,
    write_title_screening_summary,
)
from vnn_survey.venue_quality import enrich_venue_quality, write_venue_quality_summary

ProgressCallback = Callable[[str, str, int | None, int | None, str], None]

INITIAL_DISCOVERY_STAGES = [
    "Literature search",
    "Rule screening",
    "Venue enrichment",
    "Abstract enrichment",
]
MANUAL_SYNC_STAGES = [
    "Manual additions",
    "Rule screening",
    "Venue enrichment",
    "Abstract enrichment",
]
MANUAL_ENRICHMENT_STAGES = [
    "Manual additions",
    "Venue enrichment",
    "Abstract enrichment",
    "AI abstract screening",
    "Return to manual review",
]
CORPUS_ANALYSIS_STAGES = [
    "Taxonomy design",
    "Paper classification",
    "Analysis report",
]
AI_REVIEW_STAGES = [
    "AI abstract screening",
    "Recommendation summary",
    "Create manual review queue",
]
MANUAL_REVIEW_STAGES = ["Prepare recommendations", "Create manual review queue"]
PROMPT_REFINEMENT_STAGES = ["Learning from manual audit", "Prompt proposal"]
SNOWBALL_STAGES = [
    "Citation snowballing",
    "Rule screening",
    "Venue enrichment",
    "Abstract enrichment",
]


class PipelineService:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def start_initial_discovery(
        self,
        project_slug: str,
        *,
        source: str = "auto",
        source_ids: list[str] | None = None,
        limit_queries: int | None = None,
        enrich_limit: int | None = None,
        core_online: bool = True,
        use_title_llm: bool = False,
        title_batch_size: int = 100,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if source not in {"auto", "api", "sparql"}:
            raise ValueError("DBLP source must be auto, api, or sparql.")
        settings = self.store.load_project(project_slug)
        config = load_config(self.store.config_path(project_slug))
        selected_sources = list(dict.fromkeys(source_ids or settings.discovery_sources))
        if not selected_sources:
            raise ValueError("Select at least one available literature source.")
        if (
            use_title_llm
            and not self.store.has_api_key(project_slug)
            and not os.environ.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("Save or provide an OpenAI API key before AI title screening.")
        run_id = _new_run_id()
        run_dir = self.store.project_dir(project_slug) / "runs" / run_id
        processed_dir = run_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "project_slug": project_slug,
            "run_id": run_id,
            "status": "running_discovery",
            "created_at": _now(),
            "updated_at": _now(),
            "source": source,
            "sources": selected_sources,
            "options": {
                "title_llm_enabled": use_title_llm,
                "title_llm_batch_size": title_batch_size,
                "limit_queries": limit_queries,
                "enrich_limit": enrich_limit,
                "core_online": core_online,
            },
            "rounds": [],
        }
        round_state = _new_round_state(index=0, kind="initial")
        state["rounds"].append(round_state)
        self._save_state(project_slug, state)
        self.store.set_current_run(settings.slug, run_id)
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Initial discovery",
            heading="Running initial discovery",
            stages=_with_title_screening_stage(INITIAL_DISCOVERY_STAGES, use_title_llm),
            callback=progress,
            paper_count=0,
        )

        try:
            _notify(
                tracked_progress,
                "Literature search",
                "Collecting bibliographic records from the selected sources.",
            )
            manual_records = ManualPaperStore(self.store.project_dir(project_slug)).load()
            result = collect_from_sources(
                config,
                console=Console(file=io.StringIO(), no_color=True),
                source_ids=selected_sources,
                limit_queries=limit_queries,
                dblp_mode=source,
                additional_records=manual_records,
                progress_callback=_counted_item_progress(
                    tracked_progress,
                    state,
                    "Literature search",
                    "Collecting bibliographic records from the selected sources.",
                ),
            )
            save_collection(result, output_dir=run_dir)
            collection_summary = summarize(result)
            _write_json(run_dir / "run_summary.json", collection_summary)
            candidates = processed_dir / "candidate_papers.csv"
            _set_progress_paper_count(state, int(collection_summary["deduped_records"]))
            round_state["files"]["candidates"] = str(candidates)
            round_state["counts"].update(collection_summary)
            raw_count = int(collection_summary["raw_records"])
            filtered_count = int(collection_summary["filtered_records"])
            deduped_count = int(collection_summary["deduped_records"])
            record_flow_stage(
                round_state,
                key="literature_search",
                label="Literature search",
                input_count=raw_count,
                retained_count=raw_count,
                stage_type="discovery",
                details={"sources": ", ".join(selected_sources)},
            )
            record_flow_stage(
                round_state,
                key="metadata_filter",
                label="Metadata filters",
                input_count=raw_count,
                retained_count=filtered_count,
            )
            record_flow_stage(
                round_state,
                key="deduplication",
                label="Deduplication",
                input_count=filtered_count,
                retained_count=deduped_count,
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying the project's title exclusion rules.",
            )
            screened_path = processed_dir / "candidate_papers_screened.csv"
            screening_result = screen_candidates(candidates, screened_path, config.screening)
            _write_json(
                processed_dir / "screening_summary.json",
                {
                    "total": screening_result.summary.total,
                    "by_decision": dict(screening_result.summary.by_decision),
                    "by_bucket": dict(screening_result.summary.by_bucket),
                    "by_exclusion_code": dict(screening_result.summary.by_exclusion_code),
                },
            )
            rule_excluded = screening_result.summary.by_decision.get("exclude", 0)
            rule_retained = screening_result.summary.total - rule_excluded
            round_state["files"]["screened"] = str(screened_path)
            round_state["counts"]["rule_excluded"] = rule_excluded
            record_flow_stage(
                round_state,
                key="rule_screening",
                label="Rule screening",
                input_count=screening_result.summary.total,
                retained_count=rule_retained,
                excluded_count=rule_excluded,
            )
            self._save_state(project_slug, state)

            title_input, title_result = self._title_prescreen(
                project_slug,
                screened_path,
                processed_dir / "candidate_papers_title_screened.csv",
                enabled=use_title_llm,
                batch_size=title_batch_size,
                progress=tracked_progress,
            )
            round_state["counts"].update(
                {
                    "deduped_records": int(collection_summary["deduped_records"]),
                    "rule_excluded": screening_result.summary.by_decision.get("exclude", 0),
                    **_title_screening_counts(title_result),
                }
            )
            if title_result is not None:
                round_state["files"]["title_screened"] = str(title_input)
                record_flow_stage(
                    round_state,
                    key="ai_title_screening",
                    label="AI title screening",
                    input_count=title_result.summary.eligible,
                    retained_count=title_result.summary.kept_for_enrichment,
                    excluded_count=title_result.summary.excluded,
                    details={"cached": title_result.summary.cached},
                )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Resolving published versions and adding venue quality metadata.",
            )
            venue_path = processed_dir / "candidate_papers_venues.csv"
            publication_resolution_path = processed_dir / "publication_resolution.json"
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            venue_result = enrich_venue_quality(
                title_input,
                venue_path,
                venue_config,
                decisions={"include_candidate", "needs_review"},
                survey_config=config,
                publication_resolution_path=publication_resolution_path,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Venue enrichment",
                    "Resolving published versions and adding venue quality metadata.",
                ),
            )
            write_venue_quality_summary(
                venue_result.summary,
                processed_dir / "venue_quality_summary.json",
            )
            enrichment_input_count = (
                title_result.summary.kept_for_enrichment
                if title_result is not None
                else rule_retained
            )
            round_state["files"]["venues"] = str(venue_path)
            round_state["files"]["publication_resolution"] = str(publication_resolution_path)
            record_flow_stage(
                round_state,
                key="venue_enrichment",
                label="Venue enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "CORE ranks found": getattr(
                        venue_result.summary,
                        "conferences_with_core_rank",
                        0,
                    ),
                    "impact factors found": getattr(
                        venue_result.summary,
                        "journals_with_impact_factor",
                        0,
                    ),
                    "preprints checked": getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "published versions resolved": getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                },
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Looking up missing abstracts through the configured provider chain.",
            )
            enriched_path = processed_dir / "candidate_papers_enriched.csv"
            enrichment_result = enrich_candidates(
                venue_path,
                enriched_path,
                config.enrichment,
                decisions={"include_candidate", "needs_review"},
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Looking up missing abstracts through the configured provider chain.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                processed_dir / "abstract_enrichment_summary.json",
            )
            round_state["files"]["enriched"] = str(enriched_path)
            record_flow_stage(
                round_state,
                key="abstract_enrichment",
                label="Abstract enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "abstracts found": enrichment_result.summary.with_abstract,
                    "attempted": enrichment_result.summary.attempted,
                    "API requests": getattr(enrichment_result.summary, "api_requests", 0),
                    "batch requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "cache hits": getattr(enrichment_result.summary, "cache_hits", 0),
                    "429 retries": getattr(
                        enrichment_result.summary,
                        "rate_limit_retries",
                        0,
                    ),
                },
            )

            round_state["status"] = "discovery_complete"
            round_state["counts"] = {
                **collection_summary,
                "manual_records": len(manual_records),
                "rule_excluded": screening_result.summary.by_decision.get("exclude", 0),
                **_title_screening_counts(title_result),
                "publication_resolution_attempted": getattr(
                    venue_result.summary,
                    "publication_resolution_attempted",
                    0,
                ),
                "published_versions_resolved": getattr(
                    venue_result.summary,
                    "published_versions_resolved",
                    0,
                ),
                "abstracts_found": enrichment_result.summary.with_abstract,
                "abstracts_attempted": enrichment_result.summary.attempted,
                "abstract_api_requests": getattr(
                    enrichment_result.summary,
                    "api_requests",
                    0,
                ),
                "abstract_batch_requests": getattr(
                    enrichment_result.summary,
                    "batch_requests",
                    0,
                ),
                "abstract_cache_hits": getattr(
                    enrichment_result.summary,
                    "cache_hits",
                    0,
                ),
                "abstract_rate_limit_retries": getattr(
                    enrichment_result.summary,
                    "rate_limit_retries",
                    0,
                ),
                "abstract_rate_limit_wait_seconds": (
                    getattr(
                        enrichment_result.summary,
                        "rate_limit_wait_seconds",
                        0,
                    )
                ),
            }
            state["status"] = "awaiting_ai_or_review"
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Discovery complete",
                message="The candidate set is ready for review preparation.",
                paper_count=int(collection_summary["deduped_records"]),
            )
            return state
        except TaskCancelled:
            self._mark_cancelled(project_slug, state, round_state)
            return state
        except Exception as exc:
            self._mark_failed(project_slug, state, round_state, exc)
            raise

    def resume_initial_discovery(
        self,
        project_slug: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Continue an interrupted initial run from its persisted stage outputs."""

        settings = self.store.load_project(project_slug)
        state = self.load_current_state(project_slug)
        round_state = _get_round(state, 0)
        if round_state.get("kind") != "initial":
            raise RuntimeError("The current run does not contain an initial discovery round.")
        if round_state.get("status") == "discovery_complete":
            return state

        options = state.get("options", {})
        source = str(state.get("source") or "auto")
        selected_sources = list(state.get("sources") or settings.discovery_sources or ["dblp"])
        limit_queries = _optional_int(options.get("limit_queries"))
        enrich_limit = _optional_int(options.get("enrich_limit"))
        core_online = bool(options.get("core_online", True))
        use_title_llm = bool(options.get("title_llm_enabled", False))
        title_batch_size = int(options.get("title_llm_batch_size", 100))
        if (
            use_title_llm
            and not self.store.has_api_key(project_slug)
            and not os.environ.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("Save or provide an OpenAI API key before AI title screening.")

        config = load_config(self.store.config_path(project_slug))
        run_dir = self.store.project_dir(project_slug) / "runs" / state["run_id"]
        processed_dir = run_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        candidates = processed_dir / "candidate_papers.csv"
        screened_path = processed_dir / "candidate_papers_screened.csv"
        title_path = processed_dir / "candidate_papers_title_screened.csv"
        venue_path = processed_dir / "candidate_papers_venues.csv"
        enriched_path = processed_dir / "candidate_papers_enriched.csv"

        round_state["status"] = "running"
        state["status"] = "running_discovery"
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Resume initial discovery",
            heading="Resuming initial discovery",
            stages=_with_title_screening_stage(INITIAL_DISCOVERY_STAGES, use_title_llm),
            callback=progress,
            paper_count=_csv_row_count(candidates) if candidates.exists() else 0,
        )

        try:
            manual_records = ManualPaperStore(self.store.project_dir(project_slug)).load()
            _notify(
                tracked_progress,
                "Literature search",
                "Reusing saved discovery records when available.",
            )
            if not candidates.exists():
                result = collect_from_sources(
                    config,
                    console=Console(file=io.StringIO(), no_color=True),
                    source_ids=selected_sources,
                    limit_queries=limit_queries,
                    dblp_mode=source,
                    additional_records=manual_records,
                    progress_callback=_counted_item_progress(
                        tracked_progress,
                        state,
                        "Literature search",
                        "Collecting bibliographic records from the selected sources.",
                    ),
                )
                save_collection(result, output_dir=run_dir)
                collection_summary = summarize(result)
                _write_json(run_dir / "run_summary.json", collection_summary)
            else:
                collection_summary = _read_json(run_dir / "run_summary.json")
                deduped = _csv_row_count(candidates)
                collection_summary = {
                    **round_state.get("counts", {}),
                    **collection_summary,
                    "raw_records": int(collection_summary.get("raw_records", deduped)),
                    "filtered_records": int(collection_summary.get("filtered_records", deduped)),
                    "deduped_records": deduped,
                }

            raw_count = int(collection_summary["raw_records"])
            filtered_count = int(collection_summary["filtered_records"])
            deduped_count = int(collection_summary["deduped_records"])
            round_state["files"]["candidates"] = str(candidates)
            round_state["counts"].update(collection_summary)
            record_flow_stage(
                round_state,
                key="literature_search",
                label="Literature search",
                input_count=raw_count,
                retained_count=raw_count,
                stage_type="discovery",
                details={"sources": ", ".join(selected_sources)},
            )
            record_flow_stage(
                round_state,
                key="metadata_filter",
                label="Metadata filters",
                input_count=raw_count,
                retained_count=filtered_count,
            )
            record_flow_stage(
                round_state,
                key="deduplication",
                label="Deduplication",
                input_count=filtered_count,
                retained_count=deduped_count,
            )
            _set_progress_paper_count(state, deduped_count)
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying the project's title exclusion rules.",
            )
            screening_result = screen_candidates(candidates, screened_path, config.screening)
            _write_json(
                processed_dir / "screening_summary.json",
                {
                    "total": screening_result.summary.total,
                    "by_decision": dict(screening_result.summary.by_decision),
                    "by_bucket": dict(screening_result.summary.by_bucket),
                    "by_exclusion_code": dict(screening_result.summary.by_exclusion_code),
                },
            )
            rule_excluded = screening_result.summary.by_decision.get("exclude", 0)
            rule_retained = screening_result.summary.total - rule_excluded
            round_state["files"]["screened"] = str(screened_path)
            round_state["counts"]["rule_excluded"] = rule_excluded
            record_flow_stage(
                round_state,
                key="rule_screening",
                label="Rule screening",
                input_count=screening_result.summary.total,
                retained_count=rule_retained,
                excluded_count=rule_excluded,
            )
            self._save_state(project_slug, state)

            title_input, title_result = self._title_prescreen(
                project_slug,
                screened_path,
                title_path,
                enabled=use_title_llm,
                batch_size=title_batch_size,
                progress=tracked_progress,
            )
            round_state["counts"].update(_title_screening_counts(title_result))
            if title_result is not None:
                round_state["files"]["title_screened"] = str(title_input)
                record_flow_stage(
                    round_state,
                    key="ai_title_screening",
                    label="AI title screening",
                    input_count=title_result.summary.eligible,
                    retained_count=title_result.summary.kept_for_enrichment,
                    excluded_count=title_result.summary.excluded,
                    details={"cached": title_result.summary.cached},
                )
            enrichment_input_count = (
                title_result.summary.kept_for_enrichment
                if title_result is not None
                else rule_retained
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Reusing saved venue metadata when available.",
            )
            venue_summary = _read_json(processed_dir / "venue_quality_summary.json")
            publication_resolution_path = processed_dir / "publication_resolution.json"
            if not venue_path.exists() or not publication_resolution_path.exists():
                venue_config = replace(config.venue_quality, core_online_enabled=core_online)
                venue_result = enrich_venue_quality(
                    title_input,
                    venue_path,
                    venue_config,
                    decisions={"include_candidate", "needs_review"},
                    survey_config=config,
                    publication_resolution_path=publication_resolution_path,
                    progress_callback=_item_progress(
                        tracked_progress,
                        "Venue enrichment",
                        "Resolving published versions and adding venue quality metadata.",
                    ),
                )
                write_venue_quality_summary(
                    venue_result.summary,
                    processed_dir / "venue_quality_summary.json",
                )
                venue_summary = {
                    "conferences_with_core_rank": getattr(
                        venue_result.summary,
                        "conferences_with_core_rank",
                        0,
                    ),
                    "journals_with_impact_factor": getattr(
                        venue_result.summary,
                        "journals_with_impact_factor",
                        0,
                    ),
                    "publication_resolution_attempted": getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "published_versions_resolved": getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                }
            round_state["files"]["venues"] = str(venue_path)
            round_state["files"]["publication_resolution"] = str(publication_resolution_path)
            record_flow_stage(
                round_state,
                key="venue_enrichment",
                label="Venue enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "CORE ranks found": int(venue_summary.get("conferences_with_core_rank", 0)),
                    "impact factors found": int(
                        venue_summary.get("journals_with_impact_factor", 0)
                    ),
                    "preprints checked": int(
                        venue_summary.get("publication_resolution_attempted", 0)
                    ),
                    "published versions resolved": int(
                        venue_summary.get("published_versions_resolved", 0)
                    ),
                },
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Continuing unfinished lookups through the configured provider chain.",
            )
            enrichment_result = enrich_candidates(
                venue_path,
                enriched_path,
                config.enrichment,
                decisions={"include_candidate", "needs_review"},
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Continuing unfinished lookups through the configured provider chain.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                processed_dir / "abstract_enrichment_summary.json",
            )
            round_state["files"]["enriched"] = str(enriched_path)
            round_state["counts"].update(
                {
                    "manual_records": len(manual_records),
                    "publication_resolution_attempted": int(
                        venue_summary.get("publication_resolution_attempted", 0)
                    ),
                    "published_versions_resolved": int(
                        venue_summary.get("published_versions_resolved", 0)
                    ),
                    "abstracts_found": enrichment_result.summary.with_abstract,
                    "abstracts_attempted": enrichment_result.summary.attempted,
                    "abstract_api_requests": enrichment_result.summary.api_requests,
                    "abstract_batch_requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "abstract_cache_hits": enrichment_result.summary.cache_hits,
                    "abstract_rate_limit_retries": (enrichment_result.summary.rate_limit_retries),
                    "abstract_rate_limit_wait_seconds": (
                        enrichment_result.summary.rate_limit_wait_seconds
                    ),
                }
            )
            record_flow_stage(
                round_state,
                key="abstract_enrichment",
                label="Abstract enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "abstracts found": enrichment_result.summary.with_abstract,
                    "attempted": enrichment_result.summary.attempted,
                    "API requests": enrichment_result.summary.api_requests,
                    "batch requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "cache hits": enrichment_result.summary.cache_hits,
                    "429 retries": enrichment_result.summary.rate_limit_retries,
                },
            )

            round_state["status"] = "discovery_complete"
            state["status"] = "awaiting_ai_or_review"
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Discovery complete",
                message="The candidate set is ready for review preparation.",
                paper_count=deduped_count,
            )
            return state
        except TaskCancelled:
            self._mark_cancelled(project_slug, state, round_state)
            return state
        except Exception as exc:
            self._mark_failed(project_slug, state, round_state, exc)
            raise

    def sync_manual_additions(
        self,
        project_slug: str,
        *,
        enrich_limit: int | None = None,
        core_online: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        run_options = state.get("options", {})
        use_title_llm = bool(run_options.get("title_llm_enabled", False))
        title_batch_size = int(run_options.get("title_llm_batch_size", 100))
        if (
            use_title_llm
            and not self.store.has_api_key(project_slug)
            and not os.environ.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("Save or provide an OpenAI API key before AI title screening.")
        initial_round = _get_round(state, 0)
        if initial_round.get("files", {}).get("audit"):
            raise RuntimeError(
                "Manual papers cannot be synchronized after the initial review queue is created. "
                "Start a new initial run to include them."
            )
        candidate_value = initial_round.get("files", {}).get("candidates")
        if not candidate_value:
            raise RuntimeError("Run initial discovery before synchronizing manual papers.")
        manual_records = ManualPaperStore(self.store.project_dir(project_slug)).load()

        config = load_config(self.store.config_path(project_slug))
        candidates_path = Path(candidate_value)
        processed_dir = candidates_path.parent
        _, candidate_rows = read_csv(candidates_path)
        candidate_records = [
            record
            for row in candidate_rows
            if (record := _without_manual_provenance(PaperRecord.from_dict(row))) is not None
        ]
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Manual additions",
            heading="Synchronizing manual additions",
            stages=_with_title_screening_stage(MANUAL_SYNC_STAGES, use_title_llm),
            callback=progress,
            paper_count=len(candidate_records),
        )

        try:
            _notify(
                tracked_progress,
                "Manual additions",
                "Merging manually added papers and removing duplicates.",
            )
            merged = dedupe_records([*candidate_records, *manual_records])
            write_csv(merged, candidates_path)
            write_jsonl(merged, candidates_path.with_suffix(".jsonl"))
            write_jsonl(
                manual_records,
                processed_dir / "manual_additions_snapshot.jsonl",
                include_raw=True,
            )
            _set_progress_paper_count(state, len(merged))
            initial_round["flow"] = [
                stage
                for stage in initial_round.get("flow", [])
                if stage.get("key") in {"literature_search", "metadata_filter", "deduplication"}
            ]
            record_flow_stage(
                initial_round,
                key="manual_additions",
                label="Manual additions",
                input_count=len(candidate_records),
                retained_count=len(merged),
                excluded_count=0,
                stage_type="discovery",
                details={"manual papers": len(manual_records)},
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying the project's title exclusion rules.",
            )
            screened_path = processed_dir / "candidate_papers_screened.csv"
            screening_result = screen_candidates(
                candidates_path,
                screened_path,
                config.screening,
            )
            _write_json(
                processed_dir / "screening_summary.json",
                {
                    "total": screening_result.summary.total,
                    "by_decision": dict(screening_result.summary.by_decision),
                    "by_bucket": dict(screening_result.summary.by_bucket),
                    "by_exclusion_code": dict(screening_result.summary.by_exclusion_code),
                },
            )
            rule_excluded = screening_result.summary.by_decision.get("exclude", 0)
            rule_retained = screening_result.summary.total - rule_excluded
            initial_round["files"]["screened"] = str(screened_path)
            record_flow_stage(
                initial_round,
                key="rule_screening",
                label="Rule screening",
                input_count=screening_result.summary.total,
                retained_count=rule_retained,
                excluded_count=rule_excluded,
            )

            title_input, title_result = self._title_prescreen(
                project_slug,
                screened_path,
                processed_dir / "candidate_papers_title_screened.csv",
                enabled=use_title_llm,
                batch_size=title_batch_size,
                progress=tracked_progress,
            )
            initial_round["counts"].update(
                {
                    "deduped_records": len(merged),
                    "rule_excluded": screening_result.summary.by_decision.get("exclude", 0),
                    **_title_screening_counts(title_result),
                }
            )
            if title_result is not None:
                initial_round["files"]["title_screened"] = str(title_input)
                record_flow_stage(
                    initial_round,
                    key="ai_title_screening",
                    label="AI title screening",
                    input_count=title_result.summary.eligible,
                    retained_count=title_result.summary.kept_for_enrichment,
                    excluded_count=title_result.summary.excluded,
                    details={"cached": title_result.summary.cached},
                )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Resolving published versions and adding venue quality metadata.",
            )
            venue_path = processed_dir / "candidate_papers_venues.csv"
            publication_resolution_path = processed_dir / "publication_resolution.json"
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            venue_result = enrich_venue_quality(
                title_input,
                venue_path,
                venue_config,
                decisions={"include_candidate", "needs_review"},
                survey_config=config,
                publication_resolution_path=publication_resolution_path,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Venue enrichment",
                    "Resolving published versions and adding venue quality metadata.",
                ),
            )
            write_venue_quality_summary(
                venue_result.summary,
                processed_dir / "venue_quality_summary.json",
            )
            enrichment_input_count = (
                title_result.summary.kept_for_enrichment
                if title_result is not None
                else rule_retained
            )
            initial_round["files"]["venues"] = str(venue_path)
            initial_round["files"]["publication_resolution"] = str(publication_resolution_path)
            record_flow_stage(
                initial_round,
                key="venue_enrichment",
                label="Venue enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "CORE ranks found": getattr(
                        venue_result.summary,
                        "conferences_with_core_rank",
                        0,
                    ),
                    "impact factors found": getattr(
                        venue_result.summary,
                        "journals_with_impact_factor",
                        0,
                    ),
                    "preprints checked": getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "published versions resolved": getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                },
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Looking up missing abstracts through the configured provider chain.",
            )
            enriched_path = processed_dir / "candidate_papers_enriched.csv"
            enrichment_result = enrich_candidates(
                venue_path,
                enriched_path,
                config.enrichment,
                decisions={"include_candidate", "needs_review"},
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Looking up missing abstracts through the configured provider chain.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                processed_dir / "abstract_enrichment_summary.json",
            )
            initial_round["files"]["enriched"] = str(enriched_path)
            record_flow_stage(
                initial_round,
                key="abstract_enrichment",
                label="Abstract enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "abstracts found": enrichment_result.summary.with_abstract,
                    "attempted": enrichment_result.summary.attempted,
                    "API requests": getattr(enrichment_result.summary, "api_requests", 0),
                    "batch requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "cache hits": getattr(enrichment_result.summary, "cache_hits", 0),
                    "429 retries": getattr(
                        enrichment_result.summary,
                        "rate_limit_retries",
                        0,
                    ),
                },
            )

            initial_round["status"] = "discovery_complete"
            initial_round["files"].update(
                {
                    "venues": str(venue_path),
                    "screened": str(screened_path),
                    **({"title_screened": str(title_input)} if title_result is not None else {}),
                    "enriched": str(enriched_path),
                }
            )
            initial_round["counts"].update(
                {
                    "deduped_records": len(merged),
                    "manual_records": len(manual_records),
                    "rule_excluded": screening_result.summary.by_decision.get("exclude", 0),
                    **_title_screening_counts(title_result),
                    "publication_resolution_attempted": getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "published_versions_resolved": getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                    "abstracts_found": enrichment_result.summary.with_abstract,
                    "abstracts_attempted": enrichment_result.summary.attempted,
                    "abstract_api_requests": enrichment_result.summary.api_requests,
                    "abstract_batch_requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "abstract_cache_hits": enrichment_result.summary.cache_hits,
                }
            )
            state["status"] = "awaiting_ai_or_review"
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Manual additions synchronized",
                message=f"The candidate set now contains {len(merged)} unique papers.",
                paper_count=len(merged),
            )
            return state
        except TaskCancelled:
            self._mark_cancelled(project_slug, state, initial_round)
            return state
        except Exception as exc:
            self._mark_failed(project_slug, state, initial_round, exc)
            raise

    def prepare_round_for_review(
        self,
        project_slug: str,
        round_index: int,
        *,
        use_llm: bool,
        llm_limit: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        round_state = _get_round(state, round_index)
        if round_state.get("status") not in {
            "discovery_complete",
            "citation_incomplete",
            "ready_for_review",
            "failed",
        }:
            raise RuntimeError("This round is not ready for AI screening or review preparation.")
        if not round_state.get("files", {}).get("enriched"):
            raise RuntimeError("This round does not have an enriched candidate file.")
        config = load_config(self.store.config_path(project_slug))
        processed_dir = Path(round_state["files"]["enriched"]).parent
        suffix = "" if round_index == 0 else f"_round_{round_index}"
        enriched_path = Path(round_state["files"]["enriched"])
        if round_state.get("kind") == "snowball":
            strip_per_paper_citation_diagnostics(enriched_path)
        replay_requested = _round_prompt_replay_requested(state, round_state)
        if (
            (use_llm or replay_requested)
            and not self.store.has_api_key(project_slug)
            and not os.environ.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("Save or provide an OpenAI API key before AI screening.")
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="AI abstract screening and review" if use_llm else "Review preparation",
            heading="Preparing the review queue",
            stages=_with_prompt_replay_stage(
                AI_REVIEW_STAGES if use_llm else MANUAL_REVIEW_STAGES,
                replay_requested,
            ),
            callback=progress,
            paper_count=_round_paper_count(round_state),
        )

        try:
            if replay_requested:
                _notify(
                    tracked_progress,
                    "Re-screen initial AI exclusions",
                    "Applying the approved prompt to initial AI exclusions.",
                )
                replay_files, replay_counts = self._replay_initial_ai_exclusions(
                    project_slug=project_slug,
                    state=state,
                    round_index=round_index,
                    enriched_path=enriched_path,
                    config=config,
                    processed_dir=processed_dir,
                    progress=tracked_progress,
                )
                round_state["files"].update(replay_files)
                round_state["counts"].update(replay_counts)
                record_flow_stage(
                    round_state,
                    key="prompt_replay",
                    label="Re-screen initial AI exclusions",
                    input_count=replay_counts["replayed"],
                    retained_count=replay_counts["recovered"],
                    excluded_count=replay_counts["reexcluded"],
                    details={
                        "human-reviewed removed before AI": replay_counts[
                            "replay_reviewed_removed"
                        ],
                        "eligible after review removal": replay_counts["replay_eligible"],
                        "failed": replay_counts["failed"],
                    },
                )
                self._save_state(project_slug, state)

            if use_llm:
                _notify(
                    tracked_progress,
                    "AI abstract screening",
                    "Analyzing candidate abstracts with the configured model.",
                )
                llm_path = processed_dir / f"candidate_papers_llm_screened{suffix}.csv"
                llm_result = llm_screen_candidates(
                    enriched_path,
                    llm_path,
                    config.llm_screening,
                    limit=llm_limit,
                    progress_callback=_item_progress(
                        tracked_progress,
                        "AI abstract screening",
                        "Analyzing candidate abstracts with the configured model.",
                    ),
                )
                write_llm_screening_summary(
                    llm_result.summary,
                    processed_dir / f"llm_screening_summary{suffix}.json",
                )
                recommendation_path = processed_dir / f"final_screening_recommendations{suffix}.csv"
                report_path = processed_dir / f"llm_screening_report{suffix}.md"
                final_summary_path = processed_dir / f"final_screening_summary{suffix}.json"
                _notify(
                    tracked_progress,
                    "Recommendation summary",
                    "Summarizing AI recommendations for human review.",
                )
                summarize_llm_screening(
                    llm_path,
                    report_path,
                    recommendation_path,
                    final_summary_path,
                )
                round_state["files"]["llm_screened"] = str(llm_path)
                round_state["files"]["llm_report"] = str(report_path)
                round_state["counts"]["llm_screened"] = llm_result.summary.attempted
                round_state["counts"]["llm_failed"] = llm_result.summary.by_status.get("failed", 0)
                round_state["counts"]["llm_api_requests"] = llm_result.summary.api_requests
                round_state["counts"]["llm_batch_requests"] = llm_result.summary.batch_requests
                round_state["counts"]["llm_cache_hits"] = llm_result.summary.cache_hits
                llm_excluded = llm_result.summary.by_decision.get("exclude", 0)
                llm_retained = llm_result.summary.by_decision.get(
                    "include", 0
                ) + llm_result.summary.by_decision.get("maybe", 0)
                record_flow_stage(
                    round_state,
                    key="ai_abstract_screening",
                    label="AI abstract screening",
                    input_count=llm_result.summary.eligible,
                    retained_count=llm_retained,
                    excluded_count=llm_excluded,
                    details={
                        "failed": llm_result.summary.by_status.get("failed", 0),
                        "API requests": llm_result.summary.api_requests,
                        "batch requests": llm_result.summary.batch_requests,
                        "cache hits": llm_result.summary.cache_hits,
                    },
                )
            else:
                _notify(
                    tracked_progress,
                    "Prepare recommendations",
                    "Creating a human-only review queue.",
                )
                recommendation_path = processed_dir / f"final_screening_recommendations{suffix}.csv"
                create_manual_recommendations(enriched_path, recommendation_path)

            _notify(
                tracked_progress,
                "Create manual review queue",
                "Deduplicating and creating the manual audit file.",
            )
            audit_dir = self.store.project_dir(project_slug) / "audits" / state["run_id"]
            audit_path = audit_dir / f"round_{round_index}.csv"
            previous_audits = [
                Path(item["files"]["audit"])
                for item in state["rounds"]
                if int(item["index"]) < round_index and item.get("files", {}).get("audit")
            ]
            _, queue_count = create_audit_queue(
                recommendation_path,
                audit_path,
                previous_audit_paths=previous_audits,
            )
            audit_checkpoint = round_state.get("files", {}).get("audit_checkpoint")
            if audit_checkpoint and Path(audit_checkpoint).exists():
                queue_count = _restore_audit_checkpoint(
                    audit_path,
                    Path(audit_checkpoint),
                ).total
            round_state["files"].update(
                {
                    "recommendations": str(recommendation_path),
                    "pool": str(recommendation_path),
                    "audit": str(audit_path),
                }
            )
            round_state["counts"]["audit_queue"] = queue_count
            recommendation_count = _csv_row_count(recommendation_path)
            record_flow_stage(
                round_state,
                key="audit_queue",
                label="Human review queue",
                input_count=recommendation_count,
                retained_count=queue_count,
                excluded_count=max(recommendation_count - queue_count, 0),
                stage_type="review",
            )
            failed_coverage_seeds = _failed_seed_coverage_count(round_state)
            round_state["status"] = (
                "converged"
                if round_index > 0 and queue_count == 0 and failed_coverage_seeds == 0
                else "ready_for_review"
            )
            state["status"] = (
                "converged" if round_state["status"] == "converged" else "awaiting_manual_review"
            )
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Review queue ready",
                message="No new review candidates were found."
                if queue_count == 0
                else f"{queue_count} papers require human decisions.",
                paper_count=queue_count,
            )
            return state
        except TaskCancelled:
            self._mark_cancelled(project_slug, state, round_state)
            return state
        except Exception as exc:
            self._mark_failed(project_slug, state, round_state, exc)
            raise

    def start_snowball_discovery(
        self,
        project_slug: str,
        *,
        citation_providers: list[str] | None = None,
        provider_strategy: str | None = None,
        max_backward_per_seed: int = 0,
        max_forward_per_seed: int = 0,
        enrich_limit: int | None = None,
        core_online: bool = True,
        use_title_llm: bool = False,
        title_batch_size: int = 100,
        replay_initial_exclusions: bool = False,
        target_seed: dict[str, str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        retry_round = _retryable_snowball_round(state)
        if target_seed is None and retry_round and isinstance(retry_round.get("target_seed"), dict):
            target_seed = dict(retry_round["target_seed"])
        normalized_target_seed = _normalize_target_seed(target_seed)
        targeted_snowball = normalized_target_seed is not None
        replay_state = _ensure_prompt_replay_state(state)
        replay_completed_in_retry_round = bool(
            retry_round
            and replay_state.get("status") == "completed"
            and str(replay_state.get("replay_round", "")) == str(retry_round.get("index", ""))
        )
        if (
            retry_round
            and state.get("options", {}).get("replay_initial_exclusions")
            and replay_completed_in_retry_round
        ):
            replay_initial_exclusions = True
        if (
            use_title_llm
            and not self.store.has_api_key(project_slug)
            and not os.environ.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("Save or provide an OpenAI API key before AI title screening.")
        if replay_initial_exclusions:
            if replay_state.get("status") not in {"pending", "completed"}:
                raise RuntimeError(
                    "Approve a refined screening prompt before replaying AI exclusions."
                )
            if replay_state.get("status") == "completed" and not replay_completed_in_retry_round:
                raise RuntimeError("The initial AI exclusions were already replayed once.")
            if not self.store.has_api_key(project_slug) and not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "Save or provide an OpenAI API key before replaying AI exclusions."
                )
        latest_state_round = state["rounds"][-1]
        if (
            retry_round is None
            and latest_state_round.get("kind") == "snowball"
            and not latest_state_round.get("files", {}).get("audit")
            and latest_state_round.get("status")
            in {"discovery_complete", "citation_incomplete", "ready_for_review"}
        ):
            raise RuntimeError(
                "Prepare the current snowball round for manual review before starting "
                "another round."
            )
        completed_rounds = [
            item
            for item in state["rounds"]
            if item.get("files", {}).get("audit") and item is not retry_round
        ]
        if not completed_rounds:
            raise RuntimeError("Complete the initial review before snowballing.")
        latest_round = completed_rounds[-1]
        _, _, audit_summary = load_audit(Path(latest_round["files"]["audit"]))
        if audit_summary.unreviewed:
            raise RuntimeError("Finish every paper in the current audit round before snowballing.")

        project_dir = self.store.project_dir(project_slug)
        pending_round_index = (
            int(retry_round["index"])
            if retry_round is not None
            else max(int(item["index"]) for item in state["rounds"]) + 1
        )
        audit_paths = [Path(item["files"]["audit"]) for item in completed_rounds]
        cumulative_path = project_dir / "audits" / state["run_id"] / "cumulative.csv"
        cumulative_included_path = project_dir / "audits" / state["run_id"] / "included.csv"
        build_cumulative_audit(audit_paths, cumulative_path, cumulative_included_path)
        included_path = (
            project_dir / "audits" / state["run_id"] / f"round_{latest_round['index']}_included.csv"
        )
        filter_manual_includes(Path(latest_round["files"]["audit"]), included_path)

        if targeted_snowball:
            seed_input_path = (
                project_dir
                / "seeds"
                / f"{state['run_id']}_targeted_round_{pending_round_index}.csv"
            )
            target_fields = list(normalized_target_seed)
            write_audit_csv(seed_input_path, [normalized_target_seed], target_fields)
            seed_source_label = "targeted_single_paper"
        else:
            included_count = _write_incremental_snowball_candidates(
                included_path,
                included_path,
                prior_paths=audit_paths[:-1],
            )
            if not included_count:
                raise RuntimeError(
                    "The latest review round has no newly included papers to use as snowball seeds."
                )
            seed_input_path = included_path
            seed_source_label = f"manual_audit_round_{latest_round['index']}_include"

        seed_path = project_dir / "seeds" / f"{state['run_id']}_round_{pending_round_index}.yaml"
        seed_result = export_seed_papers_from_csv(
            seed_input_path,
            seed_path,
            source_label=seed_source_label,
        )
        if not seed_result.seeds:
            raise RuntimeError("The selected paper could not be converted into a snowball seed.")
        config = load_config(self.store.config_path(project_slug))
        selected_providers = list(dict.fromkeys(citation_providers or config.snowballing.providers))
        if not selected_providers:
            raise RuntimeError("Select at least one citation provider before snowballing.")
        unsupported = [
            provider
            for provider in selected_providers
            if provider not in {"semantic_scholar", "opencitations", "openalex"}
        ]
        if unsupported:
            raise RuntimeError(f"Unsupported citation providers: {', '.join(unsupported)}")
        selected_strategy = (
            str(provider_strategy or config.snowballing.provider_strategy).strip().lower()
        )
        if selected_strategy not in {"merge", "failover"}:
            raise RuntimeError("Citation provider strategy must be merge or failover.")
        query_providers = _retry_provider_order(retry_round, selected_providers)
        config = replace(
            config,
            snowballing=replace(
                config.snowballing,
                providers=query_providers,
                provider_strategy=selected_strategy,
            ),
        )
        if retry_round is not None:
            round_state = retry_round
            round_index = int(round_state["index"])
            retry_count = int(round_state.get("counts", {}).get("retry_count") or 0) + 1
            existing_files = dict(round_state.get("files", {}))
            audit_value = existing_files.get("audit")
            if audit_value and Path(audit_value).exists():
                audit_checkpoint = (
                    project_dir
                    / "audits"
                    / state["run_id"]
                    / f"round_{round_index}_before_retry_{retry_count}.csv"
                )
                audit_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                copy2(audit_value, audit_checkpoint)
                existing_files["audit_checkpoint"] = str(audit_checkpoint)
            round_state["checkpoint_files"] = dict(existing_files)
            retained_files = {
                key: value
                for key, value in existing_files.items()
                if key in {"snowballed", "seeds", "audit_checkpoint"}
            }
            round_state.update(
                {
                    "status": "running",
                    "files": retained_files,
                    "error": "",
                }
            )
            round_state.setdefault("counts", {}).update(
                {
                    "seeds": len(seed_result.seeds),
                    "retry_count": retry_count,
                }
            )
        else:
            round_index = pending_round_index
            round_state = _new_round_state(index=round_index, kind="snowball")
            round_state["counts"]["seeds"] = len(seed_result.seeds)
            state["rounds"].append(round_state)
        round_state["snowball_mode"] = "targeted" if targeted_snowball else "incremental"
        round_state["replay_initial_exclusions"] = bool(replay_initial_exclusions)
        if normalized_target_seed:
            round_state["target_seed"] = dict(normalized_target_seed)
            round_state["counts"]["target_seed_title"] = normalized_target_seed["title"]
        state["status"] = "running_snowball"
        state.setdefault("options", {}).update(
            {
                "title_llm_enabled": use_title_llm,
                "title_llm_batch_size": title_batch_size,
                "replay_initial_exclusions": replay_initial_exclusions,
                "snowball_complete_citations": (
                    max_backward_per_seed <= 0 and max_forward_per_seed <= 0
                ),
                "snowball_backward_limit": max(max_backward_per_seed, 0),
                "snowball_forward_limit": max(max_forward_per_seed, 0),
                "snowball_enrich_limit": enrich_limit,
                "snowball_core_online": core_online,
                "snowball_citation_providers": selected_providers,
                "snowball_provider_strategy": selected_strategy,
                "snowball_mode": "targeted" if targeted_snowball else "incremental",
            }
        )
        self._save_state(project_slug, state)

        run_dir = project_dir / "runs" / state["run_id"]
        processed_dir = run_dir / "processed"
        previous_pool = Path(latest_round["files"].get("pool") or latest_round["files"]["audit"])
        snowball_input = previous_pool
        checkpoint_value = round_state.get("files", {}).get("snowballed")
        if retry_round is not None and checkpoint_value and Path(checkpoint_value).exists():
            snowball_input = Path(checkpoint_value)
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Citation snowballing",
            heading="Running citation snowballing",
            stages=_with_title_screening_stage(SNOWBALL_STAGES, use_title_llm),
            callback=progress,
            paper_count=_csv_row_count(snowball_input),
        )
        snowball_path = processed_dir / f"candidate_papers_snowballed_round_{round_index}.csv"

        try:
            _notify(
                tracked_progress,
                "Citation snowballing",
                "Collecting references and citing works from the selected providers.",
            )
            round_state.setdefault("files", {}).update(
                {
                    "snowballed": str(snowball_path),
                    "seeds": str(seed_path),
                }
            )
            self._save_state(project_slug, state)
            snowball_result = snowball_candidates(
                snowball_input,
                snowball_path,
                config,
                seed_papers_path=seed_path,
                max_backward_per_seed=max_backward_per_seed,
                max_forward_per_seed=max_forward_per_seed,
                include_seed_papers=True,
                progress_callback=_counted_item_progress(
                    tracked_progress,
                    state,
                    "Citation snowballing",
                    "Collecting references and citing works from the selected providers.",
                ),
            )
            snowball_result = _merge_retry_provider_summary(
                snowball_result,
                retry_round=retry_round,
                selected_providers=selected_providers,
            )
            new_candidates_path = (
                processed_dir / f"candidate_papers_snowball_new_round_{round_index}.csv"
            )
            round_added_rows = _write_incremental_snowball_candidates(
                snowball_path,
                new_candidates_path,
                prior_paths=[previous_pool, *audit_paths],
            )
            write_snowballing_summary(
                snowball_result.summary,
                processed_dir / f"snowballing_round_{round_index}_summary.json",
            )
            coverage_path = processed_dir / f"snowball_seed_coverage_round_{round_index}.csv"
            write_seed_coverage_report(snowball_result.summary, coverage_path)
            coverage_by_status = Counter(
                str(item.get("coverage_status") or "failed")
                for item in snowball_result.summary.seed_diagnostics
            )
            coverage_issue_seeds = coverage_by_status.get("partial", 0) + coverage_by_status.get(
                "failed", 0
            )
            round_state["files"]["snowballed"] = str(snowball_path)
            round_state["files"]["snowball_new"] = str(new_candidates_path)
            round_state["files"]["seeds"] = str(seed_path)
            round_state["files"]["seed_coverage"] = str(coverage_path)
            record_flow_stage(
                round_state,
                key="citation_snowballing",
                label="Citation snowballing",
                input_count=len(seed_result.seeds),
                retained_count=round_added_rows,
                excluded_count=0,
                stage_type="discovery",
                details={
                    "new papers": round_added_rows,
                    "resolved seeds": snowball_result.summary.seeds_resolved,
                    "references available": (snowball_result.summary.references_available),
                    "references fetched": snowball_result.summary.references_fetched,
                    "citations available": snowball_result.summary.citations_available,
                    "citations fetched": snowball_result.summary.citations_fetched,
                    "truncated seeds": (
                        snowball_result.summary.backward_truncated_seeds
                        + snowball_result.summary.forward_truncated_seeds
                    ),
                    "provider strategy": snowball_result.summary.provider_strategy,
                    "providers": list(snowball_result.summary.provider_order),
                    "provider successes": dict(snowball_result.summary.provider_successes),
                    "provider failures": dict(snowball_result.summary.provider_failures),
                    "complete seed coverage": coverage_by_status.get("complete", 0),
                    "partial seed coverage": coverage_by_status.get("partial", 0),
                    "failed seed coverage": coverage_by_status.get("failed", 0),
                },
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying project exclusion terms to new records.",
            )
            screened_path = processed_dir / f"candidate_papers_screened_round_{round_index}.csv"
            screening_result = screen_candidates(
                new_candidates_path,
                screened_path,
                config.screening,
            )
            rule_excluded = screening_result.summary.by_decision.get("exclude", 0)
            rule_retained = screening_result.summary.total - rule_excluded
            round_state["files"]["screened"] = str(screened_path)
            record_flow_stage(
                round_state,
                key="rule_screening",
                label="Rule screening",
                input_count=screening_result.summary.total,
                retained_count=rule_retained,
                excluded_count=rule_excluded,
            )

            title_input, title_result = self._title_prescreen(
                project_slug,
                screened_path,
                processed_dir / f"candidate_papers_title_screened_round_{round_index}.csv",
                enabled=use_title_llm,
                batch_size=title_batch_size,
                progress=tracked_progress,
            )
            round_state["counts"].update(
                {
                    "pool_rows": round_added_rows,
                    "historical_pool_rows": snowball_result.summary.output_rows,
                    "rule_excluded": screening_result.summary.by_decision.get("exclude", 0),
                    **_title_screening_counts(title_result),
                }
            )
            if title_result is not None:
                round_state["files"]["title_screened"] = str(title_input)
                record_flow_stage(
                    round_state,
                    key="ai_title_screening",
                    label="AI title screening",
                    input_count=title_result.summary.eligible,
                    retained_count=title_result.summary.kept_for_enrichment,
                    excluded_count=title_result.summary.excluded,
                    details={"cached": title_result.summary.cached},
                )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Updating publication metadata for the expanded pool.",
            )
            venue_path = processed_dir / f"candidate_papers_venues_round_{round_index}.csv"
            publication_resolution_path = (
                processed_dir / f"publication_resolution_round_{round_index}.json"
            )
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            venue_result = enrich_venue_quality(
                title_input,
                venue_path,
                venue_config,
                decisions={"include_candidate", "needs_review"},
                survey_config=config,
                publication_resolution_path=publication_resolution_path,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Venue enrichment",
                    "Updating publication metadata for the expanded pool.",
                ),
            )
            write_venue_quality_summary(
                venue_result.summary,
                processed_dir / f"venue_quality_round_{round_index}_summary.json",
            )
            enrichment_input_count = (
                title_result.summary.kept_for_enrichment
                if title_result is not None
                else rule_retained
            )
            round_state["files"]["venues"] = str(venue_path)
            round_state["files"]["publication_resolution"] = str(publication_resolution_path)
            record_flow_stage(
                round_state,
                key="venue_enrichment",
                label="Venue enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "CORE ranks found": getattr(
                        venue_result.summary,
                        "conferences_with_core_rank",
                        0,
                    ),
                    "impact factors found": getattr(
                        venue_result.summary,
                        "journals_with_impact_factor",
                        0,
                    ),
                    "preprints checked": getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "published versions resolved": getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                },
            )
            self._save_state(project_slug, state)

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Looking up abstracts for newly discovered papers.",
            )
            enriched_path = processed_dir / f"candidate_papers_enriched_round_{round_index}.csv"
            enrichment_result = enrich_candidates(
                venue_path,
                enriched_path,
                config.enrichment,
                decisions={"include_candidate", "needs_review"},
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Looking up abstracts for newly discovered papers.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                processed_dir / f"abstract_enrichment_round_{round_index}_summary.json",
            )
            round_state["files"]["enriched"] = str(enriched_path)
            record_flow_stage(
                round_state,
                key="abstract_enrichment",
                label="Abstract enrichment",
                input_count=enrichment_input_count,
                retained_count=enrichment_input_count,
                stage_type="enrichment",
                details={
                    "abstracts found": enrichment_result.summary.with_abstract,
                    "attempted": enrichment_result.summary.attempted,
                    "API requests": getattr(enrichment_result.summary, "api_requests", 0),
                    "batch requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "cache hits": getattr(enrichment_result.summary, "cache_hits", 0),
                    "429 retries": getattr(
                        enrichment_result.summary,
                        "rate_limit_retries",
                        0,
                    ),
                },
            )

            round_state["status"] = "discovery_complete"
            audit_checkpoint = round_state.get("files", {}).get("audit_checkpoint")
            round_state["files"] = {
                "snowballed": str(snowball_path),
                "snowball_new": str(new_candidates_path),
                "seed_coverage": str(coverage_path),
                "screened": str(screened_path),
                **({"title_screened": str(title_input)} if title_result is not None else {}),
                "venues": str(venue_path),
                "publication_resolution": str(publication_resolution_path),
                "enriched": str(enriched_path),
                "seeds": str(seed_path),
                **({"audit_checkpoint": audit_checkpoint} if audit_checkpoint else {}),
            }
            round_state["counts"].update(
                {
                    "pool_rows": round_added_rows,
                    "historical_pool_rows": snowball_result.summary.output_rows,
                    "added_rows": round_added_rows,
                    "resolved_seeds": snowball_result.summary.seeds_resolved,
                    "references_available": (snowball_result.summary.references_available),
                    "references_fetched": snowball_result.summary.references_fetched,
                    "citations_available": snowball_result.summary.citations_available,
                    "citations_fetched": snowball_result.summary.citations_fetched,
                    "backward_truncated_seeds": (snowball_result.summary.backward_truncated_seeds),
                    "forward_truncated_seeds": (snowball_result.summary.forward_truncated_seeds),
                    "citation_providers": list(snowball_result.summary.provider_order),
                    "provider_strategy": snowball_result.summary.provider_strategy,
                    "provider_successes": dict(snowball_result.summary.provider_successes),
                    "provider_failures": dict(snowball_result.summary.provider_failures),
                    "coverage_complete_seeds": coverage_by_status.get("complete", 0),
                    "coverage_partial_seeds": coverage_by_status.get("partial", 0),
                    "coverage_failed_seeds": coverage_by_status.get("failed", 0),
                    "coverage_issue_seeds": coverage_issue_seeds,
                    "rule_excluded": screening_result.summary.by_decision.get("exclude", 0),
                    **_title_screening_counts(title_result),
                    "publication_resolution_attempted": getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "published_versions_resolved": getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                    "abstracts_found": enrichment_result.summary.with_abstract,
                    "abstracts_attempted": enrichment_result.summary.attempted,
                    "abstract_api_requests": enrichment_result.summary.api_requests,
                    "abstract_batch_requests": getattr(
                        enrichment_result.summary,
                        "batch_requests",
                        0,
                    ),
                    "abstract_cache_hits": enrichment_result.summary.cache_hits,
                }
            )
            state["status"] = "awaiting_ai_or_review"
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Snowball discovery complete",
                message=(
                    f"Candidates are ready for review preparation; {coverage_issue_seeds} "
                    "seed papers have citation coverage warnings."
                    if coverage_issue_seeds
                    else "New records are ready for AI screening or review."
                ),
                paper_count=_csv_row_count(enriched_path),
            )
            return state
        except TaskCancelled:
            _ensure_checkpoint_file(snowball_input, snowball_path)
            self._mark_cancelled(project_slug, state, round_state)
            return state
        except Exception as exc:
            _ensure_checkpoint_file(snowball_input, snowball_path)
            self._mark_failed(project_slug, state, round_state, exc)
            raise

    def _replay_initial_ai_exclusions(
        self,
        *,
        project_slug: str,
        state: dict[str, Any],
        round_index: int,
        enriched_path: Path,
        config: SurveyConfig,
        processed_dir: Path,
        progress: ProgressCallback | None,
    ) -> tuple[dict[str, str], dict[str, int]]:
        refinement = state.get("prompt_refinement", {})
        replay = _ensure_prompt_replay_state(state)
        approved_value = replay.get("approved_prompt_path") or refinement.get(
            "approved_prompt_path"
        )
        if not approved_value or not Path(approved_value).exists():
            raise RuntimeError("The approved screening prompt file is unavailable.")
        initial_exclusions, filter_counts = _initial_ai_exclusion_summary(state)
        if not initial_exclusions:
            counts = {
                "replayed": 0,
                "recovered": 0,
                "reexcluded": 0,
                "failed": 0,
                "replay_source_exclusions": filter_counts["deduplicated_exclusions"],
                "replay_reviewed_removed": filter_counts["reviewed_removed"],
                "replay_eligible": filter_counts["eligible"],
            }
            replay.update(
                {
                    "status": "completed",
                    "replay_round": round_index,
                    "replayed_at": _now(),
                    **counts,
                }
            )
            refinement.update({"replay_status": "completed", **counts})
            return {}, counts

        fieldnames, current_rows = read_csv(enriched_path)
        current_indexes = _row_alias_indexes(current_rows)
        replay_input_rows: list[dict[str, str]] = []
        for row in initial_exclusions:
            match_index = _matching_row_index(current_indexes, row)
            replay_input_rows.append(
                dict(current_rows[match_index] if match_index is not None else row)
            )
        revision_id = str(
            replay.get("refinement_id") or refinement.get("refinement_id") or "approved"
        )
        prefix = f"prompt_replay_round_{round_index}"
        input_path = processed_dir / f"{prefix}_input.csv"
        screened_path = processed_dir / f"{prefix}_screened.csv"
        recommendations_path = processed_dir / f"{prefix}_recommendations.csv"
        report_path = processed_dir / f"{prefix}_report.md"
        final_summary_path = processed_dir / f"{prefix}_summary.json"
        request_summary_path = processed_dir / f"{prefix}_request_summary.json"
        input_fields = list(dict.fromkeys(key for row in replay_input_rows for key in row))
        write_audit_csv(input_path, replay_input_rows, input_fields)
        replay_config = replace(
            config.llm_screening,
            model=self.store.load_project(project_slug).prompt_replay_model,
            system_prompt_path=Path(approved_value),
            prompt_version=f"prompt-refinement-{revision_id}",
        )
        decisions = {str(row.get("auto_screening_decision") or "") for row in replay_input_rows}
        llm_result = llm_screen_candidates(
            input_path,
            screened_path,
            replay_config,
            decisions=decisions,
            overwrite=True,
            progress_callback=_item_progress(
                progress,
                "Re-screen initial AI exclusions",
                "Applying the approved prompt to initial AI exclusions.",
            ),
        )
        write_llm_screening_summary(llm_result.summary, request_summary_path)
        recommendation_result = summarize_llm_screening(
            screened_path,
            report_path,
            recommendations_path,
            final_summary_path,
        )

        recovered = 0
        reexcluded = 0
        failed = 0
        current_indexes = _row_alias_indexes(current_rows)
        for row in recommendation_result.rows:
            is_failed = row.get("llm_status") == "failed"
            is_reexcluded = row.get("llm_decision") == "exclude" and not is_failed
            replay_decision = "reexcluded" if is_reexcluded else "recovered"
            if is_reexcluded:
                row.update(
                    {
                        "final_recommendation": "auto_exclude",
                        "final_priority": "8",
                        "final_reason": ("The approved refined prompt excluded this paper again."),
                    }
                )
            row.update(
                {
                    "auto_screening_decision": ("exclude" if is_reexcluded else "needs_review"),
                    "prompt_replay_decision": replay_decision,
                    "prompt_refinement_id": revision_id,
                }
            )
            failed += int(is_failed)
            reexcluded += int(is_reexcluded)
            recovered += int(not is_reexcluded)
            match_index = _matching_row_index(current_indexes, row)
            if match_index is None:
                if is_reexcluded:
                    continue
                current_rows.append(dict(row))
                for key in paper_keys(row):
                    current_indexes[key] = len(current_rows) - 1
                continue
            current_rows[match_index].update(row)
            for key in paper_keys(current_rows[match_index]):
                current_indexes[key] = match_index

        output_fields = list(
            dict.fromkeys([*fieldnames, *(key for row in current_rows for key in row)])
        )
        write_audit_csv(enriched_path, current_rows, output_fields)
        counts = {
            "replayed": len(replay_input_rows),
            "recovered": recovered,
            "reexcluded": reexcluded,
            "failed": failed,
            "replay_source_exclusions": filter_counts["deduplicated_exclusions"],
            "replay_reviewed_removed": filter_counts["reviewed_removed"],
            "replay_eligible": filter_counts["eligible"],
        }
        files = {
            "prompt_replay_input": str(input_path),
            "prompt_replay_screened": str(screened_path),
            "prompt_replay_recommendations": str(recommendations_path),
            "prompt_replay_report": str(report_path),
        }
        replay.update(
            {
                "status": "completed",
                "replay_round": round_index,
                "replayed_at": _now(),
                **counts,
                "replay_files": files,
            }
        )
        refinement.update({"replay_status": "completed", **counts})
        return files, counts

    def _title_prescreen(
        self,
        project_slug: str,
        input_path: Path,
        output_path: Path,
        *,
        enabled: bool,
        batch_size: int,
        progress: ProgressCallback | None,
    ) -> tuple[Path, TitleScreeningResult | None]:
        if not enabled:
            return input_path, None
        settings = self.store.load_project(project_slug)
        api_key = self.store.read_api_key(project_slug) or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Save or provide an OpenAI API key before AI title screening.")
        _notify(
            progress,
            "AI title screening",
            "Screening titles in high-recall batches before abstract enrichment.",
        )
        client = OpenAIResearchClient(
            base_url=settings.llm_base_url,
            api_key=api_key,
            model=settings.title_screening_model,
            timeout_seconds=30,
            retries=3,
        )
        result = screen_titles_with_llm(
            input_path,
            output_path,
            client=client,
            research_question=settings.research_question,
            scope_description=settings.scope_description,
            inclusion_criteria=settings.inclusion_criteria,
            exclusion_criteria=settings.exclusion_criteria,
            model=settings.title_screening_model,
            cache_dir=self.store.project_dir(project_slug) / "cache" / "title_screening",
            batch_size=batch_size,
            progress_callback=_item_progress(
                progress,
                "AI title screening",
                "Screening titles in high-recall batches before abstract enrichment.",
            ),
        )
        write_title_screening_summary(
            result.summary,
            output_path.with_name(f"{output_path.stem}_summary.json"),
        )
        return output_path, result

    def estimate_llm_usage(
        self,
        project_slug: str,
        round_index: int,
        *,
        llm_limit: int | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        round_state = _get_round(state, round_index)
        path = Path(round_state["files"]["enriched"])
        _, rows = read_csv(path)
        eligible = [
            row
            for row in rows
            if row.get("auto_screening_decision") in {"include_candidate", "needs_review"}
            and not row.get("llm_decision")
        ]
        if llm_limit is not None:
            eligible = eligible[: max(int(llm_limit), 0)]
        prompt_chars = len(self.store.system_prompt_path(project_slug).read_text(encoding="utf-8"))
        config = load_config(self.store.config_path(project_slug))
        batch_size = max(config.llm_screening.batch_size, 1)
        estimated_requests = math.ceil(len(eligible) / batch_size)
        input_chars = prompt_chars * estimated_requests + sum(
            len(row.get("title", "")) + len((row.get("abstract") or "")[:5000]) for row in eligible
        )
        estimated_input_tokens = math.ceil(input_chars * 1.15 / 4)
        maximum_output_tokens = sum(
            min(
                max(config.llm_screening.max_output_tokens, batch_papers * 600),
                32000,
            )
            for batch_papers in _batch_sizes(len(eligible), batch_size)
        )
        estimated_output_tokens = min(
            len(eligible) * output_tokens_per_paper(),
            maximum_output_tokens,
        )
        settings = self.store.load_project(project_slug)
        price = get_model_price(
            settings.llm_model,
            cache_path=self._model_price_cache_path(project_slug),
        )
        estimated_cost = (
            estimate_token_cost(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                price=price,
            )
            if price
            else None
        )
        maximum_cost = (
            estimate_token_cost(
                input_tokens=estimated_input_tokens,
                output_tokens=maximum_output_tokens,
                price=price,
            )
            if price
            else None
        )
        return {
            "papers": len(eligible),
            "model": settings.llm_model,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "maximum_output_tokens": maximum_output_tokens,
            "batch_size": batch_size,
            "estimated_requests": estimated_requests,
            "estimated_cost_usd": estimated_cost,
            "maximum_cost_usd": maximum_cost,
            "price": (
                {
                    "input_per_million": price.input_per_million,
                    "output_per_million": price.output_per_million,
                    "source": price.source,
                    "source_url": price.source_url,
                    "updated_at": price.updated_at,
                }
                if price
                else None
            ),
        }

    def refresh_llm_model_price(self, project_slug: str):
        settings = self.store.load_project(project_slug)
        if not is_openai_api_base_url(settings.llm_base_url):
            raise RuntimeError(
                "Live price refresh is available only for the official OpenAI API. "
                "Add custom-provider prices to configs/model_pricing.yaml."
            )
        return refresh_openai_model_price(
            settings.llm_model,
            cache_path=self._model_price_cache_path(project_slug),
        )

    def prompt_replay_pending_for_round(self, project_slug: str, round_index: int) -> bool:
        state = self.load_current_state(project_slug)
        return _round_prompt_replay_requested(state, _get_round(state, round_index))

    def _model_price_cache_path(self, project_slug: str) -> Path:
        return self.store.project_dir(project_slug) / "cache" / "model_pricing.json"

    def update_audit(
        self,
        project_slug: str,
        round_index: int,
        updates: list[dict[str, str]],
    ):
        state = self.load_current_state(project_slug)
        round_state = _get_round(state, round_index)
        summary = update_audit_rows(Path(round_state["files"]["audit"]), updates)
        _apply_audit_summary(round_state, summary)
        _invalidate_derived_outputs(state)
        self._save_state(project_slug, state)
        return summary

    def reconcile_snowball_audit(self, project_slug: str, round_index: int) -> int:
        state = self.load_current_state(project_slug)
        round_state = _get_round(state, round_index)
        audit_value = round_state.get("files", {}).get("audit")
        if round_state.get("kind") != "snowball" or not audit_value:
            return 0

        seen: set[str] = set()
        for item in state.get("rounds", []):
            if int(item.get("index", -1)) >= round_index:
                continue
            previous_audit = item.get("files", {}).get("audit")
            if not previous_audit or not Path(previous_audit).exists():
                continue
            _, previous_rows = read_csv(Path(previous_audit))
            for row in previous_rows:
                seen.update(paper_keys(row))

        audit_path = Path(audit_value)
        strip_per_paper_citation_diagnostics(audit_path)
        fieldnames, rows = read_csv(audit_path)
        emitted: set[str] = set()
        kept_rows: list[dict[str, str]] = []
        for row in rows:
            keys = paper_keys(row)
            if keys & seen or keys & emitted:
                continue
            kept_rows.append(row)
            emitted.update(keys)
        removed = len(rows) - len(kept_rows)
        if not removed:
            return 0

        backup_path = audit_path.with_name(f"{audit_path.stem}_before_duplicate_cleanup.csv")
        if not backup_path.exists():
            copy2(audit_path, backup_path)
        write_audit_csv(audit_path, kept_rows, fieldnames)
        summary = summarize_audit(kept_rows)
        round_state["files"]["audit_pre_cleanup"] = str(backup_path)
        round_state.setdefault("counts", {})["audit_duplicates_removed"] = removed
        round_state["counts"]["audit_queue"] = summary.total
        _apply_audit_summary(round_state, summary)
        if round_state is state["rounds"][-1]:
            round_state["status"] = (
                "converged"
                if summary.total == 0 and _failed_seed_coverage_count(round_state) == 0
                else "ready_for_review"
            )
            state["status"] = (
                "converged" if round_state["status"] == "converged" else "awaiting_manual_review"
            )
        _invalidate_derived_outputs(state)
        self._save_state(project_slug, state)
        return removed

    def prompt_refinement_overview(self, project_slug: str) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        audit_paths = _audit_paths(state)
        _, audit_rows = _cumulative_audit_rows(state, reviewed_only=False)
        summary = summarize_audit(audit_rows)
        _, replay_counts = _initial_ai_exclusion_summary(state)
        if not audit_paths:
            return {
                "available": False,
                "audit_total": 0,
                "unreviewed": 0,
                "initial_ai_exclusions": 0,
                "reviewed_removed_before_replay": 0,
                "audit_rounds": 0,
                "refinement": state.get("prompt_refinement", {}),
            }
        return {
            "available": True,
            "audit_total": summary.total,
            "unreviewed": summary.unreviewed,
            "initial_ai_exclusions": replay_counts["eligible"],
            "reviewed_removed_before_replay": replay_counts["reviewed_removed"],
            "audit_rounds": len(audit_paths),
            "refinement": state.get("prompt_refinement", {}),
        }

    def generate_prompt_refinement(
        self,
        project_slug: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        audit_paths = _audit_paths(state)
        if not audit_paths:
            raise RuntimeError("Complete a manual review round before refining the prompt.")
        audit_fields, audit_rows = _cumulative_audit_rows(state, reviewed_only=False)
        audit_summary = summarize_audit(audit_rows)
        if audit_summary.unreviewed:
            raise RuntimeError(
                "Finish every paper in the current audit history before refining the prompt."
            )
        settings = self.store.load_project(project_slug)
        api_key = self.store.read_api_key(project_slug) or os.environ.get("OPENAI_API_KEY", "")
        client = OpenAIResearchClient(
            base_url=settings.llm_base_url,
            api_key=api_key,
            model=settings.prompt_refinement_model,
        )
        prompt_path = self.store.system_prompt_path(project_slug)
        old_prompt = prompt_path.read_text(encoding="utf-8")
        refinement_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = (
            self.store.project_dir(project_slug)
            / "configs"
            / "prompts"
            / "refinements"
            / refinement_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = output_dir / "reviewed_papers.csv"
        write_audit_csv(audit_path, audit_rows, audit_fields)
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Prompt refinement",
            heading="Learning from cumulative manual audits",
            stages=PROMPT_REFINEMENT_STAGES,
            callback=progress,
            paper_count=len(audit_rows),
        )

        try:
            _notify(
                tracked_progress,
                "Learning from manual audit",
                "Comparing AI recommendations with all completed human decisions and notes.",
                len(audit_rows),
                len(audit_rows),
            )
            result = build_prompt_refinement(
                client=client,
                old_prompt=old_prompt,
                audit_rows=audit_rows,
                research_question=settings.research_question,
                scope_description=settings.scope_description,
                inclusion_criteria=settings.inclusion_criteria,
                exclusion_criteria=settings.exclusion_criteria,
                output_dir=output_dir,
            )
            _notify(
                tracked_progress,
                "Prompt proposal",
                "Saving the proposed prompt for human approval.",
            )
            previous_refinement = state.get("prompt_refinement")
            if isinstance(previous_refinement, dict) and previous_refinement.get("refinement_id"):
                history = state.setdefault("prompt_refinement_history", [])
                if not any(
                    item.get("refinement_id") == previous_refinement.get("refinement_id")
                    for item in history
                    if isinstance(item, dict)
                ):
                    history.append(dict(previous_refinement))
            replay_state = _ensure_prompt_replay_state(state)
            state["prompt_refinement"] = {
                "refinement_id": refinement_id,
                "status": "proposed",
                "source_round": max(
                    int(item.get("index", 0))
                    for item in state.get("rounds", [])
                    if item.get("files", {}).get("audit")
                ),
                "source_rounds": [
                    int(item.get("index", 0))
                    for item in state.get("rounds", [])
                    if item.get("files", {}).get("audit")
                ],
                "model": settings.prompt_refinement_model,
                "generated_at": _now(),
                "audit_path": str(audit_path),
                "audit_sources": [str(path) for path in audit_paths],
                "audit_sha256": _audit_fingerprint(audit_paths),
                "baseline_prompt_sha256": _text_sha256(old_prompt.strip()),
                "rows_total": result.rows_total,
                "rows_used": result.rows_used,
                "proposal_path": str(result.proposal_path),
                "baseline_prompt_path": str(result.baseline_prompt_path),
                "proposed_prompt_path": str(result.proposed_prompt_path),
                "feedback_path": str(result.feedback_path),
                "feedback_csv_path": str(audit_path),
                "change_summary": result.change_summary,
                "replay_status": replay_state.get("status", "not_available"),
            }
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Prompt proposal ready",
                message="Review and approve the proposed screening prompt.",
                paper_count=len(audit_rows),
            )
            return state
        except TaskCancelled:
            self._mark_progress_cancelled(project_slug, state)
            return state
        except Exception as exc:
            self._mark_progress_failed(project_slug, state, exc)
            raise

    def load_prompt_refinement_proposal(self, project_slug: str) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        proposal_value = state.get("prompt_refinement", {}).get("proposal_path")
        if not proposal_value:
            raise RuntimeError("No prompt refinement proposal is available.")
        return load_prompt_refinement(Path(proposal_value))

    def approve_prompt_refinement(
        self,
        project_slug: str,
        proposed_prompt: str,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        refinement = state.get("prompt_refinement", {})
        if refinement.get("status") != "proposed":
            raise RuntimeError("No pending prompt proposal is available for approval.")
        source_values = refinement.get("audit_sources")
        if isinstance(source_values, list):
            source_paths = [Path(value) for value in source_values]
            current_paths = _audit_paths(state)
            audit_changed = (
                [str(path.resolve()) for path in source_paths]
                != [str(path.resolve()) for path in current_paths]
                or any(not path.exists() for path in source_paths)
                or _audit_fingerprint(source_paths) != refinement.get("audit_sha256")
            )
        else:
            audit_path = Path(refinement["audit_path"])
            audit_changed = not audit_path.exists() or _file_sha256(audit_path) != refinement.get(
                "audit_sha256"
            )
        if audit_changed:
            raise RuntimeError(
                "The audited feedback changed after this proposal was generated. "
                "Generate a new proposal."
            )
        current_prompt = self.store.system_prompt_path(project_slug).read_text(encoding="utf-8")
        if _text_sha256(current_prompt.strip()) != refinement.get("baseline_prompt_sha256"):
            raise RuntimeError(
                "The screening prompt changed after this proposal was generated. "
                "Generate a new proposal."
            )
        approved_prompt = proposed_prompt.strip()
        if not approved_prompt:
            raise ValueError("The system prompt cannot be empty.")
        approved_path = Path(refinement["proposal_path"]).with_name("approved_prompt.txt")
        approved_path.write_text(approved_prompt + "\n", encoding="utf-8")
        self.store.save_system_prompt(project_slug, approved_prompt)
        replay = _ensure_prompt_replay_state(state)
        if replay.get("status") not in {"pending", "completed"}:
            replay.update(
                {
                    "status": "pending",
                    "refinement_id": refinement.get("refinement_id", ""),
                    "approved_prompt_path": str(approved_path),
                    "approved_at": _now(),
                }
            )
        refinement.update(
            {
                "status": "approved",
                "approved_at": _now(),
                "approved_prompt_path": str(approved_path),
                "approved_prompt_sha256": _text_sha256(approved_prompt),
                "replay_status": replay.get("status", "not_available"),
            }
        )
        self._save_state(project_slug, state)
        return state

    def reject_prompt_refinement(self, project_slug: str) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        refinement = state.get("prompt_refinement", {})
        if refinement.get("status") != "proposed":
            raise RuntimeError("No pending prompt proposal is available for rejection.")
        replay = _ensure_prompt_replay_state(state)
        refinement.update(
            {
                "status": "rejected",
                "rejected_at": _now(),
                "replay_status": replay.get("status", "not_available"),
            }
        )
        self._save_state(project_slug, state)
        return state

    def prompt_replay_overview(self, project_slug: str) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        refinement = state.get("prompt_refinement", {})
        replay = _ensure_prompt_replay_state(state)
        _, filter_counts = _initial_ai_exclusion_summary(state)
        return {
            "approved": replay.get("status") in {"pending", "completed"},
            "refinement_id": replay.get("refinement_id", ""),
            "replay_status": replay.get("status", "not_available"),
            "source_exclusions": filter_counts["deduplicated_exclusions"],
            "reviewed_removed": filter_counts["reviewed_removed"],
            "eligible": filter_counts["eligible"],
            "replay_round": replay.get("replay_round"),
            "latest_refinement_id": refinement.get("refinement_id", ""),
        }

    def add_manual_paper(
        self,
        project_slug: str,
        record: PaperRecord,
        note: str = "",
        *,
        round_index: int | None = None,
    ) -> dict[str, Any]:
        """Persist a known paper and queue it for the manual enrichment loop."""

        manual_store = ManualPaperStore(self.store.project_dir(project_slug))
        saved, collection_added = manual_store.add(record, note)
        state = self.current_state_or_none(project_slug)
        if state is None:
            return {
                "status": "saved_for_discovery",
                "collection_added": collection_added,
            }

        audit_rounds = [
            item for item in state.get("rounds", []) if item.get("files", {}).get("audit")
        ]
        if not audit_rounds:
            return {
                "status": "saved_for_discovery",
                "collection_added": collection_added,
            }

        for item in audit_rounds:
            _, existing_rows = read_csv(Path(item["files"]["audit"]))
            if any(_audit_row_matches_record(row, saved) for row in existing_rows):
                return {
                    "status": "already_in_review",
                    "round_index": int(item["index"]),
                    "collection_added": collection_added,
                }

        target_round = _select_audit_round(audit_rounds, round_index)
        target_round["counts"]["manual_records"] = len(manual_store.load())
        target_round["counts"]["manual_pending"] = len(
            _manual_enrichment_targets(
                manual_store.load(),
                _audit_rows_by_round(audit_rounds),
                int(target_round["index"]),
            )
        )
        self._save_state(project_slug, state)
        return {
            "status": "queued_for_enrichment",
            "round_index": int(target_round["index"]),
            "collection_added": collection_added,
            "pending": target_round["counts"]["manual_pending"],
        }

    def manual_enrichment_status(
        self,
        project_slug: str,
        round_index: int,
    ) -> dict[str, int]:
        state = self.load_current_state(project_slug)
        audit_rounds = [
            item for item in state.get("rounds", []) if item.get("files", {}).get("audit")
        ]
        target_round = _select_audit_round(audit_rounds, round_index)
        manual_records = ManualPaperStore(self.store.project_dir(project_slug)).load()
        rows_by_round = _audit_rows_by_round(audit_rounds)
        targets = _manual_enrichment_targets(
            manual_records,
            rows_by_round,
            int(target_round["index"]),
        )
        target_rows = dict(rows_by_round)[int(target_round["index"])]
        return {
            "saved": len(manual_records),
            "pending": len(targets),
            "enriched": sum(
                _is_direct_manual_audit_row(row) and _manual_enrichment_completed(row)
                for row in target_rows
            ),
        }

    def enrich_manual_additions(
        self,
        project_slug: str,
        round_index: int,
        *,
        enrich_limit: int | None = None,
        core_online: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Enrich queued researcher additions and return them to an audit round."""

        state = self.load_current_state(project_slug)
        audit_rounds = [
            item for item in state.get("rounds", []) if item.get("files", {}).get("audit")
        ]
        target_round = _select_audit_round(audit_rounds, round_index)
        manual_store = ManualPaperStore(self.store.project_dir(project_slug))
        manual_records = manual_store.load()
        targets = _manual_enrichment_targets(
            manual_records,
            _audit_rows_by_round(audit_rounds),
            int(target_round["index"]),
        )
        if not targets:
            raise RuntimeError("No manually added papers are waiting for enrichment.")
        if not self.store.has_api_key(project_slug) and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Save or provide an OpenAI API key before screening manual additions."
            )

        audit_path = Path(target_round["files"]["audit"])
        _, current_audit_rows, current_summary = load_audit(audit_path)
        config = load_config(self.store.config_path(project_slug))
        batch_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        batch_dir = (
            self.store.project_dir(project_slug)
            / "runs"
            / state["run_id"]
            / "manual_enrichment"
            / f"round_{round_index}"
            / batch_id
        )
        input_path = batch_dir / "manual_input.csv"
        venue_path = batch_dir / "manual_venues.csv"
        enriched_path = batch_dir / "manual_enriched.csv"
        llm_path = batch_dir / "manual_llm_screened.csv"
        recommendation_path = batch_dir / "manual_recommendations.csv"
        llm_report_path = batch_dir / "manual_llm_screening_report.md"
        final_summary_path = batch_dir / "manual_llm_screening_summary.json"
        input_rows = [_manual_enrichment_input(record) for record in targets]
        input_fields = list(dict.fromkeys(key for row in input_rows for key in row))
        write_audit_csv(input_path, input_rows, input_fields)

        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Manual paper enrichment",
            heading="Enriching manually added papers",
            stages=MANUAL_ENRICHMENT_STAGES,
            callback=progress,
            paper_count=current_summary.total,
        )
        try:
            _notify(
                tracked_progress,
                "Manual additions",
                "Deduplicating researcher additions before enrichment.",
                len(targets),
                len(targets),
            )
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            publication_resolution_path = batch_dir / "publication_resolution.json"
            venue_result = enrich_venue_quality(
                input_path,
                venue_path,
                venue_config,
                survey_config=config,
                publication_resolution_path=publication_resolution_path,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Venue enrichment",
                    "Resolving published versions and adding venue quality metadata.",
                ),
            )
            write_venue_quality_summary(
                venue_result.summary,
                batch_dir / "venue_quality_summary.json",
            )
            enrichment_result = enrich_candidates(
                venue_path,
                enriched_path,
                config.enrichment,
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Looking up missing abstracts through the configured provider chain.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                batch_dir / "abstract_enrichment_summary.json",
            )
            _notify(
                tracked_progress,
                "AI abstract screening",
                "Analyzing every manually added paper before human review.",
            )
            llm_result = llm_screen_candidates(
                enriched_path,
                llm_path,
                config.llm_screening,
                overwrite=True,
                progress_callback=_item_progress(
                    tracked_progress,
                    "AI abstract screening",
                    "Analyzing manually added papers with the configured model.",
                ),
            )
            write_llm_screening_summary(
                llm_result.summary,
                batch_dir / "manual_llm_request_summary.json",
            )
            recommendation_result = summarize_llm_screening(
                llm_path,
                llm_report_path,
                recommendation_path,
                final_summary_path,
            )

            _notify(
                tracked_progress,
                "Return to manual review",
                "Returning every manually added paper to the selected review round.",
            )
            fieldnames, audit_rows = read_csv(audit_path)
            appended = 0
            for screened_row in recommendation_result.rows:
                match_index = next(
                    (
                        index
                        for index, audit_row in enumerate(audit_rows)
                        if _audit_rows_match(audit_row, screened_row)
                    ),
                    None,
                )
                if match_index is None:
                    prepared = _manual_audit_row(
                        PaperRecord.from_dict(screened_row),
                        screened_row.get("manual_note", ""),
                    )
                    prepared.update(screened_row)
                    audit_rows.append(prepared)
                    match_index = len(audit_rows) - 1
                    appended += 1
                else:
                    decision = audit_rows[match_index].get("manual_decision", "")
                    notes = audit_rows[match_index].get("manual_notes", "")
                    audit_rows[match_index].update(screened_row)
                    audit_rows[match_index]["manual_decision"] = decision
                    audit_rows[match_index]["manual_notes"] = notes
                audit_rows[match_index].update(
                    {
                        "auto_screening_decision": "needs_review",
                        "auto_screening_reason": "Added directly by the researcher.",
                        "manual_review_added": "true",
                        "manual_review_required": "true",
                        "manual_enrichment_status": "completed",
                        "manual_enriched_at": _now(),
                    }
                )

            output_fields = list(
                dict.fromkeys([*fieldnames, *(key for row in audit_rows for key in row)])
            )
            write_audit_csv(audit_path, audit_rows, output_fields)
            summary = summarize_audit(audit_rows)
            _apply_audit_summary(target_round, summary)
            enriched_manual_rows = [
                row
                for row in audit_rows
                if _is_direct_manual_audit_row(row) and _manual_enrichment_completed(row)
            ]
            target_round["counts"].update(
                {
                    "audit_queue": summary.total,
                    "manual_records": len(manual_records),
                    "manual_pending": 0,
                    "manual_enriched": len(enriched_manual_rows),
                    "manual_review_additions": len(enriched_manual_rows),
                    "manual_publication_resolution_attempted": int(
                        target_round["counts"].get("manual_publication_resolution_attempted") or 0
                    )
                    + getattr(
                        venue_result.summary,
                        "publication_resolution_attempted",
                        0,
                    ),
                    "manual_published_versions_resolved": int(
                        target_round["counts"].get("manual_published_versions_resolved") or 0
                    )
                    + getattr(
                        venue_result.summary,
                        "published_versions_resolved",
                        0,
                    ),
                    "manual_abstracts_found": sum(
                        bool((row.get("abstract") or "").strip()) for row in enriched_manual_rows
                    ),
                    "manual_llm_screened": sum(
                        bool((row.get("llm_decision") or "").strip())
                        for row in enriched_manual_rows
                    ),
                    "manual_llm_excluded": sum(
                        row.get("llm_decision") == "exclude" for row in enriched_manual_rows
                    ),
                    "manual_llm_failed": sum(
                        row.get("llm_status") == "failed" for row in enriched_manual_rows
                    ),
                    "abstract_api_requests": int(
                        target_round["counts"].get("abstract_api_requests") or 0
                    )
                    + enrichment_result.summary.api_requests,
                    "abstract_batch_requests": int(
                        target_round["counts"].get("abstract_batch_requests") or 0
                    )
                    + enrichment_result.summary.batch_requests,
                    "abstract_cache_hits": int(
                        target_round["counts"].get("abstract_cache_hits") or 0
                    )
                    + enrichment_result.summary.cache_hits,
                    "abstract_rate_limit_retries": int(
                        target_round["counts"].get("abstract_rate_limit_retries") or 0
                    )
                    + enrichment_result.summary.rate_limit_retries,
                    "abstract_rate_limit_wait_seconds": float(
                        target_round["counts"].get("abstract_rate_limit_wait_seconds") or 0
                    )
                    + enrichment_result.summary.rate_limit_wait_seconds,
                }
            )
            target_round["files"]["manual_enrichment_latest"] = str(enriched_path)
            target_round["files"]["manual_publication_resolution_latest"] = str(
                publication_resolution_path
            )
            target_round["files"]["manual_llm_screened_latest"] = str(llm_path)
            target_round["files"]["manual_llm_report_latest"] = str(llm_report_path)
            target_round["files"]["manual_recommendations_latest"] = str(recommendation_path)
            _record_manual_enrichment_flow(target_round, audit_rows)
            target_round["status"] = "ready_for_review"
            target_round["error"] = ""
            state["status"] = "awaiting_manual_review"
            _set_progress_paper_count(state, summary.total)
            _invalidate_derived_outputs(state)
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Return to manual review",
                message=(
                    f"Enriched {len(targets)} manually added papers; "
                    f"{appended} entered the review queue."
                ),
                paper_count=summary.total,
            )
            return state
        except TaskCancelled:
            self._mark_progress_cancelled(project_slug, state)
            return state
        except Exception as exc:
            self._mark_progress_failed(project_slug, state, exc)
            raise

    def remove_manual_paper(
        self,
        project_slug: str,
        record: PaperRecord,
    ) -> dict[str, Any]:
        """Remove a saved paper and any audit row created by manual enrichment."""

        manual_store = ManualPaperStore(self.store.project_dir(project_slug))
        collection_removed = manual_store.remove(record.dedupe_key())
        manual_records = manual_store.load()
        state = self.current_state_or_none(project_slug)
        if state is None:
            return {
                "status": "removed_from_collection",
                "collection_removed": collection_removed,
                "review_rows_removed": 0,
            }

        review_rows_removed = 0
        for round_state in state.get("rounds", []):
            audit_value = round_state.get("files", {}).get("audit")
            if not audit_value:
                continue
            audit_path = Path(audit_value)
            fieldnames, rows = read_csv(audit_path)
            retained = [
                row
                for row in rows
                if not (_is_direct_manual_audit_row(row) and _audit_row_matches_record(row, record))
            ]
            removed_here = len(rows) - len(retained)
            if not removed_here:
                continue

            review_rows_removed += removed_here
            write_audit_csv(audit_path, retained, fieldnames)
            summary = summarize_audit(retained)
            round_state["counts"]["audit_queue"] = summary.total
            round_state["counts"]["manual_records"] = len(manual_records)
            enriched_manual_rows = [
                row
                for row in retained
                if _is_direct_manual_audit_row(row) and _manual_enrichment_completed(row)
            ]
            round_state["counts"]["manual_enriched"] = len(enriched_manual_rows)
            round_state["counts"]["manual_review_additions"] = len(enriched_manual_rows)
            round_state["counts"]["manual_abstracts_found"] = sum(
                bool((row.get("abstract") or "").strip()) for row in enriched_manual_rows
            )
            round_state["counts"]["manual_llm_screened"] = sum(
                bool((row.get("llm_decision") or "").strip()) for row in enriched_manual_rows
            )
            round_state["counts"]["manual_llm_excluded"] = sum(
                row.get("llm_decision") == "exclude" for row in enriched_manual_rows
            )
            round_state["counts"]["manual_llm_failed"] = sum(
                row.get("llm_status") == "failed" for row in enriched_manual_rows
            )
            _apply_audit_summary(round_state, summary)
            _record_manual_enrichment_flow(round_state, retained)

        if collection_removed:
            audit_rounds = [
                item for item in state.get("rounds", []) if item.get("files", {}).get("audit")
            ]
            rows_by_round = _audit_rows_by_round(audit_rounds)
            for round_state in audit_rounds:
                round_state["counts"]["manual_records"] = len(manual_records)
                if "manual_pending" in round_state["counts"]:
                    round_state["counts"]["manual_pending"] = len(
                        _manual_enrichment_targets(
                            manual_records,
                            rows_by_round,
                            int(round_state["index"]),
                        )
                    )

        if collection_removed or review_rows_removed:
            progress_count = state.get("progress", {}).get("paper_count")
            if review_rows_removed and progress_count not in (None, ""):
                _set_progress_paper_count(
                    state,
                    max(int(progress_count) - review_rows_removed, 0),
                )
            _invalidate_derived_outputs(state)
            self._save_state(project_slug, state)

        return {
            "status": ("removed_from_review" if review_rows_removed else "removed_from_collection"),
            "collection_removed": collection_removed,
            "review_rows_removed": review_rows_removed,
        }

    def generate_exports(self, project_slug: str) -> dict[str, Path]:
        state = self.load_current_state(project_slug)
        project_dir = self.store.project_dir(project_slug)
        audit_paths: list[Path] = []
        for item in state["rounds"]:
            audit_value = item.get("files", {}).get("audit")
            if not audit_value:
                continue
            audit_path = Path(audit_value)
            if item.get("kind") == "snowball":
                strip_per_paper_citation_diagnostics(audit_path)
            audit_paths.append(audit_path)
        if not audit_paths:
            raise RuntimeError("No manual audit is available for export.")
        export_dir = project_dir / "exports" / state["run_id"]
        cumulative_path = export_dir / "final_audit.csv"
        included_path = export_dir / "final_included_papers.csv"
        _, unique, included = build_cumulative_audit(audit_paths, cumulative_path, included_path)
        _, included_rows = read_csv(included_path)
        report_path = export_dir / "final_report.md"
        report_path.write_text(
            _build_final_report(
                project_name=self.store.load_project(project_slug).name,
                run_id=state["run_id"],
                audited=unique,
                included_rows=included_rows,
                rounds=len(audit_paths),
            ),
            encoding="utf-8",
        )
        state["exports"] = {
            "audit": str(cumulative_path),
            "included": str(included_path),
            "report": str(report_path),
            "included_count": included,
        }
        record_flow_stage(
            state["rounds"][-1],
            key="final_corpus",
            label="Final corpus",
            input_count=unique,
            retained_count=included,
            excluded_count=max(unique - included, 0),
            stage_type="review",
            details={"audit rounds": len(audit_paths)},
        )
        self._save_state(project_slug, state)
        return {"audit": cumulative_path, "included": included_path, "report": report_path}

    def analyze_final_corpus(
        self,
        project_slug: str,
        *,
        criteria: str = "",
        model: str = "",
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        included_value = state.get("exports", {}).get("included")
        if not included_value or not Path(included_value).exists():
            raise RuntimeError("Generate final exports before analyzing the corpus.")
        rows = load_csv_rows(Path(included_value))
        if not rows:
            raise RuntimeError("The final included corpus is empty.")
        settings = self.store.load_project(project_slug)
        selected_model = model.strip() or settings.corpus_analysis_model
        api_key = self.store.read_api_key(project_slug) or os.environ.get("OPENAI_API_KEY", "")
        client = OpenAIResearchClient(
            base_url=settings.llm_base_url,
            api_key=api_key,
            model=selected_model,
        )
        analyzer = CorpusAnalyzer(client)
        analysis_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = (
            self.store.project_dir(project_slug) / "analysis" / state["run_id"] / analysis_id
        )
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Corpus analysis",
            heading="Analyzing the final corpus",
            stages=CORPUS_ANALYSIS_STAGES,
            callback=progress,
            paper_count=len(rows),
        )

        try:
            result = analyzer.analyze(
                rows=rows,
                research_question=settings.research_question,
                scope_description=settings.scope_description,
                criteria=criteria,
                output_dir=output_dir,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Paper classification",
                    "Classifying every paper with the fixed taxonomy.",
                ),
                stage_callback=lambda stage, message: _notify(
                    tracked_progress,
                    stage,
                    message,
                ),
            )
            state["corpus_analysis"] = {
                "analysis_id": analysis_id,
                "model": selected_model,
                "criteria": criteria.strip(),
                "paper_count": len(rows),
                "taxonomy": str(result.taxonomy_path),
                "classifications": str(result.classifications_path),
                "report": str(result.report_path),
                "created_at": _now(),
            }
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Corpus analysis complete",
                message="Corpus classification artifacts are ready.",
                paper_count=len(rows),
            )
            return state
        except TaskCancelled:
            self._mark_progress_cancelled(project_slug, state)
            return state
        except Exception as exc:
            self._mark_progress_failed(project_slug, state, exc)
            raise

    def load_current_state(self, project_slug: str) -> dict[str, Any]:
        settings = self.store.load_project(project_slug)
        if not settings.current_run_id:
            raise FileNotFoundError("This project does not have a run yet.")
        path = self._state_path(project_slug, settings.current_run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def current_state_or_none(self, project_slug: str) -> dict[str, Any] | None:
        try:
            return self.load_current_state(project_slug)
        except FileNotFoundError:
            return None

    def run_log_path(self, project_slug: str) -> Path:
        state = self.load_current_state(project_slug)
        state_path = self._state_path(project_slug, state["run_id"])
        log_path = state_path.with_name("run_log.json")
        if not log_path.exists():
            _write_run_log(state_path, state)
        return log_path

    def _state_path(self, project_slug: str, run_id: str) -> Path:
        return self.store.project_dir(project_slug) / "runs" / run_id / "state.json"

    def _save_state(self, project_slug: str, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        state_path = self._state_path(project_slug, state["run_id"])
        _write_json(state_path, state)
        _write_json(state_path.with_name("flow_summary.json"), flow_summary_payload(state))
        state_path.with_name("flow_diagram.svg").write_text(
            build_flow_svg(state),
            encoding="utf-8",
        )
        _write_run_log(state_path, state)

    def _begin_progress(
        self,
        project_slug: str,
        state: dict[str, Any],
        *,
        operation: str,
        heading: str,
        stages: list[str],
        callback: ProgressCallback | None,
        paper_count: int,
    ) -> ProgressCallback:
        now = _now()
        state["progress"] = {
            "operation": operation,
            "heading": heading,
            "status": "running",
            "stages": list(stages),
            "stage": stages[0],
            "message": "Waiting for the first stage.",
            "completed": None,
            "total": None,
            "current": "",
            "paper_count": paper_count,
            "started_at": now,
            "updated_at": now,
        }
        self._save_state(project_slug, state)

        def update(
            stage: str,
            message: str,
            completed: int | None,
            total: int | None,
            current: str,
        ) -> None:
            raise_if_cancelled()
            progress_state = state["progress"]
            progress_state.update(
                {
                    "stage": stage,
                    "message": message,
                    "completed": completed,
                    "total": total,
                    "current": current,
                    "updated_at": _now(),
                }
            )
            self._save_state(project_slug, state)
            _notify(callback, stage, message, completed, total, current)

        return update

    def _complete_progress(
        self,
        project_slug: str,
        state: dict[str, Any],
        callback: ProgressCallback | None,
        *,
        stage: str,
        message: str,
        paper_count: int,
    ) -> None:
        raise_if_cancelled()
        progress_state = state.get("progress", {})
        progress_state.update(
            {
                "status": "completed",
                "stage": stage,
                "message": message,
                "completed": None,
                "total": None,
                "current": "",
                "paper_count": paper_count,
                "updated_at": _now(),
            }
        )
        state["progress"] = progress_state
        self._save_state(project_slug, state)
        _notify(callback, stage, message)

    def _mark_failed(
        self,
        project_slug: str,
        state: dict[str, Any],
        round_state: dict[str, Any],
        exc: Exception,
    ) -> None:
        round_state["status"] = "failed"
        round_state["error"] = str(exc)
        state["status"] = "failed"
        progress_state = state.get("progress", {})
        progress_state.update(
            {
                "status": "failed",
                "message": str(exc),
                "current": "",
                "updated_at": _now(),
            }
        )
        state["progress"] = progress_state
        self._save_state(project_slug, state)

    def _mark_cancelled(
        self,
        project_slug: str,
        state: dict[str, Any],
        round_state: dict[str, Any],
    ) -> None:
        if round_state.get("status") == "running":
            round_state["status"] = "cancelled"
        state["status"] = "cancelled"
        self._mark_progress_cancelled(project_slug, state)

    def _mark_progress_cancelled(
        self,
        project_slug: str,
        state: dict[str, Any],
    ) -> None:
        progress_state = state.get("progress", {})
        progress_state.update(
            {
                "status": "cancelled",
                "message": "Run stopped by user. Completed files have been retained.",
                "current": "",
                "updated_at": _now(),
            }
        )
        state["progress"] = progress_state
        self._save_state(project_slug, state)

    def _mark_progress_failed(
        self,
        project_slug: str,
        state: dict[str, Any],
        exc: Exception,
    ) -> None:
        progress_state = state.get("progress", {})
        progress_state.update(
            {
                "status": "failed",
                "message": str(exc),
                "current": "",
                "updated_at": _now(),
            }
        )
        state["progress"] = progress_state
        self._save_state(project_slug, state)


def test_openai_connection(base_url: str, api_key: str, timeout: int = 20) -> tuple[bool, str]:
    key = api_key.strip()
    if not key:
        return False, "Enter an API key first."
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, f"Connection failed: {exc}"
    return True, "Connection succeeded."


def list_openai_models(
    base_url: str,
    api_key: str,
    timeout: int = 20,
) -> tuple[list[str], str]:
    key = api_key.strip()
    if not key:
        return [], "Enter an API key first."
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        values = response.json().get("data", [])
    except (requests.RequestException, ValueError) as exc:
        return [], f"Connection failed: {exc}"
    model_ids = sorted(
        {
            str(item.get("id") or "").strip()
            for item in values
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
    )
    return model_ids, "Connection succeeded."


def _manual_audit_row(record: PaperRecord, note: str) -> dict[str, str]:
    row = {str(key): "" if value is None else str(value) for key, value in record.to_row().items()}
    row.update(
        {
            "auto_screening_decision": "needs_review",
            "auto_screening_reason": "Added directly by the researcher.",
            "final_recommendation": "manual_review",
            "final_priority": "1",
            "final_reason": "Added directly by the researcher for human review.",
            "manual_decision": "",
            "manual_notes": note.strip(),
            "manual_review_added": "true",
        }
    )
    return row


def _normalize_target_seed(
    value: dict[str, str] | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    title = str(value.get("title") or "").strip()
    if not title:
        raise ValueError("A paper title is required for single-paper snowballing.")
    fields = [
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "url",
        "dblp_key",
        "publication_type",
        "source",
        "provider_id",
        "abstract_provider_id",
    ]
    normalized = {field: str(value.get(field) or "").strip() for field in fields}
    normalized.update(
        {
            "title": title,
            "manual_decision": "include",
            "manual_notes": "Single-paper snowball target selected by the researcher.",
        }
    )
    return normalized


def _initial_ai_exclusion_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    rows, _ = _initial_ai_exclusion_summary(state)
    return rows


def _initial_ai_exclusion_summary(
    state: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    initial_round = next(
        (item for item in state.get("rounds", []) if int(item.get("index", -1)) == 0),
        None,
    )
    if initial_round is None:
        return [], _empty_replay_filter_counts()
    llm_value = initial_round.get("files", {}).get("llm_screened")
    if not llm_value or not Path(llm_value).exists():
        return [], _empty_replay_filter_counts()

    _, reviewed_rows = _cumulative_audit_rows(state, reviewed_only=True)
    reviewed_keys = {key for row in reviewed_rows for key in paper_keys(row)}
    _, llm_rows = read_csv(Path(llm_value))
    source_exclusions = [row for row in llm_rows if row.get("llm_decision") == "exclude"]
    exclusions: list[dict[str, str]] = []
    emitted: set[str] = set()
    reviewed_removed = 0
    for row in llm_rows:
        if row.get("llm_decision") != "exclude":
            continue
        keys = paper_keys(row)
        if keys & emitted:
            continue
        emitted.update(keys)
        if keys & reviewed_keys:
            reviewed_removed += 1
            continue
        exclusions.append(row)
    return exclusions, {
        "source_exclusions": len(source_exclusions),
        "deduplicated_exclusions": len(exclusions) + reviewed_removed,
        "reviewed_removed": reviewed_removed,
        "eligible": len(exclusions),
    }


def _empty_replay_filter_counts() -> dict[str, int]:
    return {
        "source_exclusions": 0,
        "deduplicated_exclusions": 0,
        "reviewed_removed": 0,
        "eligible": 0,
    }


def _audit_paths(state: dict[str, Any]) -> list[Path]:
    return [
        Path(value)
        for round_state in state.get("rounds", [])
        if (value := round_state.get("files", {}).get("audit")) and Path(value).exists()
    ]


def _cumulative_audit_rows(
    state: dict[str, Any],
    *,
    reviewed_only: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    alias_indexes: dict[str, int] = {}
    for path in _audit_paths(state):
        input_fields, audit_rows = read_csv(path)
        fieldnames.extend(field for field in input_fields if field not in fieldnames)
        for row in audit_rows:
            if reviewed_only and not _is_human_reviewed(row):
                continue
            keys = paper_keys(row)
            match_index = next(
                (alias_indexes[key] for key in keys if key in alias_indexes),
                None,
            )
            if match_index is None:
                match_index = len(rows)
                rows.append(dict(row))
            else:
                rows[match_index].update(row)
            for key in paper_keys(rows[match_index]):
                alias_indexes[key] = match_index
    return fieldnames, rows


def _is_human_reviewed(row: dict[str, str]) -> bool:
    decision = str(row.get("manual_decision") or "").strip().lower()
    return decision not in {"", "later"}


def _row_alias_indexes(rows: list[dict[str, str]]) -> dict[str, int]:
    return {key: index for index, row in enumerate(rows) for key in paper_keys(row)}


def _matching_row_index(
    indexes: dict[str, int],
    row: dict[str, str],
) -> int | None:
    return next((indexes[key] for key in paper_keys(row) if key in indexes), None)


def _audit_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_prompt_replay_state(state: dict[str, Any]) -> dict[str, Any]:
    replay = state.get("prompt_replay")
    if isinstance(replay, dict):
        return replay
    refinement = state.get("prompt_refinement", {})
    replay = {
        "status": refinement.get("replay_status", "not_available"),
        "refinement_id": refinement.get("refinement_id", ""),
        "approved_prompt_path": refinement.get("approved_prompt_path", ""),
        "replay_round": refinement.get("replay_round"),
        "replayed_at": refinement.get("replayed_at", ""),
        "replayed": int(refinement.get("replayed") or 0),
        "recovered": int(refinement.get("recovered") or 0),
        "reexcluded": int(refinement.get("reexcluded") or 0),
        "failed": int(refinement.get("failed") or 0),
    }
    state["prompt_replay"] = replay
    return replay


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manual_enrichment_input(record: PaperRecord) -> dict[str, str]:
    row = {str(key): "" if value is None else str(value) for key, value in record.to_row().items()}
    row.update(
        {
            "auto_screening_decision": "needs_review",
            "manual_review_added": "true",
            "manual_enrichment_status": "pending",
        }
    )
    return row


def _audit_row_matches_record(row: dict[str, str], record: PaperRecord) -> bool:
    try:
        existing = PaperRecord.from_dict(row)
    except (TypeError, ValueError):
        return False
    return len(dedupe_records([existing, record])) == 1


def _audit_rows_match(first: dict[str, str], second: dict[str, str]) -> bool:
    try:
        return _audit_row_matches_record(first, PaperRecord.from_dict(second))
    except (TypeError, ValueError):
        return False


def _is_direct_manual_audit_row(row: dict[str, str]) -> bool:
    marker = (row.get("manual_review_added") or "").strip().lower()
    if marker in {"true", "1", "yes"}:
        return True
    return row.get("final_reason") == "Added directly by the researcher for human review."


def _manual_enrichment_completed(row: dict[str, str]) -> bool:
    return (row.get("manual_enrichment_status") or "").strip().lower() == "completed"


def _select_audit_round(
    audit_rounds: list[dict[str, Any]],
    round_index: int | None,
) -> dict[str, Any]:
    if not audit_rounds:
        raise RuntimeError("No manual review round is available.")
    if round_index is None:
        return audit_rounds[-1]
    target = next(
        (item for item in audit_rounds if int(item["index"]) == round_index),
        None,
    )
    if target is None:
        raise RuntimeError(f"Review round {round_index} is not available.")
    return target


def _audit_rows_by_round(
    audit_rounds: list[dict[str, Any]],
) -> list[tuple[int, list[dict[str, str]]]]:
    return [
        (int(round_state["index"]), read_csv(Path(round_state["files"]["audit"]))[1])
        for round_state in audit_rounds
    ]


def _manual_enrichment_targets(
    manual_records: list[PaperRecord],
    rows_by_round: list[tuple[int, list[dict[str, str]]]],
    target_round_index: int,
) -> list[PaperRecord]:
    targets: list[PaperRecord] = []
    for record in manual_records:
        matches = [
            (round_index, row)
            for round_index, rows in rows_by_round
            for row in rows
            if _audit_row_matches_record(row, record)
        ]
        if not matches:
            targets.append(record)
            continue
        target_match = next(
            (
                row
                for round_index, row in matches
                if round_index == target_round_index
                and _is_direct_manual_audit_row(row)
                and not _manual_enrichment_completed(row)
            ),
            None,
        )
        if target_match is not None:
            targets.append(dedupe_records([record, PaperRecord.from_dict(target_match)])[0])
    return dedupe_records(targets)


def _record_manual_enrichment_flow(
    round_state: dict[str, Any],
    audit_rows: list[dict[str, str]],
) -> None:
    loop_keys = {
        "manual_review_additions",
        "manual_loop_additions",
        "manual_venue_enrichment",
        "manual_abstract_enrichment",
        "manual_ai_abstract_screening",
        "manual_return_to_review",
    }
    base_stages = [
        stage for stage in round_state.get("flow", []) if stage.get("key") not in loop_keys
    ]
    manual_rows = [
        row
        for row in audit_rows
        if _is_direct_manual_audit_row(row) and _manual_enrichment_completed(row)
    ]
    if not manual_rows:
        round_state["flow"] = base_stages
        return

    count = len(manual_rows)
    abstracts_found = sum(bool((row.get("abstract") or "").strip()) for row in manual_rows)
    ranked = sum(bool(row.get("core_rank") or row.get("impact_factor")) for row in manual_rows)
    ai_screened = sum(bool((row.get("llm_decision") or "").strip()) for row in manual_rows)
    ai_excluded = sum(row.get("llm_decision") == "exclude" for row in manual_rows)
    ai_failed = sum(row.get("llm_status") == "failed" for row in manual_rows)
    loop_holder: dict[str, Any] = {"flow": []}
    record_flow_stage(
        loop_holder,
        key="manual_loop_additions",
        label="Researcher additions",
        input_count=count,
        retained_count=count,
        excluded_count=0,
        stage_type="discovery",
        details={"submitted": count},
    )
    record_flow_stage(
        loop_holder,
        key="manual_venue_enrichment",
        label="Manual venue enrichment",
        input_count=count,
        retained_count=count,
        stage_type="enrichment",
        details={"rank or IF found": ranked},
    )
    record_flow_stage(
        loop_holder,
        key="manual_abstract_enrichment",
        label="Manual abstract enrichment",
        input_count=count,
        retained_count=count,
        stage_type="enrichment",
        details={"abstracts found": abstracts_found},
    )
    record_flow_stage(
        loop_holder,
        key="manual_ai_abstract_screening",
        label="Manual AI abstract screening",
        input_count=count,
        retained_count=count,
        excluded_count=0,
        stage_type="enrichment",
        details={
            "AI screened": ai_screened,
            "AI exclude recommendations": ai_excluded,
            "failed": ai_failed,
        },
    )
    record_flow_stage(
        loop_holder,
        key="manual_return_to_review",
        label="Return to manual review",
        input_count=count,
        retained_count=count,
        stage_type="review",
        details={"review queue total": len(audit_rows)},
        loop_to="human_audit",
    )
    insert_at = next(
        (index + 1 for index, stage in enumerate(base_stages) if stage.get("key") == "human_audit"),
        len(base_stages),
    )
    base_stages[insert_at:insert_at] = loop_holder["flow"]
    round_state["flow"] = base_stages


def _apply_audit_summary(
    round_state: dict[str, Any],
    summary: AuditSummary,
) -> None:
    round_state["counts"]["reviewed"] = summary.reviewed
    round_state["counts"]["unreviewed"] = summary.unreviewed
    included = summary.by_decision.get("include", 0) + summary.by_decision.get("include_related", 0)
    record_flow_stage(
        round_state,
        key="human_audit",
        label="Human audit",
        input_count=summary.total,
        retained_count=included,
        excluded_count=summary.by_decision.get("exclude", 0),
        stage_type="review",
        details={"pending": summary.unreviewed},
    )


def _invalidate_derived_outputs(state: dict[str, Any]) -> None:
    state.pop("exports", None)
    state.pop("corpus_analysis", None)
    for round_state in state.get("rounds", []):
        round_state["flow"] = [
            stage for stage in round_state.get("flow", []) if stage.get("key") != "final_corpus"
        ]


def _citation_provider_failure_count(round_state: dict[str, Any]) -> int:
    values = round_state.get("counts", {}).get("provider_failures", {}) or {}
    if not isinstance(values, dict):
        return 0
    return sum(max(int(value or 0), 0) for value in values.values())


def _failed_seed_coverage_count(round_state: dict[str, Any]) -> int:
    counts = round_state.get("counts", {})
    if "coverage_failed_seeds" in counts:
        return max(int(counts.get("coverage_failed_seeds") or 0), 0)
    successes = counts.get("provider_successes", {}) or {}
    if isinstance(successes, dict) and any(int(value or 0) > 0 for value in successes.values()):
        return 0
    return _citation_provider_failure_count(round_state)


def _retry_provider_order(
    retry_round: dict[str, Any] | None,
    selected_providers: list[str],
) -> list[str]:
    if retry_round is None:
        return selected_providers
    counts = retry_round.get("counts", {})
    successes = counts.get("provider_successes", {}) or {}
    failures = counts.get("provider_failures", {}) or {}
    if not isinstance(successes, dict) or not isinstance(failures, dict):
        return selected_providers
    pending = [
        provider
        for provider in selected_providers
        if int(failures.get(provider, 0) or 0) > 0 or int(successes.get(provider, 0) or 0) <= 0
    ]
    return pending or selected_providers


def _merge_retry_provider_summary(
    result: SnowballingResult,
    *,
    retry_round: dict[str, Any] | None,
    selected_providers: list[str],
) -> SnowballingResult:
    if retry_round is None:
        return result
    prior_values = retry_round.get("counts", {}).get("provider_successes", {}) or {}
    prior_successes = Counter(
        {
            provider: int(prior_values.get(provider, 0) or 0)
            for provider in selected_providers
            if isinstance(prior_values, dict) and int(prior_values.get(provider, 0) or 0) > 0
        }
    )
    prior_successes.update(result.summary.provider_successes)
    unresolved_failures = Counter(
        {
            provider: int(result.summary.provider_failures.get(provider, 0) or 0)
            for provider in selected_providers
            if int(result.summary.provider_failures.get(provider, 0) or 0) > 0
        }
    )
    unresolved_errors = {
        provider: list(result.summary.provider_errors.get(provider, []))
        for provider in unresolved_failures
    }
    summary = replace(
        result.summary,
        provider_order=tuple(selected_providers),
        provider_successes=prior_successes,
        provider_failures=unresolved_failures,
        provider_errors=unresolved_errors,
    )
    return replace(result, summary=summary)


def _ensure_checkpoint_file(source: Path, target: Path) -> None:
    if target.exists() or not source.exists() or source == target:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, target)


def _write_incremental_snowball_candidates(
    snowball_path: Path,
    output_path: Path,
    *,
    prior_paths: list[Path],
) -> int:
    seen: set[str] = set()
    for path in dict.fromkeys(prior_paths):
        if not path.exists():
            continue
        _, prior_rows = read_csv(path)
        for row in prior_rows:
            seen.update(paper_keys(row))

    fieldnames, rows = read_csv(snowball_path)
    incremental_rows: list[dict[str, str]] = []
    for row in rows:
        keys = paper_keys(row)
        if keys & seen:
            continue
        incremental_rows.append(row)
        seen.update(keys)
    write_audit_csv(output_path, incremental_rows, fieldnames)
    return len(incremental_rows)


def _restore_audit_checkpoint(audit_path: Path, checkpoint_path: Path) -> AuditSummary:
    current_fields, current_rows = read_csv(audit_path)
    checkpoint_fields, checkpoint_rows = read_csv(checkpoint_path)
    current_by_key = {paper_key(row): row for row in current_rows}
    for checkpoint_row in checkpoint_rows:
        key = paper_key(checkpoint_row)
        current = current_by_key.get(key)
        if current is not None:
            current["manual_decision"] = checkpoint_row.get("manual_decision", "")
            current["manual_notes"] = checkpoint_row.get("manual_notes", "")
            continue
        if not any(
            (
                checkpoint_row.get("manual_decision"),
                checkpoint_row.get("manual_notes"),
                checkpoint_row.get("manual_review_added"),
            )
        ):
            continue
        restored = dict(checkpoint_row)
        current_rows.append(restored)
        current_by_key[key] = restored
    fields = list(
        dict.fromkeys([*current_fields, *checkpoint_fields, "manual_decision", "manual_notes"])
    )
    write_audit_csv(audit_path, current_rows, fields)
    return summarize_audit(current_rows)


def _new_round_state(index: int, kind: str) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "status": "running",
        "created_at": _now(),
        "files": {},
        "counts": {},
        "flow": [],
        "error": "",
    }


def _retryable_snowball_round(state: dict[str, Any]) -> dict[str, Any] | None:
    rounds = state.get("rounds", [])
    if not rounds:
        return None
    latest = rounds[-1]
    if latest.get("kind") != "snowball":
        return None
    if latest.get("status") in {"failed", "cancelled"}:
        return latest
    return None


def _get_round(state: dict[str, Any], index: int) -> dict[str, Any]:
    for item in state.get("rounds", []):
        if int(item.get("index", -1)) == index:
            return item
    raise KeyError(f"Run does not contain round {index}.")


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    completed: int | None = None,
    total: int | None = None,
    current: str = "",
) -> None:
    if callback:
        callback(stage, message, completed, total, current)


def _item_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> Callable[[int, int, str], None] | None:
    if callback is None:
        return None

    def update(completed: int, total: int, current: str) -> None:
        _notify(callback, stage, message, completed, total, current)

    return update


def _counted_item_progress(
    callback: ProgressCallback | None,
    state: dict[str, Any],
    stage: str,
    message: str,
) -> Callable[[int, int, str, int], None] | None:
    if callback is None:
        return None

    def update(completed: int, total: int, current: str, paper_count: int) -> None:
        _set_progress_paper_count(state, paper_count)
        _notify(callback, stage, message, completed, total, current)

    return update


def _set_progress_paper_count(state: dict[str, Any], paper_count: int) -> None:
    state.setdefault("progress", {})["paper_count"] = max(int(paper_count), 0)


def _batch_sizes(total: int, batch_size: int) -> list[int]:
    safe_total = max(int(total), 0)
    safe_batch_size = max(int(batch_size), 1)
    return [
        min(safe_batch_size, safe_total - start)
        for start in range(0, safe_total, safe_batch_size)
    ]


def _round_prompt_replay_requested(
    state: dict[str, Any],
    round_state: dict[str, Any],
) -> bool:
    replay_state = _ensure_prompt_replay_state(state)
    requested = round_state.get(
        "replay_initial_exclusions",
        round_state.get("kind") == "snowball"
        and state.get("options", {}).get("replay_initial_exclusions"),
    )
    return bool(requested and replay_state.get("status") == "pending")


def _with_title_screening_stage(stages: list[str], enabled: bool) -> list[str]:
    values = list(stages)
    if enabled:
        rule_index = values.index("Rule screening")
        values.insert(rule_index + 1, "AI title screening")
    return values


def _with_prompt_replay_stage(stages: list[str], enabled: bool) -> list[str]:
    values = list(stages)
    if enabled:
        values.insert(0, "Re-screen initial AI exclusions")
    return values


def _title_screening_counts(result: TitleScreeningResult | None) -> dict[str, int]:
    if result is None:
        return {}
    return {
        "title_screened": result.summary.eligible,
        "title_excluded": result.summary.excluded,
        "title_kept": result.summary.kept_for_enrichment,
        "title_screening_batches": result.summary.batches,
        "title_screening_cached": result.summary.cached,
    }


def _without_manual_provenance(record: PaperRecord) -> PaperRecord | None:
    automatic_sources = [source for source in record.discovery_sources if source != "manual"]
    if not automatic_sources:
        return None
    automatic_queries = [query for query in record.discovery_queries if query != "manual addition"]
    return replace(
        record,
        source=record.source if record.source != "manual" else automatic_sources[0],
        query=record.query
        if record.query != "manual addition"
        else (automatic_queries[0] if automatic_queries else ""),
        discovery_sources=automatic_sources,
        discovery_queries=automatic_queries,
        manual_added=False,
        manual_note=None,
    )


def _round_paper_count(round_state: dict[str, Any]) -> int:
    counts = round_state.get("counts", {})
    for key in ("pool_rows", "deduped_records"):
        value = counts.get(key)
        if value not in (None, ""):
            return max(int(value), 0)
    files = round_state.get("files", {})
    for key in ("pool", "enriched", "screened", "venues", "candidates"):
        value = files.get(key)
        if value and Path(value).exists():
            return _csv_row_count(Path(value))
    return 0


def _csv_row_count(path: Path) -> int:
    return len(read_csv(path)[1])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _write_run_log(state_path: Path, state: dict[str, Any]) -> None:
    events_path = state_path.with_name("run_events.jsonl")
    latest_round = state.get("rounds", [])[-1] if state.get("rounds") else {}
    progress = state.get("progress", {})
    event = {
        "state_status": state.get("status", ""),
        "round_index": latest_round.get("index"),
        "round_status": latest_round.get("status", ""),
        "operation": progress.get("operation", ""),
        "progress_status": progress.get("status", ""),
        "stage": progress.get("stage", ""),
        "message": progress.get("message", ""),
        "error": latest_round.get("error", ""),
    }
    signature = hashlib.sha256(
        json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    previous_signature = ""
    if events_path.exists():
        try:
            last_line = next(
                line
                for line in reversed(events_path.read_text(encoding="utf-8").splitlines())
                if line.strip()
            )
            previous_signature = str(json.loads(last_line).get("signature") or "")
        except (OSError, StopIteration, ValueError, TypeError):
            previous_signature = ""
    if signature != previous_signature:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"timestamp": _now(), "signature": signature, **event},
                    ensure_ascii=False,
                )
                + "\n"
            )

    events: list[dict[str, Any]] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                value.pop("signature", None)
                events.append(value)
    _write_json(
        state_path.with_name("run_log.json"),
        {
            "schema_version": 1,
            "exported_at": _now(),
            "project_slug": state.get("project_slug", ""),
            "run_id": state.get("run_id", ""),
            "events": events,
            "state": state,
        },
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _build_final_report(
    *,
    project_name: str,
    run_id: str,
    audited: int,
    included_rows: list[dict[str, str]],
    rounds: int,
) -> str:
    by_year = Counter(row.get("year") or "Unknown" for row in included_rows)
    by_venue = Counter(row.get("venue_type") or "unknown" for row in included_rows)
    lines = [
        f"# {project_name}: Final Survey Corpus",
        "",
        f"- Run: `{run_id}`",
        f"- Audit rounds: {rounds}",
        f"- Unique audited papers: {audited}",
        f"- Included or related papers: {len(included_rows)}",
        "",
        "## Included Papers by Year",
        "",
        "| Year | Papers |",
        "|---|---:|",
    ]
    lines.extend(f"| {year} | {count} |" for year, count in sorted(by_year.items()))
    lines.extend(
        ["", "## Included Papers by Venue Type", "", "| Venue type | Papers |", "|---|---:|"]
    )
    lines.extend(f"| {venue} | {count} |" for venue, count in by_venue.most_common())
    lines.extend(
        ["", "## Included Corpus", "", "| Year | Title | Venue | Decision |", "|---:|---|---|---|"]
    )
    for row in sorted(
        included_rows, key=lambda item: (item.get("year", "9999"), item.get("title", ""))
    ):
        title = (row.get("title") or "").replace("|", "\\|")
        venue = (row.get("venue") or "").replace("|", "\\|")
        lines.append(
            f"| {row.get('year', '')} | {title} | {venue} | {row.get('manual_decision', '')} |"
        )
    lines.append("")
    lines.append(
        "All inclusion decisions in this report were made or confirmed by a human reviewer."
    )
    lines.append("")
    return "\n".join(lines)
