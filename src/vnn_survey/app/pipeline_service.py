from __future__ import annotations

import io
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import get_ident
from typing import Any

import requests
from rich.console import Console

from vnn_survey.ai_research import CorpusAnalyzer, OpenAIResearchClient, load_csv_rows
from vnn_survey.app.audit import (
    build_cumulative_audit,
    create_audit_queue,
    create_manual_recommendations,
    load_audit,
    read_csv,
    update_audit_rows,
)
from vnn_survey.app.manual_papers import ManualPaperStore
from vnn_survey.app.project_store import ProjectStore
from vnn_survey.config import load_config
from vnn_survey.enrichment import enrich_candidates, write_enrichment_summary
from vnn_survey.export import write_csv, write_jsonl
from vnn_survey.llm_screening import llm_screen_candidates, write_llm_screening_summary
from vnn_survey.llm_summary import summarize_llm_screening
from vnn_survey.models import PaperRecord
from vnn_survey.pipeline import (
    collect_from_sources,
    dedupe_records,
    save_collection,
    summarize,
)
from vnn_survey.screening import screen_candidates
from vnn_survey.snowballing import (
    export_seed_papers_from_csv,
    snowball_candidates,
    write_snowballing_summary,
)
from vnn_survey.venue_quality import enrich_venue_quality, write_venue_quality_summary

ProgressCallback = Callable[[str, str, int | None, int | None, str], None]

INITIAL_DISCOVERY_STAGES = [
    "Literature search",
    "Venue enrichment",
    "Rule screening",
    "Abstract enrichment",
]
MANUAL_SYNC_STAGES = [
    "Manual additions",
    "Venue enrichment",
    "Rule screening",
    "Abstract enrichment",
]
CORPUS_ANALYSIS_STAGES = [
    "Taxonomy design",
    "Paper classification",
    "Analysis report",
]
AI_REVIEW_STAGES = ["AI screening", "Recommendation summary", "Audit queue"]
MANUAL_REVIEW_STAGES = ["Review preparation", "Audit queue"]
SNOWBALL_STAGES = [
    "Citation snowballing",
    "Venue enrichment",
    "Rule screening",
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
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if source not in {"auto", "api", "sparql"}:
            raise ValueError("DBLP source must be auto, api, or sparql.")
        settings = self.store.load_project(project_slug)
        config = load_config(self.store.config_path(project_slug))
        selected_sources = list(dict.fromkeys(source_ids or settings.discovery_sources))
        if not selected_sources:
            raise ValueError("Select at least one available literature source.")
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
            stages=INITIAL_DISCOVERY_STAGES,
            callback=progress,
            paper_count=0,
        )

        try:
            _notify(
                tracked_progress,
                "Literature search",
                "Collecting bibliographic records from the selected sources.",
            )
            manual_records = ManualPaperStore(
                self.store.project_dir(project_slug)
            ).load()
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

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Adding publication type, CORE rank, and IF.",
            )
            venue_path = processed_dir / "candidate_papers_venues.csv"
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            venue_result = enrich_venue_quality(
                candidates,
                venue_path,
                venue_config,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Venue enrichment",
                    "Adding publication type, CORE rank, and IF.",
                ),
            )
            write_venue_quality_summary(
                venue_result.summary,
                processed_dir / "venue_quality_summary.json",
            )

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying the project's title exclusion rules.",
            )
            screened_path = processed_dir / "candidate_papers_screened.csv"
            screening_result = screen_candidates(venue_path, screened_path, config.screening)
            _write_json(
                processed_dir / "screening_summary.json",
                {
                    "total": screening_result.summary.total,
                    "by_decision": dict(screening_result.summary.by_decision),
                    "by_bucket": dict(screening_result.summary.by_bucket),
                    "by_exclusion_code": dict(screening_result.summary.by_exclusion_code),
                },
            )

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Looking up abstracts through OpenAlex.",
            )
            enriched_path = processed_dir / "candidate_papers_enriched.csv"
            enrichment_result = enrich_candidates(
                screened_path,
                enriched_path,
                config.enrichment,
                decisions={"include_candidate", "needs_review"},
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Looking up abstracts through OpenAlex.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                processed_dir / "abstract_enrichment_summary.json",
            )

            round_state["status"] = "discovery_complete"
            round_state["files"] = {
                "candidates": str(candidates),
                "venues": str(venue_path),
                "screened": str(screened_path),
                "enriched": str(enriched_path),
            }
            round_state["counts"] = {
                **collection_summary,
                "manual_records": len(manual_records),
                "abstracts_found": enrichment_result.summary.with_abstract,
                "abstracts_attempted": enrichment_result.summary.attempted,
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
            if (record := _without_manual_provenance(PaperRecord.from_dict(row)))
            is not None
        ]
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Manual additions",
            heading="Synchronizing manual additions",
            stages=MANUAL_SYNC_STAGES,
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

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Adding publication type, CORE rank, and IF.",
            )
            venue_path = processed_dir / "candidate_papers_venues.csv"
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            venue_result = enrich_venue_quality(
                candidates_path,
                venue_path,
                venue_config,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Venue enrichment",
                    "Adding publication type, CORE rank, and IF.",
                ),
            )
            write_venue_quality_summary(
                venue_result.summary,
                processed_dir / "venue_quality_summary.json",
            )

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying the project's title exclusion rules.",
            )
            screened_path = processed_dir / "candidate_papers_screened.csv"
            screening_result = screen_candidates(
                venue_path,
                screened_path,
                config.screening,
            )
            _write_json(
                processed_dir / "screening_summary.json",
                {
                    "total": screening_result.summary.total,
                    "by_decision": dict(screening_result.summary.by_decision),
                    "by_bucket": dict(screening_result.summary.by_bucket),
                    "by_exclusion_code": dict(
                        screening_result.summary.by_exclusion_code
                    ),
                },
            )

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Looking up abstracts through OpenAlex.",
            )
            enriched_path = processed_dir / "candidate_papers_enriched.csv"
            enrichment_result = enrich_candidates(
                screened_path,
                enriched_path,
                config.enrichment,
                decisions={"include_candidate", "needs_review"},
                limit=enrich_limit,
                progress_callback=_item_progress(
                    tracked_progress,
                    "Abstract enrichment",
                    "Looking up abstracts through OpenAlex.",
                ),
            )
            write_enrichment_summary(
                enrichment_result.summary,
                processed_dir / "abstract_enrichment_summary.json",
            )

            initial_round["status"] = "discovery_complete"
            initial_round["files"].update(
                {
                    "venues": str(venue_path),
                    "screened": str(screened_path),
                    "enriched": str(enriched_path),
                }
            )
            initial_round["counts"].update(
                {
                    "deduped_records": len(merged),
                    "manual_records": len(manual_records),
                    "abstracts_found": enrichment_result.summary.with_abstract,
                    "abstracts_attempted": enrichment_result.summary.attempted,
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
        if round_state.get("status") not in {"discovery_complete", "ready_for_review", "failed"}:
            raise RuntimeError("This round is not ready for AI screening or review preparation.")
        if not round_state.get("files", {}).get("enriched"):
            raise RuntimeError("This round does not have an enriched candidate file.")
        config = load_config(self.store.config_path(project_slug))
        processed_dir = Path(round_state["files"]["enriched"]).parent
        suffix = "" if round_index == 0 else f"_round_{round_index}"
        enriched_path = Path(round_state["files"]["enriched"])
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="AI screening" if use_llm else "Review preparation",
            heading="Preparing the review queue",
            stages=AI_REVIEW_STAGES if use_llm else MANUAL_REVIEW_STAGES,
            callback=progress,
            paper_count=_round_paper_count(round_state),
        )

        try:
            if use_llm:
                if not self.store.has_api_key(project_slug) and not os.environ.get(
                    "OPENAI_API_KEY"
                ):
                    raise RuntimeError("Save or provide an OpenAI API key before AI screening.")
                _notify(
                    tracked_progress,
                    "AI screening",
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
                        "AI screening",
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
            else:
                _notify(
                    tracked_progress,
                    "Review preparation",
                    "Creating a human-only review queue.",
                )
                recommendation_path = processed_dir / f"final_screening_recommendations{suffix}.csv"
                create_manual_recommendations(enriched_path, recommendation_path)

            _notify(
                tracked_progress,
                "Audit queue",
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
            round_state["files"].update(
                {
                    "recommendations": str(recommendation_path),
                    "pool": str(recommendation_path),
                    "audit": str(audit_path),
                }
            )
            round_state["counts"]["audit_queue"] = queue_count
            round_state["status"] = (
                "converged" if round_index > 0 and queue_count == 0 else "ready_for_review"
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
                paper_count=_round_paper_count(round_state),
            )
            return state
        except Exception as exc:
            self._mark_failed(project_slug, state, round_state, exc)
            raise

    def start_snowball_discovery(
        self,
        project_slug: str,
        *,
        max_backward_per_seed: int = 30,
        max_forward_per_seed: int = 30,
        enrich_limit: int | None = None,
        core_online: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = self.load_current_state(project_slug)
        completed_rounds = [item for item in state["rounds"] if item.get("files", {}).get("audit")]
        if not completed_rounds:
            raise RuntimeError("Complete the initial review before snowballing.")
        latest_round = completed_rounds[-1]
        _, _, audit_summary = load_audit(Path(latest_round["files"]["audit"]))
        if audit_summary.unreviewed:
            raise RuntimeError("Finish every paper in the current audit round before snowballing.")

        project_dir = self.store.project_dir(project_slug)
        audit_paths = [Path(item["files"]["audit"]) for item in completed_rounds]
        cumulative_path = project_dir / "audits" / state["run_id"] / "cumulative.csv"
        included_path = project_dir / "audits" / state["run_id"] / "included.csv"
        _, _, included_count = build_cumulative_audit(audit_paths, cumulative_path, included_path)
        if not included_count:
            raise RuntimeError("At least one paper must be included before snowballing.")

        seed_path = project_dir / "seeds" / f"{state['run_id']}_current.yaml"
        seed_result = export_seed_papers_from_csv(
            included_path,
            seed_path,
            source_label="manual_audit_include",
        )
        round_index = max(int(item["index"]) for item in state["rounds"]) + 1
        round_state = _new_round_state(index=round_index, kind="snowball")
        round_state["counts"]["seeds"] = len(seed_result.seeds)
        state["rounds"].append(round_state)
        state["status"] = "running_snowball"
        self._save_state(project_slug, state)

        config = load_config(self.store.config_path(project_slug))
        run_dir = project_dir / "runs" / state["run_id"]
        processed_dir = run_dir / "processed"
        previous_pool = Path(latest_round["files"]["pool"])
        tracked_progress = self._begin_progress(
            project_slug,
            state,
            operation="Citation snowballing",
            heading="Running citation snowballing",
            stages=SNOWBALL_STAGES,
            callback=progress,
            paper_count=_csv_row_count(previous_pool),
        )

        try:
            _notify(
                tracked_progress,
                "Citation snowballing",
                "Collecting references and citing works from OpenAlex.",
            )
            snowball_path = processed_dir / f"candidate_papers_snowballed_round_{round_index}.csv"
            snowball_result = snowball_candidates(
                previous_pool,
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
                    "Collecting references and citing works from OpenAlex.",
                ),
            )
            write_snowballing_summary(
                snowball_result.summary,
                processed_dir / f"snowballing_round_{round_index}_summary.json",
            )

            _notify(
                tracked_progress,
                "Venue enrichment",
                "Updating publication metadata for the expanded pool.",
            )
            venue_path = processed_dir / f"candidate_papers_venues_round_{round_index}.csv"
            venue_config = replace(config.venue_quality, core_online_enabled=core_online)
            venue_result = enrich_venue_quality(
                snowball_path,
                venue_path,
                venue_config,
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

            _notify(
                tracked_progress,
                "Rule screening",
                "Applying project exclusion terms to new records.",
            )
            screened_path = processed_dir / f"candidate_papers_screened_round_{round_index}.csv"
            screen_candidates(venue_path, screened_path, config.screening)

            _notify(
                tracked_progress,
                "Abstract enrichment",
                "Looking up abstracts for newly discovered papers.",
            )
            enriched_path = processed_dir / f"candidate_papers_enriched_round_{round_index}.csv"
            enrichment_result = enrich_candidates(
                screened_path,
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

            round_state["status"] = "discovery_complete"
            round_state["files"] = {
                "snowballed": str(snowball_path),
                "venues": str(venue_path),
                "screened": str(screened_path),
                "enriched": str(enriched_path),
                "seeds": str(seed_path),
            }
            round_state["counts"].update(
                {
                    "pool_rows": snowball_result.summary.output_rows,
                    "added_rows": snowball_result.summary.added_rows,
                    "resolved_seeds": snowball_result.summary.seeds_resolved,
                    "abstracts_found": enrichment_result.summary.with_abstract,
                }
            )
            state["status"] = "awaiting_ai_or_review"
            self._save_state(project_slug, state)
            self._complete_progress(
                project_slug,
                state,
                progress,
                stage="Snowball discovery complete",
                message="New records are ready for AI screening or review.",
                paper_count=snowball_result.summary.output_rows,
            )
            return state
        except Exception as exc:
            self._mark_failed(project_slug, state, round_state, exc)
            raise

    def estimate_llm_usage(self, project_slug: str, round_index: int) -> dict[str, int]:
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
        prompt_chars = len(self.store.system_prompt_path(project_slug).read_text(encoding="utf-8"))
        input_chars = sum(
            prompt_chars + len(row.get("title", "")) + len((row.get("abstract") or "")[:5000])
            for row in eligible
        )
        config = load_config(self.store.config_path(project_slug))
        return {
            "papers": len(eligible),
            "estimated_input_tokens": math.ceil(input_chars / 4),
            "maximum_output_tokens": len(eligible) * config.llm_screening.max_output_tokens,
        }

    def update_audit(
        self,
        project_slug: str,
        round_index: int,
        updates: list[dict[str, str]],
    ):
        state = self.load_current_state(project_slug)
        round_state = _get_round(state, round_index)
        summary = update_audit_rows(Path(round_state["files"]["audit"]), updates)
        round_state["counts"]["reviewed"] = summary.reviewed
        round_state["counts"]["unreviewed"] = summary.unreviewed
        self._save_state(project_slug, state)
        return summary

    def generate_exports(self, project_slug: str) -> dict[str, Path]:
        state = self.load_current_state(project_slug)
        project_dir = self.store.project_dir(project_slug)
        audit_paths = [
            Path(item["files"]["audit"])
            for item in state["rounds"]
            if item.get("files", {}).get("audit")
        ]
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
        api_key = self.store.read_api_key(project_slug) or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        client = OpenAIResearchClient(
            base_url=settings.llm_base_url,
            api_key=api_key,
            model=selected_model,
        )
        analyzer = CorpusAnalyzer(client)
        analysis_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = (
            self.store.project_dir(project_slug)
            / "analysis"
            / state["run_id"]
            / analysis_id
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

    def _state_path(self, project_slug: str, run_id: str) -> Path:
        return self.store.project_dir(project_slug) / "runs" / run_id / "state.json"

    def _save_state(self, project_slug: str, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        _write_json(self._state_path(project_slug, state["run_id"]), state)

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


def _new_round_state(index: int, kind: str) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "status": "running",
        "created_at": _now(),
        "files": {},
        "counts": {},
        "error": "",
    }


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


def _without_manual_provenance(record: PaperRecord) -> PaperRecord | None:
    automatic_sources = [
        source for source in record.discovery_sources if source != "manual"
    ]
    if not automatic_sources:
        return None
    automatic_queries = [
        query for query in record.discovery_queries if query != "manual addition"
    ]
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
