import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

from vnn_survey.ai_research import CorpusAnalysisResult
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.app.task_manager import TaskCancelled, TaskManager, raise_if_cancelled
from vnn_survey.models import PaperRecord
from vnn_survey.pipeline import CollectionResult
from vnn_survey.title_screening import TitleScreeningResult, TitleScreeningSummary


def test_human_only_round_preparation_and_export(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Service Test",
        research_question="What is in scope?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "test-run"
    processed = store.project_dir(project.slug) / "runs" / run_id / "processed"
    enriched = processed / "candidate_papers_enriched.csv"
    enriched.parent.mkdir(parents=True, exist_ok=True)
    with enriched.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "year",
                "doi",
                "auto_screening_decision",
                "abstract",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "A Paper",
                "year": "2025",
                "doi": "10.1/paper",
                "auto_screening_decision": "include_candidate",
                "abstract": "An abstract.",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_ai_or_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "source": "auto",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "discovery_complete",
                "created_at": "2026-01-01T00:00:00",
                "files": {"enriched": str(enriched)},
                "counts": {},
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    progress_events: list[tuple[str, int | None, int | None]] = []
    prepared = service.prepare_round_for_review(
        project.slug,
        0,
        use_llm=False,
        progress=lambda stage, _message, completed, total, _current: progress_events.append(
            (stage, completed, total)
        ),
    )
    audit_path = Path(prepared["rounds"][0]["files"]["audit"])
    assert prepared["rounds"][0]["counts"]["audit_queue"] == 1
    assert audit_path.exists()
    assert [event[0] for event in progress_events] == [
        "Review preparation",
        "Audit queue",
        "Review queue ready",
    ]
    persisted = service.load_current_state(project.slug)["progress"]
    assert persisted["operation"] == "Review preparation"
    assert persisted["status"] == "completed"
    assert persisted["stage"] == "Review queue ready"
    assert persisted["paper_count"] == 1

    service.update_audit(
        project.slug,
        0,
        [
            {
                "title": "A Paper",
                "year": "2025",
                "doi": "10.1/paper",
                "manual_decision": "include",
                "manual_notes": "In scope.",
            }
        ],
    )
    exports = service.generate_exports(project.slug)
    assert exports["included"].exists()
    assert "A Paper" in exports["report"].read_text(encoding="utf-8")


def test_final_corpus_analysis_persists_outputs_and_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Analysis Test",
        research_question="How are papers grouped?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    service = PipelineService(store)
    run_id = "analysis-run"
    included = store.project_dir(project.slug) / "exports" / run_id / "included.csv"
    included.parent.mkdir(parents=True, exist_ok=True)
    with included.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "year", "doi"])
        writer.writeheader()
        writer.writerow({"title": "Paper A", "year": "2025", "doi": "10.1/a"})
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [],
        "exports": {"included": str(included)},
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    class FakeAnalyzer:
        def __init__(self, _client) -> None:
            pass

        def analyze(self, **kwargs) -> CorpusAnalysisResult:
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            taxonomy = output_dir / "taxonomy.json"
            classifications = output_dir / "classifications.csv"
            report = output_dir / "report.md"
            taxonomy.write_text(
                json.dumps(
                    {
                        "title": "Taxonomy",
                        "categories": [
                            {"id": "a", "label": "A"},
                            {"id": "b", "label": "B"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            classifications.write_text(
                "paper_id,title,primary_category\n1,Paper A,a\n",
                encoding="utf-8",
            )
            report.write_text("# Taxonomy\n", encoding="utf-8")
            kwargs["stage_callback"]("Taxonomy design", "Designing taxonomy.")
            kwargs["stage_callback"]("Paper classification", "Classifying papers.")
            kwargs["progress_callback"](1, 1, "Paper A")
            kwargs["stage_callback"]("Analysis report", "Writing report.")
            return CorpusAnalysisResult(
                taxonomy={"title": "Taxonomy"},
                classifications=[],
                taxonomy_path=taxonomy,
                classifications_path=classifications,
                report_path=report,
            )

    monkeypatch.setattr("vnn_survey.app.pipeline_service.CorpusAnalyzer", FakeAnalyzer)
    analyzed = service.analyze_final_corpus(
        project.slug,
        criteria="By method",
        model="gpt-test",
    )

    assert analyzed["corpus_analysis"]["model"] == "gpt-test"
    assert Path(analyzed["corpus_analysis"]["report"]).exists()
    assert analyzed["progress"]["status"] == "completed"
    assert analyzed["progress"]["paper_count"] == 1


def test_cancelled_discovery_is_persisted_without_becoming_a_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Cancellation Test",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    manager = TaskManager(max_workers=1)
    collecting = Event()

    def wait_for_cancellation(*_args, **_kwargs):
        collecting.set()
        while True:
            raise_if_cancelled()
            sleep(0.01)

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.collect_from_sources",
        wait_for_cancellation,
    )

    try:
        assert manager.start(
            project.slug,
            "initial_discovery",
            service.start_initial_discovery,
            project.slug,
        )
        assert collecting.wait(timeout=2)
        assert manager.cancel(project.slug)

        deadline = monotonic() + 2
        while manager.is_running(project.slug) and monotonic() < deadline:
            sleep(0.01)

        state = service.load_current_state(project.slug)
        assert state["status"] == "cancelled"
        assert state["progress"]["status"] == "cancelled"
        assert state["rounds"][0]["status"] == "cancelled"
        assert not state["rounds"][0]["error"]
    finally:
        manager.shutdown()


def test_cancelled_initial_discovery_resumes_same_run_from_saved_venue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Resume Test",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    record = PaperRecord(title="Checkpoint Paper", source="test", query="query", year=2025)
    collection = CollectionResult([record], [record], [record], {}, {})
    collect_calls = 0

    def fake_collect(*_args, **_kwargs):
        nonlocal collect_calls
        collect_calls += 1
        return collection

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.collect_from_sources",
        fake_collect,
    )
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.write_venue_quality_summary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.write_enrichment_summary",
        lambda *_args, **_kwargs: None,
    )

    def fake_venue(input_path, output_path, *_args, **_kwargs):
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(
            summary=SimpleNamespace(
                conferences_with_core_rank=0,
                journals_with_impact_factor=0,
            )
        )

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.enrich_venue_quality",
        fake_venue,
    )
    enrichment_calls = 0

    def fake_enrichment(input_path, output_path, *_args, **_kwargs):
        nonlocal enrichment_calls
        enrichment_calls += 1
        shutil.copyfile(input_path, output_path)
        if enrichment_calls == 1:
            raise TaskCancelled("stop after venue")
        return SimpleNamespace(
            summary=SimpleNamespace(
                with_abstract=0,
                attempted=1,
                api_requests=1,
                cache_hits=0,
                rate_limit_retries=0,
                rate_limit_wait_seconds=0,
            )
        )

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.enrich_candidates",
        fake_enrichment,
    )

    stopped = service.start_initial_discovery(project.slug, source_ids=["dblp"])
    run_id = stopped["run_id"]
    assert stopped["status"] == "cancelled"
    assert Path(stopped["rounds"][0]["files"]["venues"]).exists()

    resumed = service.resume_initial_discovery(project.slug)

    assert resumed["run_id"] == run_id
    assert resumed["status"] == "awaiting_ai_or_review"
    assert collect_calls == 1
    assert enrichment_calls == 2
    assert [stage["key"] for stage in resumed["rounds"][0]["flow"]][-1] == (
        "abstract_enrichment"
    )


def test_initial_discovery_uses_title_decisions_before_abstract_enrichment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Title Pipeline",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["transformer verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    service = PipelineService(store)
    records = [
        PaperRecord(title="Keep", source="test", query="query", year=2025),
        PaperRecord(title="Exclude", source="test", query="query", year=2024),
    ]
    collection = CollectionResult(records, records, records, {}, {})
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.collect_from_sources",
        lambda *_args, **_kwargs: collection,
    )

    def fake_title_prescreen(
        _self,
        _project_slug,
        input_path,
        output_path,
        **_kwargs,
    ):
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        rows[0]["title_llm_decision"] = "include"
        rows[1]["title_llm_decision"] = "exclude"
        rows[1]["auto_screening_decision"] = "exclude"
        fields.append("title_llm_decision")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary = TitleScreeningSummary(
            total=2,
            eligible=2,
            api_screened=2,
            cached=0,
            batches=1,
            by_decision=Counter({"include": 1, "exclude": 1}),
        )
        return output_path, TitleScreeningResult(rows, summary)

    monkeypatch.setattr(PipelineService, "_title_prescreen", fake_title_prescreen)
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.write_venue_quality_summary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.write_enrichment_summary",
        lambda *_args, **_kwargs: None,
    )

    def fake_venue(input_path, output_path, *_args, **_kwargs):
        assert input_path.name == "candidate_papers_title_screened.csv"
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(summary=object())

    attempted: list[str] = []

    def fake_enrichment(input_path, output_path, *_args, **_kwargs):
        assert input_path.name == "candidate_papers_venues.csv"
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        attempted.extend(
            row["title"]
            for row in rows
            if row["auto_screening_decision"] != "exclude"
        )
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(
            summary=SimpleNamespace(with_abstract=1, attempted=len(attempted))
        )

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.enrich_venue_quality",
        fake_venue,
    )
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.enrich_candidates",
        fake_enrichment,
    )

    state = service.start_initial_discovery(
        project.slug,
        source_ids=["dblp"],
        use_title_llm=True,
        core_online=False,
    )

    assert attempted == ["Keep"]
    assert state["rounds"][0]["counts"]["title_excluded"] == 1
    assert state["rounds"][0]["counts"]["abstracts_attempted"] == 1
