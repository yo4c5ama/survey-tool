import csv
import json
from pathlib import Path

from vnn_survey.ai_research import CorpusAnalysisResult
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore


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
