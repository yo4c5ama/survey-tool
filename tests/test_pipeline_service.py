import csv
from pathlib import Path

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
