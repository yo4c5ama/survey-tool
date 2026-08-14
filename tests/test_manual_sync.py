from pathlib import Path

from vnn_survey.app.audit import read_csv
from vnn_survey.app.manual_papers import ManualPaperStore, create_manual_record
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.export import write_csv
from vnn_survey.models import PaperRecord


def test_manual_papers_can_be_added_and_removed_before_review(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Manual Sync",
        research_question="What is relevant?",
        scope_description="Test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "test-run"
    processed = store.project_dir(project.slug) / "runs" / run_id / "processed"
    candidates = processed / "candidate_papers.csv"
    write_csv(
        [
            PaperRecord(
                title="Automatically Found",
                source="dblp",
                query="verification",
                year=2025,
                doi="10.1000/auto",
            )
        ],
        candidates,
    )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_ai_or_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "source": "auto",
        "sources": ["dblp"],
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "discovery_complete",
                "created_at": "2026-01-01T00:00:00",
                "files": {"candidates": str(candidates)},
                "counts": {"deduped_records": 1},
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)
    manual_store = ManualPaperStore(store.project_dir(project.slug))
    manual, _ = manual_store.add(
        create_manual_record(
            title="Manually Added",
            year=2024,
            doi="10.1000/manual",
        ),
        "Known paper",
    )

    synced = service.sync_manual_additions(
        project.slug,
        enrich_limit=0,
        core_online=False,
    )

    assert synced["rounds"][0]["counts"]["deduped_records"] == 2
    _, rows = read_csv(candidates)
    assert {row["title"] for row in rows} == {"Automatically Found", "Manually Added"}

    assert manual_store.remove(manual.dedupe_key()) is True
    resynced = service.sync_manual_additions(
        project.slug,
        enrich_limit=0,
        core_online=False,
    )
    assert resynced["rounds"][0]["counts"]["deduped_records"] == 1
    _, rows = read_csv(candidates)
    assert [row["title"] for row in rows] == ["Automatically Found"]
