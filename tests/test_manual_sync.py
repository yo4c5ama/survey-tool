from pathlib import Path

from vnn_survey.app.audit import read_csv
from vnn_survey.app.audit import write_csv as write_audit_csv
from vnn_survey.app.manual_papers import ManualPaperStore, create_manual_record
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.export import write_csv as write_paper_csv
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
    write_paper_csv(
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


def test_manual_paper_enters_existing_review_and_updates_flow(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Live Manual Review",
        research_question="What is relevant?",
        scope_description="Test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "live-manual-run"
    audit_path = store.project_dir(project.slug) / "audits" / run_id / "round_0.csv"
    write_audit_csv(
        audit_path,
        [
            {
                "title": "Automatically Found",
                "year": "2025",
                "doi": "10.1000/auto",
                "manual_decision": "include",
                "manual_notes": "In scope.",
            }
        ],
        ["title", "year", "doi", "manual_decision", "manual_notes"],
    )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "progress": {"paper_count": 1},
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "created_at": "2026-01-01T00:00:00",
                "files": {"audit": str(audit_path)},
                "counts": {"audit_queue": 1, "reviewed": 1, "unreviewed": 0},
                "flow": [
                    {
                        "key": "audit_queue",
                        "label": "Human review queue",
                        "type": "review",
                        "input": 1,
                        "retained": 1,
                        "excluded": 0,
                        "details": {},
                    },
                    {
                        "key": "human_audit",
                        "label": "Human audit",
                        "type": "review",
                        "input": 1,
                        "retained": 1,
                        "excluded": 0,
                        "details": {"pending": 0},
                    },
                    {
                        "key": "final_corpus",
                        "label": "Final corpus",
                        "type": "review",
                        "input": 1,
                        "retained": 1,
                        "excluded": 0,
                        "details": {},
                    },
                ],
                "error": "",
            }
        ],
        "exports": {"included": "stale.csv"},
        "corpus_analysis": {"report": "stale.md"},
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)
    manual = create_manual_record(
        title="Manually Added to Review",
        year=2024,
        doi="10.1000/manual-review",
    )

    result = service.add_manual_paper(project.slug, manual, "Known relevant paper")

    assert result == {
        "status": "added_to_review",
        "round_index": 0,
        "collection_added": True,
        "audit_total": 2,
    }
    _, rows = read_csv(audit_path)
    assert [row["title"] for row in rows] == [
        "Automatically Found",
        "Manually Added to Review",
    ]
    assert rows[1]["manual_added"] == "True"
    assert rows[1]["manual_review_added"] == "true"
    assert rows[1]["manual_notes"] == "Known relevant paper"

    updated = service.load_current_state(project.slug)
    updated_round = updated["rounds"][0]
    assert updated_round["counts"]["audit_queue"] == 2
    assert updated_round["counts"]["reviewed"] == 1
    assert updated_round["counts"]["unreviewed"] == 1
    assert updated["progress"]["paper_count"] == 2
    assert "exports" not in updated
    assert "corpus_analysis" not in updated
    assert [stage["key"] for stage in updated_round["flow"]] == [
        "audit_queue",
        "manual_review_additions",
        "human_audit",
    ]
    assert updated_round["flow"][1]["retained"] == 2

    second = create_manual_record(
        title="A Second Manual Paper",
        year=2023,
        doi="10.1000/manual-review-2",
    )
    assert service.add_manual_paper(project.slug, second)["status"] == "added_to_review"
    manual_stage = service.load_current_state(project.slug)["rounds"][0]["flow"][1]
    assert manual_stage["input"] == 1
    assert manual_stage["retained"] == 3
    assert manual_stage["details"]["added by researcher"] == 2

    duplicate = service.add_manual_paper(project.slug, manual, "Duplicate attempt")
    assert duplicate["status"] == "already_in_review"
    assert len(read_csv(audit_path)[1]) == 3

    removed = service.remove_manual_paper(project.slug, manual)
    assert removed == {
        "status": "removed_from_review",
        "collection_removed": True,
        "review_rows_removed": 1,
    }
    _, retained_rows = read_csv(audit_path)
    assert [row["title"] for row in retained_rows] == [
        "Automatically Found",
        "A Second Manual Paper",
    ]
    after_removal = service.load_current_state(project.slug)
    after_removal_round = after_removal["rounds"][0]
    assert after_removal["progress"]["paper_count"] == 2
    assert after_removal_round["counts"]["audit_queue"] == 2
    assert after_removal_round["counts"]["manual_review_additions"] == 1
    assert after_removal_round["flow"][1]["retained"] == 2


def test_saved_manual_papers_are_reconciled_into_an_existing_review(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Legacy Manual Paper",
        research_question="What is relevant?",
        scope_description="Test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    legacy = create_manual_record(
        title="Saved Before Live Review",
        year=2024,
        doi="10.1000/legacy-manual",
        note="Known paper",
    )
    ManualPaperStore(store.project_dir(project.slug)).add(legacy, "Known paper")
    service = PipelineService(store)
    run_id = "legacy-manual-run"
    audit_path = store.project_dir(project.slug) / "audits" / run_id / "round_0.csv"
    write_audit_csv(
        audit_path,
        [{"title": "Existing", "year": "2025", "manual_decision": ""}],
        ["title", "year", "manual_decision"],
    )
    store.set_current_run(project.slug, run_id)
    service._save_state(
        project.slug,
        {
            "project_slug": project.slug,
            "run_id": run_id,
            "status": "awaiting_manual_review",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "progress": {"paper_count": 1},
            "rounds": [
                {
                    "index": 0,
                    "kind": "initial",
                    "status": "ready_for_review",
                    "created_at": "2026-01-01T00:00:00",
                    "files": {"audit": str(audit_path)},
                    "counts": {"audit_queue": 1},
                    "flow": [],
                    "error": "",
                }
            ],
        },
    )

    first = service.reconcile_saved_manual_papers(project.slug, 0)
    second = service.reconcile_saved_manual_papers(project.slug, 0)

    assert first == {"added": 1, "already_present": 0}
    assert second == {"added": 0, "already_present": 1}
    assert [row["title"] for row in read_csv(audit_path)[1]] == [
        "Existing",
        "Saved Before Live Review",
    ]
