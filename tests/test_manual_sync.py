from pathlib import Path

from vnn_survey import llm_screening
from vnn_survey.app.audit import read_csv
from vnn_survey.app.audit import write_csv as write_audit_csv
from vnn_survey.app.manual_papers import ManualPaperStore, create_manual_record
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.export import write_csv as write_paper_csv
from vnn_survey.models import PaperRecord


def _install_fake_manual_screening(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, config) -> None:
            self.config = config

        def screen(self, row: dict[str, str]) -> dict[str, object]:
            excluded = row["title"] in {
                "Manually Added to Review",
                "Saved Before Live Review",
            }
            return {
                "decision": "exclude" if excluded else "include",
                "scope": "out_of_scope" if excluded else "in_scope",
                "confidence": 0.95,
                "reason": "Test exclusion" if excluded else "Test inclusion",
                "evidence": row["title"],
                "_response_id": "manual-screening-test",
            }

    monkeypatch.setattr(llm_screening, "OpenAIResponsesClient", FakeClient)


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


def test_manual_paper_is_enriched_before_entering_existing_review(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Live Manual Review",
        research_question="What is relevant?",
        scope_description="Test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    _install_fake_manual_screening(monkeypatch)
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
        "status": "queued_for_enrichment",
        "round_index": 0,
        "collection_added": True,
        "pending": 1,
    }
    _, rows = read_csv(audit_path)
    assert [row["title"] for row in rows] == ["Automatically Found"]
    assert service.manual_enrichment_status(project.slug, 0) == {
        "saved": 1,
        "pending": 1,
        "enriched": 0,
    }

    service.enrich_manual_additions(
        project.slug,
        0,
        enrich_limit=0,
        core_online=False,
    )

    _, rows = read_csv(audit_path)
    assert [row["title"] for row in rows] == [
        "Automatically Found",
        "Manually Added to Review",
    ]
    assert rows[1]["manual_added"] == "True"
    assert rows[1]["manual_review_added"] == "true"
    assert rows[1]["manual_enrichment_status"] == "completed"
    assert rows[1]["manual_notes"] == "Known relevant paper"
    assert rows[1]["llm_decision"] == "exclude"
    assert rows[1]["llm_confidence"] == "0.950"
    assert rows[1]["llm_reason"] == "Test exclusion"
    assert rows[1]["final_recommendation"] == "likely_exclude"
    assert rows[1]["manual_review_required"] == "true"

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
        "human_audit",
        "manual_loop_additions",
        "manual_venue_enrichment",
        "manual_abstract_enrichment",
        "manual_ai_abstract_screening",
        "manual_return_to_review",
    ]
    assert updated_round["flow"][-1]["loop_to"] == "human_audit"
    assert updated_round["flow"][2]["retained"] == 1
    assert updated_round["flow"][-2]["retained"] == 1
    assert updated_round["flow"][-2]["excluded"] == 0
    assert updated_round["flow"][-2]["details"]["AI exclude recommendations"] == 1
    assert updated_round["counts"]["manual_llm_excluded"] == 1

    second = create_manual_record(
        title="A Second Manual Paper",
        year=2023,
        doi="10.1000/manual-review-2",
    )
    assert service.add_manual_paper(project.slug, second)["status"] == (
        "queued_for_enrichment"
    )
    service.enrich_manual_additions(
        project.slug,
        0,
        enrich_limit=0,
        core_online=False,
    )
    manual_stage = service.load_current_state(project.slug)["rounds"][0]["flow"][2]
    assert manual_stage["input"] == 2
    assert manual_stage["retained"] == 2
    assert manual_stage["details"]["submitted"] == 2

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
    assert after_removal_round["flow"][2]["retained"] == 1

    pending = create_manual_record(
        title="Pending Manual Paper",
        year=2022,
        doi="10.1000/manual-pending",
    )
    service.add_manual_paper(project.slug, pending)
    assert service.load_current_state(project.slug)["rounds"][0]["counts"][
        "manual_pending"
    ] == 1
    pending_removal = service.remove_manual_paper(project.slug, pending)
    assert pending_removal == {
        "status": "removed_from_collection",
        "collection_removed": True,
        "review_rows_removed": 0,
    }
    assert service.load_current_state(project.slug)["rounds"][0]["counts"][
        "manual_pending"
    ] == 0


def test_legacy_direct_manual_paper_enters_enrichment_loop_without_duplication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Legacy Manual Paper",
        research_question="What is relevant?",
        scope_description="Test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    _install_fake_manual_screening(monkeypatch)
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
        [
            {"title": "Existing", "year": "2025", "manual_decision": ""},
            {
                "title": "Saved Before Live Review",
                "year": "2024",
                "doi": "10.1000/legacy-manual",
                "manual_review_added": "true",
                "manual_decision": "",
                "manual_notes": "Known paper",
            },
        ],
        [
            "title",
            "year",
            "doi",
            "manual_review_added",
            "manual_decision",
            "manual_notes",
        ],
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
                    "counts": {"audit_queue": 2},
                    "flow": [],
                    "error": "",
                }
            ],
        },
    )

    before = service.manual_enrichment_status(project.slug, 0)
    service.enrich_manual_additions(
        project.slug,
        0,
        enrich_limit=0,
        core_online=False,
    )
    after = service.manual_enrichment_status(project.slug, 0)

    assert before == {"saved": 1, "pending": 1, "enriched": 0}
    assert after == {"saved": 1, "pending": 0, "enriched": 1}
    assert [row["title"] for row in read_csv(audit_path)[1]] == [
        "Existing",
        "Saved Before Live Review",
    ]
    assert read_csv(audit_path)[1][1]["manual_enrichment_status"] == "completed"
    assert read_csv(audit_path)[1][1]["llm_decision"] == "exclude"
