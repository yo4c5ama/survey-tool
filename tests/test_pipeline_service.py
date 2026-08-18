import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest
import yaml

import vnn_survey.app.pipeline_service as pipeline_service_module
from vnn_survey.ai_research import CorpusAnalysisResult
from vnn_survey.app.audit import load_audit, read_csv
from vnn_survey.app.pipeline_service import (
    AI_REVIEW_STAGES,
    SNOWBALL_STAGES,
    PipelineService,
    _csv_unique_paper_count,
    _with_prompt_replay_stage,
    _with_title_screening_stage,
    _write_json,
    _write_incremental_snowball_candidates,
)
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.app.project_transfer import import_projects_backup
from vnn_survey.app.task_manager import TaskCancelled, TaskManager, raise_if_cancelled
from vnn_survey.llm_screening import LlmScreeningResult, LlmScreeningSummary
from vnn_survey.models import PaperRecord
from vnn_survey.pipeline import CollectionResult
from vnn_survey.snowballing import SnowballingResult, SnowballingSummary
from vnn_survey.title_screening import TitleScreeningResult, TitleScreeningSummary


def test_unique_paper_count_preserves_aliases_from_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "aliases.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "year", "doi"])
        writer.writeheader()
        writer.writerows(
            [
                {"title": "Shared Title", "year": "2024", "doi": ""},
                {"title": "Shared Title", "year": "2025", "doi": "10.1/bridge"},
                {"title": "Published Title", "year": "2025", "doi": "10.1/bridge"},
            ]
        )

    assert _csv_unique_paper_count(path) == 1


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
        "Prepare recommendations",
        "Create manual review queue",
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


def test_snowball_discovery_stops_after_five_core_stages() -> None:
    stages = _with_title_screening_stage(SNOWBALL_STAGES, True)

    assert stages == [
        "Citation snowballing",
        "Rule screening",
        "AI title screening",
        "Venue enrichment",
        "Abstract enrichment",
    ]
    assert "Re-screen initial AI exclusions" not in stages
    assert _with_prompt_replay_stage(AI_REVIEW_STAGES, True)[0] == (
        "Re-screen initial AI exclusions"
    )


def test_historical_replay_runs_during_review_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Deferred replay",
        research_question="What is in scope?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    service = PipelineService(store)
    run_id = "deferred-replay-run"
    processed = store.project_dir(project.slug) / "runs" / run_id / "processed"
    enriched = processed / "candidate_papers_enriched_round_1.csv"
    enriched.parent.mkdir(parents=True, exist_ok=True)
    with enriched.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "abstract", "auto_screening_decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Snowball candidate",
                "abstract": "A formal verification paper.",
                "auto_screening_decision": "needs_review",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_ai_or_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "prompt_replay": {"status": "pending"},
        "rounds": [
            {
                "index": 1,
                "kind": "snowball",
                "status": "discovery_complete",
                "replay_initial_exclusions": True,
                "files": {"enriched": str(enriched)},
                "counts": {},
                "flow": [],
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    def fake_replay(_self, **_kwargs):
        _kwargs["state"]["prompt_replay"]["status"] = "completed"
        return {}, {
            "replayed": 3,
            "recovered": 2,
            "reexcluded": 1,
            "failed": 0,
            "replay_source_exclusions": 3,
            "replay_reviewed_removed": 0,
            "replay_eligible": 3,
        }

    monkeypatch.setattr(PipelineService, "_replay_initial_ai_exclusions", fake_replay)
    progress_stages: list[str] = []

    prepared = service.prepare_round_for_review(
        project.slug,
        1,
        use_llm=False,
        progress=lambda stage, *_args: progress_stages.append(stage),
    )

    assert progress_stages == [
        "Re-screen initial AI exclusions",
        "Prepare recommendations",
        "Create manual review queue",
        "Review queue ready",
    ]
    assert prepared["rounds"][0]["counts"]["replayed"] == 3
    assert Path(prepared["rounds"][0]["files"]["audit"]).exists()


def test_ai_abstract_screening_persists_a_manual_review_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="AI review queue",
        research_question="What is in scope?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    service = PipelineService(store)
    run_id = "ai-review-run"
    processed = store.project_dir(project.slug) / "runs" / run_id / "processed"
    enriched = processed / "candidate_papers_enriched.csv"
    enriched.parent.mkdir(parents=True, exist_ok=True)
    with enriched.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "abstract", "auto_screening_decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Verified Transformer",
                "abstract": "A formal verification method for a Transformer.",
                "auto_screening_decision": "needs_review",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_ai_or_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "discovery_complete",
                "files": {"enriched": str(enriched)},
                "counts": {},
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    estimate = service.estimate_llm_usage(project.slug, 0)
    assert estimate["papers"] == 1
    assert estimate["estimated_cost_usd"] > 0
    assert estimate["maximum_cost_usd"] >= estimate["estimated_cost_usd"]

    def fake_screen(input_path: Path, output_path: Path, *_args, **_kwargs):
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        for row in rows:
            row.update(
                {
                    "llm_decision": "include",
                    "llm_scope": "transformer_verification",
                    "llm_confidence": "0.95",
                    "llm_reason": "The abstract states formal verification.",
                    "llm_evidence": "formal verification method",
                    "llm_status": "screened",
                }
            )
        output_fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rows)
        return LlmScreeningResult(
            rows=rows,
            summary=LlmScreeningSummary(
                total=1,
                eligible=1,
                attempted=1,
                by_status=Counter({"screened": 1}),
                by_decision=Counter({"include": 1}),
                by_scope=Counter({"transformer_verification": 1}),
                api_requests=1,
                batch_requests=1,
            ),
        )

    monkeypatch.setattr("vnn_survey.app.pipeline_service.llm_screen_candidates", fake_screen)

    prepared = service.prepare_round_for_review(project.slug, 0, use_llm=True)
    audit_path = Path(prepared["rounds"][0]["files"]["audit"])
    _, rows, summary = load_audit(audit_path)

    assert summary.total == 1
    assert rows[0]["title"] == "Verified Transformer"
    assert rows[0]["llm_decision"] == "include"
    assert prepared["rounds"][0]["status"] == "ready_for_review"


def test_snowball_uses_only_latest_review_round_includes_as_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Incremental Seeds",
        research_question="Which papers?",
        scope_description="Test latest-round snowball seeds.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "incremental-seeds"
    run_dir = store.project_dir(project.slug) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fields = ["title", "year", "doi", "manual_decision"]
    round_0 = run_dir / "round_0.csv"
    round_1 = run_dir / "round_1.csv"
    with round_0.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "title": "Old Included Seed",
                "year": "2023",
                "doi": "10.1/old",
                "manual_decision": "include",
            }
        )
    with round_1.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Newest Included Seed",
                    "year": "2025",
                    "doi": "10.1/new",
                    "manual_decision": "include_related",
                },
                {
                    "title": "Newest Excluded Paper",
                    "year": "2025",
                    "doi": "10.1/excluded",
                    "manual_decision": "exclude",
                },
                {
                    "title": "Old Included Seed",
                    "year": "2024",
                    "doi": "10.1/old-published",
                    "manual_decision": "include",
                },
            ]
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(round_0), "pool": str(round_0)},
                "counts": {},
                "flow": [],
                "error": "",
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "ready_for_review",
                "files": {"audit": str(round_1), "pool": str(round_1)},
                "counts": {},
                "flow": [],
                "error": "",
            },
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)
    captured_seeds: list[dict[str, str]] = []
    initial_progress_counts: list[int] = []

    def capture_seeds(*_args, seed_papers_path, **kwargs):
        progress = service.load_current_state(project.slug)["progress"]
        initial_progress_counts.append(int(progress["paper_count"]))
        kwargs["progress_callback"](0, 1, "", 3)
        initial_progress_counts.append(
            int(service.load_current_state(project.slug)["progress"]["paper_count"])
        )
        kwargs["progress_callback"](1, 1, "Newest Included Seed", 8)
        initial_progress_counts.append(
            int(service.load_current_state(project.slug)["progress"]["paper_count"])
        )
        payload = yaml.safe_load(Path(seed_papers_path).read_text(encoding="utf-8"))
        captured_seeds.extend(payload["seed_papers"])
        raise RuntimeError("stop after seed capture")

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.snowball_candidates",
        capture_seeds,
    )

    with pytest.raises(RuntimeError, match="stop after seed capture"):
        service.start_snowball_discovery(
            project.slug,
            citation_providers=["semantic_scholar"],
            core_online=False,
        )

    assert [seed["title"] for seed in captured_seeds] == ["Newest Included Seed"]
    assert initial_progress_counts == [0, 0, 5]


def test_single_paper_snowball_exports_only_the_selected_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Targeted Snowball",
        research_question="Which papers?",
        scope_description="A targeted citation recovery test.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "targeted-snowball"
    run_dir = store.project_dir(project.slug) / "runs" / run_id
    audit = run_dir / "round_0_audit.csv"
    pool = run_dir / "round_0_pool.csv"
    fields = ["title", "year", "doi", "manual_decision", "auto_screening_decision"]
    rows = [
        {
            "title": "Previously Reviewed Seed",
            "year": "2024",
            "doi": "10.1/previous",
            "manual_decision": "include",
            "auto_screening_decision": "include_candidate",
        }
    ]
    audit.parent.mkdir(parents=True, exist_ok=True)
    for path in (audit, pool):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(audit), "pool": str(pool)},
                "counts": {},
                "flow": [],
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)
    captured_seeds: list[dict[str, str]] = []

    def capture_target(*_args, seed_papers_path, **_kwargs):
        payload = yaml.safe_load(Path(seed_papers_path).read_text(encoding="utf-8"))
        captured_seeds.extend(payload["seed_papers"])
        raise RuntimeError("stop after targeted seed capture")

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.snowball_candidates",
        capture_target,
    )

    with pytest.raises(RuntimeError, match="stop after targeted seed capture"):
        service.start_snowball_discovery(
            project.slug,
            citation_providers=["semantic_scholar"],
            core_online=False,
            target_seed={
                "title": "Specific Failed Paper",
                "doi": "10.1/specific",
                "year": "2025",
            },
        )

    assert captured_seeds == [
        {
            "title": "Specific Failed Paper",
            "source": "targeted_single_paper",
            "notes": (
                "year=2025; manual_decision=include; manual_notes=Single-paper snowball "
                "target selected by the researcher."
            ),
            "doi": "10.1/specific",
        }
    ]
    persisted = service.load_current_state(project.slug)
    targeted_round = persisted["rounds"][-1]
    assert targeted_round["snowball_mode"] == "targeted"
    assert targeted_round["target_seed"]["title"] == "Specific Failed Paper"
    assert Path(targeted_round["files"]["seeds"]).name.endswith("round_1.yaml")


def test_every_saved_state_updates_the_exportable_run_log(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Run Log",
        research_question="What happened?",
        scope_description="Run log test.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    state = {
        "project_slug": project.slug,
        "run_id": "logged-run",
        "status": "running_snowball",
        "rounds": [
            {
                "index": 1,
                "kind": "snowball",
                "status": "running",
                "files": {},
                "counts": {},
                "flow": [],
                "error": "",
            }
        ],
        "progress": {
            "operation": "Citation snowballing",
            "status": "running",
            "stage": "Citation snowballing",
            "message": "Collecting citations.",
        },
    }
    store.set_current_run(project.slug, state["run_id"])
    service._save_state(project.slug, state)
    state["status"] = "awaiting_ai_or_review"
    state["rounds"][0]["status"] = "discovery_complete"
    state["progress"].update({"status": "completed", "message": "Citation collection completed."})
    service._save_state(project.slug, state)

    payload = json.loads(service.run_log_path(project.slug).read_text(encoding="utf-8"))
    assert payload["run_id"] == "logged-run"
    assert payload["state"]["status"] == "awaiting_ai_or_review"
    assert [event["progress_status"] for event in payload["events"]] == [
        "running",
        "completed",
    ]


def test_atomic_json_write_retries_transient_windows_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"status": "old"}', encoding="utf-8")
    original_replace = pipeline_service_module.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            error = PermissionError(13, "Access is denied", str(destination))
            error.winerror = 5
            raise error
        return original_replace(source, destination)

    monkeypatch.setattr(pipeline_service_module.os, "replace", flaky_replace)
    monkeypatch.setattr(pipeline_service_module, "sleep", lambda _seconds: None)

    _write_json(path, {"status": "new"})

    assert attempts == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "new"}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_run_log_write_failure_does_not_abort_state_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Locked run log",
        research_question="Which papers?",
        scope_description="Test an auxiliary Windows file lock.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    state = {
        "project_slug": project.slug,
        "run_id": "locked-log-run",
        "status": "running_discovery",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "running",
                "files": {},
                "counts": {},
                "flow": [],
                "error": "",
            }
        ],
    }

    def locked_log(*_args, **_kwargs):
        error = PermissionError(13, "Access is denied", "run_log.json")
        error.winerror = 5
        raise error

    monkeypatch.setattr(pipeline_service_module, "_write_run_log", locked_log)

    service._save_state(project.slug, state)

    persisted = json.loads(
        (
            store.project_dir(project.slug)
            / "runs"
            / state["run_id"]
            / "state.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["status"] == "running_discovery"
    assert persisted["rounds"][0]["status"] == "running"
    assert (
        store.project_dir(project.slug)
        / "runs"
        / state["run_id"]
        / "flow_summary.json"
    ).exists()


def test_delete_snowball_round_cascades_and_creates_restorable_backup(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Round deletion",
        research_question="Which papers?",
        scope_description="Test round history deletion.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.create_project(
        name="Unrelated project",
        research_question="What else?",
        scope_description="Must not enter the automatic backup.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["testing"])],
    )
    service = PipelineService(store)
    run_id = "round-deletion-run"
    project_dir = store.project_dir(project.slug)
    processed = project_dir / "runs" / run_id / "processed"
    audit_dir = project_dir / "audits" / run_id
    seed_dir = project_dir / "seeds"
    processed.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    fields = ["title", "year", "doi", "manual_decision", "manual_notes"]

    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    round_0_audit = audit_dir / "round_0.csv"
    round_1_audit = audit_dir / "round_1.csv"
    round_2_audit = audit_dir / "round_2.csv"
    write_rows(
        round_0_audit,
        [
            {
                "title": "Initial included paper",
                "year": "2023",
                "doi": "10.1/initial",
                "manual_decision": "include",
                "manual_notes": "Keep",
            }
        ],
    )
    write_rows(
        round_1_audit,
        [
            {
                "title": "First snowball paper",
                "year": "2024",
                "doi": "10.1/round-1",
                "manual_decision": "include",
                "manual_notes": "Keep",
            }
        ],
    )
    write_rows(
        round_2_audit,
        [
            {
                "title": "Second snowball paper",
                "year": "2025",
                "doi": "10.1/round-2",
                "manual_decision": "",
                "manual_notes": "",
            }
        ],
    )
    initial_candidates = processed / "candidate_papers.csv"
    round_1_new = processed / "candidate_papers_snowball_new_round_1.csv"
    round_2_new = processed / "candidate_papers_snowball_new_round_2.csv"
    write_rows(initial_candidates, [])
    write_rows(round_1_new, [])
    write_rows(round_2_new, [])
    round_1_summary = processed / "snowballing_round_1_summary.json"
    round_2_summary = processed / "snowballing_round_2_summary.json"
    round_1_summary.write_text("{}", encoding="utf-8")
    round_2_summary.write_text("{}", encoding="utf-8")
    round_1_seed = seed_dir / f"{run_id}_round_1.yaml"
    round_2_seed = seed_dir / f"{run_id}_round_2.yaml"
    round_1_seed.write_text("papers: []\n", encoding="utf-8")
    round_2_seed.write_text("papers: []\n", encoding="utf-8")
    manual_round_2 = project_dir / "runs" / run_id / "manual_enrichment" / "round_2"
    manual_round_2.mkdir(parents=True)
    (manual_round_2 / "manual_enriched.csv").write_text("title\nPaper\n", encoding="utf-8")

    export_path = project_dir / "exports" / run_id / "final_included_papers.csv"
    analysis_path = project_dir / "analysis" / run_id / "analysis-1" / "report.md"
    export_path.parent.mkdir(parents=True)
    analysis_path.parent.mkdir(parents=True)
    export_path.write_text("title\nOld export\n", encoding="utf-8")
    analysis_path.write_text("Old analysis", encoding="utf-8")
    approved_prompt = project_dir / "configs" / "prompts" / "approved_prompt.txt"
    approved_prompt.parent.mkdir(parents=True, exist_ok=True)
    approved_prompt.write_text("Approved prompt\n", encoding="utf-8")

    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {
                    "candidates": str(initial_candidates),
                    "audit": str(round_0_audit),
                },
                "counts": {"deduped_records": 1, "audit_queue": 1},
                "flow": [],
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "ready_for_review",
                "files": {
                    "snowball_new": str(round_1_new),
                    "seeds": str(round_1_seed),
                    "audit": str(round_1_audit),
                },
                "counts": {"pool_rows": 1, "audit_queue": 1},
                "flow": [],
            },
            {
                "index": 2,
                "kind": "snowball",
                "status": "ready_for_review",
                "files": {
                    "snowball_new": str(round_2_new),
                    "seeds": str(round_2_seed),
                    "audit": str(round_2_audit),
                },
                "counts": {"pool_rows": 1, "audit_queue": 1},
                "flow": [],
            },
        ],
        "prompt_refinement": {
            "refinement_id": "refined-prompt",
            "status": "approved",
            "source_round": 1,
            "source_rounds": [0, 1],
            "approved_prompt_path": str(approved_prompt),
            "approved_at": "2026-01-01T00:00:00",
            "replay_status": "completed",
        },
        "prompt_replay": {
            "status": "completed",
            "refinement_id": "old-prompt",
            "approved_prompt_path": str(approved_prompt),
            "replay_round": 1,
            "replayed": 10,
            "recovered": 2,
        },
        "exports": {"included": str(export_path)},
        "corpus_analysis": {"report": str(analysis_path)},
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    result = service.delete_snowball_round(project.slug, 1)
    persisted = service.load_current_state(project.slug)

    assert result["deleted_rounds"] == [1, 2]
    assert result["latest_round"] == 0
    assert result["cleanup_failures"] == []
    assert [item["index"] for item in persisted["rounds"]] == [0]
    assert persisted["status"] == "awaiting_manual_review"
    assert persisted["progress"]["stage"] == "Round deletion complete"
    assert persisted["prompt_replay"]["status"] == "pending"
    assert persisted["prompt_replay"]["refinement_id"] == "refined-prompt"
    assert persisted["prompt_refinement"]["source_data_stale"] is True
    assert "exports" not in persisted
    assert "corpus_analysis" not in persisted
    assert initial_candidates.exists()
    assert round_0_audit.exists()
    assert not round_1_audit.exists()
    assert not round_2_audit.exists()
    assert not round_1_new.exists()
    assert not round_2_new.exists()
    assert not round_1_summary.exists()
    assert not round_2_summary.exists()
    assert not round_1_seed.exists()
    assert not round_2_seed.exists()
    assert not manual_round_2.exists()
    assert not export_path.exists()
    assert not analysis_path.exists()

    _, cumulative_rows = read_csv(audit_dir / "cumulative.csv")
    _, included_rows = read_csv(audit_dir / "included.csv")
    assert [row["title"] for row in cumulative_rows] == ["Initial included paper"]
    assert [row["title"] for row in included_rows] == ["Initial included paper"]

    backup_path = Path(result["backup_path"])
    assert backup_path.exists()
    with zipfile.ZipFile(backup_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        names = set(archive.namelist())
    assert manifest["project_count"] == 1
    assert manifest["projects"] == [{"slug": project.slug, "name": project.name}]
    assert f"projects/{project.slug}/audits/{run_id}/round_2.csv" in names

    restored = ProjectStore(tmp_path / "restored" / "projects", tmp_path / "restored" / "secrets")
    imported = import_projects_backup(restored, backup_path)
    assert imported.imported == (project.slug,)
    restored_state = PipelineService(restored).load_current_state(project.slug)
    assert [item["index"] for item in restored_state["rounds"]] == [0, 1, 2]


def test_incremental_snowball_candidates_exclude_every_previously_seen_paper(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior.csv"
    full = tmp_path / "snowball_full.csv"
    incremental = tmp_path / "snowball_new.csv"
    fields = ["title", "year", "doi", "provider_id"]
    with prior.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "title": "Previously Reviewed Paper",
                "year": "2023",
                "doi": "10.1/preprint",
                "provider_id": "old-id",
            }
        )
    with full.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Previously Reviewed Paper",
                    "year": "2024",
                    "doi": "10.1/published",
                    "provider_id": "new-id",
                },
                {
                    "title": "New Citation Paper",
                    "year": "2025",
                    "doi": "10.1/new-citation",
                    "provider_id": "citation-id",
                },
            ]
        )

    added = _write_incremental_snowball_candidates(
        full,
        incremental,
        prior_paths=[prior],
    )
    _, rows, _ = load_audit(incremental)

    assert added == 1
    assert [row["title"] for row in rows] == ["New Citation Paper"]


def test_existing_snowball_audit_is_backed_up_and_pruned_against_earlier_rounds(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Audit Cleanup",
        research_question="Which papers?",
        scope_description="Test cross-round cleanup.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "audit-cleanup"
    audit_dir = store.project_dir(project.slug) / "audits" / run_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    round_0 = audit_dir / "round_0.csv"
    round_1 = audit_dir / "round_1.csv"
    fields = [
        "title",
        "year",
        "doi",
        "manual_decision",
        "manual_notes",
        "snowball_provider",
        "snowball_coverage_status",
        "snowball_missing_providers",
        "snowball_coverage_notes",
    ]
    with round_0.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "title": "Already Reviewed",
                "year": "2023",
                "doi": "10.1/old",
                "manual_decision": "include",
                "manual_notes": "Original decision",
            }
        )
    with round_1.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Already Reviewed",
                    "year": "2024",
                    "doi": "10.1/new-version",
                    "manual_decision": "",
                    "manual_notes": "",
                },
                {
                    "title": "Only New Paper",
                    "year": "2025",
                    "doi": "10.1/only-new",
                    "manual_decision": "",
                    "manual_notes": "",
                },
                {
                    "title": "Only New Paper",
                    "year": "2026",
                    "doi": "10.1/only-new-version",
                    "manual_decision": "",
                    "manual_notes": "",
                },
            ]
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(round_0)},
                "counts": {},
                "flow": [],
                "error": "",
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "ready_for_review",
                "files": {"audit": str(round_1)},
                "counts": {"coverage_failed_seeds": 0},
                "flow": [],
                "error": "",
            },
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    removed = service.reconcile_snowball_audit(project.slug, 1)
    persisted = service.load_current_state(project.slug)
    _, rows, _ = load_audit(round_1)

    assert removed == 2
    assert [row["title"] for row in rows] == ["Only New Paper"]
    assert {
        "snowball_provider",
        "snowball_coverage_status",
        "snowball_missing_providers",
        "snowball_coverage_notes",
    }.isdisjoint(rows[0])
    assert Path(persisted["rounds"][1]["files"]["audit_pre_cleanup"]).exists()


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
    stopped["rounds"][0]["error"] = "stale error from the interrupted attempt"
    service._save_state(project.slug, stopped)

    resumed = service.resume_initial_discovery(project.slug)

    assert resumed["run_id"] == run_id
    assert resumed["status"] == "awaiting_ai_or_review"
    assert resumed["rounds"][0]["error"] == ""
    assert collect_calls == 1
    assert enrichment_calls == 2
    assert [stage["key"] for stage in resumed["rounds"][0]["flow"]][-1] == ("abstract_enrichment")


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
            row["title"] for row in rows if row["auto_screening_decision"] != "exclude"
        )
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(summary=SimpleNamespace(with_abstract=1, attempted=len(attempted)))

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


def test_failed_snowball_retry_reuses_the_same_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Snowball Retry",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "snowball-retry"
    run_dir = store.project_dir(project.slug) / "runs" / run_id
    audit = run_dir / "audit.csv"
    pool = run_dir / "pool.csv"
    partial = run_dir / "partial_snowball.csv"
    partial_audit = run_dir / "partial_audit.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    for path, include_decision in [(audit, True), (pool, False)]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["title", "year", "doi", "manual_decision"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "title": "Seed Paper",
                    "year": "2025",
                    "doi": "10.1/seed",
                    "manual_decision": "include" if include_decision else "",
                }
            )
    shutil.copyfile(pool, partial)
    with partial_audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "doi", "manual_decision", "manual_notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Partial Paper",
                "year": "2024",
                "doi": "10.1/partial",
                "manual_decision": "exclude",
                "manual_notes": "Keep this decision across retry.",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "failed",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "awaiting_manual_review",
                "files": {"audit": str(audit), "pool": str(pool)},
                "counts": {},
                "flow": [],
                "error": "",
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "failed",
                "files": {
                    "snowballed": str(partial),
                    "audit": str(partial_audit),
                },
                "counts": {
                    "seeds": 1,
                    "citation_providers": ["semantic_scholar", "opencitations"],
                    "provider_successes": {"opencitations": 2},
                    "provider_failures": {"semantic_scholar": 1},
                },
                "flow": [],
                "error": "OpenAlex snowball request failed",
            },
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    attempted_inputs: list[Path] = []

    def fail_snowball(input_path, *_args, **_kwargs):
        attempted_inputs.append(Path(input_path))
        raise RuntimeError("Semantic Scholar snowball request failed (HTTP 429)")

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.snowball_candidates",
        fail_snowball,
    )

    with pytest.raises(RuntimeError, match="HTTP 429"):
        service.start_snowball_discovery(project.slug, core_online=False)

    retried = service.load_current_state(project.slug)
    assert len(retried["rounds"]) == 2
    assert retried["rounds"][-1]["index"] == 1
    assert retried["rounds"][-1]["counts"]["retry_count"] == 1
    assert retried["rounds"][-1]["status"] == "failed"
    assert attempted_inputs == [partial]
    checkpoint = Path(retried["rounds"][-1]["files"]["snowballed"])
    assert checkpoint.exists()
    assert "Seed Paper" in checkpoint.read_text(encoding="utf-8")
    audit_checkpoint = Path(retried["rounds"][-1]["files"]["audit_checkpoint"])
    assert audit_checkpoint.exists()
    assert "Keep this decision across retry." in audit_checkpoint.read_text(encoding="utf-8")


def test_partial_provider_failure_completes_discovery_and_can_enter_review(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Partial Snowball Resume",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "partial-snowball-resume"
    run_dir = store.project_dir(project.slug) / "runs" / run_id
    audit = run_dir / "round_0_audit.csv"
    pool = run_dir / "round_0_pool.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    fields = ["title", "year", "doi", "manual_decision", "auto_screening_decision"]
    row = {
        "title": "Seed Paper",
        "year": "2025",
        "doi": "10.1/seed",
        "manual_decision": "include",
        "auto_screening_decision": "include_candidate",
    }
    for path in (audit, pool):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(audit), "pool": str(pool)},
                "counts": {},
                "flow": [],
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    provider_orders: list[list[str]] = []
    input_paths: list[Path] = []

    def fake_snowball(input_path, output_path, config, *_args, **_kwargs):
        provider_orders.append(list(config.snowballing.providers))
        input_paths.append(Path(input_path))
        if Path(input_path) != Path(output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
        first_attempt = len(provider_orders) == 1
        summary = SnowballingSummary(
            input_rows=1,
            input_unique_rows=1,
            output_rows=1,
            seeds_loaded=1,
            seeds_resolved=1,
            added_rows=0,
            merged_rows=0,
            references_available=0,
            references_fetched=0,
            citations_available=0,
            citations_fetched=0,
            backward_truncated_seeds=0,
            forward_truncated_seeds=0,
            by_relation=Counter(),
            by_source=Counter(),
            by_provider=Counter(),
            provider_order=tuple(config.snowballing.providers),
            provider_strategy=config.snowballing.provider_strategy,
            provider_successes=Counter(
                {"opencitations": 2} if first_attempt else {"semantic_scholar": 2}
            ),
            provider_failures=Counter({"semantic_scholar": 1} if first_attempt else {}),
            provider_errors=({"semantic_scholar": ["temporary failure"]} if first_attempt else {}),
            seed_diagnostics=[],
        )
        return SnowballingResult([], summary)

    def fake_screen(input_path, output_path, *_args, **_kwargs):
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(summary=SimpleNamespace(total=1, by_decision=Counter()))

    def fake_venue(input_path, output_path, *_args, **_kwargs):
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(summary=SimpleNamespace())

    def fake_enrichment(input_path, output_path, *_args, **_kwargs):
        shutil.copyfile(input_path, output_path)
        return SimpleNamespace(
            summary=SimpleNamespace(
                with_abstract=0,
                attempted=1,
                api_requests=0,
                batch_requests=0,
                cache_hits=0,
                rate_limit_retries=0,
            )
        )

    monkeypatch.setattr("vnn_survey.app.pipeline_service.snowball_candidates", fake_snowball)
    monkeypatch.setattr("vnn_survey.app.pipeline_service.screen_candidates", fake_screen)
    monkeypatch.setattr("vnn_survey.app.pipeline_service.enrich_venue_quality", fake_venue)
    monkeypatch.setattr("vnn_survey.app.pipeline_service.enrich_candidates", fake_enrichment)
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.write_venue_quality_summary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.write_enrichment_summary",
        lambda *_args, **_kwargs: None,
    )

    completed = service.start_snowball_discovery(
        project.slug,
        citation_providers=["semantic_scholar", "opencitations"],
        core_online=False,
    )

    assert len(completed["rounds"]) == 2
    assert completed["rounds"][1]["status"] == "discovery_complete"
    assert completed["status"] == "awaiting_ai_or_review"
    assert completed["rounds"][1]["counts"]["provider_failures"] == {"semantic_scholar": 1}
    assert provider_orders == [["semantic_scholar", "opencitations"]]
    assert input_paths == [pool]

    with pytest.raises(RuntimeError, match="Prepare the current snowball round"):
        service.start_snowball_discovery(
            project.slug,
            citation_providers=["semantic_scholar", "opencitations"],
            core_online=False,
        )

    prepared = service.prepare_round_for_review(project.slug, 1, use_llm=False)
    assert Path(prepared["rounds"][1]["files"]["audit"]).exists()


def test_provider_failure_does_not_block_review_preparation(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Incomplete Citation Coverage",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "provider-incomplete"
    run_dir = store.project_dir(project.slug) / "runs" / run_id
    audit = run_dir / "round_0_audit.csv"
    enriched = run_dir / "round_1_enriched.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "doi", "manual_decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Seed Paper",
                "year": "2025",
                "doi": "10.1/seed",
                "manual_decision": "include",
            }
        )
    with enriched.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "year",
                "doi",
                "auto_screening_decision",
                "snowball_relations",
                "snowball_seed_titles",
                "snowball_provider",
                "snowball_coverage_status",
                "snowball_missing_providers",
                "snowball_coverage_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "New Snowball Candidate",
                "year": "2024",
                "doi": "10.1/new-candidate",
                "auto_screening_decision": "include_candidate",
                "snowball_relations": "forward",
                "snowball_seed_titles": "Seed Paper",
                "snowball_provider": "openalex",
                "snowball_coverage_status": "partial",
                "snowball_missing_providers": "semantic_scholar",
                "snowball_coverage_notes": "semantic_scholar: rate limited",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_ai_or_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(audit)},
                "counts": {},
                "flow": [],
                "error": "",
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "citation_incomplete",
                "files": {"enriched": str(enriched)},
                "counts": {
                    "provider_failures": {"semantic_scholar": 1},
                    "provider_successes": {"openalex": 38},
                    "citation_providers": [
                        "semantic_scholar",
                        "opencitations",
                    ],
                },
                "flow": [],
                "error": "",
            },
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    prepared = service.prepare_round_for_review(
        project.slug,
        1,
        use_llm=False,
    )

    assert Path(prepared["rounds"][1]["files"]["audit"]).exists()
    assert prepared["rounds"][1]["status"] == "ready_for_review"
    assert prepared["rounds"][1]["counts"]["provider_failures"] == {"semantic_scholar": 1}
    _, audit_rows, audit_summary = load_audit(Path(prepared["rounds"][1]["files"]["audit"]))
    assert audit_summary.total == 1
    assert audit_rows[0]["title"] == "New Snowball Candidate"
    hidden_fields = {
        "snowball_provider",
        "snowball_coverage_status",
        "snowball_missing_providers",
        "snowball_coverage_notes",
    }
    assert hidden_fields.isdisjoint(audit_rows[0])
    with enriched.open("r", encoding="utf-8", newline="") as handle:
        assert hidden_fields.isdisjoint(csv.DictReader(handle).fieldnames or [])


def test_review_preparation_restores_decisions_from_retry_checkpoint(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Restore Retry Audit",
        research_question="Which papers?",
        scope_description="A test scope.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "restore-retry-audit"
    run_dir = store.project_dir(project.slug) / "runs" / run_id
    initial_audit = run_dir / "round_0_audit.csv"
    enriched = run_dir / "round_1_enriched.csv"
    checkpoint = run_dir / "round_1_checkpoint.csv"
    initial_audit.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(
        initial_audit,
        [
            {
                "title": "Seed Paper",
                "year": "2025",
                "doi": "10.1/seed",
                "manual_decision": "include",
            }
        ],
    )
    candidate = {
        "title": "Previously Reviewed Candidate",
        "year": "2024",
        "doi": "10.1/candidate",
        "auto_screening_decision": "include_candidate",
    }
    write_rows(enriched, [candidate])
    write_rows(
        checkpoint,
        [
            {
                **candidate,
                "manual_decision": "exclude",
                "manual_notes": "Restored reviewer note.",
            }
        ],
    )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_ai_or_review",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(initial_audit)},
                "counts": {},
                "flow": [],
                "error": "",
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "discovery_complete",
                "files": {
                    "enriched": str(enriched),
                    "audit_checkpoint": str(checkpoint),
                },
                "counts": {"provider_failures": {}},
                "flow": [],
                "error": "",
            },
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)

    prepared = service.prepare_round_for_review(project.slug, 1, use_llm=False)
    _, rows, summary = load_audit(Path(prepared["rounds"][1]["files"]["audit"]))

    assert summary.reviewed == 1
    assert rows[0]["manual_decision"] == "exclude"
    assert rows[0]["manual_notes"] == "Restored reviewer note."
