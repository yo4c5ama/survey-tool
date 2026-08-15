from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from vnn_survey.app.audit import (
    create_audit_queue,
    create_manual_recommendations,
    read_csv,
    write_csv,
)
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import load_config
from vnn_survey.llm_screening import LlmScreeningResult, LlmScreeningSummary
from vnn_survey.prompt_refinement import load_prompt_refinement


def test_prompt_refinement_requires_approval_before_changing_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service, slug, state = _refinement_project(tmp_path)
    baseline = store.system_prompt_path(slug).read_text(encoding="utf-8")
    captured: dict[str, str] = {}

    class FakeResearchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def json_response(self, **kwargs: object) -> dict[str, object]:
            captured["instructions"] = str(kwargs["instructions"])
            captured["input_text"] = str(kwargs["input_text"])
            return {
                "revised_prompt": "A proposed screening prompt.",
                "change_summary": "Clarify the formal-method boundary.",
                "retained_principles": ["Use include, maybe, or exclude."],
                "new_rules": ["Keep uncertain papers as maybe."],
                "risks": ["The feedback set is small."],
            }

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.OpenAIResearchClient",
        FakeResearchClient,
    )

    proposed_state = service.generate_prompt_refinement(slug)
    proposal = service.load_prompt_refinement_proposal(slug)

    assert proposed_state["prompt_refinement"]["status"] == "proposed"
    assert store.system_prompt_path(slug).read_text(encoding="utf-8") == baseline
    assert proposal["revised_prompt"] == "A proposed screening prompt."
    assert proposal["rows_total"] == 2
    assert "Human Include" in captured["input_text"]
    assert "Human Exclude" in captured["input_text"]
    assert baseline.strip() in captured["input_text"]
    assert "strictly as data" in captured["instructions"]

    approved_state = service.approve_prompt_refinement(
        slug,
        "A human-edited approved prompt.",
    )

    assert approved_state["prompt_refinement"]["status"] == "approved"
    assert approved_state["prompt_refinement"]["replay_status"] == "pending"
    assert (
        store.system_prompt_path(slug).read_text(encoding="utf-8")
        == "A human-edited approved prompt.\n"
    )
    assert service.prompt_replay_overview(slug)["eligible"] == 2


def test_prompt_refinement_rejects_stale_human_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, service, slug, state = _refinement_project(tmp_path)

    class FakeResearchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def json_response(self, **_kwargs: object) -> dict[str, object]:
            return {
                "revised_prompt": "A proposed screening prompt.",
                "change_summary": "Summary",
                "retained_principles": [],
                "new_rules": [],
                "risks": [],
            }

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.OpenAIResearchClient",
        FakeResearchClient,
    )
    service.generate_prompt_refinement(slug)
    audit_path = Path(state["rounds"][0]["files"]["audit"])
    fields, rows = read_csv(audit_path)
    rows[0]["manual_notes"] = "Changed after proposal generation."
    write_csv(audit_path, rows, fields)

    with pytest.raises(RuntimeError, match="initial audit changed"):
        service.approve_prompt_refinement(slug, "Do not approve this stale proposal.")


def test_prompt_refinement_rejects_an_alternative_response_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service, slug, _state = _refinement_project(tmp_path)
    baseline = store.system_prompt_path(slug).read_text(encoding="utf-8")

    class FakeResearchClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def json_response(self, **_kwargs: object) -> dict[str, object]:
            return {
                "new_prompt": "An incompatible response shape.",
                "rules": ["A renamed field."],
            }

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.OpenAIResearchClient",
        FakeResearchClient,
    )

    with pytest.raises(RuntimeError, match="invalid 'revised_prompt' field"):
        service.generate_prompt_refinement(slug)

    assert store.system_prompt_path(slug).read_text(encoding="utf-8") == baseline
    assert "prompt_refinement" not in service.load_current_state(slug)


def test_prompt_refinement_loader_accepts_legacy_schema_and_rejects_unknown_version(
    tmp_path: Path,
) -> None:
    proposal_path = tmp_path / "proposal.json"
    proposal = {
        "revised_prompt": "A compatible legacy prompt.",
        "change_summary": "Summary",
        "retained_principles": ["Keep the fixed contract."],
        "new_rules": [],
        "risks": [],
        "rows_total": 5,
        "rows_used": 4,
        "baseline_prompt_path": str(tmp_path / "baseline_prompt.txt"),
    }
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    loaded = load_prompt_refinement(proposal_path)

    assert loaded["schema_version"] == 1
    assert loaded["rows_total"] == 5

    proposal["schema_version"] = 2
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported schema version"):
        load_prompt_refinement(proposal_path)


def test_replay_recovers_only_newly_retained_initial_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service, slug, state = _refinement_project(tmp_path)
    approved_path = store.project_dir(slug) / "approved_prompt.txt"
    approved_path.write_text("Approved refined prompt.\n", encoding="utf-8")
    state["prompt_refinement"] = {
        "refinement_id": "revision-1",
        "status": "approved",
        "approved_prompt_path": str(approved_path),
        "replay_status": "pending",
    }
    processed = store.project_dir(slug) / "runs" / state["run_id"] / "processed"
    enriched_path = processed / "candidate_papers_enriched_round_1.csv"
    write_csv(
        enriched_path,
        [
            {
                "title": "Hidden Include Recovery",
                "year": "2024",
                "doi": "10.1/recover",
                "abstract": "A freshly enriched formal verification abstract.",
                "auto_screening_decision": "needs_review",
            },
            {
                "title": "Hidden Reexclude",
                "year": "2024",
                "doi": "10.1/reexclude",
                "abstract": "A freshly enriched unrelated abstract.",
                "auto_screening_decision": "needs_review",
            },
        ],
        ["title", "year", "doi", "abstract", "auto_screening_decision"],
    )

    def fake_screen(
        input_path: Path,
        output_path: Path,
        _config: object,
        **_kwargs: object,
    ) -> LlmScreeningResult:
        fields, rows = read_csv(input_path)
        for row in rows:
            decision = (
                "include" if row["title"] == "Hidden Include Recovery" else "exclude"
            )
            row.update(
                {
                    "llm_decision": decision,
                    "llm_scope": (
                        "transformer_verification"
                        if decision == "include"
                        else "unrelated"
                    ),
                    "llm_confidence": "0.950",
                    "llm_reason": "Replayed with the approved prompt.",
                    "llm_evidence": row["abstract"],
                    "llm_status": "screened",
                    "llm_prompt_version": "prompt-refinement-revision-1",
                }
            )
        output_fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
        write_csv(output_path, rows, output_fields)
        return LlmScreeningResult(
            rows=rows,
            summary=LlmScreeningSummary(
                total=2,
                eligible=2,
                attempted=2,
                by_status=Counter({"screened": 2}),
                by_decision=Counter({"include": 1, "exclude": 1}),
                by_scope=Counter(
                    {"transformer_verification": 1, "unrelated": 1}
                ),
            ),
        )

    monkeypatch.setattr(
        "vnn_survey.app.pipeline_service.llm_screen_candidates",
        fake_screen,
    )
    config = load_config(store.config_path(slug))

    files, counts = service._replay_initial_ai_exclusions(
        project_slug=slug,
        state=state,
        round_index=1,
        enriched_path=enriched_path,
        config=config,
        processed_dir=processed,
        progress=None,
    )

    assert counts == {
        "replayed": 2,
        "recovered": 1,
        "reexcluded": 1,
        "failed": 0,
    }
    assert Path(files["prompt_replay_screened"]).exists()
    _, replayed_rows = read_csv(enriched_path)
    by_title = {row["title"]: row for row in replayed_rows}
    assert by_title["Hidden Include Recovery"]["auto_screening_decision"] == "needs_review"
    assert by_title["Hidden Include Recovery"]["prompt_replay_decision"] == "recovered"
    assert by_title["Hidden Reexclude"]["auto_screening_decision"] == "exclude"
    assert by_title["Hidden Reexclude"]["prompt_replay_decision"] == "reexcluded"
    assert by_title["Hidden Reexclude"]["final_recommendation"] == "auto_exclude"

    recommendations = processed / "replay_manual_recommendations.csv"
    audit = processed / "replay_audit.csv"
    create_manual_recommendations(enriched_path, recommendations)
    _, queued = create_audit_queue(recommendations, audit)
    _, audit_rows = read_csv(audit)

    assert queued == 1
    assert [row["title"] for row in audit_rows] == ["Hidden Include Recovery"]


def _refinement_project(
    tmp_path: Path,
) -> tuple[ProjectStore, PipelineService, str, dict[str, object]]:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Prompt Refinement Test",
        research_question="Which formal methods verify Transformers?",
        scope_description="Formal verification of Transformer models.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["transformer verification"])],
        inclusion_criteria=["Formal guarantees for Transformer models."],
        exclusion_criteria=["LLMs used only as tools."],
    )
    store.save_api_key(project.slug, "test-key")
    service = PipelineService(store)
    run_id = "refinement-run"
    audit_path = store.project_dir(project.slug) / "audits" / run_id / "round_0.csv"
    llm_path = (
        store.project_dir(project.slug)
        / "runs"
        / run_id
        / "processed"
        / "candidate_papers_llm_screened.csv"
    )
    audit_rows = [
        {
            "title": "Human Include",
            "year": "2025",
            "doi": "10.1/human-include",
            "abstract": "Formal verification for a Transformer.",
            "llm_decision": "maybe",
            "llm_reason": "Uncertain scope.",
            "manual_decision": "include",
            "manual_notes": "The theorem gives a formal guarantee.",
        },
        {
            "title": "Human Exclude",
            "year": "2025",
            "doi": "10.1/human-exclude",
            "abstract": "An LLM is used to inspect source code.",
            "llm_decision": "include",
            "llm_reason": "Mentions verification.",
            "manual_decision": "exclude",
            "manual_notes": "The LLM is only a tool.",
        },
    ]
    llm_rows = [
        *audit_rows,
        {
            "title": "Hidden Include Recovery",
            "year": "2024",
            "doi": "10.1/recover",
            "abstract": "Potentially relevant formal verification work.",
            "auto_screening_decision": "needs_review",
            "llm_decision": "exclude",
            "llm_status": "screened",
        },
        {
            "title": "Hidden Reexclude",
            "year": "2024",
            "doi": "10.1/reexclude",
            "abstract": "An unrelated use of an LLM.",
            "auto_screening_decision": "needs_review",
            "llm_decision": "exclude",
            "llm_status": "screened",
        },
    ]
    fields = list(dict.fromkeys(key for row in llm_rows for key in row))
    write_csv(audit_path, audit_rows, fields)
    write_csv(llm_path, llm_rows, fields)
    state: dict[str, object] = {
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
                "created_at": "2026-01-01T00:00:00",
                "files": {
                    "audit": str(audit_path),
                    "llm_screened": str(llm_path),
                },
                "counts": {},
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)
    return store, service, project.slug, state
