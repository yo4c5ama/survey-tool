import csv
import logging
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from vnn_survey.ai_research import PaperWorkspace
from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.app.task_manager import TaskManager, TaskSnapshot


def test_app_opens_project_creation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(tmp_path / "projects"))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(tmp_path / "secrets"))
    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert any(title.value == "SurveyFlow" for title in app.title)
    assert any(item.label == "Backup and restore" for item in app.expander)
    assert any(item.key == "project_backup_upload" for item in app.file_uploader)
    assert any("yoac" in item.value and "Codex" in item.value for item in app.markdown)


def test_app_switches_between_all_interface_languages(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(tmp_path / "projects"))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(tmp_path / "secrets"))
    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    expected_subheaders = {
        "zh": "创建综述项目",
        "ja": "レビュー・プロジェクトを作成",
        "ko": "문헌 검토 프로젝트 만들기",
        "en": "Create a survey project",
    }
    project_name = next(item for item in app.text_input if item.key == "create_project_name")
    project_name.set_value("Persistent research input")
    app.run(timeout=20)
    for language, expected in expected_subheaders.items():
        language_picker = next(item for item in app.selectbox if item.key == "ui_language")
        language_picker.set_value(language)
        app.run(timeout=20)
        assert not app.exception
        assert any(header.value == expected for header in app.subheader)
        project_name = next(item for item in app.text_input if item.key == "create_project_name")
        assert project_name.value == "Persistent research input"


def test_create_project_recommends_sources_for_selected_field(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(tmp_path / "projects"))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(tmp_path / "secrets"))
    st.cache_resource.clear()
    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    domain = next(item for item in app.selectbox if item.key == "create_research_domain")
    sources = next(item for item in app.multiselect if item.key == "create_discovery_sources")
    assert domain.value == "computer_science"
    assert sources.value == ["dblp", "openalex", "arxiv"]

    domain.set_value("arts_design")
    app.run(timeout=20)

    sources = next(item for item in app.multiselect if item.key == "create_discovery_sources")
    assert sources.value == ["openalex", "crossref"]


def test_manual_review_allows_paper_additions_before_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    store.create_project(
        name="Manual workspace",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("manual_review")
    app.run(timeout=20)

    assert not app.exception
    assert any(title.value == "Manual review" for title in app.title)
    assert any(item.label == "Add papers" for item in app.expander)
    assert any(
        item.key == "manual_lookup_title_manual-workspace" for item in app.text_input
    )


def test_snowball_page_shows_selectable_round_history_and_delete_control(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Snowball history",
        research_question="Which papers?",
        scope_description="Test history UI.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    service = PipelineService(store)
    run_id = "history-ui-run"
    project_dir = store.project_dir(project.slug)
    audit_dir = project_dir / "audits" / run_id
    processed = project_dir / "runs" / run_id / "processed"
    audit_dir.mkdir(parents=True)
    processed.mkdir(parents=True)

    audit_fields = ["title", "year", "manual_decision", "manual_notes"]
    initial_audit = audit_dir / "round_0.csv"
    snowball_audit = audit_dir / "round_1.csv"
    for path, title in [
        (initial_audit, "Initial seed"),
        (snowball_audit, "New paper"),
    ]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "title": title,
                    "year": "2025",
                    "manual_decision": "include",
                    "manual_notes": "Keep",
                }
            )
    snowball_new = processed / "candidate_papers_snowball_new_round_1.csv"
    with snowball_new.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "snowball_relations"])
        writer.writeheader()
        writer.writerow({"title": "New paper", "snowball_relations": "backward; forward"})
    seed_path = project_dir / "seeds" / f"{run_id}_round_1.yaml"
    seed_path.write_text("papers: []\n", encoding="utf-8")

    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "awaiting_manual_review",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(initial_audit)},
                "counts": {"audit_queue": 1},
                "flow": [],
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "ready_for_review",
                "snowball_mode": "incremental",
                "files": {
                    "audit": str(snowball_audit),
                    "snowball_new": str(snowball_new),
                    "seeds": str(seed_path),
                },
                "counts": {
                    "seeds": 1,
                    "added_rows": 1,
                    "audit_queue": 1,
                    "coverage_complete_seeds": 1,
                },
                "flow": [],
            },
        ],
    }
    store.set_current_run(project.slug, run_id)
    service._save_state(project.slug, state)
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("snowball")
    app.run(timeout=20)

    assert not app.exception
    history = next(
        item for item in app.selectbox if item.key == f"snowball_history_round_{run_id}"
    )
    assert history.value == 1
    assert any(item.label == "Delete selected round" for item in app.expander)
    assert any("Round files" in item.value for item in app.markdown)


def test_manual_review_embeds_live_paper_addition_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Integrated Review",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    run_id = "integrated-review-run"
    audit = store.project_dir(project.slug) / "audits" / run_id / "round_0.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "year",
                "abstract",
                "url",
                "manual_decision",
                "manual_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "A Review Paper",
                "year": "2025",
                "abstract": "An abstract.",
                "url": "https://example.org/review-paper.pdf",
                "manual_decision": "",
                "manual_notes": "",
            }
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
                "created_at": "2026-01-01T00:00:00",
                "files": {"audit": str(audit)},
                "counts": {"audit_queue": 1},
                "flow": [],
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    PipelineService(store)._save_state(project.slug, state)
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("manual_review")
    app.run(timeout=20)

    assert not app.exception
    assert any(item.label == "Add papers" for item in app.expander)
    assert any(
        item.key == "manual_lookup_title_integrated-review" for item in app.text_input
    )
    assert any(
        item.key == "manual_enrichment_start_integrated-review_0"
        for item in app.button
    )
    assert any("saved automatically" in item.value for item in app.caption)
    assert any(
        'class="sf-paper-source-link"' in item.value
        and 'href="https://example.org/review-paper.pdf"' in item.value
        for item in app.markdown
    )


def test_manual_review_selects_a_newly_created_audit_round(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.WARNING)
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Live audit rounds",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    run_id = "live-audit-run"
    audit_dir = store.project_dir(project.slug) / "audits" / run_id
    audit_dir.mkdir(parents=True, exist_ok=True)

    def write_audit(round_index: int, title: str) -> Path:
        path = audit_dir / f"round_{round_index}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["title", "year", "abstract", "manual_decision", "manual_notes"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "title": title,
                    "year": "2025",
                    "abstract": "An abstract.",
                    "manual_decision": "",
                    "manual_notes": "",
                }
            )
        return path

    round_zero = write_audit(0, "Initial paper")
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
                "files": {"audit": str(round_zero)},
                "counts": {"audit_queue": 1},
                "flow": [],
                "error": "",
            }
        ],
    }
    store.set_current_run(project.slug, run_id)
    service = PipelineService(store)
    service._save_state(project.slug, state)
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("manual_review")
    app.run(timeout=20)
    selector = next(item for item in app.selectbox if item.key == f"audit_round_{run_id}")
    assert selector.value == 0

    round_one = write_audit(1, "New AI-screened paper")
    state["rounds"].append(
        {
            "index": 1,
            "kind": "snowball",
            "status": "ready_for_review",
            "files": {"audit": str(round_one)},
            "counts": {"audit_queue": 1},
            "flow": [],
            "error": "",
        }
    )
    service._save_state(project.slug, state)

    app.run(timeout=20)
    selector = next(item for item in app.selectbox if item.key == f"audit_round_{run_id}")
    assert selector.options == ["0", "1"]
    assert selector.value == 1
    assert not any(
        "was created with a default value but also had its value set" in record.getMessage()
        for record in caplog.records
    )


def test_ai_research_workspace_opens_for_exported_corpus(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="AI workspace",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    run_id = "exported-run"
    included = store.project_dir(project.slug) / "exports" / run_id / "included.csv"
    included.parent.mkdir(parents=True, exist_ok=True)
    with included.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "authors", "year", "venue", "doi", "abstract"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "A Final Paper",
                "authors": "A. Author",
                "year": "2025",
                "venue": "A Venue",
                "doi": "10.1000/final",
                "abstract": "An abstract.",
            }
        )
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
    PipelineService(store)._save_state(project.slug, state)
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("ai_research")
    app.run(timeout=20)

    assert not app.exception
    assert any(tab.label == "Paper analysis" for tab in app.tabs)
    assert any(tab.label == "Corpus classification" for tab in app.tabs)
    assert any("A Final Paper" in header.value for header in app.markdown)
    assert any(button.label == "Analyze paper" for button in app.button)
    assert not app.chat_input

    paper = {
        "title": "A Final Paper",
        "authors": "A. Author",
        "year": "2025",
        "venue": "A Venue",
        "doi": "10.1000/final",
        "abstract": "An abstract.",
    }
    PaperWorkspace(store.project_dir(project.slug)).save_analysis(
        paper,
        content="## English\nA concise saved briefing.",
        model="gpt-test",
        response_id="resp_saved",
        source_kind="metadata",
        source_url="https://doi.org/10.1000/final",
        interface_language="en",
    )
    app.run(timeout=20)

    assert not app.exception
    assert any("A concise saved briefing." in item.value for item in app.markdown)
    assert any(
        item.placeholder == "Ask a question about this paper" for item in app.chat_input
    )

    language = next(item for item in app.selectbox if item.key == "ui_language")
    language.set_value("zh")
    app.run(timeout=20)

    assert not app.exception
    assert any(tab.label == "论文分析" for tab in app.tabs)
    assert any(item.value == "AI 论文导读" for item in app.subheader)
    assert any(item.placeholder == "针对这篇论文提问" for item in app.chat_input)


def test_run_center_restores_saved_progress_and_paper_count(monkeypatch, tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Persistent run",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    run_id = "saved-run"
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "running_discovery",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "running",
                "created_at": "2026-01-01T00:00:00",
                "files": {},
                "counts": {},
                "error": "",
            }
        ],
        "progress": {
            "operation": "Initial discovery",
            "heading": "Running initial discovery",
            "status": "running",
            "stages": [
                "DBLP search",
                "Venue enrichment",
                "Rule screening",
                "Abstract enrichment",
            ],
            "stage": "Abstract enrichment",
            "message": "Looking up abstracts through OpenAlex.",
            "completed": 4,
            "total": 10,
            "current": "A saved paper",
            "paper_count": 37,
            "started_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:01:00",
        },
    }
    store.set_current_run(project.slug, run_id)
    PipelineService(store)._save_state(project.slug, state)
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("run_center")
    app.run(timeout=20)

    assert not app.exception
    assert any(header.value == "Current run" for header in app.subheader)
    paper_metric = next(metric for metric in app.metric if metric.label == "Papers collected")
    assert paper_metric.value == "10"
    progress_text = [item.proto.text for item in app.get("progress")]
    assert any("Stage 4 of 4: Abstract enrichment" in text for text in progress_text)
    assert all("papers collected" not in text for text in progress_text)


def test_live_progress_controls_render_only_in_the_relevant_module(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Scoped progress",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    run_id = "scoped-progress-run"
    audit = store.project_dir(project.slug) / "audits" / run_id / "round_0.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "manual_decision", "manual_notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "A Reviewed Paper",
                "manual_decision": "include",
                "manual_notes": "In scope.",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "running",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "created_at": "2026-01-01T00:00:00",
                "files": {"audit": str(audit)},
                "counts": {"audit_queue": 1, "reviewed": 1},
                "flow": [],
                "error": "",
            }
        ],
        "progress": {
            "operation": "Prompt refinement",
            "status": "running",
            "stages": ["Prompt refinement"],
            "stage": "Prompt refinement",
            "message": "Analyzing review decisions.",
            "completed": 1,
            "total": 2,
            "paper_count": 1,
        },
    }
    store.set_current_run(project.slug, run_id)
    PipelineService(store)._save_state(project.slug, state)
    monkeypatch.setattr(
        TaskManager,
        "snapshot",
        lambda self, slug: TaskSnapshot(
            operation="prompt_refinement",
            started_at="2026-01-01T00:00:00",
            running=True,
            error="",
            cancel_requested=False,
            cancelled=False,
            can_restart=False,
        ),
    )
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("manual_review")
    app.run(timeout=20)

    assert not app.exception
    stop_keys = {item.key for item in app.button if item.label == "Stop run"}
    assert stop_keys == {f"stop_run_prompt_refinement_{project.slug}"}


def test_manual_review_tracks_review_queue_creation_until_the_new_round_is_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Review preparation progress",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    run_id = "review-preparation-progress"
    audit = store.project_dir(project.slug) / "audits" / run_id / "round_0.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "manual_decision", "manual_notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Previously reviewed paper",
                "manual_decision": "include",
                "manual_notes": "Keep visible while the next round runs.",
            }
        )
    state = {
        "project_slug": project.slug,
        "run_id": run_id,
        "status": "running",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "rounds": [
            {
                "index": 0,
                "kind": "initial",
                "status": "ready_for_review",
                "files": {"audit": str(audit)},
                "counts": {"audit_queue": 1, "reviewed": 1},
                "flow": [],
                "error": "",
            },
            {
                "index": 1,
                "kind": "snowball",
                "status": "discovery_complete",
                "files": {},
                "counts": {},
                "flow": [],
                "error": "",
            }
        ],
        "progress": {
            "operation": "AI abstract screening and review",
            "status": "running",
            "stages": ["AI abstract screening", "Create manual review queue"],
            "stage": "AI abstract screening",
            "message": "Analyzing abstracts.",
            "completed": 1,
            "total": 4,
            "paper_count": 4,
        },
    }
    store.set_current_run(project.slug, run_id)
    PipelineService(store)._save_state(project.slug, state)
    monkeypatch.setattr(
        TaskManager,
        "snapshot",
        lambda self, slug: TaskSnapshot(
            operation="review_preparation",
            started_at="2026-01-01T00:00:00",
            running=True,
            error="",
            cancel_requested=False,
            cancelled=False,
            can_restart=False,
        ),
    )
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("manual_review")
    app.run(timeout=20)

    assert not app.exception
    stop_keys = {item.key for item in app.button if item.label == "Stop run"}
    assert stop_keys == {f"stop_run_manual_review_{project.slug}"}
    selector = next(item for item in app.selectbox if item.key == f"audit_round_{run_id}")
    assert selector.value == 0
    assert not any(item.key == f"prepare_review_1_{project.slug}" for item in app.button)


def test_ai_prompt_can_be_regenerated_after_widget_is_created(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Prompt reset",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("ai_settings")
    app.run(timeout=20)

    prompt = next(
        item for item in app.text_area if item.key == f"ai_prompt_{project.slug}"
    )
    prompt.set_value("A temporary custom prompt")
    app.run(timeout=20)
    regenerate = next(item for item in app.button if item.label == "Regenerate from scope")
    regenerate.click()
    app.run(timeout=20)

    assert not app.exception
    prompt = next(
        item for item in app.text_area if item.key == f"ai_prompt_{project.slug}"
    )
    expected = store.system_prompt_path(project.slug).read_text(encoding="utf-8")
    assert prompt.value.rstrip("\n") == expected.rstrip("\n")
    assert prompt.value != "A temporary custom prompt"


def test_run_center_offers_title_prescreen_when_api_key_is_saved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects_root = tmp_path / "projects"
    secrets_root = tmp_path / "secrets"
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(projects_root))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(secrets_root))
    store = ProjectStore(projects_root, secrets_root)
    project = store.create_project(
        name="Title Prescreen UI",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    store.save_api_key(project.slug, "test-key")
    st.cache_resource.clear()

    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    workspace = next(item for item in app.radio if item.key == "workspace_page")
    workspace.set_value("run_center")
    app.run(timeout=20)

    title_toggle = next(
        item
        for item in app.toggle
        if item.key == f"title_llm_{project.slug}_False"
    )
    batch_size = next(
        item
        for item in app.number_input
        if item.key == f"title_batch_{project.slug}_False"
    )
    assert not app.exception
    assert title_toggle.value is True
    assert batch_size.value == 100
