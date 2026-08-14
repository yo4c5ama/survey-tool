from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from vnn_survey.app.pipeline_service import PipelineService
from vnn_survey.app.project_store import KeywordGroup, ProjectStore


def test_app_opens_project_creation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VNN_SURVEY_APP_DATA", str(tmp_path / "projects"))
    monkeypatch.setenv("VNN_SURVEY_APP_SECRETS", str(tmp_path / "secrets"))
    app_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert any(title.value == "SurveyFlow" for title in app.title)


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
    assert paper_metric.value == "37"
    assert any("37 papers collected" in item.proto.text for item in app.get("progress"))
