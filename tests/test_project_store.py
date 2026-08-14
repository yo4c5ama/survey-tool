from pathlib import Path

from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import load_config


def test_project_store_generates_isolated_generic_config(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Formal NLP",
        research_question="Which methods certify NLP systems?",
        scope_description="Formal guarantees for NLP models.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[
            KeywordGroup("model", ["language model", "transformer"]),
            KeywordGroup("method", ["verification", "certification"]),
        ],
        inclusion_criteria=["The model is the verification target."],
        exclusion_criteria=["The model is only a tool."],
    )

    config = load_config(store.config_path(project.slug))

    assert project.slug == "formal-nlp"
    assert config.screening.profile == "generic"
    assert config.years.start == 2020
    assert config.years.end == 2026
    assert config.build_queries() == [
        "transformer verification|certification",
        "language model verification|certification",
    ]
    assert (
        "Which methods certify NLP systems?" in store.system_prompt_path(project.slug).read_text()
    )


def test_api_key_is_outside_project_and_owner_only(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Key Test",
        research_question="Test",
        scope_description="Test",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )

    key_path = store.save_api_key(project.slug, "test-secret")

    assert key_path.read_text() == "test-secret"
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert "test-secret" not in store.config_path(project.slug).read_text()
