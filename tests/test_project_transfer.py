import json
import zipfile
from pathlib import Path

import pytest

from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.app.project_transfer import (
    ProjectTransferError,
    create_projects_backup,
    import_projects_backup,
)


def _create_project(store: ProjectStore, name: str):
    return store.create_project(
        name=name,
        research_question="Which papers are relevant?",
        scope_description="A portable test project.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )


def test_backup_restores_all_projects_and_rebases_machine_paths(tmp_path: Path) -> None:
    source = ProjectStore(tmp_path / "old" / "projects", tmp_path / "old" / "secrets")
    first = _create_project(source, "First survey")
    second = _create_project(source, "Second survey")
    run_id = "portable-run"
    run_dir = source.project_dir(first.slug) / "runs" / run_id
    output_path = run_dir / "processed" / "papers.csv"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("title\nPortable paper\n", encoding="utf-8")
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "rounds": [{"files": {"candidates": str(output_path.resolve())}}],
            }
        ),
        encoding="utf-8",
    )
    cache_file = source.project_dir(first.slug) / "cache" / "provider" / "cached.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("{}", encoding="utf-8")
    source.save_api_key(first.slug, "secret-key")

    backup = create_projects_backup(source, output_dir=tmp_path / "exports")

    with zipfile.ZipFile(backup.path) as archive:
        names = set(archive.namelist())
    assert backup.project_count == 2
    assert f"projects/{first.slug}/runs/{run_id}/state.json" in names
    assert f"projects/{second.slug}/project.json" in names
    assert not any(name.startswith("secrets/") for name in names)
    assert not any("/cache/" in name for name in names)

    restored = ProjectStore(tmp_path / "new" / "projects", tmp_path / "new" / "secrets")
    result = import_projects_backup(restored, backup.path)

    assert result.imported == (first.slug, second.slug)
    assert result.skipped == ()
    assert {project.slug for project in restored.list_projects()} == {
        first.slug,
        second.slug,
    }
    restored_state = json.loads(
        (restored.project_dir(first.slug) / "runs" / run_id / "state.json").read_text(
            encoding="utf-8"
        )
    )
    rebased = Path(restored_state["rounds"][0]["files"]["candidates"])
    expected_path = (
        restored.project_dir(first.slug).resolve()
        / "runs"
        / run_id
        / "processed"
        / "papers.csv"
    )
    assert rebased == expected_path
    pipeline = restored.config_path(first.slug).read_text(encoding="utf-8")
    assert str(restored.project_dir(first.slug).resolve()) in pipeline
    assert str(source.project_dir(first.slug).resolve()) not in pipeline
    assert not restored.has_api_key(first.slug)


def test_backup_can_restore_secrets_and_replace_an_existing_project(tmp_path: Path) -> None:
    source = ProjectStore(tmp_path / "source" / "projects", tmp_path / "source" / "secrets")
    project = _create_project(source, "Replace me")
    source.save_api_key(project.slug, "restored-secret")
    marker = source.project_dir(project.slug) / "manual" / "source.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("from backup", encoding="utf-8")
    backup = create_projects_backup(
        source,
        include_secrets=True,
        output_dir=tmp_path / "exports",
    )

    target = ProjectStore(tmp_path / "target" / "projects", tmp_path / "target" / "secrets")
    existing = _create_project(target, "Replace me")
    stale = target.project_dir(existing.slug) / "stale.txt"
    stale.write_text("stale", encoding="utf-8")
    target.save_api_key(existing.slug, "old-secret")

    skipped = import_projects_backup(target, backup.path)
    assert skipped.imported == ()
    assert skipped.skipped == (project.slug,)
    assert stale.exists()
    assert target.read_api_key(project.slug) == "old-secret"

    replaced = import_projects_backup(
        target,
        backup.path,
        conflict="replace",
        restore_secrets=True,
    )
    assert replaced.imported == (project.slug,)
    assert replaced.restored_secret_projects == (project.slug,)
    assert not stale.exists()
    assert (target.project_dir(project.slug) / "manual" / "source.txt").exists()
    assert target.read_api_key(project.slug) == "restored-secret"


def test_import_rejects_unsafe_or_non_surveyflow_archives(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "surveyflow-project-backup",
                    "format_version": 1,
                    "projects": [{"slug": "sample", "name": "Sample"}],
                }
            ),
        )

    with pytest.raises(ProjectTransferError, match="unsafe file path"):
        import_projects_backup(store, archive_path)
