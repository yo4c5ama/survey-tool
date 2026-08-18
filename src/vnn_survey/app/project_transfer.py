from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from vnn_survey.app.project_store import ProjectSettings, ProjectStore

BACKUP_FORMAT = "surveyflow-project-backup"
BACKUP_FORMAT_VERSION = 1
MAX_ARCHIVE_FILES = 100_000
MAX_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024
_SLUG_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class ProjectTransferError(ValueError):
    """Raised when a SurveyFlow backup cannot be created or restored safely."""


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    project_count: int
    file_count: int
    source_bytes: int
    includes_secrets: bool
    includes_caches: bool


@dataclass(frozen=True, slots=True)
class ImportResult:
    imported: tuple[str, ...]
    skipped: tuple[str, ...]
    restored_secret_projects: tuple[str, ...]
    backup_includes_secrets: bool


def create_projects_backup(
    store: ProjectStore,
    *,
    include_secrets: bool = False,
    include_caches: bool = False,
    output_dir: Path | None = None,
    project_slugs: list[str] | None = None,
) -> BackupResult:
    projects = store.list_projects()
    if project_slugs is not None:
        requested = set(project_slugs)
        projects = [project for project in projects if project.slug in requested]
        missing = requested - {project.slug for project in projects}
        if missing:
            raise ProjectTransferError(
                f"Unknown project(s): {', '.join(sorted(missing))}."
            )
    if not projects:
        raise ProjectTransferError("Create at least one project before exporting a backup.")

    destination = Path(output_dir or store.root.parent / "backups")
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = destination / f"SurveyFlow-projects-{timestamp}.zip"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    file_count = 0
    source_bytes = 0

    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for project in projects:
                count, size = _write_tree(
                    archive,
                    store.project_dir(project.slug),
                    PurePosixPath("projects") / project.slug,
                    include_caches=include_caches,
                )
                file_count += count
                source_bytes += size
                if include_secrets:
                    count, size = _write_tree(
                        archive,
                        store.secrets_root / project.slug,
                        PurePosixPath("secrets") / project.slug,
                        include_caches=True,
                    )
                    file_count += count
                    source_bytes += size

            manifest = {
                "format": BACKUP_FORMAT,
                "format_version": BACKUP_FORMAT_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "project_count": len(projects),
                "projects": [
                    {"slug": project.slug, "name": project.name} for project in projects
                ],
                "includes_secrets": include_secrets,
                "includes_caches": include_caches,
            }
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
        temporary.replace(output_path)
        output_path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return BackupResult(
        path=output_path,
        project_count=len(projects),
        file_count=file_count,
        source_bytes=source_bytes,
        includes_secrets=include_secrets,
        includes_caches=include_caches,
    )


def save_uploaded_backup(store: ProjectStore, content: bytes) -> Path:
    if not content.startswith(b"PK"):
        raise ProjectTransferError("The uploaded file is not a ZIP backup.")
    destination = store.root.parent / "backups"
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = destination / f"SurveyFlow-import-{timestamp}.zip"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    path.chmod(0o600)
    return path


def import_projects_backup(
    store: ProjectStore,
    archive_path: Path,
    *,
    conflict: str = "skip",
    restore_secrets: bool = False,
) -> ImportResult:
    if conflict not in {"skip", "replace"}:
        raise ValueError("conflict must be 'skip' or 'replace'.")

    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectTransferError("The uploaded file is not a valid ZIP backup.") from exc

    with archive:
        manifest = _validate_archive(archive)
        project_entries = manifest["projects"]
        with tempfile.TemporaryDirectory(
            prefix=".surveyflow-import-",
            dir=store.root.parent,
        ) as temporary_value:
            staging_root = Path(temporary_value)
            _extract_archive(archive, staging_root)
            _validate_staged_projects(staging_root, project_entries)
            return _install_projects(
                store,
                staging_root,
                project_entries,
                conflict=conflict,
                restore_secrets=restore_secrets,
                backup_includes_secrets=bool(manifest.get("includes_secrets")),
            )


def _write_tree(
    archive: zipfile.ZipFile,
    root: Path,
    archive_root: PurePosixPath,
    *,
    include_caches: bool,
) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    file_count = 0
    source_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not include_caches and relative.parts and relative.parts[0] == "cache":
            continue
        if path.is_symlink() or not path.is_file():
            continue
        archive.write(path, (archive_root / PurePosixPath(*relative.parts)).as_posix())
        file_count += 1
        source_bytes += path.stat().st_size
    return file_count, source_bytes


def _validate_archive(archive: zipfile.ZipFile) -> dict[str, Any]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_FILES:
        raise ProjectTransferError("The backup contains too many files.")
    if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
        raise ProjectTransferError("The uncompressed backup is too large.")

    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ProjectTransferError("The backup contains an unsafe file path.")
        if path.parts[0] not in {"manifest.json", "projects", "secrets"}:
            raise ProjectTransferError("The backup contains an unsupported top-level entry.")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ProjectTransferError("Symbolic links are not allowed in project backups.")

    try:
        manifest = json.loads(archive.read("manifest.json"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProjectTransferError("The backup manifest is missing or invalid.") from exc
    if not isinstance(manifest, dict):
        raise ProjectTransferError("The backup manifest is invalid.")
    if manifest.get("format") != BACKUP_FORMAT:
        raise ProjectTransferError("This ZIP is not a SurveyFlow project backup.")
    if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
        raise ProjectTransferError("This SurveyFlow backup version is not supported.")

    projects = manifest.get("projects")
    if not isinstance(projects, list) or not projects:
        raise ProjectTransferError("The backup does not contain any projects.")
    slugs: set[str] = set()
    for item in projects:
        if not isinstance(item, dict):
            raise ProjectTransferError("The project list in the backup is invalid.")
        slug = str(item.get("slug") or "")
        if not _SLUG_PATTERN.fullmatch(slug) or slug in slugs:
            raise ProjectTransferError("The backup contains an invalid project identifier.")
        slugs.add(slug)

    damaged = archive.testzip()
    if damaged:
        raise ProjectTransferError(f"The backup is damaged at {damaged}.")
    return manifest


def _extract_archive(archive: zipfile.ZipFile, staging_root: Path) -> None:
    for info in archive.infolist():
        if info.is_dir():
            continue
        relative = PurePosixPath(info.filename)
        target = staging_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _validate_staged_projects(
    staging_root: Path,
    project_entries: list[dict[str, Any]],
) -> None:
    for item in project_entries:
        slug = str(item["slug"])
        metadata_path = staging_root / "projects" / slug / "project.json"
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProjectTransferError(
                f"Project metadata is missing or invalid for {slug}."
            ) from exc
        if not isinstance(value, dict) or str(value.get("slug") or "") != slug:
            raise ProjectTransferError(f"Project metadata does not match {slug}.")
        ProjectSettings.from_dict(value)


def _install_projects(
    store: ProjectStore,
    staging_root: Path,
    project_entries: list[dict[str, Any]],
    *,
    conflict: str,
    restore_secrets: bool,
    backup_includes_secrets: bool,
) -> ImportResult:
    imported: list[str] = []
    skipped: list[str] = []
    restored_secret_projects: list[str] = []
    rollback_root = staging_root / "rollback"

    for item in project_entries:
        slug = str(item["slug"])
        source = staging_root / "projects" / slug
        target = store.project_dir(slug)
        if target.exists() and conflict == "skip":
            skipped.append(slug)
            continue

        previous_project = rollback_root / "projects" / slug
        previous_secrets = rollback_root / "secrets" / slug
        secret_source = staging_root / "secrets" / slug
        secret_target = store.secrets_root / slug
        replaced_project = False
        replaced_secrets = False
        installed_secrets = False
        try:
            if target.exists():
                previous_project.parent.mkdir(parents=True, exist_ok=True)
                target.replace(previous_project)
                replaced_project = True
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)

            if restore_secrets and backup_includes_secrets and secret_source.exists():
                if secret_target.exists():
                    previous_secrets.parent.mkdir(parents=True, exist_ok=True)
                    secret_target.replace(previous_secrets)
                    replaced_secrets = True
                secret_target.parent.mkdir(parents=True, exist_ok=True)
                secret_source.replace(secret_target)
                installed_secrets = True
                _protect_secret_files(secret_target)
                restored_secret_projects.append(slug)

            _rebase_project_json_paths(target, slug)
            store.refresh_project_configs(slug)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            if replaced_project and previous_project.exists():
                previous_project.replace(target)
            if installed_secrets:
                shutil.rmtree(secret_target, ignore_errors=True)
            if replaced_secrets:
                if previous_secrets.exists():
                    previous_secrets.replace(secret_target)
            raise
        else:
            shutil.rmtree(previous_project, ignore_errors=True)
            shutil.rmtree(previous_secrets, ignore_errors=True)
            imported.append(slug)

    return ImportResult(
        imported=tuple(imported),
        skipped=tuple(skipped),
        restored_secret_projects=tuple(restored_secret_projects),
        backup_includes_secrets=backup_includes_secrets,
    )


def _rebase_project_json_paths(project_dir: Path, slug: str) -> None:
    for path in project_dir.rglob("*.json"):
        relative = path.relative_to(project_dir)
        if relative.parts and relative.parts[0] == "cache":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        rebased = _rebase_value(value, slug, project_dir.resolve())
        if rebased == value:
            continue
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(rebased, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _rebase_value(value: Any, slug: str, project_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rebase_value(item, slug, project_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_value(item, slug, project_dir) for item in value]
    if not isinstance(value, str):
        return value
    if not (value.startswith("/") or _WINDOWS_ABSOLUTE.match(value)):
        return value
    parts = [part for part in re.split(r"[\\/]", value) if part]
    indexes = [index for index, part in enumerate(parts) if part == slug]
    if not indexes:
        return value
    return str(project_dir.joinpath(*parts[indexes[-1] + 1 :]))


def _protect_secret_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
