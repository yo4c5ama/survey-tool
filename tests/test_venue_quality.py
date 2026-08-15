import csv
import json
from dataclasses import replace
from pathlib import Path

from vnn_survey import venue_quality
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import load_config
from vnn_survey.models import PaperRecord


def test_arxiv_record_is_promoted_to_safely_matched_publication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _survey_config(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    report_path = tmp_path / "publication_resolution.json"
    _write_rows(input_path, [_arxiv_row()])

    formal_record = PaperRecord(
        title="Formal Transformer Verification",
        source="dblp",
        query="Formal Transformer Verification",
        year=2025,
        authors=["Alice Smith", "Bob Lee"],
        venue="International Conference on Verification",
        doi="10.1000/formal-transformer",
        url="https://doi.org/10.1000/formal-transformer",
        dblp_key="conf/icv/SmithL25",
        publication_type="Inproceedings",
    )
    monkeypatch.setattr(
        venue_quality,
        "search_title_candidates",
        lambda *args, **kwargs: ([formal_record], {}),
    )

    result = venue_quality.enrich_venue_quality(
        input_path,
        output_path,
        replace(config.venue_quality, core_online_enabled=False),
        survey_config=config,
        publication_resolution_path=report_path,
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert row["venue"] == "International Conference on Verification"
    assert row["doi"] == "10.1000/formal-transformer"
    assert row["year"] == "2025"
    assert row["publication_type"] == "Inproceedings"
    assert row["dblp_key"] == "conf/icv/SmithL25"
    assert row["venue_type"] == "conference"
    assert row["discovery_sources"] == "openalex; dblp"
    assert "publication_resolution_status" not in row
    assert result.summary.publication_resolution_attempted == 1
    assert result.summary.published_versions_resolved == 1
    assert result.summary.arxiv == 0
    assert report["resolved"] == 1
    assert report["records"][0]["match"]["source"] == "dblp"


def test_same_title_with_different_authors_remains_arxiv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _survey_config(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    report_path = tmp_path / "publication_resolution.json"
    _write_rows(input_path, [_arxiv_row()])

    unrelated_record = PaperRecord(
        title="Formal Transformer Verification",
        source="crossref",
        query="Formal Transformer Verification",
        year=2025,
        authors=["Carol Jones"],
        venue="Journal of Unrelated Results",
        doi="10.1000/unrelated",
        publication_type="journal-article",
    )
    monkeypatch.setattr(
        venue_quality,
        "search_title_candidates",
        lambda *args, **kwargs: ([unrelated_record], {}),
    )

    result = venue_quality.enrich_venue_quality(
        input_path,
        output_path,
        replace(config.venue_quality, core_online_enabled=False),
        survey_config=config,
        publication_resolution_path=report_path,
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert row["venue"] == "arXiv"
    assert row["doi"] == "10.48550/arXiv.2401.00001"
    assert row["venue_type"] == "arxiv"
    assert result.summary.publication_resolution_attempted == 1
    assert result.summary.published_versions_resolved == 0
    assert report["unresolved"] == 1
    assert report["records"][0]["candidate_evidence"][0]["author_overlap"] is False


def test_formal_dblp_key_takes_precedence_over_stale_arxiv_metadata(
    tmp_path: Path,
) -> None:
    config = _survey_config(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    row = _arxiv_row()
    row["dblp_key"] = "conf/icv/SmithL25"
    _write_rows(input_path, [row])

    result = venue_quality.enrich_venue_quality(
        input_path,
        output_path,
        replace(config.venue_quality, core_online_enabled=False),
    )

    assert result.rows[0]["venue_type"] == "conference"
    assert result.summary.conferences == 1
    assert result.summary.arxiv == 0


def _survey_config(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Venue Resolution",
        research_question="Which papers were formally published?",
        scope_description="Formal verification of Transformer models.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["transformer verification"])],
    )
    return load_config(store.config_path(project.slug))


def _arxiv_row() -> dict[str, str]:
    return {
        "title": "Formal Transformer Verification",
        "source": "openalex",
        "query": "transformer verification",
        "year": "2024",
        "authors": "Alice Smith; Bob Lee",
        "venue": "arXiv",
        "doi": "10.48550/arXiv.2401.00001",
        "url": "https://arxiv.org/abs/2401.00001",
        "dblp_key": "",
        "publication_type": "preprint",
        "discovery_sources": "openalex",
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
