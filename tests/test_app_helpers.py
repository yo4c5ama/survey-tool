import csv
from math import nan
from pathlib import Path

import pandas as pd
import streamlit as st

from vnn_survey.app.main import (
    _audit_rows_changed,
    _download_button,
    _has_known_impact_factor,
    _impact_factor_text,
    _looks_like_pdf_url,
    _paper_external_url,
    _paper_metadata,
    _snowball_direction_counts,
)


def test_download_button_does_not_rerun_the_app(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_download_button(*_args, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(st, "download_button", fake_download_button)

    assert _download_button("Download", data=b"content", on_click="rerun")
    assert captured["on_click"] == "ignore"


def test_snowball_direction_counts_include_bidirectional_overlap(tmp_path: Path) -> None:
    path = tmp_path / "snowball_new.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "snowball_relations"])
        writer.writeheader()
        writer.writerows(
            [
                {"title": "Reference", "snowball_relations": "backward"},
                {"title": "Citation", "snowball_relations": "forward"},
                {"title": "Both", "snowball_relations": "backward; forward"},
                {"title": "Target seed", "snowball_relations": "seed"},
            ]
        )

    assert _snowball_direction_counts({"files": {"snowball_new": str(path)}}) == {
        "new": 4,
        "backward": 2,
        "forward": 2,
        "both": 1,
    }


def test_paper_metadata_formats_numeric_year_and_metrics() -> None:
    metadata = _paper_metadata(
        {
            "authors": "Ada Lovelace; Grace Hopper",
            "year": 2025,
            "venue": "Verification Conference",
            "core_rank": "A",
            "impact_factor": 4.2,
        }
    )

    assert metadata == (
        "Ada Lovelace; Grace Hopper · 2025 · Verification Conference · CORE A · IF 4.2"
    )


def test_paper_metadata_omits_missing_values() -> None:
    assert _paper_metadata({"authors": None, "year": None, "venue": nan}) == ""


def test_unknown_impact_factor_values_are_not_displayed() -> None:
    assert _impact_factor_text("N/A") == ""
    assert _impact_factor_text("not found") == ""
    assert _impact_factor_text(0) == ""
    assert _paper_metadata({"title": "Paper", "impact_factor": "unknown"}) == ""
    assert not _has_known_impact_factor(pd.DataFrame({"impact_factor": ["", "N/A", 0]}))
    assert _has_known_impact_factor(pd.DataFrame({"impact_factor": ["", 4.2]}))


def test_paper_external_url_prefers_safe_url_then_falls_back_to_doi() -> None:
    assert (
        _paper_external_url(
            {"url": "https://example.org/paper.pdf", "doi": "10.1000/fallback"}
        )
        == "https://example.org/paper.pdf"
    )
    assert _paper_external_url({"doi": "doi:10.1000/example"}) == (
        "https://doi.org/10.1000/example"
    )
    assert _paper_external_url({"url": "javascript:alert(1)"}) == ""
    assert _looks_like_pdf_url("https://example.org/paper.pdf?download=1")
    assert not _looks_like_pdf_url("https://example.org/paper")


def test_audit_rows_changed_only_detects_review_field_edits() -> None:
    original = pd.DataFrame(
        [
            {
                "title": "A Paper",
                "manual_decision": "include",
                "manual_notes": "In scope.",
            }
        ]
    )

    assert not _audit_rows_changed(
        original,
        [{"manual_decision": "include", "manual_notes": "In scope."}],
    )
    assert _audit_rows_changed(
        original,
        [{"manual_decision": "exclude", "manual_notes": "Out of scope."}],
    )
