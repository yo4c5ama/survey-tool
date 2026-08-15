from math import nan

import pandas as pd

from vnn_survey.app.main import _audit_rows_changed, _paper_metadata


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
