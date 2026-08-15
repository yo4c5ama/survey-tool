from math import nan

from vnn_survey.app.main import _paper_metadata


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
