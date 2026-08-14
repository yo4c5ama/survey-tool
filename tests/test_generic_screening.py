from vnn_survey.config import ScreeningConfig
from vnn_survey.screening import annotate_row


def test_generic_screening_keeps_query_matches() -> None:
    row = annotate_row(
        {"title": "Certified Models for Clinical Prediction"},
        ScreeningConfig(profile="generic"),
    )

    assert row["auto_screening_decision"] == "include_candidate"
    assert row["auto_screening_bucket"] == "project_query_match"


def test_generic_screening_applies_user_title_exclusions() -> None:
    row = annotate_row(
        {"title": "A Tutorial on Certified Models"},
        ScreeningConfig(profile="generic", exclude_terms=["tutorial"]),
    )

    assert row["auto_screening_decision"] == "exclude"
    assert row["exclusion_code"] == "user_exclusion_term"
