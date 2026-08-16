import csv
from pathlib import Path

from vnn_survey.app.audit import (
    build_cumulative_audit,
    create_audit_queue,
    create_manual_recommendations,
    load_audit,
    update_audit_rows,
)


def test_audit_queue_update_and_cumulative_export(tmp_path: Path) -> None:
    recommendations = tmp_path / "recommendations.csv"
    with recommendations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "doi", "final_recommendation"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Keep Me",
                    "year": "2025",
                    "doi": "10.1/keep",
                    "final_recommendation": "manual_include_review",
                },
                {
                    "title": "Skip Me",
                    "year": "2024",
                    "doi": "10.1/skip",
                    "final_recommendation": "likely_exclude",
                },
            ]
        )

    audit_path = tmp_path / "round_0.csv"
    _, queued = create_audit_queue(recommendations, audit_path)
    assert queued == 1

    summary = update_audit_rows(
        audit_path,
        [
            {
                "title": "Keep Me",
                "year": "2025",
                "doi": "10.1/keep",
                "manual_decision": "include",
                "manual_notes": "Directly relevant.",
            }
        ],
    )
    assert summary.reviewed == 1
    assert summary.unreviewed == 0

    cumulative = tmp_path / "cumulative.csv"
    included = tmp_path / "included.csv"
    total, unique, kept = build_cumulative_audit([audit_path], cumulative, included)
    assert (total, unique, kept) == (1, 1, 1)
    _, rows, _ = load_audit(included)
    assert rows[0]["title"] == "Keep Me"


def test_human_only_queue_respects_preliminary_exclusions(tmp_path: Path) -> None:
    screened = tmp_path / "screened.csv"
    with screened.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "auto_screening_decision"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Keep for Human Review",
                    "year": "2025",
                    "auto_screening_decision": "include_candidate",
                },
                {
                    "title": "Excluded by Title Prescreen",
                    "year": "2024",
                    "auto_screening_decision": "exclude",
                },
            ]
        )

    recommendations = tmp_path / "recommendations.csv"
    create_manual_recommendations(screened, recommendations)
    _, queued = create_audit_queue(recommendations, tmp_path / "audit.csv")

    assert queued == 1


def test_audit_queue_excludes_previously_reviewed_title_across_identifier_changes(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "round_0.csv"
    with previous.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "doi", "manual_decision"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Already Reviewed Paper",
                "year": "2023",
                "doi": "https://doi.org/10.1/preprint",
                "manual_decision": "include",
            }
        )

    recommendations = tmp_path / "round_1_recommendations.csv"
    with recommendations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "doi", "final_recommendation"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Already Reviewed Paper",
                    "year": "2024",
                    "doi": "10.1/published",
                    "final_recommendation": "manual_review",
                },
                {
                    "title": "Genuinely New Paper",
                    "year": "2025",
                    "doi": "10.1/new",
                    "final_recommendation": "manual_review",
                },
            ]
        )

    audit = tmp_path / "round_1.csv"
    _, queued = create_audit_queue(
        recommendations,
        audit,
        previous_audit_paths=[previous],
    )
    _, rows, _ = load_audit(audit)

    assert queued == 1
    assert [row["title"] for row in rows] == ["Genuinely New Paper"]
