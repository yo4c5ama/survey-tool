import csv
from pathlib import Path

from vnn_survey.app.audit import (
    build_cumulative_audit,
    create_audit_queue,
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
