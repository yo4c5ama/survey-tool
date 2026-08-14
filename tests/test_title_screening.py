import csv
import json
from pathlib import Path

from vnn_survey.title_screening import screen_titles_with_llm


class FakeTitleClient:
    def __init__(self) -> None:
        self.calls = 0

    def json_response(self, **kwargs):
        self.calls += 1
        papers = json.loads(kwargs["input_text"].split("\n", 1)[1])
        results = []
        for paper in papers:
            title = paper["title"]
            if "Power Transformer" in title:
                decision = "exclude"
            elif "Ambiguous" in title:
                decision = "maybe"
            else:
                decision = "include"
            results.append(
                {
                    "paper_id": paper["paper_id"],
                    "decision": decision,
                    "reason": f"Title decision: {decision}",
                }
            )
        return {"results": results}


def test_title_screening_batches_caches_and_filters_before_enrichment(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "screened.csv"
    output_path = tmp_path / "title_screened.csv"
    rows = [
        {
            "title": "Formal Verification of Transformers",
            "year": "2025",
            "auto_screening_decision": "include_candidate",
        },
        {
            "title": "Power Transformer Fault Analysis",
            "year": "2024",
            "auto_screening_decision": "include_candidate",
        },
        {
            "title": "An Ambiguous Attention Method",
            "year": "2023",
            "auto_screening_decision": "include_candidate",
        },
        {
            "title": "Already Excluded",
            "year": "2022",
            "auto_screening_decision": "exclude",
        },
    ]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    client = FakeTitleClient()
    result = screen_titles_with_llm(
        input_path,
        output_path,
        client=client,
        research_question="Which Transformer methods are formally verified?",
        scope_description="Transformer verification",
        inclusion_criteria=["Formal guarantee"],
        exclusion_criteria=["Electrical transformers"],
        model="test-model",
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )

    assert client.calls == 2
    assert result.summary.eligible == 3
    assert result.summary.excluded == 1
    assert result.summary.kept_for_enrichment == 2
    assert result.rows[0]["auto_screening_decision"] == "include_candidate"
    assert result.rows[1]["auto_screening_decision"] == "exclude"
    assert result.rows[1]["exclusion_code"] == "title_llm_exclude"
    assert result.rows[2]["auto_screening_decision"] == "needs_review"
    assert result.rows[3]["title_llm_status"] == "skipped_rule"

    cached_client = FakeTitleClient()
    cached_result = screen_titles_with_llm(
        input_path,
        tmp_path / "cached_output.csv",
        client=cached_client,
        research_question="Which Transformer methods are formally verified?",
        scope_description="Transformer verification",
        inclusion_criteria=["Formal guarantee"],
        exclusion_criteria=["Electrical transformers"],
        model="test-model",
        cache_dir=tmp_path / "cache",
        batch_size=2,
    )

    assert cached_client.calls == 0
    assert cached_result.summary.cached == 3
    assert all(
        row["title_llm_status"] in {"cached", "skipped_rule"}
        for row in cached_result.rows
    )
