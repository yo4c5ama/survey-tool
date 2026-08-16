import csv
import io
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from vnn_survey import llm_screening, pipeline
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import EnrichmentConfig, LlmScreeningConfig, load_config
from vnn_survey.enrichment import enrich_candidates
from vnn_survey.models import PaperRecord


def _write_candidates(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "auto_screening_decision", "abstract"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "title": "Paper One",
                    "auto_screening_decision": "include_candidate",
                    "abstract": "",
                },
                {
                    "title": "Paper Two",
                    "auto_screening_decision": "needs_review",
                    "abstract": "",
                },
            ]
        )


def test_abstract_enrichment_reports_item_progress(tmp_path: Path) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source)
    updates: list[tuple[int, int, str]] = []

    enrich_candidates(
        source,
        tmp_path / "enriched.csv",
        EnrichmentConfig(providers=[]),
        decisions={"include_candidate", "needs_review"},
        progress_callback=lambda completed, total, title: updates.append((completed, total, title)),
    )

    assert updates == [(0, 2, ""), (1, 2, "Paper One"), (2, 2, "Paper Two")]


def test_llm_screening_reports_item_progress(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source)
    updates: list[tuple[int, int, str]] = []

    class FakeClient:
        def __init__(self, config: LlmScreeningConfig) -> None:
            self.config = config

        def screen(self, row: dict[str, str]) -> dict[str, object]:
            return {
                "decision": "include",
                "scope": "in_scope",
                "confidence": 0.9,
                "reason": "Relevant",
                "evidence": row["title"],
                "_response_id": "test",
            }

    monkeypatch.setattr(llm_screening, "OpenAIResponsesClient", FakeClient)
    llm_screening.llm_screen_candidates(
        source,
        tmp_path / "screened.csv",
        LlmScreeningConfig(request_delay_seconds=0),
        progress_callback=lambda completed, total, title: updates.append((completed, total, title)),
    )

    assert updates == [
        (0, 2, "Existing results"),
        (2, 2, "Batch 1/1; papers 1-2; API 0; cache 0"),
    ]


def test_dblp_progress_reports_deduplicated_candidate_count(monkeypatch, tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Progress count",
        research_question="Which papers?",
        scope_description="Test scope",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    config = replace(
        load_config(store.config_path(project.slug)),
        extra_queries=["second query"],
    )

    def fake_search_query(**kwargs):
        first = PaperRecord(
            title="Paper One",
            source="dblp",
            query=kwargs["query"],
            year=2025,
            doi="10.1/one",
        )
        if kwargs["query"] != "second query":
            return [first]
        return [
            first,
            PaperRecord(
                title="Paper Two",
                source="dblp",
                query=kwargs["query"],
                year=2024,
                doi="10.1/two",
            ),
        ]

    monkeypatch.setattr(pipeline, "_search_query", fake_search_query)
    updates: list[tuple[int, int, str, int]] = []
    pipeline.collect_from_dblp(
        config,
        console=Console(file=io.StringIO()),
        progress_callback=lambda completed, total, query, count: updates.append(
            (completed, total, query, count)
        ),
    )

    assert [update[3] for update in updates] == [0, 1, 2]
