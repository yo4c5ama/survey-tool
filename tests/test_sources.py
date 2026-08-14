import io
from pathlib import Path

from rich.console import Console

from vnn_survey.app.manual_papers import ManualPaperStore, create_manual_record
from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import expand_query_alternatives, load_config
from vnn_survey.models import PaperRecord
from vnn_survey.pipeline import collect_from_sources
from vnn_survey.source_catalog import load_source_catalog
from vnn_survey.sources import SourceSearchResult


def _project_store(tmp_path: Path) -> tuple[ProjectStore, str]:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Multi Source",
        research_question="What is relevant?",
        scope_description="A cross-disciplinary test.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
        research_domain="general",
        discovery_sources=["openalex", "crossref"],
    )
    return store, project.slug


def test_domain_catalog_recommends_only_available_sources() -> None:
    catalog = load_source_catalog()

    assert catalog.recommended_sources("computer_science") == [
        "dblp",
        "openalex",
        "arxiv",
    ]
    assert catalog.sources["europeana"].status == "planned"
    assert "europeana" not in catalog.available_source_ids()
    assert catalog.profiles["arts_design"].localized_label("zh") == "艺术 / 设计 / 建筑"


def test_dblp_pipe_alternatives_expand_for_other_sources() -> None:
    assert expand_query_alternatives("transformer|bert formal verification") == [
        "transformer formal verification",
        "bert formal verification",
    ]


def test_project_persists_domain_and_sources_in_pipeline_config(tmp_path: Path) -> None:
    store, slug = _project_store(tmp_path)

    project = store.load_project(slug)
    config = load_config(store.config_path(slug))

    assert project.research_domain == "general"
    assert project.discovery_sources == ["openalex", "crossref"]
    assert config.discovery.sources == ["openalex", "crossref"]
    assert project.paper_qa_model
    assert project.corpus_analysis_model


def test_multi_source_collection_merges_duplicate_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, slug = _project_store(tmp_path)
    config = load_config(store.config_path(slug))

    class FakeProvider:
        def __init__(self, source_id: str) -> None:
            self.source_id = source_id

        def search(self, query: str, limit: int | None = None) -> SourceSearchResult:
            return SourceSearchResult(
                [
                    PaperRecord(
                        title="A Shared Paper",
                        source=self.source_id,
                        query=query,
                        year=2025,
                        doi="10.1000/shared",
                    )
                ]
            )

    monkeypatch.setattr(
        "vnn_survey.pipeline.create_provider",
        lambda source_id, _config, dblp_mode="auto": FakeProvider(source_id),
    )
    manual = create_manual_record(title="A Shared Paper", doi="10.1000/shared")
    result = collect_from_sources(
        config,
        Console(file=io.StringIO()),
        source_ids=["openalex", "crossref"],
        additional_records=[manual],
    )

    assert len(result.raw_records) == 3
    assert len(result.deduped_records) == 1
    assert set(result.deduped_records[0].discovery_sources) == {
        "openalex",
        "crossref",
        "manual",
    }
    assert result.deduped_records[0].manual_added is True


def test_manual_store_updates_duplicates_and_preserves_notes(tmp_path: Path) -> None:
    store = ManualPaperStore(tmp_path)
    first = create_manual_record(title="Known Work", doi="10.1000/known")
    second = PaperRecord(
        title="Known Work",
        source="crossref",
        query="Known Work",
        year=2024,
        doi="10.1000/known",
        venue="A Journal",
    )

    _, added_first = store.add(first, "Suggested by reviewer")
    saved, added_second = store.add(second, "Metadata confirmed")

    assert added_first is True
    assert added_second is False
    assert len(store.load()) == 1
    assert saved.venue == "A Journal"
    assert saved.manual_added is True
    assert "manual" in saved.discovery_sources
