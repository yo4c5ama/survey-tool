import io
import json
from pathlib import Path

from rich.console import Console

from vnn_survey import sources
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
    assert project.title_screening_model
    assert project.prompt_refinement_model
    assert project.prompt_replay_model
    assert config.enrichment.providers == [
        "arxiv",
        "pubmed",
        "crossref",
        "semantic_scholar",
        "openalex",
    ]
    assert config.enrichment.batch_size == 100
    assert project.llm_screen_batch_size == 20
    assert config.llm_screening.batch_size == 20


def test_legacy_project_migrates_abstract_provider_config(tmp_path: Path) -> None:
    store, slug = _project_store(tmp_path)
    project_path = store.project_dir(slug) / "project.json"
    value = json.loads(project_path.read_text(encoding="utf-8"))
    value.pop("abstract_providers")
    value.pop("abstract_batch_size")
    value.pop("llm_screen_batch_size")
    value.pop("scholarly_api_email")
    legacy_model = value["llm_model"]
    value.pop("title_screening_model")
    value.pop("prompt_refinement_model")
    value.pop("prompt_replay_model")
    project_path.write_text(json.dumps(value), encoding="utf-8")

    project = store.load_project(slug)
    config = load_config(store.config_path(slug))

    assert project.abstract_providers[0] == "arxiv"
    assert config.enrichment.providers == project.abstract_providers
    assert project.llm_screen_batch_size == 20
    assert config.llm_screening.batch_size == 20
    assert project.title_screening_model == legacy_model
    assert project.prompt_refinement_model == legacy_model
    assert project.prompt_replay_model == legacy_model


def test_discovery_parsers_retain_native_abstracts() -> None:
    arxiv_payload = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2501.00001v2</id>
        <published>2025-01-01T00:00:00Z</published>
        <title>A Formal Transformer</title>
        <summary>  A sound verification method.  </summary>
        <author><name>A. Researcher</name></author>
      </entry>
    </feed>"""
    arxiv_record = sources._parse_arxiv_feed(arxiv_payload, "formal transformer")[0]

    pubmed_payload = """<PubmedArticleSet><PubmedArticle>
      <MedlineCitation><PMID>12345</PMID><Article>
        <ArticleTitle>A Medical Model</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Context.</AbstractText>
          <AbstractText Label="METHODS">A verified method.</AbstractText>
        </Abstract>
        <Journal><Title>Journal</Title></Journal>
      </Article></MedlineCitation>
      <PubmedData><ArticleIdList>
        <ArticleId IdType="doi">10.1000/test</ArticleId>
      </ArticleIdList></PubmedData>
    </PubmedArticle></PubmedArticleSet>"""
    pubmed_record = sources._parse_pubmed_xml(pubmed_payload, "medical model")[0]

    assert arxiv_record.abstract == "A sound verification method."
    assert arxiv_record.abstract_source == "arxiv"
    assert pubmed_record.abstract == "BACKGROUND: Context. METHODS: A verified method."
    assert pubmed_record.abstract_source == "pubmed"


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
