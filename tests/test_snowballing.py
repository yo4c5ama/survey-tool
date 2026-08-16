from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import SnowballingConfig, SurveyConfig, load_config
from vnn_survey.snowballing import (
    OpenAlexSnowballClient,
    OpenCitationsSnowballClient,
    RetrievedWorks,
    SemanticScholarSnowballClient,
    _request_json,
    snowball_candidates,
    write_seed_coverage_report,
    write_snowballing_summary,
)


def test_referenced_works_fetches_every_id_in_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        params = dict(kwargs["params"])
        calls.append(params)
        ids = str(params["filter"]).removeprefix("openalex:").split("|")
        return {
            "results": [_work(work_id, title=f"Reference {work_id}") for work_id in reversed(ids)]
        }

    monkeypatch.setattr(client, "_request", fake_request)
    seed = {
        "id": "https://openalex.org/WSEED",
        "referenced_works": [f"https://openalex.org/W{index}" for index in range(205)],
        "referenced_works_count": 205,
    }

    result = client.referenced_works(seed, limit=None)

    assert result.available == 205
    assert not result.truncated
    assert len(result.works) == 205
    assert [_short_id(row) for row in result.works] == [f"W{index}" for index in range(205)]
    assert [call["per_page"] for call in calls] == [100, 100, 5]


def test_citing_works_pages_all_results_newest_first(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        params = dict(kwargs["params"])
        calls.append(params)
        if params["cursor"] == "*":
            start, end, next_cursor = 0, 100, "next-page"
        else:
            start, end, next_cursor = 100, 150, None
        return {
            "meta": {"count": 150, "next_cursor": next_cursor},
            "results": [
                _work(f"W{index}", title=f"Citation {index}") for index in range(start, end)
            ],
        }

    monkeypatch.setattr(client, "_request", fake_request)

    result = client.citing_works(_work("WSEED", cited_by_count=149), limit=None)

    assert result.available == 150
    assert len(result.works) == 150
    assert not result.truncated
    assert [call["cursor"] for call in calls] == ["*", "next-page"]
    assert all(call["per_page"] == 100 for call in calls)
    assert all(call["sort"] == "publication_date:desc" for call in calls)

    calls.clear()
    limited = client.citing_works(_work("WSEED", cited_by_count=150), limit=30)
    assert len(limited.works) == 30
    assert limited.available == 150
    assert limited.truncated
    assert len(calls) == 1
    assert calls[0]["per_page"] == 30


def test_snowball_summary_records_complete_seed_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Complete Snowball",
        research_question="Which papers?",
        scope_description="Test complete citation retrieval.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    config = load_config(store.config_path(project.slug))
    assert config.snowballing.max_backward_per_seed == 0
    assert config.snowballing.max_forward_per_seed == 0
    assert config.snowballing.cache_ttl_hours == 24
    assert config.snowballing.providers == ["semantic_scholar", "opencitations"]
    assert config.snowballing.provider_strategy == "merge"
    config = replace(
        config,
        snowballing=replace(
            config.snowballing,
            providers=["openalex"],
            provider_strategy="failover",
        ),
    )
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")

    input_path = tmp_path / "input.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "year", "doi"])
        writer.writeheader()
    seed_path = tmp_path / "seeds.yaml"
    seed_path.write_text(
        yaml.safe_dump({"seed_papers": [{"title": "Seed Paper"}]}),
        encoding="utf-8",
    )
    limits: list[tuple[int | None, int | None]] = []

    class FakeClient:
        provider_id = "openalex"

        def __init__(self, _config: SnowballingConfig, **_kwargs: object) -> None:
            pass

        def resolve_seed(self, _seed: object) -> dict[str, object]:
            return _work("WSEED", title="Seed Paper", cited_by_count=2)

        def referenced_works(
            self,
            _work_value: dict[str, object],
            limit: int | None,
        ) -> RetrievedWorks:
            limits.append((limit, None))
            return RetrievedWorks(
                works=[_work("WR1", title="Reference One")],
                available=1,
                truncated=False,
            )

        def citing_works(
            self,
            _work_value: dict[str, object],
            limit: int | None,
        ) -> RetrievedWorks:
            previous_backward, _ = limits[-1]
            limits[-1] = (previous_backward, limit)
            return RetrievedWorks(
                works=[
                    _work("WC1", title="Citation One"),
                    _work("WC2", title="Citation Two"),
                ],
                available=2,
                truncated=False,
            )

    monkeypatch.setattr("vnn_survey.snowballing.OpenAlexSnowballClient", FakeClient)
    output_path = tmp_path / "snowballed.csv"
    result = snowball_candidates(
        input_path,
        output_path,
        config,
        seed_papers_path=seed_path,
        max_backward_per_seed=0,
        max_forward_per_seed=0,
    )

    assert limits == [(None, None)]
    assert result.summary.seeds_resolved == 1
    assert result.summary.references_available == 1
    assert result.summary.references_fetched == 1
    assert result.summary.citations_available == 2
    assert result.summary.citations_fetched == 2
    assert result.summary.backward_truncated_seeds == 0
    assert result.summary.forward_truncated_seeds == 0
    assert result.summary.added_rows == 4

    summary_path = tmp_path / "snowballing_summary.json"
    write_snowballing_summary(result.summary, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["seed_diagnostics"] == [
        {
            "seed_title": "Seed Paper",
            "seed_id": "https://openalex.org/WSEED",
            "resolved": True,
            "coverage_status": "complete",
            "missing_providers": [],
            "providers_resolved": ["openalex"],
            "providers_used": ["openalex"],
            "reference_providers": ["openalex"],
            "citation_providers": ["openalex"],
            "provider_errors": {},
            "references_available": 1,
            "references_fetched": 1,
            "citations_available": 2,
            "citations_fetched": 2,
            "backward_truncated": False,
            "forward_truncated": False,
        }
    ]
    assert result.summary.provider_order == ("openalex",)
    assert result.summary.provider_successes == {"openalex": 2}
    coverage_path = tmp_path / "seed_coverage.csv"
    write_seed_coverage_report(result.summary, coverage_path)
    with coverage_path.open("r", encoding="utf-8", newline="") as handle:
        coverage_rows = list(csv.DictReader(handle))
    assert coverage_rows[0]["coverage_status"] == "complete"


def test_openalex_cache_expires_so_new_citations_can_be_seen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    calls = 0

    class FakeResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, int]:
            return {"request": calls}

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(client.session, "get", fake_get)
    request = {
        "cache_key": "citation-page",
        "url": "https://api.openalex.org/works",
        "params": {"filter": "cites:WSEED"},
    }

    assert client._request(**request) == {"request": 1}
    assert client._request(**request) == {"request": 1}
    assert calls == 1

    cache_path = next((tmp_path / "cache").glob("*.json"))
    os.utime(cache_path, (0, 0))

    assert client._request(**request) == {"request": 2}
    assert calls == 2


def test_openalex_queries_are_limited_to_the_survey_years(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_OPENALEX_KEY", "test-key")
    client = OpenAlexSnowballClient(
        SnowballingConfig(
            cache_dir=tmp_path / "cache",
            request_delay_seconds=0,
            openalex_api_key_env="TEST_OPENALEX_KEY",
        ),
        year_start=2020,
        year_end=2026,
    )
    filters: list[str] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        params = dict(kwargs["params"])
        filters.append(str(params["filter"]))
        return {"meta": {"count": 0, "next_cursor": None}, "results": []}

    monkeypatch.setattr(client, "_request", fake_request)
    seed = _work("WSEED", cited_by_count=100)
    seed["referenced_works"] = ["https://openalex.org/WREF"]

    client.referenced_works(seed, limit=None)
    citations = client.citing_works(seed, limit=None)

    assert all("from_publication_date:2020-01-01" in value for value in filters)
    assert all("to_publication_date:2026-12-31" in value for value in filters)
    assert citations.available == 0


def test_openalex_rate_limit_error_preserves_diagnostics_without_exposing_key() -> None:
    class FakeResponse:
        status_code = 429
        headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Limit": "1.00",
            "X-RateLimit-Reset": "3600",
        }
        text = ""

        def json(self) -> dict[str, str]:
            return {
                "error": "Rate limit exceeded",
                "message": "Daily allowance exhausted",
            }

    class FakeSession:
        def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    with pytest.raises(RuntimeError) as captured:
        _request_json(
            session=FakeSession(),
            url="https://api.openalex.org/works",
            params={"api_key": "secret-test-key"},
            timeout=30,
            retries=3,
            delay=0,
        )

    message = str(captured.value)
    assert "HTTP 429" in message
    assert "Daily allowance exhausted" in message
    assert "remaining 0 of 1.00" in message
    assert "reset in 3600 seconds" in message
    assert "secret-test-key" not in message


def test_semantic_scholar_citation_requests_use_year_filter_and_pagination(
    tmp_path: Path,
) -> None:
    client = SemanticScholarSnowballClient(
        SnowballingConfig(cache_dir=tmp_path / "cache", request_delay_seconds=0),
        year_start=2020,
        year_end=2026,
    )
    calls: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        params = dict(kwargs["params"])
        calls.append(params)
        return {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "s2-citation",
                        "title": "Semantic Scholar Citation",
                        "year": 2025,
                    }
                }
            ]
        }

    client._request = fake_request  # type: ignore[method-assign]
    result = client.citing_works(
        {"paperId": "s2-seed", "title": "Seed"},
        limit=None,
    )

    assert [item["title"] for item in result.works] == ["Semantic Scholar Citation"]
    assert calls[0]["publicationDateOrYear"] == "2020:2026"
    assert calls[0]["limit"] == 1000


def test_opencitations_resolves_doi_edges_and_filters_years(tmp_path: Path) -> None:
    client = OpenCitationsSnowballClient(
        SnowballingConfig(cache_dir=tmp_path / "cache", request_delay_seconds=0),
        year_start=2020,
        year_end=2026,
    )

    def fake_request(
        cache_key: str,
        url: str,
        params: dict[str, object],
        *,
        metadata: bool = False,
    ) -> list[dict[str, str]]:
        del url, params
        if metadata:
            return [
                {
                    "id": "doi:10.1/in-range omid:br/1",
                    "title": "Open Citation",
                    "author": "A. Author; B. Author",
                    "pub_date": "2024",
                    "venue": "A Venue [issn:1234-5678]",
                    "type": "journal article",
                }
            ]
        assert cache_key.startswith("citations:")
        return [
            {"citing": "doi:10.1/in-range omid:br/1", "creation": "2024-01-01"},
            {"citing": "doi:10.1/old", "creation": "2010-01-01"},
        ]

    client._request = fake_request  # type: ignore[method-assign]
    result = client.citing_works(
        {"doi": "10.1/seed", "title": "Seed"},
        limit=None,
    )

    assert len(result.works) == 1
    assert result.works[0]["doi"] == "10.1/in-range"
    assert result.works[0]["venue"] == "A Venue"


def test_failover_records_failed_request_and_continues_with_next_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, input_path, seed_path = _snowball_fixture(tmp_path)
    config = replace(
        config,
        snowballing=replace(
            config.snowballing,
            providers=["semantic_scholar", "opencitations"],
            provider_strategy="failover",
        ),
    )

    class FailingClient:
        provider_id = "semantic_scholar"

        def resolve_seed(self, _seed: object) -> dict[str, object]:
            return _work("S2SEED", title="Seed Paper")

        def referenced_works(self, *_args, **_kwargs) -> RetrievedWorks:
            raise RuntimeError("Semantic Scholar snowball request failed (HTTP 429)")

        def citing_works(self, *_args, **_kwargs) -> RetrievedWorks:
            return RetrievedWorks([], 0, False)

    class WorkingClient:
        provider_id = "opencitations"

        def resolve_seed(self, _seed: object) -> dict[str, object]:
            return {"_provider_id": "doi:10.1/seed", "title": "Seed Paper"}

        def referenced_works(self, *_args, **_kwargs) -> RetrievedWorks:
            work = _work("OCREF", title="Fallback Reference")
            work["doi"] = "10.1/reference"
            return RetrievedWorks([work], 1, False)

        def citing_works(self, *_args, **_kwargs) -> RetrievedWorks:
            return RetrievedWorks([], 0, False)

    monkeypatch.setattr(
        "vnn_survey.snowballing._create_snowball_clients",
        lambda *_args, **_kwargs: ([FailingClient(), WorkingClient()], {}),
    )
    result = snowball_candidates(
        input_path,
        tmp_path / "failover.csv",
        config,
        seed_papers_path=seed_path,
    )

    assert result.summary.provider_failures["semantic_scholar"] == 1
    assert result.summary.provider_successes["semantic_scholar"] == 1
    assert result.summary.provider_successes["opencitations"] == 1
    assert result.summary.seed_diagnostics[0]["coverage_status"] == "partial"
    assert any(row["title"] == "Fallback Reference" for row in result.rows)


def test_provider_failure_on_one_seed_does_not_disable_later_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, input_path, seed_path = _snowball_fixture(tmp_path)
    seed_path.write_text(
        yaml.safe_dump(
            {
                "seed_papers": [
                    {"title": "First Seed"},
                    {"title": "Second Seed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        config,
        snowballing=replace(
            config.snowballing,
            providers=["semantic_scholar"],
            provider_strategy="merge",
        ),
    )
    reference_calls: list[str] = []

    class SometimesFailingClient:
        provider_id = "semantic_scholar"

        def resolve_seed(self, seed) -> dict[str, object]:
            return _work(f"ID-{seed.title}", title=seed.title)

        def referenced_works(self, work, *_args, **_kwargs) -> RetrievedWorks:
            reference_calls.append(str(work["title"]))
            if work["title"] == "First Seed":
                raise RuntimeError("temporary seed-specific failure")
            return RetrievedWorks([], 0, False)

        def citing_works(self, *_args, **_kwargs) -> RetrievedWorks:
            return RetrievedWorks([], 0, False)

    monkeypatch.setattr(
        "vnn_survey.snowballing._create_snowball_clients",
        lambda *_args, **_kwargs: ([SometimesFailingClient()], {}),
    )

    result = snowball_candidates(
        input_path,
        tmp_path / "per_seed_failure.csv",
        config,
        seed_papers_path=seed_path,
    )

    assert reference_calls == ["First Seed", "Second Seed"]
    assert [item["coverage_status"] for item in result.summary.seed_diagnostics] == [
        "partial",
        "complete",
    ]


def test_all_provider_failure_is_reported_per_seed_without_aborting_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, input_path, seed_path = _snowball_fixture(tmp_path)
    config = replace(
        config,
        snowballing=replace(
            config.snowballing,
            providers=["semantic_scholar", "opencitations"],
        ),
    )
    with input_path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=["title", "year", "doi"]).writerow(
            {"title": "Existing Paper", "year": "2024", "doi": "10.1/existing"}
        )

    class FailingClient:
        def __init__(self, provider_id: str) -> None:
            self.provider_id = provider_id

        def resolve_seed(self, _seed: object) -> dict[str, object]:
            raise RuntimeError(f"{self.provider_id} unavailable")

        def referenced_works(self, *_args, **_kwargs) -> RetrievedWorks:
            raise AssertionError("Unresolved seeds must not request references")

        def citing_works(self, *_args, **_kwargs) -> RetrievedWorks:
            raise AssertionError("Unresolved seeds must not request citations")

    monkeypatch.setattr(
        "vnn_survey.snowballing._create_snowball_clients",
        lambda *_args, **_kwargs: (
            [FailingClient("semantic_scholar"), FailingClient("opencitations")],
            {},
        ),
    )
    output_path = tmp_path / "failed_checkpoint.csv"

    result = snowball_candidates(
        input_path,
        output_path,
        config,
        seed_papers_path=seed_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert output_path.exists()
    assert [row["title"] for row in rows] == ["Existing Paper"]
    assert result.summary.seed_diagnostics[0]["coverage_status"] == "failed"
    assert result.summary.seed_diagnostics[0]["missing_providers"] == [
        "semantic_scholar",
        "opencitations",
    ]


def test_merge_strategy_unions_provider_provenance_without_duplicate_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, input_path, seed_path = _snowball_fixture(tmp_path)
    config = replace(
        config,
        snowballing=replace(
            config.snowballing,
            providers=["semantic_scholar", "opencitations"],
            provider_strategy="merge",
        ),
    )

    class MergeClient:
        def __init__(self, provider_id: str) -> None:
            self.provider_id = provider_id

        def resolve_seed(self, _seed: object) -> dict[str, object]:
            return {"_provider_id": f"{self.provider_id}:seed", "title": "Seed Paper"}

        def referenced_works(self, *_args, **_kwargs) -> RetrievedWorks:
            work = _work(f"{self.provider_id}:shared", title="Shared Reference")
            work["doi"] = "10.1/shared"
            return RetrievedWorks([work], 1, False)

        def citing_works(self, *_args, **_kwargs) -> RetrievedWorks:
            return RetrievedWorks([], 0, False)

    monkeypatch.setattr(
        "vnn_survey.snowballing._create_snowball_clients",
        lambda *_args, **_kwargs: (
            [MergeClient("semantic_scholar"), MergeClient("opencitations")],
            {},
        ),
    )
    output_path = tmp_path / "merged.csv"
    result = snowball_candidates(
        input_path,
        output_path,
        config,
        seed_papers_path=seed_path,
        include_seed_papers=False,
    )

    shared = [row for row in result.rows if row["doi"] == "10.1/shared"]
    assert len(shared) == 1
    assert shared[0]["snowball_provider"] == "semantic_scholar; opencitations"
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        exported_fields = set(csv.DictReader(handle).fieldnames or [])
    assert {
        "snowball_provider",
        "snowball_coverage_status",
        "snowball_missing_providers",
        "snowball_coverage_notes",
    }.isdisjoint(exported_fields)


def _snowball_fixture(
    tmp_path: Path,
) -> tuple[SurveyConfig, Path, Path]:
    store = ProjectStore(tmp_path / "projects", tmp_path / "secrets")
    project = store.create_project(
        name="Provider Strategy",
        research_question="Which papers?",
        scope_description="Test provider strategy.",
        year_start=2020,
        year_end=2026,
        keyword_groups=[KeywordGroup("topic", ["verification"])],
    )
    input_path = tmp_path / "input.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "year", "doi"])
        writer.writeheader()
    seed_path = tmp_path / "seeds.yaml"
    seed_path.write_text(
        yaml.safe_dump({"seed_papers": [{"title": "Seed Paper", "doi": "10.1/seed"}]}),
        encoding="utf-8",
    )
    return load_config(store.config_path(project.slug)), input_path, seed_path


def _client(tmp_path: Path, monkeypatch) -> OpenAlexSnowballClient:
    monkeypatch.setenv("TEST_OPENALEX_KEY", "test-key")
    return OpenAlexSnowballClient(
        SnowballingConfig(
            cache_dir=tmp_path / "cache",
            request_delay_seconds=0,
            openalex_api_key_env="TEST_OPENALEX_KEY",
        )
    )


def _work(
    work_id: str,
    *,
    title: str = "A Paper",
    cited_by_count: int = 0,
) -> dict[str, object]:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "publication_year": 2025,
        "authorships": [],
        "locations": [],
        "referenced_works": [],
        "cited_by_count": cited_by_count,
    }


def _short_id(row: dict[str, object]) -> str:
    return str(row["id"]).rsplit("/", maxsplit=1)[-1]
