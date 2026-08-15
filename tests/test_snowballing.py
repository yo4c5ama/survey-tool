from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import yaml

from vnn_survey.app.project_store import KeywordGroup, ProjectStore
from vnn_survey.config import SnowballingConfig, load_config
from vnn_survey.snowballing import (
    OpenAlexSnowballClient,
    RetrievedWorks,
    snowball_candidates,
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
            "results": [
                _work(work_id, title=f"Reference {work_id}")
                for work_id in reversed(ids)
            ]
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
    assert [_short_id(row) for row in result.works] == [
        f"W{index}" for index in range(205)
    ]
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
                _work(f"W{index}", title=f"Citation {index}")
                for index in range(start, end)
            ],
        }

    monkeypatch.setattr(client, "_request", fake_request)

    result = client.citing_works(_work("WSEED", cited_by_count=149), limit=None)

    assert result.available == 150
    assert len(result.works) == 150
    assert not result.truncated
    assert [call["cursor"] for call in calls] == ["*", "next-page"]
    assert all(call["per_page"] == 100 for call in calls)
    assert all(call["sort"] == "-publication_date" for call in calls)

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
        def __init__(self, _config: SnowballingConfig) -> None:
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
            "seed_id": "WSEED",
            "resolved": True,
            "references_available": 1,
            "references_fetched": 1,
            "citations_available": 2,
            "citations_fetched": 2,
            "backward_truncated": False,
            "forward_truncated": False,
        }
    ]


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
