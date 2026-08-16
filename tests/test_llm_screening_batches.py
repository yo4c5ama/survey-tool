import csv
import json as json_module
from pathlib import Path
from typing import Any

from vnn_survey.config import LlmScreeningConfig
from vnn_survey.llm_screening import llm_screen_candidates


def _write_candidates(path: Path, titles: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "abstract", "auto_screening_decision"],
        )
        writer.writeheader()
        for title in titles:
            writer.writerow(
                {
                    "title": title,
                    "abstract": f"Abstract for {title}",
                    "auto_screening_decision": "include_candidate",
                }
            )


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, *, fail_title: str = "") -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[list[dict[str, str]]] = []
        self.fail_title = fail_title

    def post(self, url: str, *, json: dict[str, Any], timeout: int) -> _FakeResponse:
        del url, timeout
        papers = json_module.loads(json["input"].splitlines()[-1])
        self.calls.append(papers)
        if self.fail_title and any(self.fail_title in paper["content"] for paper in papers):
            return _FakeResponse({"id": "bad-response", "output_text": "not-json"})
        results = [
            {
                "paper_id": paper["paper_id"],
                "decision": "include",
                "scope": "in_scope",
                "confidence": 0.9,
                "reason": "Relevant to the configured scope.",
                "evidence": paper["content"],
            }
            for paper in papers
        ]
        return _FakeResponse(
            {
                "id": f"response-{len(self.calls)}",
                "output_text": json_module.dumps({"results": results}),
            }
        )


def test_abstract_screening_batches_and_reuses_per_paper_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source, [f"Paper {index}" for index in range(5)])
    session = _FakeSession()
    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    monkeypatch.setattr("vnn_survey.llm_screening.requests.Session", lambda: session)
    config = LlmScreeningConfig(
        api_key_env="TEST_OPENAI_KEY",
        cache_dir=tmp_path / "cache",
        batch_size=2,
        retries=1,
        request_delay_seconds=0,
    )

    first = llm_screen_candidates(source, tmp_path / "first.csv", config)

    assert [len(call) for call in session.calls] == [2, 2, 1]
    assert first.summary.api_requests == 3
    assert first.summary.batch_requests == 2
    assert first.summary.cache_hits == 0
    assert {row["llm_status"] for row in first.rows} == {"screened"}

    session.calls.clear()
    second = llm_screen_candidates(source, tmp_path / "second.csv", config)

    assert session.calls == []
    assert second.summary.api_requests == 0
    assert second.summary.batch_requests == 0
    assert second.summary.cache_hits == 5
    assert {row["llm_status"] for row in second.rows} == {"cached"}


def test_failed_batch_is_split_until_only_bad_paper_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidates.csv"
    _write_candidates(source, ["Good One", "Bad Paper", "Good Two"])
    session = _FakeSession(fail_title="Bad Paper")
    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    monkeypatch.setattr("vnn_survey.llm_screening.requests.Session", lambda: session)
    config = LlmScreeningConfig(
        api_key_env="TEST_OPENAI_KEY",
        cache_dir=tmp_path / "cache",
        batch_size=3,
        retries=1,
        request_delay_seconds=0,
    )

    result = llm_screen_candidates(source, tmp_path / "screened.csv", config)

    by_title = {row["title"]: row for row in result.rows}
    assert by_title["Good One"]["llm_status"] == "screened"
    assert by_title["Good Two"]["llm_status"] == "screened"
    assert by_title["Bad Paper"]["llm_status"] == "failed"
    assert result.summary.by_status == {"screened": 2, "failed": 1}
    assert result.summary.api_requests == 5
    assert result.summary.batch_requests == 2
