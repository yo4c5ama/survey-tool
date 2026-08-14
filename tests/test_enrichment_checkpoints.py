from __future__ import annotations

import csv
from pathlib import Path

from vnn_survey import enrichment
from vnn_survey.config import EnrichmentConfig


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_openalex_prefetches_dois_in_configured_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    source = tmp_path / "candidates.csv"
    rows = [
        {
            "title": f"Paper {index}",
            "year": "2025",
            "doi": f"10.1000/{index}",
            "auto_screening_decision": "include_candidate",
            "abstract": "",
        }
        for index in range(60)
    ]
    _write_rows(source, rows)
    requests: list[str] = []

    def fake_request(self, cache_key, url, params):
        requests.append(cache_key)
        assert url == enrichment.OPENALEX_WORKS_URL
        assert cache_key.startswith("doi-batch:")
        doi_urls = params["filter"].removeprefix("doi:").split("|")
        return {
            "results": [
                {
                    "id": f"https://openalex.org/W{index}",
                    "doi": doi_url,
                    "title": f"Paper {int(doi_url.rsplit('/', 1)[-1])}",
                    "abstract_inverted_index": {"Abstract": [0]},
                }
                for index, doi_url in enumerate(doi_urls)
            ]
        }

    monkeypatch.setattr(enrichment.OpenAlexClient, "_request", fake_request)
    result = enrichment.enrich_candidates(
        source,
        tmp_path / "enriched.csv",
        EnrichmentConfig(providers=["openalex"], request_delay_seconds=0),
        decisions={"include_candidate"},
    )

    assert len(requests) == 1
    assert result.summary.attempted == 60
    assert result.summary.with_abstract == 60


def test_partial_enrichment_output_is_reused(tmp_path: Path) -> None:
    source = tmp_path / "candidates.csv"
    output = tmp_path / "enriched.csv"
    rows = [
        {
            "title": "Finished Paper",
            "year": "2025",
            "doi": "10.1000/finished",
            "auto_screening_decision": "include_candidate",
            "abstract": "",
        },
        {
            "title": "Pending Paper",
            "year": "2025",
            "doi": "10.1000/pending",
            "auto_screening_decision": "include_candidate",
            "abstract": "",
        },
    ]
    _write_rows(source, rows)
    partial = [
        {
            **rows[0],
            "abstract_status": "not_found",
            "abstract_checked_at": "earlier",
            "abstract_provider_chain": "openalex",
        },
        {**rows[1], "abstract_status": "", "abstract_checked_at": ""},
    ]
    _write_rows(output, partial)

    result = enrichment.enrich_candidates(
        source,
        output,
        EnrichmentConfig(providers=["openalex"]),
        decisions={"include_candidate"},
    )

    assert result.summary.attempted == 1
    assert result.rows[0]["abstract_status"] == "not_found"
    assert result.rows[1]["abstract_status"] == "failed"


def test_request_diagnostics_record_rate_limit_wait(monkeypatch) -> None:
    sleeps: list[float] = []

    class Response:
        def __init__(self, status_code: int, headers: dict[str, str]) -> None:
            self.status_code = status_code
            self.headers = headers

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"ok": True}

    class Session:
        def __init__(self) -> None:
            self.responses = [
                Response(429, {"Retry-After": "2"}),
                Response(200, {"X-RateLimit-Remaining": "999"}),
            ]

        def request(self, *_args, **_kwargs):
            return self.responses.pop(0)

    monkeypatch.setattr(enrichment, "cancellable_sleep", sleeps.append)
    diagnostics = enrichment.RequestDiagnostics()
    payload = enrichment._request_json(
        Session(),
        "GET",
        "https://example.test",
        {},
        timeout=1,
        retries=2,
        delay=0,
        diagnostics=diagnostics,
    )

    assert payload == {"ok": True}
    assert diagnostics.api_requests == 2
    assert diagnostics.rate_limit_retries == 1
    assert diagnostics.rate_limit_wait_seconds == 2
    assert diagnostics.rate_limit_remaining == "999"
    assert sleeps == [2.0, 0]


def test_provider_cascade_skips_native_abstracts_and_only_forwards_missing_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidates.csv"
    rows = [
        {
            "title": "Native Paper",
            "doi": "10.1000/native",
            "abstract": "Already returned by discovery.",
            "abstract_source": "arxiv",
            "auto_screening_decision": "include_candidate",
        },
        {
            "title": "First Match",
            "doi": "10.1000/first",
            "abstract": "",
            "abstract_source": "",
            "auto_screening_decision": "include_candidate",
        },
        {
            "title": "Second Match",
            "doi": "10.1000/second",
            "abstract": "",
            "abstract_source": "",
            "auto_screening_decision": "include_candidate",
        },
    ]
    _write_rows(source, rows)
    prefetched: list[tuple[str, list[str]]] = []

    class FirstClient(enrichment.AbstractClient):
        source = "first"

        def prefetch(self, values, progress_callback=None):
            prefetched.append((self.source, [row["title"] for row in values]))

        def find(self, row):
            if row["title"] == "First Match":
                return enrichment.AbstractMatch(
                    source=self.source,
                    status="enriched",
                    abstract="First abstract",
                )
            return enrichment.AbstractMatch(source=self.source, status="not_found")

    class SecondClient(enrichment.AbstractClient):
        source = "second"

        def prefetch(self, values, progress_callback=None):
            prefetched.append((self.source, [row["title"] for row in values]))

        def find(self, row):
            return enrichment.AbstractMatch(
                source=self.source,
                status="enriched",
                abstract="Second abstract",
            )

    monkeypatch.setattr(
        enrichment,
        "_build_clients",
        lambda *_args: [FirstClient(), SecondClient()],
    )
    result = enrichment.enrich_candidates(
        source,
        tmp_path / "enriched.csv",
        EnrichmentConfig(providers=["first", "second"]),
        decisions={"include_candidate"},
    )

    assert prefetched == [
        ("first", ["First Match", "Second Match"]),
        ("second", ["Second Match"]),
    ]
    assert result.summary.attempted == 2
    assert result.summary.with_abstract == 3
    assert [row["abstract_source"] for row in result.rows] == ["arxiv", "first", "second"]


def test_semantic_scholar_prefetch_batches_up_to_five_hundred(monkeypatch) -> None:
    client = enrichment.SemanticScholarClient(
        EnrichmentConfig(providers=["semantic_scholar"], batch_size=500),
        enrichment.RequestDiagnostics(),
    )
    batches: list[list[str]] = []

    def fake_batch(paper_ids):
        batches.append(paper_ids)
        return [
            {
                "paperId": paper_id,
                "title": paper_id,
                "abstract": "Abstract",
                "externalIds": {"DOI": paper_id.removeprefix("DOI:")},
            }
            for paper_id in paper_ids
        ]

    monkeypatch.setattr(client, "_request_batch", fake_batch)
    client.prefetch(
        [
            {"title": f"Paper {index}", "doi": f"10.1000/{index}"}
            for index in range(501)
        ]
    )

    assert [len(batch) for batch in batches] == [500, 1]


def test_crossref_jats_abstract_is_normalized() -> None:
    assert enrichment._clean_markup(
        '<jats:p xmlns:jats="http://www.ncbi.nlm.nih.gov/JATS1">'
        "A <jats:bold>formal</jats:bold> result.</jats:p>"
    ) == "A formal result."
