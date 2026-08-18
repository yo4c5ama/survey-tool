import re
from pathlib import Path

import pytest
import requests

from vnn_survey.ai_research import CorpusAnalyzer, OpenAIResearchClient, PaperWorkspace


def test_paper_workspace_saves_pdf_and_conversation_memory(tmp_path: Path) -> None:
    workspace = PaperWorkspace(tmp_path)
    paper = {"title": "A Paper", "doi": "10.1000/paper"}

    pdf_path = workspace.save_pdf(paper, b"%PDF-1.4\nsmall test document")
    workspace.save_file_id(paper, "file_123")
    messages = workspace.append_exchange(
        paper,
        question="What is the contribution?",
        answer="A grounded answer.",
        model="gpt-test",
        response_id="resp_123",
    )

    assert pdf_path.exists()
    assert workspace.pdf_path(paper) == pdf_path
    assert workspace.file_id(paper) == "file_123"
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert workspace.load_conversation(paper)[1]["model"] == "gpt-test"

    saved_analysis = workspace.save_analysis(
        paper,
        content="## English\nA clear briefing.",
        model="gpt-test",
        response_id="resp_analysis",
        source_kind="uploaded_pdf",
        source_url="https://example.org/paper.pdf",
        interface_language="en",
    )
    assert saved_analysis["source_kind"] == "uploaded_pdf"
    assert workspace.load_analysis(paper)["content"].startswith("## English")

    workspace.clear_conversation(paper)
    assert workspace.load_conversation(paper) == []


def test_paper_workspace_rejects_non_pdf(tmp_path: Path) -> None:
    workspace = PaperWorkspace(tmp_path)

    with pytest.raises(ValueError, match="valid PDF"):
        workspace.save_pdf({"title": "Bad file"}, b"plain text")


def test_paper_analysis_sends_remote_source_all_metadata_and_bilingual_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAIResearchClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-test",
    )
    payloads: list[dict] = []

    def fake_post(payload: dict) -> dict:
        payloads.append(payload)
        return {"id": "resp_analysis", "output_text": "## English\nBriefing"}

    monkeypatch.setattr(client, "_post_response", fake_post)
    result = client.analyze_paper(
        paper={
            "title": "A Formal Method",
            "authors": "A. Author",
            "abstract": "We verify a transformer with a sound abstraction.",
            "url": "https://example.org/papers/formal-method",
            "manual_decision": "include",
            "reviewer_note": "Central verification paper.",
        },
        output_language="zh",
    )

    assert result.source_kind == "linked_page"
    assert result.source_url == "https://example.org/papers/formal-method"
    content = payloads[0]["input"][0]["content"]
    assert content[0] == {
        "type": "input_file",
        "file_url": "https://example.org/papers/formal-method",
    }
    metadata = content[1]["text"]
    assert "sound abstraction" in metadata
    assert "Manual Decision:\ninclude" in metadata
    assert "Reviewer Note:\nCentral verification paper." in metadata
    assert "English first" in payloads[0]["instructions"]
    assert "Simplified Chinese" in payloads[0]["instructions"]


def test_paper_analysis_falls_back_to_metadata_when_link_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAIResearchClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-test",
    )
    payloads: list[dict] = []

    def fake_post(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            response = requests.Response()
            response.status_code = 400
            raise requests.HTTPError("unreadable file URL", response=response)
        return {"id": "resp_fallback", "output_text": "## English\nMetadata briefing"}

    monkeypatch.setattr(client, "_post_response", fake_post)
    result = client.analyze_paper(
        paper={
            "title": "A Paper",
            "abstract": "Available evidence.",
            "url": "https://example.org/paper-page",
        },
        output_language="en",
    )

    assert result.source_kind == "metadata"
    assert any(item.get("type") == "input_file" for item in payloads[0]["input"][0]["content"])
    assert not any(
        item.get("type") == "input_file" for item in payloads[1]["input"][0]["content"]
    )


def test_corpus_analyzer_uses_one_taxonomy_for_every_batch(tmp_path: Path) -> None:
    class FakeClient:
        model = "gpt-test"

        def json_response(
            self,
            *,
            instructions: str,
            input_text: str,
            schema: dict,
            max_output_tokens: int,
        ) -> dict:
            if schema["name"] == "survey_taxonomy":
                return {
                    "title": "Method taxonomy",
                    "overview": "Two stable categories.",
                    "categories": [
                        {
                            "id": "formal",
                            "label": "Formal",
                            "description": "Formal methods.",
                            "inclusion_signals": "Proof or solver.",
                        },
                        {
                            "id": "empirical",
                            "label": "Empirical",
                            "description": "Empirical studies.",
                            "inclusion_signals": "Experiments only.",
                        },
                    ],
                }
            paper_ids = re.findall(r"paper_id: ([a-f0-9]+)", input_text)
            return {
                "classifications": [
                    {
                        "paper_id": paper_id,
                        "primary_category": "formal",
                        "secondary_categories": [],
                        "rationale": "Uses a proof method.",
                    }
                    for paper_id in paper_ids
                ]
            }

    rows = [
        {
            "title": f"Paper {index}",
            "doi": f"10.1000/{index}",
            "year": "2025",
            "abstract": "A proof-based method.",
        }
        for index in range(23)
    ]
    progress: list[tuple[int, int]] = []
    stages: list[str] = []
    result = CorpusAnalyzer(FakeClient()).analyze(
        rows=rows,
        research_question="Which methods?",
        scope_description="Verification papers.",
        criteria="Classify by method.",
        output_dir=tmp_path / "analysis",
        progress_callback=lambda completed, total, _title: progress.append(
            (completed, total)
        ),
        stage_callback=lambda stage, _message: stages.append(stage),
    )

    assert len(result.classifications) == 23
    assert {item["primary_category"] for item in result.classifications} == {"formal"}
    assert stages == ["Taxonomy design", "Paper classification", "Analysis report"]
    assert progress == [(0, 23), (20, 23), (23, 23)]
    assert result.taxonomy_path.exists()
    assert result.classifications_path.exists()
    assert "Method taxonomy" in result.report_path.read_text(encoding="utf-8")
