from vnn_survey.app.run_flow import (
    build_flow_svg,
    flow_summary_payload,
    record_flow_stage,
    round_flow_stages,
)


def test_flow_stages_are_replaced_by_key_and_exported() -> None:
    round_state = {"index": 0, "kind": "initial", "status": "running", "flow": []}
    record_flow_stage(
        round_state,
        key="deduplication",
        label="Deduplication",
        input_count=100,
        retained_count=80,
    )
    record_flow_stage(
        round_state,
        key="deduplication",
        label="Deduplication",
        input_count=110,
        retained_count=85,
    )
    state = {"run_id": "run-1", "status": "running", "rounds": [round_state]}

    assert len(round_state["flow"]) == 1
    assert round_state["flow"][0]["excluded"] == 25
    assert flow_summary_payload(state)["rounds"][0]["stages"][0]["retained"] == 85
    svg = build_flow_svg(state)
    assert svg.startswith("<svg")
    assert "Deduplication" in svg
    assert "25 excluded" in svg


def test_flow_svg_hides_enrichment_and_places_manual_additions_before_audit() -> None:
    round_state = {"index": 0, "kind": "initial", "status": "ready", "flow": []}
    record_flow_stage(
        round_state,
        key="abstract_enrichment",
        label="Abstract enrichment",
        input_count=10,
        retained_count=10,
        stage_type="enrichment",
    )
    record_flow_stage(
        round_state,
        key="ai_abstract_screening",
        label="AI abstract screening",
        input_count=10,
        retained_count=8,
    )
    record_flow_stage(
        round_state,
        key="human_audit",
        label="Human audit",
        input_count=10,
        retained_count=7,
        stage_type="review",
    )
    record_flow_stage(
        round_state,
        key="manual_loop_additions",
        label="Researcher additions",
        input_count=2,
        retained_count=2,
        stage_type="discovery",
        details={"submitted": 2},
    )
    record_flow_stage(
        round_state,
        key="manual_venue_enrichment",
        label="Manual venue enrichment",
        input_count=2,
        retained_count=2,
        stage_type="enrichment",
    )
    state = {"run_id": "run-loop", "status": "ready", "rounds": [round_state]}

    displayed = round_flow_stages(round_state)
    assert [stage["key"] for stage in displayed] == [
        "ai_abstract_screening",
        "manual_additions",
        "human_audit",
    ]
    assert displayed[1]["retained"] == 10
    assert displayed[1]["details"] == {"manual papers": 2}
    svg = build_flow_svg(state)
    assert "Abstract enrichment" not in svg
    assert "Manual venue enrichment" not in svg
    assert svg.index("Manual additions") < svg.index("Human audit")
