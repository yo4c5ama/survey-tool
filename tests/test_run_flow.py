from vnn_survey.app.run_flow import build_flow_svg, flow_summary_payload, record_flow_stage


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


def test_flow_svg_draws_manual_enrichment_loop() -> None:
    round_state = {"index": 0, "kind": "initial", "status": "ready", "flow": []}
    record_flow_stage(
        round_state,
        key="human_audit",
        label="Human audit",
        input_count=10,
        retained_count=8,
        stage_type="review",
    )
    record_flow_stage(
        round_state,
        key="manual_return_to_review",
        label="Return to review",
        input_count=2,
        retained_count=2,
        stage_type="review",
        loop_to="human_audit",
    )
    state = {"run_id": "run-loop", "status": "ready", "rounds": [round_state]}

    assert round_state["flow"][1]["loop_to"] == "human_audit"
    svg = build_flow_svg(state)
    assert "manual enrichment loop" in svg
    assert 'stroke-dasharray="5 3"' in svg
