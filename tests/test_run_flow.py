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
