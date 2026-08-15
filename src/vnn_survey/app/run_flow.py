from __future__ import annotations

from html import escape
from typing import Any


def record_flow_stage(
    round_state: dict[str, Any],
    *,
    key: str,
    label: str,
    input_count: int,
    retained_count: int,
    excluded_count: int | None = None,
    stage_type: str = "filter",
    details: dict[str, int | str] | None = None,
    loop_to: str | None = None,
) -> None:
    """Insert or replace a persisted literature-flow stage for one round."""

    input_value = max(int(input_count), 0)
    retained_value = max(int(retained_count), 0)
    excluded_value = (
        max(input_value - retained_value, 0)
        if excluded_count is None
        else max(int(excluded_count), 0)
    )
    stage = {
        "key": key,
        "label": label,
        "type": stage_type,
        "input": input_value,
        "retained": retained_value,
        "excluded": excluded_value,
        "details": dict(details or {}),
    }
    if loop_to:
        stage["loop_to"] = loop_to
    stages = round_state.setdefault("flow", [])
    for index, existing in enumerate(stages):
        if existing.get("key") == key:
            stages[index] = stage
            return
    stages.append(stage)


def flow_summary_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": state.get("run_id", ""),
        "status": state.get("status", ""),
        "updated_at": state.get("updated_at", ""),
        "rounds": [
            {
                "index": round_state.get("index", index),
                "kind": round_state.get("kind", ""),
                "status": round_state.get("status", ""),
                "stages": list(round_state.get("flow", [])),
            }
            for index, round_state in enumerate(state.get("rounds", []))
        ],
    }


def build_flow_svg(state: dict[str, Any]) -> str:
    rounds = [
        (round_state, round_flow_stages(round_state))
        for round_state in state.get("rounds", [])
    ]
    rounds = [(round_state, stages) for round_state, stages in rounds if stages]
    max_stages = max((len(stages) for _, stages in rounds), default=1)
    width = max(760, 36 + max_stages * 190)
    height = max(180, 48 + len(rounds) * 190)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" "
        "refX=\"7\" refY=\"4\" orient=\"auto\"><path d=\"M0,0 L8,4 L0,8 Z\" "
        "fill=\"#6b7280\"/></marker></defs>",
        '<text x="18" y="27" font-family="Arial, sans-serif" font-size="16" '
        'font-weight="700" fill="#18202a">Literature flow</text>',
    ]
    colors = {
        "discovery": "#4d6f91",
        "filter": "#a65353",
        "enrichment": "#397d73",
        "review": "#66723f",
    }
    for round_position, (round_state, stages) in enumerate(rounds):
        y = 52 + round_position * 190
        round_label = (
            f"Round {round_state.get('index', round_position)} · "
            f"{round_state.get('kind', '')}"
        )
        parts.append(
            f'<text x="18" y="{y + 14}" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#596273">{escape(round_label)}</text>'
        )
        for stage_index, stage in enumerate(stages):
            x = 18 + stage_index * 190
            node_y = y + 24
            color = colors.get(str(stage.get("type") or "filter"), colors["filter"])
            if stage_index:
                parts.append(
                    f'<line x1="{x - 27}" y1="{node_y + 56}" x2="{x - 9}" '
                    f'y2="{node_y + 56}" stroke="#6b7280" stroke-width="1.5" '
                    'marker-end="url(#arrow)"/>'
                )
            parts.extend(
                [
                    f'<rect x="{x}" y="{node_y}" width="162" height="116" rx="6" '
                    f'fill="#ffffff" stroke="#cbd3de" stroke-width="1"/>',
                    f'<rect x="{x}" y="{node_y}" width="162" height="4" rx="2" '
                    f'fill="{color}"/>',
                ]
            )
            for line_index, line in enumerate(_wrap_label(str(stage.get("label") or ""))):
                parts.append(
                    f'<text x="{x + 11}" y="{node_y + 24 + line_index * 14}" '
                    'font-family="Arial, sans-serif" font-size="11" font-weight="700" '
                    f'fill="#18202a">{escape(line)}</text>'
                )
            retained = max(int(stage.get("retained") or 0), 0)
            excluded = max(int(stage.get("excluded") or 0), 0)
            parts.append(
                f'<text x="{x + 11}" y="{node_y + 72}" font-family="Arial, sans-serif" '
                f'font-size="24" font-weight="700" fill="#18202a">{retained:,}</text>'
            )
            change = (
                "identified"
                if stage.get("type") == "discovery"
                else "no papers removed"
                if stage.get("type") == "enrichment"
                else f"{excluded:,} excluded"
            )
            parts.append(
                f'<text x="{x + 11}" y="{node_y + 94}" font-family="Arial, sans-serif" '
                f'font-size="10" fill="#596273">{escape(change)}</text>'
            )
        positions = {str(stage.get("key") or ""): index for index, stage in enumerate(stages)}
        for stage_index, stage in enumerate(stages):
            target_index = positions.get(str(stage.get("loop_to") or ""))
            if target_index is None or target_index >= stage_index:
                continue
            source_x = 18 + stage_index * 190 + 81
            target_x = 18 + target_index * 190 + 81
            node_bottom = y + 24 + 116
            loop_y = node_bottom + 30
            parts.append(
                f'<path d="M {source_x} {node_bottom} C {source_x} {loop_y}, '
                f'{target_x} {loop_y}, {target_x} {node_bottom}" fill="none" '
                'stroke="#397d73" stroke-width="1.8" stroke-dasharray="5 3" '
                'marker-end="url(#arrow)"/>'
            )
            parts.append(
                f'<text x="{min(source_x, target_x) + 8}" y="{loop_y - 5}" '
                'font-family="Arial, sans-serif" font-size="10" fill="#397d73">'
                "manual enrichment loop</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def round_flow_stages(round_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return persisted stages, with a count-based fallback for older saved runs."""

    persisted = round_state.get("flow")
    if persisted:
        return list(persisted)

    counts = round_state.get("counts", {})
    stages: list[dict[str, Any]] = []

    def add(
        key: str,
        label: str,
        input_count: int,
        retained_count: int,
        *,
        stage_type: str = "filter",
        details: dict[str, int | str] | None = None,
    ) -> None:
        holder = {"flow": stages}
        record_flow_stage(
            holder,
            key=key,
            label=label,
            input_count=input_count,
            retained_count=retained_count,
            stage_type=stage_type,
            details=details,
        )

    raw = _optional_int(counts.get("raw_records"))
    filtered = _optional_int(counts.get("filtered_records"))
    deduped = _optional_int(counts.get("deduped_records"))
    pool = _optional_int(counts.get("pool_rows"))
    if raw is not None:
        add("literature_search", "Literature search", raw, raw, stage_type="discovery")
    if raw is not None and filtered is not None:
        add("metadata_filter", "Metadata filters", raw, filtered)
    if filtered is not None and deduped is not None:
        add("deduplication", "Deduplication", filtered, deduped)
    current = deduped if deduped is not None else pool
    if current is None:
        return stages

    rule_excluded = _optional_int(counts.get("rule_excluded"))
    if rule_excluded is not None:
        next_count = max(current - rule_excluded, 0)
        add("rule_screening", "Rule screening", current, next_count)
        current = next_count

    title_kept = _optional_int(counts.get("title_kept"))
    if title_kept is not None:
        add("ai_title_screening", "AI title screening", current, title_kept)
        current = title_kept

    abstracts_attempted = _optional_int(counts.get("abstracts_attempted"))
    if abstracts_attempted is not None:
        add(
            "abstract_enrichment",
            "Abstract enrichment",
            current,
            current,
            stage_type="enrichment",
            details={
                "abstracts found": _optional_int(counts.get("abstracts_found")) or 0,
                "attempted": abstracts_attempted,
            },
        )

    audit_queue = _optional_int(counts.get("audit_queue"))
    if audit_queue is not None:
        add(
            "audit_queue",
            "Human review queue",
            current,
            audit_queue,
            stage_type="review",
        )
    return stages


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return None


def _wrap_label(value: str, limit: int = 21) -> list[str]:
    words = value.split()
    if not words:
        return [""]
    lines = [words[0]]
    for word in words[1:]:
        if len(lines[-1]) + len(word) + 1 <= limit:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines[:2]
