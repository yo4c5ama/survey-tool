from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vnn_survey.config import load_config
from vnn_survey.enrichment import (
    enrich_candidates,
    write_enrichment_summary,
)
from vnn_survey.final_filter import filter_final_candidates
from vnn_survey.llm_screening import (
    llm_screen_candidates,
    write_llm_screening_summary,
)
from vnn_survey.llm_summary import summarize_llm_screening
from vnn_survey.manual_audit import (
    filter_manual_includes,
    merge_manual_audits,
    prepare_audit_round,
)
from vnn_survey.pipeline import collect_from_dblp, save_collection, summarize
from vnn_survey.screening import screen_candidates
from vnn_survey.snowballing import (
    export_seed_papers_from_csv,
    snowball_candidates,
    write_snowballing_summary,
)
from vnn_survey.tracks import classify_research_tracks, write_track_summary
from vnn_survey.venue_quality import (
    enrich_venue_quality,
    write_venue_quality_summary,
)

app = typer.Typer(help="Literature collection tools for the transformer verification survey.")
console = Console()


@app.command()
def queries(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey search configuration.",
    ),
) -> None:
    """Print generated search queries."""
    survey_config = load_config(config)
    for query in survey_config.build_queries():
        console.print(query)


@app.command()
def collect_dblp(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey search configuration.",
    ),
    output_dir: Path = typer.Option(
        Path("data"),
        "--output-dir",
        "-o",
        help="Directory for raw and processed outputs.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the number of queries without calling DBLP.",
    ),
    limit_queries: int | None = typer.Option(
        None,
        "--limit-queries",
        help="Run only the first N generated queries. Useful for smoke tests.",
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help="DBLP source mode: auto, api, or sparql.",
    ),
    timestamped: bool = typer.Option(
        False,
        "--timestamped",
        help="Write outputs to data/runs/<timestamp> instead of overwriting data/raw and data/processed.",
    ),
    run_name: str | None = typer.Option(
        None,
        "--run-name",
        help="Optional run name. Uses data/runs/<run-name>, or appends it to the timestamp.",
    ),
    update_latest: bool = typer.Option(
        True,
        "--update-latest/--no-update-latest",
        help="Update data/runs/latest to point at this timestamped or named run.",
    ),
) -> None:
    """Run the DBLP candidate collection stage."""
    survey_config = load_config(config)
    if source not in {"auto", "api", "sparql"}:
        raise typer.BadParameter("source must be one of: auto, api, sparql")
    query_count = len(survey_config.build_queries())
    if limit_queries is not None:
        query_count = min(query_count, limit_queries)
    if dry_run:
        console.print(f"Generated {query_count} DBLP queries from {config}.")
        return

    resolved_output_dir = _resolve_output_dir(
        output_dir, timestamped=timestamped, run_name=run_name
    )
    console.print(f"Running DBLP collection with {query_count} queries via source={source}.")
    console.print(f"Writing outputs to {resolved_output_dir}.")
    result = collect_from_dblp(
        survey_config,
        console=console,
        limit_queries=limit_queries,
        source=source,
    )
    summary = summarize(result)
    save_collection(result, output_dir=resolved_output_dir)
    _write_run_summary(
        summary=summary,
        output_dir=resolved_output_dir,
        config=config,
        source=source,
        limit_queries=limit_queries,
        timestamped=timestamped,
        run_name=run_name,
        effective_query_count=query_count,
    )
    latest_alias = _update_latest_alias(
        output_dir=output_dir,
        target_dir=resolved_output_dir,
        enabled=update_latest,
    )
    _print_summary(summary, resolved_output_dir)
    if latest_alias:
        console.print(f"Updated latest run alias: {latest_alias} -> {resolved_output_dir.name}")


@app.command("snowball-candidates")
def snowball_candidates_cmd(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey and snowballing configuration.",
    ),
    input_path: Path = typer.Option(
        Path("data/processed/candidate_papers.csv"),
        "--input",
        "-i",
        help="Candidate CSV produced by collect-dblp.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Snowballed CSV output path. Defaults to candidate_papers_snowballed.csv.",
    ),
    seed_papers_path: Path | None = typer.Option(
        None,
        "--seed-papers",
        help="Manual seed paper YAML. Defaults to snowballing.seed_papers_path.",
    ),
    max_backward_per_seed: int | None = typer.Option(
        None,
        "--max-backward-per-seed",
        help="Override backward references per seed. Use 0 for all.",
    ),
    max_forward_per_seed: int | None = typer.Option(
        None,
        "--max-forward-per-seed",
        help="Override forward citations per seed. Use 0 for all.",
    ),
    limit_seeds: int | None = typer.Option(
        None,
        "--limit-seeds",
        help="Use only the first N seeds. Useful for smoke tests.",
    ),
    include_seed_papers: bool = typer.Option(
        True,
        "--include-seed-papers/--exclude-seed-papers",
        help="Include resolved seed papers themselves in the output candidate pool.",
    ),
) -> None:
    """Add manual seeds and backward/forward snowballing results."""
    survey_config = load_config(config)
    resolved_output = output_path or input_path.with_name("candidate_papers_snowballed.csv")
    console.print(
        "Running citation snowballing with "
        f"providers={','.join(survey_config.snowballing.providers)}; "
        f"strategy={survey_config.snowballing.provider_strategy}; "
        f"seed_file={seed_papers_path or survey_config.snowballing.seed_papers_path}; "
        f"input={input_path}"
    )
    result = snowball_candidates(
        input_path=input_path,
        output_path=resolved_output,
        config=survey_config,
        seed_papers_path=seed_papers_path,
        max_backward_per_seed=max_backward_per_seed,
        max_forward_per_seed=max_forward_per_seed,
        limit_seeds=limit_seeds,
        include_seed_papers=include_seed_papers,
    )
    summary_path = resolved_output.with_name("snowballing_summary.json")
    write_snowballing_summary(result.summary, summary_path)
    _print_snowballing_summary(result.summary, resolved_output, summary_path)


@app.command("export-seeds")
def export_seeds_cmd(
    input_path: Path = typer.Option(
        Path("data/runs/latest/processed/final_strict_candidates_with_arxiv.csv"),
        "--input",
        "-i",
        help="Strict candidate CSV to export as snowball seeds.",
    ),
    output_path: Path = typer.Option(
        Path("configs/generated/seed_papers_round0.yaml"),
        "--output",
        "-o",
        help="Seed YAML output path.",
    ),
    source_label: str = typer.Option(
        "strict_include_arxiv",
        "--source-label",
        help="Source label written into each seed item.",
    ),
) -> None:
    """Export kept strict candidates as a seed_papers YAML file."""
    result = export_seed_papers_from_csv(
        input_path=input_path,
        output_path=output_path,
        source_label=source_label,
    )
    console.print(f"Exported {len(result.seeds)} seed papers to {result.output_path}")


@app.command("filter-manual-includes")
def filter_manual_includes_cmd(
    input_path: Path = typer.Option(..., "--input", "-i", help="Manually audited CSV."),
    output_path: Path = typer.Option(..., "--output", "-o", help="Accepted rows CSV."),
) -> None:
    """Keep rows marked include/include_related/keep by manual audit."""
    total, kept = filter_manual_includes(input_path=input_path, output_path=output_path)
    console.print(f"Kept {kept} of {total} manually audited papers in {output_path}")


@app.command("prepare-audit-round")
def prepare_audit_round_cmd(
    input_path: Path = typer.Option(..., "--input", "-i", help="Current strict candidate CSV."),
    output_path: Path = typer.Option(..., "--output", "-o", help="New audit round CSV."),
    previous_audit_paths: list[Path] = typer.Option(
        [],
        "--previous-audit",
        help="Previously reviewed CSV; repeat for multiple rounds.",
    ),
) -> None:
    """Create an audit sheet containing only candidates not reviewed before."""
    total, new_rows = prepare_audit_round(
        input_path=input_path,
        output_path=output_path,
        previous_audit_paths=previous_audit_paths,
    )
    console.print(f"Prepared {new_rows} new papers from {total} candidates in {output_path}")


@app.command("merge-manual-audits")
def merge_manual_audits_cmd(
    input_paths: list[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Manual audit CSV; repeat in round order.",
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Merged audit CSV."),
) -> None:
    """Merge multiple manual audit rounds into one deduplicated audit file."""
    total, unique = merge_manual_audits(input_paths=input_paths, output_path=output_path)
    console.print(f"Merged {total} rows into {unique} unique audited papers in {output_path}")


@app.command("screen-candidates")
def screen_candidates_cmd(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey search and screening configuration.",
    ),
    input_path: Path = typer.Option(
        Path("data/processed/candidate_papers.csv"),
        "--input",
        "-i",
        help="Candidate CSV produced by collect-dblp.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Screened CSV output path. Defaults to candidate_papers_screened.csv next to input.",
    ),
) -> None:
    """Add conservative automatic screening labels to candidate papers."""
    survey_config = load_config(config)
    resolved_output = output_path or input_path.with_name("candidate_papers_screened.csv")
    result = screen_candidates(
        input_path=input_path,
        output_path=resolved_output,
        config=survey_config.screening,
    )

    table = Table(title="Screening Summary")
    table.add_column("Group")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_row("total", "papers", str(result.summary.total))
    for value, count in result.summary.by_decision.most_common():
        table.add_row("decision", value, str(count))
    for value, count in result.summary.by_bucket.most_common():
        table.add_row("bucket", value, str(count))
    for value, count in result.summary.by_exclusion_code.most_common():
        table.add_row("exclusion", value, str(count))
    console.print(table)
    console.print(f"Saved screened candidates to {resolved_output}")


@app.command("enrich-venues")
def enrich_venues_cmd(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey and venue-quality configuration.",
    ),
    input_path: Path = typer.Option(
        Path("data/processed/candidate_papers.csv"),
        "--input",
        "-i",
        help="Candidate CSV produced by collect-dblp.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Venue-enriched CSV output path. Defaults to candidate_papers_venues.csv.",
    ),
    core_online: bool | None = typer.Option(
        None,
        "--core-online/--no-core-online",
        help="Override CORE portal online lookup from the config.",
    ),
) -> None:
    """Infer venue type and attach local CORE/JIF metadata."""
    survey_config = load_config(config)
    venue_config = survey_config.venue_quality
    if core_online is not None:
        venue_config = replace(venue_config, core_online_enabled=core_online)
    resolved_output = output_path or input_path.with_name("candidate_papers_venues.csv")
    result = enrich_venue_quality(
        input_path=input_path,
        output_path=resolved_output,
        config=venue_config,
        survey_config=survey_config,
        publication_resolution_path=resolved_output.with_name(
            f"{resolved_output.stem}_publication_resolution.json"
        ),
    )
    summary_path = resolved_output.with_name("venue_quality_summary.json")
    write_venue_quality_summary(result.summary, summary_path)
    _print_venue_quality_summary(result.summary, resolved_output, summary_path)


@app.command("enrich-abstracts")
def enrich_abstracts_cmd(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey search and enrichment configuration.",
    ),
    input_path: Path = typer.Option(
        Path("data/processed/candidate_papers_screened.csv"),
        "--input",
        "-i",
        help="Candidate CSV. Prefer the screened CSV so exclude rows can be skipped.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Enriched CSV output path. Defaults to candidate_papers_enriched.csv next to input.",
    ),
    screen_decisions: str = typer.Option(
        "include_candidate,needs_review",
        "--screen-decisions",
        help="Comma-separated auto_screening_decision values to enrich, or 'all'.",
    ),
    providers: str | None = typer.Option(
        None,
        "--providers",
        help="Comma-separated providers overriding config, e.g. openalex,semantic_scholar.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Enrich only the first N eligible rows. Useful for smoke tests.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Refresh rows that already have abstracts.",
    ),
) -> None:
    """Enrich candidates with abstracts from external metadata providers."""
    survey_config = load_config(config)
    enrichment_config = survey_config.enrichment
    if providers:
        enrichment_config = replace(
            enrichment_config,
            providers=[provider.strip() for provider in providers.split(",") if provider.strip()],
        )

    resolved_output = output_path or input_path.with_name("candidate_papers_enriched.csv")
    decisions = _parse_decisions(screen_decisions)
    console.print(
        "Running abstract enrichment with "
        f"providers={','.join(enrichment_config.providers)}; "
        f"decisions={screen_decisions}; input={input_path}"
    )
    result = enrich_candidates(
        input_path=input_path,
        output_path=resolved_output,
        config=enrichment_config,
        decisions=decisions,
        limit=limit,
        overwrite=overwrite,
    )

    summary_path = resolved_output.with_name("abstract_enrichment_summary.json")
    write_enrichment_summary(result.summary, summary_path)
    _print_enrichment_summary(result.summary, resolved_output, summary_path)


@app.command("llm-screen")
def llm_screen_cmd(
    config: Path = typer.Option(
        Path("configs/transformer_verification.yaml"),
        "--config",
        "-c",
        help="Survey and LLM screening configuration.",
    ),
    input_path: Path = typer.Option(
        Path("data/processed/candidate_papers_enriched.csv"),
        "--input",
        "-i",
        help="Enriched candidate CSV.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="LLM-screened CSV output path. Defaults to candidate_papers_llm_screened.csv.",
    ),
    screen_decisions: str | None = typer.Option(
        None,
        "--screen-decisions",
        help="Comma-separated auto_screening_decision values to screen, or 'all'.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the OpenAI model in the config.",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        max=50,
        help="Override the number of abstracts sent in each AI screening request.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Screen only the first N eligible rows. Useful for smoke tests.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Refresh rows that already have LLM screening results.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show how many rows would be screened without calling OpenAI.",
    ),
) -> None:
    """Use OpenAI to classify title/abstract relevance for the survey."""
    survey_config = load_config(config)
    llm_config = survey_config.llm_screening
    if model:
        llm_config = replace(llm_config, model=model)
    if batch_size:
        llm_config = replace(llm_config, batch_size=batch_size)

    resolved_output = output_path or input_path.with_name("candidate_papers_llm_screened.csv")
    decisions = (
        set(llm_config.include_decisions)
        if screen_decisions is None
        else _parse_decisions(screen_decisions)
    )
    console.print(
        "Running OpenAI LLM screening with "
        f"model={llm_config.model}; "
        f"batch_size={llm_config.batch_size}; "
        f"decisions={screen_decisions or ','.join(llm_config.include_decisions)}; "
        f"input={input_path}"
    )
    result = llm_screen_candidates(
        input_path=input_path,
        output_path=resolved_output,
        config=llm_config,
        decisions=decisions,
        limit=limit,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    if not dry_run:
        summary_path = resolved_output.with_name("llm_screening_summary.json")
        write_llm_screening_summary(result.summary, summary_path)
        _print_llm_screening_summary(result.summary, resolved_output, summary_path)
    else:
        _print_llm_screening_summary(result.summary, resolved_output, None)


@app.command("summarize-llm-screening")
def summarize_llm_screening_cmd(
    input_path: Path = typer.Option(
        Path("data/processed/candidate_papers_llm_screened.csv"),
        "--input",
        "-i",
        help="LLM-screened CSV produced by llm-screen.",
    ),
    report_path: Path | None = typer.Option(
        None,
        "--report",
        "-r",
        help="Markdown report output path. Defaults to llm_screening_report.md next to input.",
    ),
    recommendations_path: Path | None = typer.Option(
        None,
        "--recommendations",
        help="Priority-sorted CSV output path. Defaults to final_screening_recommendations.csv.",
    ),
    summary_path: Path | None = typer.Option(
        None,
        "--summary",
        help="JSON summary output path. Defaults to final_screening_summary.json.",
    ),
    high_confidence: float = typer.Option(
        0.8,
        "--high-confidence",
        help="Confidence threshold for high-confidence include/exclude buckets.",
    ),
    max_examples: int = typer.Option(
        40,
        "--max-examples",
        help="Maximum example rows per report section.",
    ),
) -> None:
    """Summarize LLM screening outputs into a report and review queue."""
    resolved_report = report_path or input_path.with_name("llm_screening_report.md")
    resolved_recommendations = recommendations_path or input_path.with_name(
        "final_screening_recommendations.csv"
    )
    resolved_summary = summary_path or input_path.with_name("final_screening_summary.json")

    result = summarize_llm_screening(
        input_path=input_path,
        report_path=resolved_report,
        recommendations_path=resolved_recommendations,
        summary_path=resolved_summary,
        high_confidence=high_confidence,
        max_examples=max_examples,
    )

    _print_final_summary(
        result.summary,
        report_path=resolved_report,
        recommendations_path=resolved_recommendations,
        summary_path=resolved_summary,
    )


@app.command("filter-final-candidates")
def filter_final_candidates_cmd(
    input_path: Path = typer.Option(
        Path("data/runs/latest/processed/final_screening_recommendations.csv"),
        "--input",
        "-i",
        help="Final recommendation CSV produced by summarize-llm-screening.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Strict kept-candidates CSV. Defaults to final_strict_candidates.csv.",
    ),
    summary_path: Path | None = typer.Option(
        None,
        "--summary",
        help="Strict filter JSON summary. Defaults to final_strict_summary.json.",
    ),
    removed_path: Path | None = typer.Option(
        None,
        "--removed",
        help="Optional CSV path for removed rows with removal reasons.",
    ),
    exclude_arxiv: bool = typer.Option(
        True,
        "--exclude-arxiv/--include-arxiv",
        help="Exclude arXiv/CoRR rows from the strict candidate set.",
    ),
    strict_llm_formal_only: bool = typer.Option(
        True,
        "--strict-llm-formal-only/--allow-all-llm-target",
        help="For LLM-target papers, keep only formal verification/certification style papers.",
    ),
) -> None:
    """Apply strict final filters for the survey candidate set."""
    resolved_output = output_path or input_path.with_name("final_strict_candidates.csv")
    resolved_summary = summary_path or input_path.with_name("final_strict_summary.json")
    resolved_removed = removed_path or input_path.with_name("final_strict_removed.csv")
    result = filter_final_candidates(
        input_path=input_path,
        output_path=resolved_output,
        summary_path=resolved_summary,
        removed_path=resolved_removed,
        exclude_arxiv=exclude_arxiv,
        strict_llm_formal_only=strict_llm_formal_only,
    )
    _print_strict_filter_summary(
        result.summary,
        output_path=resolved_output,
        summary_path=resolved_summary,
        removed_path=resolved_removed,
    )


@app.command("classify-tracks")
def classify_tracks_cmd(
    input_path: Path = typer.Option(
        Path("data/runs/latest/processed/candidate_papers_llm_screened.csv"),
        "--input",
        "-i",
        help="CSV to classify, usually candidate_papers_llm_screened.csv.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Track-classified CSV. Defaults to candidate_papers_tracked.csv.",
    ),
) -> None:
    """Add coarse research-track labels for analysis."""
    resolved_output = output_path or input_path.with_name("candidate_papers_tracked.csv")
    result = classify_research_tracks(input_path=input_path, output_path=resolved_output)
    summary_path = resolved_output.with_name("research_track_summary.json")
    write_track_summary(result.summary, summary_path)
    _print_track_summary(result.summary, resolved_output, summary_path)


def _print_summary(summary: dict[str, object], output_dir: Path) -> None:
    table = Table(title="Collection Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in [
        "raw_records",
        "filtered_records",
        "deduped_records",
        "failed_queries",
        "fallback_queries",
    ]:
        table.add_row(key, str(summary[key]))
    console.print(table)

    console.print(f"Saved candidates to {output_dir / 'processed' / 'candidate_papers.csv'}")
    console.print(json.dumps(summary, ensure_ascii=False, indent=2))


def _print_snowballing_summary(summary, output_path: Path, summary_path: Path) -> None:
    table = Table(title="Snowballing Summary")
    table.add_column("Group")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_row("total", "input_rows", str(summary.input_rows))
    table.add_row("total", "input_unique_rows", str(summary.input_unique_rows))
    table.add_row("total", "output_rows", str(summary.output_rows))
    table.add_row("total", "added_rows", str(summary.added_rows))
    table.add_row("total", "merged_duplicate_hits", str(summary.merged_rows))
    table.add_row("seed", "loaded", str(summary.seeds_loaded))
    table.add_row("seed", "resolved", str(summary.seeds_resolved))
    table.add_row("backward", "available", str(summary.references_available))
    table.add_row("backward", "fetched", str(summary.references_fetched))
    table.add_row(
        "backward",
        "truncated_seeds",
        str(summary.backward_truncated_seeds),
    )
    table.add_row("forward", "available", str(summary.citations_available))
    table.add_row("forward", "fetched", str(summary.citations_fetched))
    table.add_row(
        "forward",
        "truncated_seeds",
        str(summary.forward_truncated_seeds),
    )
    for value, count in summary.by_relation.most_common():
        table.add_row("relation", value or "empty", str(count))
    for value, count in summary.by_source.most_common():
        table.add_row("source", value or "empty", str(count))
    for value, count in summary.provider_successes.most_common():
        table.add_row("provider success", value, str(count))
    for value, count in summary.provider_failures.most_common():
        table.add_row("provider failure", value, str(count))
    console.print(table)
    console.print(f"Saved snowballed candidates to {output_path}")
    console.print(f"Saved snowballing summary to {summary_path}")


def _print_enrichment_summary(summary, output_path: Path, summary_path: Path) -> None:
    table = Table(title="Abstract Enrichment Summary")
    table.add_column("Group")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_row("total", "rows", str(summary.total))
    table.add_row("total", "eligible_rows", str(summary.eligible))
    table.add_row("total", "rows_with_abstract", str(summary.with_abstract))
    table.add_row("this_run", "attempted", str(summary.attempted))
    table.add_row("this_run", "api_requests", str(summary.api_requests))
    table.add_row("this_run", "batch_requests", str(summary.batch_requests))
    table.add_row("this_run", "cache_hits", str(summary.cache_hits))
    table.add_row("this_run", "api_requests", str(getattr(summary, "api_requests", 0)))
    table.add_row("this_run", "batch_requests", str(getattr(summary, "batch_requests", 0)))
    table.add_row("this_run", "cache_hits", str(getattr(summary, "cache_hits", 0)))
    for value, count in summary.by_status.most_common():
        table.add_row("status", value or "empty", str(count))
    for value, count in summary.by_source.most_common():
        table.add_row("source", value or "empty", str(count))
    console.print(table)
    console.print(f"Saved enriched candidates to {output_path}")
    console.print(f"Saved enrichment summary to {summary_path}")


def _print_venue_quality_summary(summary, output_path: Path, summary_path: Path) -> None:
    table = Table(title="Venue Quality Summary")
    table.add_column("Group")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_row("total", "rows", str(summary.total))
    table.add_row("total", "conferences", str(summary.conferences))
    table.add_row("total", "conferences_with_core_rank", str(summary.conferences_with_core_rank))
    table.add_row("total", "journals", str(summary.journals))
    table.add_row("total", "journals_with_impact_factor", str(summary.journals_with_impact_factor))
    table.add_row("total", "arxiv", str(summary.arxiv))
    table.add_row(
        "publication_resolution",
        "attempted",
        str(summary.publication_resolution_attempted),
    )
    table.add_row(
        "publication_resolution",
        "resolved",
        str(summary.published_versions_resolved),
    )
    for value, count in summary.by_venue_type.most_common():
        table.add_row("venue_type", value or "empty", str(count))
    for value, count in summary.by_core_rank.most_common():
        table.add_row("core_rank", value or "empty", str(count))
    for value, count in summary.by_journal_impact_factor_band.most_common():
        table.add_row("jif_band", value or "empty", str(count))
    console.print(table)
    console.print(f"Saved venue-enriched candidates to {output_path}")
    console.print(f"Saved venue quality summary to {summary_path}")


def _print_llm_screening_summary(summary, output_path: Path, summary_path: Path | None) -> None:
    table = Table(title="LLM Screening Summary")
    table.add_column("Group")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_row("total", "rows", str(summary.total))
    table.add_row("total", "eligible_rows", str(summary.eligible))
    table.add_row("this_run", "attempted", str(summary.attempted))
    for value, count in summary.by_status.most_common():
        table.add_row("status", value or "empty", str(count))
    for value, count in summary.by_decision.most_common():
        table.add_row("decision", value, str(count))
    for value, count in summary.by_scope.most_common():
        table.add_row("scope", value, str(count))
    console.print(table)
    if summary_path:
        console.print(f"Saved LLM-screened candidates to {output_path}")
        console.print(f"Saved LLM screening summary to {summary_path}")
    else:
        console.print(f"Dry run only. No output written to {output_path}")


def _print_final_summary(
    summary,
    report_path: Path,
    recommendations_path: Path,
    summary_path: Path,
) -> None:
    table = Table(title="Final Screening Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("total_rows", str(summary.total))
    table.add_row("auto_eligible_rows", str(summary.auto_eligible))
    table.add_row("rows_with_llm_decision", str(summary.screened))
    table.add_row("unscreened_eligible_rows", str(summary.unscreened_eligible))
    for value, count in summary.by_final_recommendation.most_common():
        table.add_row(f"final:{value}", str(count))
    console.print(table)
    console.print(f"Saved Markdown report to {report_path}")
    console.print(f"Saved priority CSV to {recommendations_path}")
    console.print(f"Saved JSON summary to {summary_path}")


def _print_strict_filter_summary(
    summary,
    output_path: Path,
    summary_path: Path,
    removed_path: Path,
) -> None:
    table = Table(title="Strict Final Filter Summary")
    table.add_column("Group")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_row("total", "rows", str(summary.total))
    table.add_row("total", "kept", str(summary.kept))
    table.add_row("total", "removed", str(summary.removed))
    for value, count in summary.by_reason.most_common():
        table.add_row("reason", value or "empty", str(count))
    for value, count in summary.kept_by_scope.most_common():
        table.add_row("kept_scope", value or "empty", str(count))
    for value, count in summary.kept_by_venue_type.most_common():
        table.add_row("kept_venue", value or "empty", str(count))
    for value, count in summary.kept_by_research_track.most_common():
        table.add_row("kept_track", value or "empty", str(count))
    console.print(table)
    console.print(f"Saved strict candidates to {output_path}")
    console.print(f"Saved removed rows to {removed_path}")
    console.print(f"Saved strict summary to {summary_path}")


def _print_track_summary(summary, output_path: Path, summary_path: Path) -> None:
    table = Table(title="Research Track Summary")
    table.add_column("Track")
    table.add_column("Count", justify="right")
    table.add_row("total", str(summary.total))
    for value, count in summary.by_track.most_common():
        table.add_row(value or "empty", str(count))
    console.print(table)
    console.print(f"Saved track-classified rows to {output_path}")
    console.print(f"Saved track summary to {summary_path}")


def _parse_decisions(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _resolve_output_dir(output_dir: Path, timestamped: bool, run_name: str | None) -> Path:
    if not timestamped and not run_name:
        return output_dir

    slug = _slugify(run_name) if run_name else None
    if timestamped:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirname = f"{timestamp}-{slug}" if slug else timestamp
        return output_dir / "runs" / dirname

    return output_dir / "runs" / (slug or "unnamed-run")


def _update_latest_alias(output_dir: Path, target_dir: Path, enabled: bool) -> Path | None:
    if not enabled:
        return None

    runs_dir = output_dir / "runs"
    try:
        target_dir.relative_to(runs_dir)
    except ValueError:
        return None

    alias_path = runs_dir / "latest"
    if target_dir.name == alias_path.name:
        return None

    runs_dir.mkdir(parents=True, exist_ok=True)
    if alias_path.is_symlink():
        alias_path.unlink()
    elif alias_path.exists():
        raise RuntimeError(
            f"Cannot update latest alias because {alias_path} already exists and is not a symlink."
        )

    alias_path.symlink_to(target_dir.name, target_is_directory=True)
    return alias_path


def _write_run_summary(
    summary: dict[str, object],
    output_dir: Path,
    config: Path,
    source: str,
    limit_queries: int | None,
    timestamped: bool,
    run_name: str | None,
    effective_query_count: int,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config),
        "source": source,
        "limit_queries": limit_queries,
        "effective_query_count": effective_query_count,
        "timestamped": timestamped,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _slugify(value: str | None) -> str:
    if not value:
        return ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "unnamed-run"
