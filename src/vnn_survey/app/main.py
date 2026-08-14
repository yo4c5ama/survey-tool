from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from vnn_survey.ai_research import (
    OpenAIResearchClient,
    PaperWorkspace,
    estimate_corpus_requests,
)
from vnn_survey.app.audit import load_audit
from vnn_survey.app.i18n import LANGUAGE_NAMES, language_name, translate
from vnn_survey.app.manual_papers import ManualPaperStore, create_manual_record
from vnn_survey.app.pipeline_service import (
    PipelineService,
    list_openai_models,
    test_openai_connection,
)
from vnn_survey.app.project_store import KeywordGroup, ProjectSettings, ProjectStore
from vnn_survey.app.task_manager import TaskManager
from vnn_survey.config import expand_query_alternatives, load_config
from vnn_survey.models import PaperRecord
from vnn_survey.source_catalog import SourceCatalog, load_source_catalog
from vnn_survey.sources import search_title_candidates

APP_NAME = "SurveyFlow"
PAGE_LABELS = {
    "scope": "Scope",
    "ai_settings": "AI settings",
    "run_center": "Run center",
    "manual_additions": "Add papers",
    "manual_review": "Manual review",
    "snowball": "Snowball",
    "results": "Results",
    "ai_research": "AI research",
}

MODEL_SUGGESTIONS = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
]
CUSTOM_MODEL_OPTION = "__custom_model__"


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=None, layout="wide")
    _apply_styles()
    store = _store()
    service = PipelineService(store)

    projects = store.list_projects()
    _render_sidebar_header()
    _render_language_selector()
    if not projects:
        st.sidebar.info(_t("Create your first survey project to begin."))
        _render_create_project(store)
        return

    if st.sidebar.button(_t("New project"), width="stretch", icon=":material/add:"):
        st.session_state["create_project"] = True
    if st.session_state.get("create_project"):
        _render_create_project(store, can_cancel=True)
        return

    slugs = [project.slug for project in projects]
    selected_slug = st.session_state.get("project_slug")
    if selected_slug not in slugs:
        selected_slug = slugs[0]
    project_lookup = {project.slug: project for project in projects}
    selected_slug = st.sidebar.selectbox(
        _t("Survey project"),
        slugs,
        index=slugs.index(selected_slug),
        format_func=lambda slug: project_lookup[slug].name,
        key="project_selector",
    )
    st.session_state["project_slug"] = selected_slug
    project = store.load_project(selected_slug)

    page = st.sidebar.radio(
        _t("Workspace"),
        list(PAGE_LABELS),
        format_func=lambda page_id: _t(PAGE_LABELS[page_id]),
        label_visibility="collapsed",
        key="workspace_page",
    )
    st.sidebar.divider()
    _render_sidebar_status(project, service.current_state_or_none(project.slug))

    if page == "scope":
        _render_scope_page(store, project)
    elif page == "ai_settings":
        _render_ai_settings(store, project)
    elif page == "run_center":
        _render_run_center(store, service, project)
    elif page == "manual_additions":
        _render_manual_additions(store, service, project)
    elif page == "manual_review":
        _render_manual_review(service, project)
    elif page == "snowball":
        _render_snowball(service, project)
    elif page == "results":
        _render_results(service, project)
    elif page == "ai_research":
        _render_ai_research(store, service, project)


@st.cache_resource
def _store() -> ProjectStore:
    return ProjectStore()


@st.cache_resource
def _task_manager() -> TaskManager:
    return TaskManager()


@st.cache_resource
def _source_catalog() -> SourceCatalog:
    return load_source_catalog()


def _render_sidebar_header() -> None:
    st.sidebar.markdown(f"## {APP_NAME}")
    st.sidebar.caption(_t("Systematic literature review workspace"))


def _render_language_selector() -> None:
    st.sidebar.selectbox(
        _t("Interface language"),
        list(LANGUAGE_NAMES),
        format_func=language_name,
        key="ui_language",
    )


def _render_sidebar_status(project: ProjectSettings, state: dict[str, Any] | None) -> None:
    st.sidebar.caption(_t("Updated {value}", value=project.updated_at or _t("not yet")))
    if not state:
        st.sidebar.markdown(f"**{_t('Status')}:** {_t('Not started')}")
        return
    status = _state_label(state.get("status", "unknown"))
    st.sidebar.markdown(f"**{_t('Status')}:** {status}")
    st.sidebar.caption(_t("Run {run_id}", run_id=state.get("run_id", "")))


def _render_create_project(store: ProjectStore, can_cancel: bool = False) -> None:
    st.title(APP_NAME)
    st.subheader(_t("Create a survey project"))
    st.write(
        _t(
            "Define the research scope once. The app will generate the search configuration, "
            "screening prompt, run folders, and audit workspace."
        )
    )
    research_domain, discovery_sources = _render_domain_source_selector(
        prefix="create",
        current_domain="computer_science",
        current_sources=None,
    )
    current_year = datetime.now().year
    default_groups = pd.DataFrame(
        [
            {"Group": _t("research object"), "Terms": "transformer\nlarge language model"},
            {"Group": _t("research activity"), "Terms": "formal verification\ncertification"},
        ]
    )
    with st.form("create_project_form"):
        left, right = st.columns(2)
        with left:
            name = st.text_input(
                _t("Project name"),
                placeholder=_t("Reliable language models"),
                key="create_project_name",
            )
            question = st.text_area(
                _t("Research question"),
                placeholder=_t("Which formal methods provide guarantees for language models?"),
                height=96,
                key="create_research_question",
            )
        with right:
            start_year = st.number_input(
                _t("Start year"),
                min_value=1900,
                max_value=current_year + 1,
                value=2017,
                key="create_start_year",
            )
            end_year = st.number_input(
                _t("End year"),
                min_value=1900,
                max_value=current_year + 1,
                value=current_year,
                key="create_end_year",
            )
        scope = st.text_area(
            _t("Scope description"),
            placeholder=_t(
                "Describe the models, methods, properties, and application boundaries."
            ),
            height=110,
            key="create_scope_description",
        )
        st.markdown(f"**{_t('Keyword groups')}**")
        st.caption(_t("Terms inside a group use OR. Different groups use AND."))
        groups_frame = st.data_editor(
            default_groups,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Group": st.column_config.TextColumn(_t("Group name"), required=True),
                "Terms": st.column_config.TextColumn(
                    _t("Terms, one per line or comma-separated"),
                    required=True,
                    width="large",
                ),
            },
            key="create_keyword_groups",
        )
        inclusion = st.text_area(
            _t("Inclusion criteria, one per line"),
            placeholder=_t(
                "The paper studies the target model directly.\n"
                "The paper states a rigorous guarantee."
            ),
            height=90,
            key="create_inclusion_criteria",
        )
        exclusion = st.text_area(
            _t("Exclusion criteria, one per line"),
            placeholder=_t(
                "The target model is only used as a tool.\nThe work is purely empirical."
            ),
            height=90,
            key="create_exclusion_criteria",
        )
        title_exclusions = st.text_input(
            _t("Optional title exclusion terms"),
            placeholder=_t("survey, tutorial"),
            key="create_title_exclusions",
        )
        include_arxiv = st.checkbox(
            _t("Keep arXiv and CoRR records"), value=True, key="create_include_arxiv"
        )
        submitted = st.form_submit_button(_t("Create project"), type="primary")

    if can_cancel and st.button(_t("Cancel")):
        st.session_state["create_project"] = False
        st.rerun()
    if not submitted:
        return
    try:
        project = store.create_project(
            name=name,
            research_question=question,
            scope_description=scope,
            year_start=int(start_year),
            year_end=int(end_year),
            keyword_groups=_groups_from_frame(groups_frame),
            research_domain=research_domain,
            discovery_sources=discovery_sources,
            inclusion_criteria=_split_lines(inclusion),
            exclusion_criteria=_split_lines(exclusion),
            title_exclude_terms=_split_terms(title_exclusions),
            include_arxiv=include_arxiv,
        )
    except ValueError as exc:
        st.error(_runtime_text(str(exc)))
        return
    st.session_state["project_slug"] = project.slug
    st.session_state["project_selector"] = project.slug
    st.session_state["create_project"] = False
    st.rerun()


def _render_scope_page(store: ProjectStore, project: ProjectSettings) -> None:
    st.title(project.name)
    st.caption(_t("Research scope and search logic"))
    research_domain, discovery_sources = _render_domain_source_selector(
        prefix=f"scope_{project.slug}",
        current_domain=project.research_domain,
        current_sources=project.discovery_sources,
    )
    groups_frame = pd.DataFrame(
        [{"Group": group.name, "Terms": "\n".join(group.terms)} for group in project.keyword_groups]
    )
    with st.form("scope_form"):
        left, right = st.columns([3, 1])
        with left:
            name = st.text_input(
                _t("Project name"), value=project.name, key=f"scope_name_{project.slug}"
            )
            question = st.text_area(
                _t("Research question"),
                value=project.research_question,
                height=90,
                key=f"scope_question_{project.slug}",
            )
            scope = st.text_area(
                _t("Scope description"),
                value=project.scope_description,
                height=110,
                key=f"scope_description_{project.slug}",
            )
        with right:
            start_year = st.number_input(
                _t("Start year"),
                min_value=1900,
                max_value=2100,
                value=project.year_start,
                key=f"scope_start_year_{project.slug}",
            )
            end_year = st.number_input(
                _t("End year"),
                min_value=1900,
                max_value=2100,
                value=project.year_end,
                key=f"scope_end_year_{project.slug}",
            )
            include_arxiv = st.checkbox(
                _t("Keep arXiv / CoRR"),
                value=project.include_arxiv,
                key=f"scope_arxiv_{project.slug}",
            )
            include_informal = st.checkbox(
                _t("Keep informal records"),
                value=project.include_informal,
                key=f"scope_informal_{project.slug}",
            )
        st.markdown(f"**{_t('Keyword groups')}**")
        st.caption(_t("OR within each row; AND across rows."))
        edited_groups = st.data_editor(
            groups_frame,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Group": st.column_config.TextColumn(_t("Group name"), required=True),
                "Terms": st.column_config.TextColumn(
                    _t("Terms"), required=True, width="large"
                ),
            },
            key=f"scope_groups_{project.slug}",
        )
        inclusion = st.text_area(
            _t("Inclusion criteria, one per line"),
            value="\n".join(project.inclusion_criteria),
            height=100,
            key=f"scope_inclusion_{project.slug}",
        )
        exclusion = st.text_area(
            _t("Exclusion criteria, one per line"),
            value="\n".join(project.exclusion_criteria),
            height=100,
            key=f"scope_exclusion_{project.slug}",
        )
        title_exclusions = st.text_input(
            _t("Title exclusion terms"),
            value=", ".join(project.title_exclude_terms),
            key=f"scope_title_exclusions_{project.slug}",
        )
        saved = st.form_submit_button(_t("Save scope"), type="primary")
    if saved:
        try:
            project.name = name.strip()
            project.research_question = question.strip()
            project.scope_description = scope.strip()
            project.year_start = int(start_year)
            project.year_end = int(end_year)
            project.keyword_groups = _groups_from_frame(edited_groups)
            project.research_domain = research_domain
            project.discovery_sources = discovery_sources
            project.inclusion_criteria = _split_lines(inclusion)
            project.exclusion_criteria = _split_lines(exclusion)
            project.title_exclude_terms = _split_terms(title_exclusions)
            project.include_arxiv = include_arxiv
            project.include_informal = include_informal
            store.save_project(project)
            st.success(
                _t(
                    "Scope saved. Reset the AI prompt on the AI settings page if the criteria "
                    "changed."
                )
            )
        except ValueError as exc:
            st.error(_runtime_text(str(exc)))

    st.divider()
    st.subheader(_t("Query preview"))
    expression = _boolean_expression(project.keyword_groups)
    st.code(expression, language="text")
    try:
        queries = load_config(store.config_path(project.slug)).build_queries()
        request_count = sum(
            len(queries)
            if source_id == "dblp"
            else sum(len(expand_query_alternatives(query)) for query in queries)
            for source_id in project.discovery_sources
        )
        st.caption(
            _t(
                "The grouped expression compiles to {queries} queries across {sources} "
                "sources ({requests} requests).",
                queries=len(queries),
                sources=len(project.discovery_sources),
                requests=request_count,
            )
        )
        with st.expander(_t("Show generated queries")):
            st.code("\n".join(queries), language="text")
    except ValueError as exc:
        st.error(_t("The current query configuration is invalid: {error}", error=exc))


def _render_ai_settings(store: ProjectStore, project: ProjectSettings) -> None:
    st.title(_t("AI settings"))
    st.write(
        _t(
            "Configure separate models for screening, PDF Q&A, and corpus classification. "
            "Human reviewers retain final authority."
        )
    )
    base_url = st.text_input(
        _t("Base URL"), value=project.llm_base_url, key=f"ai_base_url_{project.slug}"
    )
    fetched_models = st.session_state.get(f"available_models_{project.slug}", [])
    model_columns = st.columns(3)
    with model_columns[0]:
        model = _render_model_selector(
            _t("Screening model"),
            project.llm_model,
            key=f"ai_screening_model_{project.slug}",
            fetched_models=fetched_models,
            help_text=_t("Used for title and abstract screening."),
        )
    with model_columns[1]:
        paper_model = _render_model_selector(
            _t("Paper Q&A model"),
            project.paper_qa_model,
            key=f"ai_paper_model_{project.slug}",
            fetched_models=fetched_models,
            help_text=_t("Used for questions about an uploaded paper PDF."),
        )
    with model_columns[2]:
        corpus_model = _render_model_selector(
            _t("Corpus analysis model"),
            project.corpus_analysis_model,
            key=f"ai_corpus_model_{project.slug}",
            fetched_models=fetched_models,
            help_text=_t("Used to design a taxonomy and classify the final corpus."),
        )
    if st.button(_t("Save model settings"), type="primary"):
        if not all([model.strip(), paper_model.strip(), corpus_model.strip()]):
            st.error(_t("Every AI task requires a model."))
        else:
            project.llm_model = model.strip()
            project.paper_qa_model = paper_model.strip()
            project.corpus_analysis_model = corpus_model.strip()
            project.llm_base_url = base_url.strip()
            store.save_project(project)
            st.success(_t("Model settings saved."))

    st.divider()
    st.subheader(_t("API key"))
    saved_key = store.read_api_key(project.slug)
    api_key = st.text_input(
        _t("OpenAI API key"),
        type="password",
        placeholder=_t("A key is already saved") if saved_key else _t("Enter your API key"),
        key=f"ai_api_key_{project.slug}",
    )
    remember = st.checkbox(
        _t("Save this key on this computer"), value=True, key=f"ai_remember_{project.slug}"
    )
    key_for_action = api_key.strip() or saved_key
    first, second, third = st.columns(3)
    with first:
        if st.button(_t("Test connection"), width="stretch"):
            ok, message = test_openai_connection(base_url, key_for_action)
            (st.success if ok else st.error)(_runtime_text(message))
    with second:
        if st.button(_t("Apply API key"), width="stretch"):
            if not api_key.strip():
                st.error(_t("Enter a new API key before applying it."))
            else:
                os.environ["OPENAI_API_KEY"] = api_key.strip()
                if remember:
                    store.save_api_key(project.slug, api_key)
                st.success(
                    _t("The API key is ready. It is never written to project YAML or logs.")
                )
    with third:
        if st.button(_t("Refresh model list"), width="stretch"):
            models, message = list_openai_models(base_url, key_for_action)
            if models:
                st.session_state[f"available_models_{project.slug}"] = models
                st.success(_t("Loaded {count} models.", count=len(models)))
                st.rerun()
            else:
                st.error(_runtime_text(message))
    st.caption(
        _t(
            "Saved keys are stored in a project-specific file under .secrets "
            "with owner-only permissions."
        )
    )

    st.divider()
    st.subheader(_t("Screening prompt"))
    prompt = store.system_prompt_path(project.slug).read_text(encoding="utf-8")
    prompt_key = f"ai_prompt_{project.slug}"
    if st.session_state.pop(f"{prompt_key}_reset_success", False):
        st.success(_t("Prompt regenerated from the current research scope."))
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = prompt
    edited_prompt = st.text_area(
        _t("System prompt"),
        height=420,
        key=prompt_key,
    )
    save_col, reset_col = st.columns([1, 1])
    with save_col:
        if st.button(_t("Save prompt"), type="primary", width="stretch"):
            try:
                store.save_system_prompt(project.slug, edited_prompt)
                st.success(
                    _t(
                        "Prompt saved. Its content hash will invalidate incompatible cached "
                        "results."
                    )
                )
            except ValueError as exc:
                st.error(_runtime_text(str(exc)))
    with reset_col:
        st.button(
            _t("Regenerate from scope"),
            width="stretch",
            on_click=_reset_ai_prompt,
            args=(store, project.slug, prompt_key),
        )


def _reset_ai_prompt(store: ProjectStore, project_slug: str, widget_key: str) -> None:
    st.session_state[widget_key] = store.reset_system_prompt(project_slug)
    st.session_state[f"{widget_key}_reset_success"] = True


def _render_run_center(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
) -> None:
    st.title(_t("Run center"))
    st.write(_t("Run discovery first. AI screening is a separate, explicitly confirmed stage."))
    state = service.current_state_or_none(project.slug)
    task = _task_manager().snapshot(project.slug)
    if state or task:
        _render_current_run_progress(project.slug)
    if state:
        _render_round_overview(state)

    if not state:
        if task and task.running:
            st.info(_t("The pipeline is starting. You can leave this page and return later."))
            return
        _initial_discovery_form(service, project)
        return

    if task and task.running:
        st.info(_t("This run continues in the background. You can safely visit another page."))
        return

    latest = state["rounds"][-1]
    status = latest.get("status")
    if status == "discovery_complete":
        _review_preparation_panel(store, service, project, latest)
    elif status == "ready_for_review":
        st.success(_t("The current review queue is ready. Continue on the Manual review page."))
    elif status == "converged":
        st.success(_t("No new papers entered the review queue. Snowballing has converged."))
    elif status == "failed":
        st.error(_runtime_text(latest.get("error") or _t("The last stage failed.")))
        if latest.get("files", {}).get("enriched"):
            _review_preparation_panel(store, service, project, latest)

    with st.expander(_t("Start a new initial run")):
        st.warning(
            _t(
                "This creates a separate run and makes it the current run. Existing files are "
                "retained."
            )
        )
        _initial_discovery_form(service, project, compact=True)


def _initial_discovery_form(
    service: PipelineService,
    project: ProjectSettings,
    compact: bool = False,
) -> None:
    st.subheader(_t("Initial discovery"))
    catalog = _source_catalog()
    available_sources = catalog.available_source_ids()
    source_ids = st.multiselect(
        _t("Literature sources"),
        available_sources,
        default=[source for source in project.discovery_sources if source in available_sources],
        format_func=lambda source_id: catalog.sources[source_id].label,
        key=f"sources_{project.slug}_{compact}",
    )
    source = "auto"
    if "dblp" in source_ids:
        source = st.selectbox(
            _t("DBLP connection mode"),
            ["auto", "sparql", "api"],
            help=_t("Auto tries the publication API and falls back to SPARQL."),
            key=f"source_{project.slug}_{compact}",
        )
    options = st.columns(3)
    with options[0]:
        query_limit = st.number_input(
            _t("Query limit"),
            min_value=0,
            value=0,
            help=_t("Use 0 for every generated query."),
            key=f"query_limit_{compact}",
        )
    with options[1]:
        abstract_limit = st.number_input(
            _t("Abstract limit"),
            min_value=0,
            value=0,
            help=_t("Use 0 for every eligible paper."),
            key=f"abstract_limit_{compact}",
        )
    with options[2]:
        core_online = st.checkbox(
            _t("Look up CORE ranks online"),
            value=True,
            key=f"core_online_{compact}",
        )
    if st.button(
        _t("Run initial discovery"),
        type="primary",
        icon=":material/play_arrow:",
        disabled=not source_ids,
        key=f"run_discovery_{compact}",
    ):
        started = _task_manager().start(
            project.slug,
            "initial_discovery",
            service.start_initial_discovery,
            project.slug,
            source=source,
            source_ids=source_ids,
            limit_queries=_none_if_zero(query_limit),
            enrich_limit=_none_if_zero(abstract_limit),
            core_online=core_online,
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))


def _render_manual_additions(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
) -> None:
    st.title(_t("Add papers"))
    st.write(
        _t(
            "Add known papers that automatic searches missed. Confirm a metadata match or "
            "create a local record; every addition is deduplicated and remains visible in "
            "the audit trail."
        )
    )
    manual_store = ManualPaperStore(store.project_dir(project.slug))
    records = manual_store.load()
    state = service.current_state_or_none(project.slug)
    task = _task_manager().snapshot(project.slug)
    if task and task.running:
        _render_current_run_progress(project.slug)

    st.subheader(_t("Saved manual papers"))
    if records:
        st.dataframe(
            pd.DataFrame([record.to_row() for record in records])[
                ["title", "authors", "year", "venue", "doi", "source", "manual_note"]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "title": st.column_config.TextColumn(_t("Title"), width="large"),
                "authors": st.column_config.TextColumn(_t("Authors"), width="large"),
                "year": st.column_config.TextColumn(_t("Year")),
                "venue": st.column_config.TextColumn(_t("Venue")),
                "doi": st.column_config.TextColumn("DOI"),
                "source": st.column_config.TextColumn(_t("Resolved source")),
                "manual_note": st.column_config.TextColumn(_t("Addition note"), width="large"),
            },
        )
        remove_index = st.selectbox(
            _t("Paper to remove"),
            range(len(records)),
            format_func=lambda index: records[index].title,
            key=f"manual_remove_{project.slug}",
        )
        if st.button(
            _t("Remove selected paper"),
            icon=":material/delete:",
            disabled=bool(task and task.running),
        ):
            manual_store.remove(records[remove_index].dedupe_key())
            st.success(_t("The manual paper was removed."))
            st.rerun()
    else:
        st.info(_t("No papers have been added manually."))

    st.divider()
    st.subheader(_t("Find a paper by title"))
    title = st.text_input(
        _t("Known paper title"),
        placeholder=_t("Enter the full or approximate title"),
        key=f"manual_lookup_title_{project.slug}",
    )
    catalog = _source_catalog()
    available_sources = catalog.available_source_ids()
    lookup_sources = st.multiselect(
        _t("Search sources"),
        available_sources,
        default=[source for source in project.discovery_sources if source in available_sources],
        format_func=lambda source_id: catalog.sources[source_id].label,
        key=f"manual_lookup_sources_{project.slug}",
    )
    if st.button(
        _t("Search for title matches"),
        icon=":material/search:",
        disabled=not title.strip() or not lookup_sources,
    ):
        with st.spinner(_t("Searching selected sources...")):
            config = load_config(store.config_path(project.slug))
            matches, errors = search_title_candidates(
                title,
                lookup_sources,
                config,
                limit_per_source=5,
            )
        st.session_state[f"manual_matches_{project.slug}"] = [
            record.to_dict(include_raw=True) for record in matches
        ]
        st.session_state[f"manual_match_errors_{project.slug}"] = errors

    matches = [
        PaperRecord.from_dict(value)
        for value in st.session_state.get(f"manual_matches_{project.slug}", [])
    ]
    errors = st.session_state.get(f"manual_match_errors_{project.slug}", {})
    for source_id, error in errors.items():
        st.warning(
            _t(
                "{source} could not be searched: {error}",
                source=catalog.sources.get(source_id).label
                if source_id in catalog.sources
                else source_id,
                error=error,
            )
        )
    if matches:
        match_index = st.selectbox(
            _t("Candidate match"),
            range(len(matches)),
            format_func=lambda index: _manual_match_label(matches[index]),
            key=f"manual_match_{project.slug}",
        )
        selected = matches[match_index]
        st.caption(_paper_metadata(selected.to_row()))
        note = st.text_input(
            _t("Addition note"),
            placeholder=_t("Why this paper should enter the candidate pool"),
            key=f"manual_match_note_{project.slug}",
        )
        if st.button(
            _t("Add selected match"),
            type="primary",
            icon=":material/add:",
        ):
            _, added = manual_store.add(selected, note)
            st.success(
                _t("Paper added to the manual collection.")
                if added
                else _t("The existing manual record was updated instead of duplicated.")
            )
            st.session_state[f"manual_matches_{project.slug}"] = []
            st.rerun()
    elif f"manual_matches_{project.slug}" in st.session_state:
        st.info(_t("No metadata match was found. Use the manual form below."))

    with st.expander(_t("Add a paper without a metadata match")):
        with st.form(f"manual_record_form_{project.slug}"):
            manual_title = st.text_input(_t("Title"))
            manual_authors = st.text_input(
                _t("Authors"),
                placeholder=_t("Separate authors with semicolons"),
            )
            columns = st.columns(2)
            with columns[0]:
                manual_year = st.number_input(
                    _t("Year"),
                    min_value=0,
                    max_value=2100,
                    value=0,
                    help=_t("Use 0 when the year is unknown."),
                )
                manual_venue = st.text_input(_t("Venue"))
                manual_type = st.text_input(_t("Publication type"))
            with columns[1]:
                manual_doi = st.text_input("DOI")
                manual_url = st.text_input(_t("URL"))
                manual_note = st.text_input(_t("Addition note"))
            submitted = st.form_submit_button(_t("Add manual record"), type="primary")
        if submitted:
            try:
                record = create_manual_record(
                    title=manual_title,
                    authors=_split_semicolon_values(manual_authors),
                    year=None if int(manual_year) == 0 else int(manual_year),
                    venue=manual_venue,
                    doi=manual_doi,
                    url=manual_url,
                    publication_type=manual_type,
                    note=manual_note,
                )
                _, added = manual_store.add(record, manual_note)
                st.success(
                    _t("Paper added to the manual collection.")
                    if added
                    else _t("The existing manual record was updated instead of duplicated.")
                )
                st.rerun()
            except ValueError as exc:
                st.error(_runtime_text(str(exc)))

    st.divider()
    st.subheader(_t("Apply to discovery"))
    if not state:
        st.info(_t("Saved papers will be included automatically in the next initial discovery."))
        return
    initial_round = next(
        (item for item in state.get("rounds", []) if int(item.get("index", -1)) == 0),
        None,
    )
    if not initial_round:
        st.info(_t("Saved papers will be included automatically in the next initial discovery."))
        return
    if initial_round.get("files", {}).get("audit"):
        st.warning(
            _t(
                "The initial review queue already exists. Start a new initial run to include "
                "newly saved papers without rewriting the audit history."
            )
        )
        return
    st.write(
        _t(
            "Synchronize the saved papers into the current candidate pool before creating "
            "the review queue. Venue, screening, and abstract fields will be refreshed."
        )
    )
    has_synced_manual = _round_has_manual_papers(initial_round)
    if st.button(
        _t("Synchronize manual papers"),
        type="primary",
        icon=":material/sync:",
        disabled=(not records and not has_synced_manual) or bool(task and task.running),
    ):
        started = _task_manager().start(
            project.slug,
            "manual_additions",
            service.sync_manual_additions,
            project.slug,
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))


def _review_preparation_panel(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
    round_state: dict[str, Any],
) -> None:
    round_index = int(round_state["index"])
    estimate = service.estimate_llm_usage(project.slug, round_index)
    st.subheader(_t("Prepare round {round_index} for review", round_index=round_index))
    metrics = st.columns(3)
    metrics[0].metric(_t("Papers eligible for AI"), f"{estimate['papers']:,}")
    metrics[1].metric(
        _t("Estimated input tokens"), f"{estimate['estimated_input_tokens']:,}"
    )
    metrics[2].metric(
        _t("Maximum output tokens"), f"{estimate['maximum_output_tokens']:,}"
    )
    use_llm = st.toggle(
        _t("Use AI abstract screening"),
        value=store.has_api_key(project.slug),
        key=f"use_llm_{round_state.get('index')}_{project.slug}",
    )
    llm_limit = st.number_input(
        _t("AI paper limit"),
        min_value=0,
        value=0,
        help=_t("Use 0 for every eligible paper. A small limit is useful for testing."),
        key=f"llm_limit_{round_state.get('index')}_{project.slug}",
    )
    if use_llm and not store.has_api_key(project.slug) and not os.environ.get("OPENAI_API_KEY"):
        st.warning(_t("Add an API key on the AI settings page before continuing."))
    button_label = (
        _t("Run AI screening and create review queue")
        if use_llm
        else _t("Create human-only review queue")
    )
    if st.button(
        button_label,
        type="primary",
        disabled=use_llm
        and not store.has_api_key(project.slug)
        and not os.environ.get("OPENAI_API_KEY"),
    ):
        started = _task_manager().start(
            project.slug,
            "review_preparation",
            service.prepare_round_for_review,
            project.slug,
            round_index,
            use_llm=use_llm,
            llm_limit=_none_if_zero(llm_limit),
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))


def _render_manual_review(service: PipelineService, project: ProjectSettings) -> None:
    st.title(_t("Manual review"))
    state = service.current_state_or_none(project.slug)
    if not state:
        st.info(_t("Run initial discovery before opening the review workspace."))
        return
    review_rounds = [item for item in state["rounds"] if item.get("files", {}).get("audit")]
    if not review_rounds:
        st.info(_t("Prepare the current round for review in the Run center."))
        return
    indexes = [int(item["index"]) for item in review_rounds]
    round_index = st.selectbox(
        _t("Audit round"),
        indexes,
        index=len(indexes) - 1,
        key=f"audit_round_{state['run_id']}",
    )
    round_state = next(item for item in review_rounds if int(item["index"]) == round_index)
    audit_path = Path(round_state["files"]["audit"])
    _, rows, summary = load_audit(audit_path)
    metrics = st.columns(5)
    metrics[0].metric(_t("Candidates"), summary.total)
    metrics[1].metric(_t("Reviewed"), summary.reviewed)
    metrics[2].metric(_t("Include"), summary.by_decision.get("include", 0))
    metrics[3].metric(_t("Related"), summary.by_decision.get("include_related", 0))
    metrics[4].metric(_t("Exclude"), summary.by_decision.get("exclude", 0))

    frame = pd.DataFrame(rows).fillna("")
    filter_col, decision_col = st.columns([2, 1])
    with filter_col:
        search = st.text_input(
            _t("Search title or abstract"), key=f"audit_search_{state['run_id']}_{round_index}"
        )
    with decision_col:
        decision_filter = st.selectbox(
            _t("Decision filter"),
            ["All", "Unreviewed", "include", "include_related", "exclude", "later"],
            format_func=_decision_label,
            key=f"audit_filter_{state['run_id']}_{round_index}",
        )
    filtered = _filter_audit_frame(frame, search, decision_filter)
    visible_columns = [
        column
        for column in [
            "title",
            "year",
            "venue",
            "venue_type",
            "core_rank",
            "impact_factor",
            "llm_decision",
            "llm_confidence",
            "llm_reason",
            "manual_decision",
            "manual_notes",
        ]
        if column in filtered.columns
    ]
    editor_frame = filtered[visible_columns].copy()
    if "manual_decision" in editor_frame:
        editor_frame["manual_decision"] = editor_frame["manual_decision"].map(_decision_label)
    if "llm_decision" in editor_frame:
        editor_frame["llm_decision"] = editor_frame["llm_decision"].map(_decision_label)
    if "venue_type" in editor_frame:
        editor_frame["venue_type"] = editor_frame["venue_type"].map(_value_label)
    edited = st.data_editor(
        editor_frame,
        hide_index=True,
        width="stretch",
        height=480,
        disabled=[
            column
            for column in visible_columns
            if column not in {"manual_decision", "manual_notes"}
        ],
        column_config={
            "title": st.column_config.TextColumn(_t("Title"), width="large"),
            "year": st.column_config.TextColumn(_t("Year")),
            "venue": st.column_config.TextColumn(_t("Venue")),
            "venue_type": st.column_config.TextColumn(_t("Venue type")),
            "core_rank": st.column_config.TextColumn(_t("CORE rank")),
            "impact_factor": st.column_config.TextColumn(_t("Impact Factor")),
            "llm_decision": st.column_config.TextColumn(_t("AI decision")),
            "llm_confidence": st.column_config.TextColumn(_t("AI confidence")),
            "manual_decision": st.column_config.SelectboxColumn(
                _t("Manual decision"),
                options=[
                    "",
                    _decision_label("include"),
                    _decision_label("include_related"),
                    _decision_label("exclude"),
                    _decision_label("later"),
                ],
                required=False,
                width="medium",
            ),
            "manual_notes": st.column_config.TextColumn(_t("Reviewer notes"), width="large"),
            "llm_reason": st.column_config.TextColumn(_t("AI reason"), width="large"),
        },
        key=f"audit_editor_{state['run_id']}_{round_index}",
    )
    if st.button(_t("Save review decisions"), type="primary", icon=":material/save:"):
        updated_rows = _merge_editor_identity(filtered, edited)
        saved_summary = service.update_audit(project.slug, round_index, updated_rows)
        st.success(
            _t(
                "Saved {reviewed} reviewed papers; {unreviewed} still require a decision.",
                reviewed=saved_summary.reviewed,
                unreviewed=saved_summary.unreviewed,
            )
        )
        st.rerun()

    st.download_button(
        _t("Download this audit CSV"),
        data=audit_path.read_bytes(),
        file_name=audit_path.name,
        mime="text/csv",
        icon=":material/download:",
    )

    st.divider()
    st.subheader(_t("Paper reader"))
    if frame.empty:
        st.info(_t("This round contains no new papers."))
        return
    labels = [f"{row.get('year', '')} · {row.get('title', '')}" for row in rows]
    selected = st.selectbox(
        _t("Paper"),
        range(len(rows)),
        format_func=lambda index: labels[index],
        key=f"paper_reader_{state['run_id']}_{round_index}",
    )
    paper = rows[selected]
    st.markdown(f"### {paper.get('title', '')}")
    st.caption(_paper_metadata(paper))
    st.markdown(f"**{_t('Abstract')}**")
    st.write(paper.get("abstract") or _t("No abstract is available."))
    if paper.get("llm_reason"):
        st.markdown(f"**{_t('AI assessment')}**")
        st.write(paper.get("llm_reason"))
        if paper.get("llm_evidence"):
            st.caption(_t("Evidence: {evidence}", evidence=paper.get("llm_evidence")))


def _render_snowball(service: PipelineService, project: ProjectSettings) -> None:
    st.title(_t("Snowball"))
    st.write(
        _t(
            "Included and related papers become seeds. Each round collects references and "
            "citations, deduplicates them against every audited paper, and creates a new review "
            "queue."
        )
    )
    state = service.current_state_or_none(project.slug)
    if not state:
        st.info(_t("Complete the initial discovery and review first."))
        return
    _render_round_overview(state)
    task = _task_manager().snapshot(project.slug)
    if task and task.running:
        _render_current_run_progress(project.slug)
        st.info(_t("This run continues in the background. You can safely visit another page."))
        return
    latest = state["rounds"][-1]
    if latest.get("status") == "converged":
        st.success(
            _t("Snowballing has converged because no new papers entered the review queue.")
        )
        return
    if not latest.get("files", {}).get("audit"):
        st.info(_t("Prepare the current round in the Run center before continuing."))
        return
    _, _, summary = load_audit(Path(latest["files"]["audit"]))
    if summary.unreviewed:
        st.warning(
            _t(
                "Finish {count} remaining manual decisions before starting the next round.",
                count=summary.unreviewed,
            )
        )
    columns = st.columns(3)
    with columns[0]:
        backward = st.number_input(
            _t("References per seed"),
            min_value=1,
            max_value=200,
            value=30,
            key=f"snowball_backward_{state['run_id']}",
        )
    with columns[1]:
        forward = st.number_input(
            _t("Citations per seed"),
            min_value=1,
            max_value=200,
            value=30,
            key=f"snowball_forward_{state['run_id']}",
        )
    with columns[2]:
        abstract_limit = st.number_input(
            _t("Abstract limit"),
            min_value=0,
            value=0,
            help=_t("Use 0 for all new candidates."),
            key=f"snowball_abstract_limit_{state['run_id']}",
        )
    core_online = st.checkbox(
        _t("Look up CORE ranks online"),
        value=True,
        key=f"snowball_core_online_{state['run_id']}",
    )
    if st.button(
        _t("Start next snowball round"),
        type="primary",
        disabled=summary.unreviewed > 0,
        icon=":material/account_tree:",
    ):
        started = _task_manager().start(
            project.slug,
            "snowball_discovery",
            service.start_snowball_discovery,
            project.slug,
            max_backward_per_seed=int(backward),
            max_forward_per_seed=int(forward),
            enrich_limit=_none_if_zero(abstract_limit),
            core_online=core_online,
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))


def _render_results(service: PipelineService, project: ProjectSettings) -> None:
    st.title(_t("Results"))
    state = service.current_state_or_none(project.slug)
    if not state:
        st.info(_t("No run is available yet."))
        return
    _render_round_overview(state)
    audit_rounds = [item for item in state["rounds"] if item.get("files", {}).get("audit")]
    if not audit_rounds:
        st.info(_t("Complete at least one manual review round before exporting results."))
        return
    if st.button(_t("Generate final exports"), type="primary", icon=":material/file_export:"):
        try:
            service.generate_exports(project.slug)
            st.success(_t("Final audit, included corpus, and Markdown report generated."))
            st.rerun()
        except Exception as exc:
            st.error(_t("Export failed: {error}", error=_runtime_text(str(exc))))
    state = service.load_current_state(project.slug)
    exports = state.get("exports", {})
    if not exports:
        return
    included_path = Path(exports["included"])
    audit_path = Path(exports["audit"])
    report_path = Path(exports["report"])
    included = pd.read_csv(included_path, keep_default_na=False)
    audited = pd.read_csv(audit_path, keep_default_na=False)
    metric_columns = st.columns(4)
    metric_columns[0].metric(_t("Audit rounds"), len(audit_rounds))
    metric_columns[1].metric(_t("Audited papers"), len(audited))
    metric_columns[2].metric(_t("Included corpus"), len(included))
    metric_columns[3].metric(
        _t("Excluded"), int((audited.get("manual_decision", "") == "exclude").sum())
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown(f"**{_t('Included papers by year')}**")
        if not included.empty and "year" in included:
            st.bar_chart(included["year"].astype(str).value_counts().sort_index())
    with chart_right:
        st.markdown(f"**{_t('Included papers by venue type')}**")
        if not included.empty and "venue_type" in included:
            venue_counts = included["venue_type"].replace("", "unknown").map(_value_label)
            st.bar_chart(venue_counts.value_counts())
    display_columns = [
        column
        for column in [
            "title",
            "authors",
            "year",
            "venue",
            "venue_type",
            "core_rank",
            "impact_factor",
            "manual_decision",
        ]
        if column in included.columns
    ]
    display_frame = included[display_columns].copy()
    if "venue_type" in display_frame:
        display_frame["venue_type"] = display_frame["venue_type"].map(_value_label)
    if "manual_decision" in display_frame:
        display_frame["manual_decision"] = display_frame["manual_decision"].map(_decision_label)
    st.dataframe(
        display_frame,
        hide_index=True,
        width="stretch",
        height=420,
        column_config=_result_column_config(),
    )
    download_columns = st.columns(3)
    with download_columns[0]:
        st.download_button(
            _t("Included papers CSV"),
            data=included_path.read_bytes(),
            file_name=included_path.name,
            mime="text/csv",
            width="stretch",
        )
    with download_columns[1]:
        st.download_button(
            _t("Complete audit CSV"),
            data=audit_path.read_bytes(),
            file_name=audit_path.name,
            mime="text/csv",
            width="stretch",
        )
    with download_columns[2]:
        st.download_button(
            _t("Final report"),
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/markdown",
            width="stretch",
        )


def _render_ai_research(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
) -> None:
    st.title(_t("AI research"))
    st.write(
        _t(
            "Study individual papers with their PDFs or classify the final reviewed corpus. "
            "AI output remains an analytical aid and should be checked by the researcher."
        )
    )
    state = service.current_state_or_none(project.slug)
    if not state or not state.get("exports", {}).get("included"):
        st.info(_t("Generate final exports on the Results page before using AI research."))
        return
    included_path = Path(state["exports"]["included"])
    if not included_path.exists():
        st.error(_t("The exported included-paper file is missing."))
        return
    included = pd.read_csv(included_path, keep_default_na=False)
    if included.empty:
        st.info(_t("The final included corpus is empty."))
        return
    paper_tab, corpus_tab = st.tabs([_t("Paper Q&A"), _t("Corpus classification")])
    with paper_tab:
        _render_paper_qa(store, project, included)
    with corpus_tab:
        _render_corpus_analysis(store, service, project, state, included)


def _render_paper_qa(
    store: ProjectStore,
    project: ProjectSettings,
    included: pd.DataFrame,
) -> None:
    rows = included.fillna("").to_dict(orient="records")
    selected_index = st.selectbox(
        _t("Paper"),
        range(len(rows)),
        format_func=lambda index: f"{rows[index].get('year', '')} · {rows[index].get('title', '')}",
        key=f"research_paper_{project.slug}",
    )
    paper = {str(key): str(value) for key, value in rows[selected_index].items()}
    st.markdown(f"### {paper.get('title', '')}")
    st.caption(_paper_metadata(paper))
    workspace = PaperWorkspace(store.project_dir(project.slug))
    pdf_path = workspace.pdf_path(paper)

    upload = st.file_uploader(
        _t("Paper PDF"),
        type=["pdf"],
        key=f"paper_pdf_{project.slug}_{workspace.paper_id(paper)}",
    )
    upload_col, status_col = st.columns([1, 2])
    with upload_col:
        if st.button(
            _t("Save PDF"),
            type="primary",
            icon=":material/upload_file:",
            disabled=upload is None,
            key=f"save_pdf_{project.slug}_{workspace.paper_id(paper)}",
        ):
            try:
                pdf_path = workspace.save_pdf(paper, upload.getvalue())
                st.success(_t("PDF saved for this paper."))
                st.rerun()
            except ValueError as exc:
                st.error(_runtime_text(str(exc)))
    with status_col:
        if pdf_path:
            st.success(_t("A PDF is available for this paper."))
        else:
            st.info(_t("Upload the paper PDF before asking questions."))

    fetched_models = st.session_state.get(f"available_models_{project.slug}", [])
    model = _render_model_selector(
        _t("Paper Q&A model"),
        project.paper_qa_model,
        key=f"paper_qa_model_{project.slug}",
        fetched_models=fetched_models,
    )
    conversation = workspace.load_conversation(paper)
    clear_col, memory_col = st.columns([1, 3])
    with clear_col:
        if st.button(
            _t("Clear conversation"),
            icon=":material/delete_sweep:",
            disabled=not conversation,
            key=f"clear_conversation_{project.slug}_{workspace.paper_id(paper)}",
        ):
            workspace.clear_conversation(paper)
            st.rerun()
    with memory_col:
        st.caption(
            _t(
                "Conversation memory is saved locally for this paper ({count} messages).",
                count=len(conversation),
            )
        )
    for message in conversation:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant" and message.get("model"):
                st.caption(message.get("model", ""))

    question = st.chat_input(
        _t("Ask a question about this paper"),
        key=f"paper_question_{project.slug}_{workspace.paper_id(paper)}",
        disabled=pdf_path is None,
    )
    if question:
        api_key = store.read_api_key(project.slug) or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            st.error(_t("Add an API key on the AI settings page before continuing."))
            return
        try:
            with st.spinner(_t("Reading the PDF and preparing an answer...")):
                client = OpenAIResearchClient(
                    base_url=project.llm_base_url,
                    api_key=api_key,
                    model=model,
                )
                answer, response_id, file_id = client.ask_pdf(
                    pdf_path=pdf_path,
                    paper=paper,
                    conversation=conversation,
                    question=question,
                    file_id=workspace.file_id(paper),
                )
                workspace.save_file_id(paper, file_id)
                workspace.append_exchange(
                    paper,
                    question=question,
                    answer=answer,
                    model=model,
                    response_id=response_id,
                )
            st.rerun()
        except Exception as exc:
            st.error(_t("Paper Q&A failed: {error}", error=str(exc)))


def _render_corpus_analysis(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
    state: dict[str, Any],
    included: pd.DataFrame,
) -> None:
    st.write(
        _t(
            "Provide classification guidance such as method, research objective, guarantee type, "
            "or application domain. Leave it empty to let the model propose a taxonomy."
        )
    )
    criteria = st.text_area(
        _t("Classification guidance (optional)"),
        height=120,
        placeholder=_t(
            "Example: classify by verified property first, then by verification method."
        ),
        key=f"corpus_criteria_{project.slug}",
    )
    fetched_models = st.session_state.get(f"available_models_{project.slug}", [])
    model = _render_model_selector(
        _t("Corpus analysis model"),
        project.corpus_analysis_model,
        key=f"corpus_analysis_model_{project.slug}",
        fetched_models=fetched_models,
    )
    metrics = st.columns(2)
    metrics[0].metric(_t("Included papers"), len(included))
    metrics[1].metric(
        _t("Estimated API requests"),
        estimate_corpus_requests(len(included)),
    )
    task = _task_manager().snapshot(project.slug)
    progress_state = (service.current_state_or_none(project.slug) or state).get(
        "progress", {}
    )
    if (task and task.running) or progress_state.get("operation") == "Corpus analysis":
        _render_current_run_progress(project.slug)
    has_key = store.has_api_key(project.slug) or bool(os.environ.get("OPENAI_API_KEY"))
    if st.button(
        _t("Analyze final corpus"),
        type="primary",
        icon=":material/account_tree:",
        disabled=not has_key or bool(task and task.running),
    ):
        started = _task_manager().start(
            project.slug,
            "corpus_analysis",
            service.analyze_final_corpus,
            project.slug,
            criteria=criteria,
            model=model,
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))
    if not has_key:
        st.warning(_t("Add an API key on the AI settings page before continuing."))

    latest_state = service.current_state_or_none(project.slug) or state
    analysis = latest_state.get("corpus_analysis", {})
    if not analysis:
        return
    taxonomy_path = Path(analysis.get("taxonomy", ""))
    classifications_path = Path(analysis.get("classifications", ""))
    report_path = Path(analysis.get("report", ""))
    if not taxonomy_path.exists() or not classifications_path.exists():
        st.warning(_t("The saved corpus analysis files are unavailable."))
        return
    taxonomy = _read_json(taxonomy_path)
    classifications = pd.read_csv(classifications_path, keep_default_na=False)
    st.divider()
    st.subheader(str(taxonomy.get("title") or _t("Corpus classification")))
    st.write(str(taxonomy.get("overview") or ""))
    category_counts = classifications["primary_category"].value_counts()
    summary_columns = st.columns(3)
    summary_columns[0].metric(_t("Papers classified"), len(classifications))
    summary_columns[1].metric(_t("Categories"), len(taxonomy.get("categories", [])))
    summary_columns[2].metric(_t("Model"), analysis.get("model", ""))
    st.bar_chart(category_counts)
    for category in taxonomy.get("categories", []):
        category_id = str(category.get("id") or "")
        count = int(category_counts.get(category_id, 0))
        st.markdown(f"**{category.get('label', category_id)} ({count})**")
        st.write(category.get("description", ""))
    st.dataframe(
        classifications[
            ["title", "year", "venue", "primary_category", "rationale"]
        ],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "title": st.column_config.TextColumn(_t("Title"), width="large"),
            "year": st.column_config.TextColumn(_t("Year")),
            "venue": st.column_config.TextColumn(_t("Venue")),
            "primary_category": st.column_config.TextColumn(
                _t("Primary category")
            ),
            "rationale": st.column_config.TextColumn(
                _t("Classification rationale"), width="large"
            ),
        },
    )
    download_columns = st.columns(3)
    with download_columns[0]:
        st.download_button(
            _t("Taxonomy JSON"),
            data=taxonomy_path.read_bytes(),
            file_name=taxonomy_path.name,
            mime="application/json",
            width="stretch",
        )
    with download_columns[1]:
        st.download_button(
            _t("Classifications CSV"),
            data=classifications_path.read_bytes(),
            file_name=classifications_path.name,
            mime="text/csv",
            width="stretch",
        )
    with download_columns[2]:
        st.download_button(
            _t("Analysis report"),
            data=report_path.read_bytes() if report_path.exists() else b"",
            file_name=report_path.name or "report.md",
            mime="text/markdown",
            width="stretch",
        )


def _render_round_overview(state: dict[str, Any]) -> None:
    rows = []
    for item in state.get("rounds", []):
        counts = item.get("counts", {})
        rows.append(
            {
                _t("Round"): item.get("index"),
                _t("Type"): _state_label(item.get("kind", "")),
                _t("Status"): _state_label(item.get("status", "")),
                _t("Pool"): counts.get("pool_rows", counts.get("deduped_records", "")),
                _t("Added"): counts.get("added_rows", ""),
                _t("Review queue"): counts.get("audit_queue", ""),
                _t("Reviewed"): counts.get("reviewed", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_domain_source_selector(
    *,
    prefix: str,
    current_domain: str,
    current_sources: list[str] | None,
) -> tuple[str, list[str]]:
    catalog = _source_catalog()
    language = st.session_state.get("ui_language", "en")
    domain_key = f"{prefix}_research_domain"
    sources_key = f"{prefix}_discovery_sources"
    if domain_key not in st.session_state:
        st.session_state[domain_key] = (
            current_domain if current_domain in catalog.profiles else "custom"
        )
    if sources_key not in st.session_state:
        st.session_state[sources_key] = current_sources or catalog.recommended_sources(
            st.session_state[domain_key]
        )

    st.subheader(_t("Research field and sources"))
    domain_id = st.selectbox(
        _t("Research field"),
        list(catalog.profiles),
        format_func=lambda profile_id: catalog.profiles[profile_id].localized_label(language),
        key=domain_key,
        on_change=_reset_sources_for_domain,
        args=(domain_key, sources_key),
    )
    profile = catalog.profiles[domain_id]
    st.caption(profile.localized_description(language))
    available_sources = catalog.available_source_ids()
    recommended = set(catalog.recommended_sources(domain_id))
    selected_sources = st.multiselect(
        _t("Literature sources"),
        available_sources,
        format_func=lambda source_id: _source_option_label(
            catalog,
            source_id,
            source_id in recommended,
        ),
        key=sources_key,
        help=_t("Recommended sources are selected automatically; you can override them."),
    )
    visible_source_ids = list(
        dict.fromkeys(
            [*selected_sources, *profile.recommended_sources, *profile.optional_sources]
        )
    )
    st.markdown(f"**{_t('Coverage')}**")
    for source_id in visible_source_ids:
        source = catalog.sources[source_id]
        role = (
            _t("Recommended")
            if source_id in profile.recommended_sources
            else _t("Optional")
        )
        availability = _t("Available") if source.available else _t("Planned")
        st.markdown(f"**{source.label}** · {role} · {availability}")
        st.write(source.localized_scope(language))
        st.caption(f"{_t('Limitations')}: {source.localized_limitation(language)}")
    if not selected_sources:
        st.warning(_t("Select at least one available literature source."))
    return domain_id, selected_sources


def _reset_sources_for_domain(domain_key: str, sources_key: str) -> None:
    catalog = _source_catalog()
    st.session_state[sources_key] = catalog.recommended_sources(
        st.session_state.get(domain_key, "general")
    )


def _source_option_label(
    catalog: SourceCatalog,
    source_id: str,
    recommended: bool,
) -> str:
    label = catalog.sources[source_id].label
    return f"{label} ({_t('Recommended')})" if recommended else label


def _manual_match_label(record: PaperRecord) -> str:
    authors = record.authors[0] if record.authors else _t("Unknown author")
    year = record.year or _t("Unknown year")
    sources = "/".join(source.upper() for source in record.discovery_sources)
    return f"{sources} · {year} · {authors} · {record.title}"


def _render_model_selector(
    label: str,
    current: str,
    *,
    key: str,
    fetched_models: list[str] | None = None,
    help_text: str | None = None,
) -> str:
    choices = list(
        dict.fromkeys(
            [
                current,
                *(fetched_models or []),
                *MODEL_SUGGESTIONS,
            ]
        )
    )
    choices = [choice for choice in choices if choice]
    choices.append(CUSTOM_MODEL_OPTION)
    selected = st.selectbox(
        label,
        choices,
        index=choices.index(current) if current in choices else 0,
        format_func=lambda value: _t("Custom model...")
        if value == CUSTOM_MODEL_OPTION
        else value,
        key=key,
        help=help_text,
    )
    if selected != CUSTOM_MODEL_OPTION:
        return selected
    return st.text_input(
        _t("Custom model name"),
        key=f"{key}_custom",
        placeholder=_t("Enter the exact API model ID"),
    ).strip()


def _round_has_manual_papers(round_state: dict[str, Any]) -> bool:
    candidate_value = round_state.get("files", {}).get("candidates")
    if not candidate_value or not Path(candidate_value).exists():
        return False
    try:
        frame = pd.read_csv(candidate_value, keep_default_na=False)
    except (OSError, pd.errors.ParserError):
        return False
    if "manual_added" in frame:
        return frame["manual_added"].astype(str).str.lower().eq("true").any()
    if "discovery_sources" in frame:
        return frame["discovery_sources"].astype(str).str.contains("manual").any()
    return False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _t(message: str, **values: object) -> str:
    return translate(message, st.session_state.get("ui_language", "en"), **values)


@st.fragment(run_every=1.0)
def _render_current_run_progress(project_slug: str) -> None:
    service = PipelineService(_store())
    state = service.current_state_or_none(project_slug)
    task = _task_manager().snapshot(project_slug)

    st.subheader(_t("Current run"))
    st.caption(_t("Live progress and the latest saved state for this project."))
    if not state:
        if task and task.running:
            st.progress(0.02, text=_t("Starting the pipeline in the background."))
        elif task and task.error:
            st.error(_t("Pipeline failed: {error}", error=_runtime_text(task.error)))
        return

    progress = state.get("progress", {})
    progress_status = str(progress.get("status") or state.get("status") or "unknown")
    operation = str(progress.get("operation") or state["rounds"][-1].get("kind") or "unknown")
    paper_count = _current_paper_count(state)
    metrics = st.columns(4)
    metrics[0].metric(_t("Run ID"), state.get("run_id", ""))
    metrics[1].metric(_t("Operation"), _runtime_text(operation))
    metrics[2].metric(_t("Status"), _state_label(progress_status))
    metrics[3].metric(_t("Papers collected"), f"{paper_count:,}")

    if not progress:
        st.info(_t("Detailed progress is available for runs started with this version."))
        return

    stages = [str(stage) for stage in progress.get("stages", [])]
    stage = str(progress.get("stage") or (stages[0] if stages else "unknown"))
    completed = progress.get("completed")
    total = progress.get("total")
    current = str(progress.get("current") or "")
    message = str(progress.get("message") or "")

    if completed is not None and total is not None:
        item_fraction = 1.0 if int(total) == 0 else float(completed) / int(total)
        item_fraction = min(max(item_fraction, 0.0), 1.0)
        item_text = _t("{completed} of {total} items", completed=completed, total=total)
    else:
        item_fraction = 1.0 if progress_status == "completed" else 0.02
        item_text = _runtime_text(message or "Waiting for the first stage.")

    if progress_status == "completed":
        overall_fraction = 1.0
        overall_text = _t("Run completed")
    else:
        stage_index = stages.index(stage) if stage in stages else 0
        overall_fraction = min(
            (stage_index + item_fraction) / max(len(stages), 1),
            1.0,
        )
        overall_text = _t(
            "Stage {current} of {total}: {stage}",
            current=stage_index + 1,
            total=max(len(stages), 1),
            stage=_runtime_text(stage),
        )

    overall_text = (
        f"{overall_text} · "
        f"{_t('{count} papers collected', count=paper_count)}"
    )
    st.progress(overall_fraction, text=overall_text)
    st.progress(
        item_fraction,
        text=f"{_runtime_text(stage)}: {item_text}",
    )
    detail = _runtime_text(message)
    if current:
        detail = f"{detail} {_t('Current item: {item}', item=current)}"
    if detail:
        st.caption(detail)
    st.caption(
        _t(
            "Started {started_at} · Last saved {updated_at}",
            started_at=progress.get("started_at", state.get("created_at", "")),
            updated_at=progress.get("updated_at", state.get("updated_at", "")),
        )
    )

    if task and task.running:
        if task.cancel_requested:
            st.button(
                _t("Stopping..."),
                icon=":material/progress_activity:",
                disabled=True,
                key=f"stop_run_{project_slug}",
            )
            st.caption(
                _t(
                    "The current item will finish safely before the run stops."
                )
            )
        elif st.button(
            _t("Stop run"),
            icon=":material/stop_circle:",
            type="secondary",
            key=f"stop_run_{project_slug}",
        ):
            _task_manager().cancel(project_slug)
            st.rerun()
    elif task and task.can_restart and (task.cancelled or task.error):
        if st.button(
            _t("Run again"),
            icon=":material/replay:",
            type="primary",
            key=f"restart_run_{project_slug}",
        ):
            if _task_manager().restart(project_slug):
                st.rerun()
            else:
                st.warning(_t("The previous task is still stopping. Please wait."))

    if task and task.running:
        if not task.cancel_requested:
            st.info(_t("Updating automatically. You can safely visit another page."))
    elif progress_status == "running":
        st.warning(
            _t(
                "This saved run is no longer active. It may have been interrupted when the "
                "app stopped."
            )
        )
    elif progress_status == "failed":
        st.error(_t("Pipeline failed: {error}", error=_runtime_text(message)))
    elif progress_status == "cancelled":
        st.warning(_t("Run stopped. Completed files have been retained."))

    marker_key = f"progress_status_{project_slug}_{state.get('run_id', '')}"
    previous_status = st.session_state.get(marker_key)
    st.session_state[marker_key] = progress_status
    if previous_status == "running" and progress_status != "running":
        st.rerun()


def _current_paper_count(state: dict[str, Any]) -> int:
    progress_count = state.get("progress", {}).get("paper_count")
    if progress_count not in (None, ""):
        return max(int(progress_count), 0)
    latest = state.get("rounds", [{}])[-1]
    counts = latest.get("counts", {})
    for key in ("pool_rows", "deduped_records", "audit_queue"):
        value = counts.get(key)
        if value not in (None, ""):
            return max(int(value), 0)
    return 0


def _state_label(value: object) -> str:
    normalized = str(value or "unknown").replace("_", " ").lower()
    translated = _t(normalized)
    return translated if translated != normalized else normalized.title()


def _decision_label(value: object) -> str:
    decision = str(value or "")
    return _t(decision) if decision else ""


def _value_label(value: object) -> str:
    raw_value = str(value or "unknown")
    return _t(raw_value)


def _decision_code(value: object) -> str:
    label = str(value or "")
    for decision in ["include", "include_related", "exclude", "later"]:
        if label == _decision_label(decision):
            return decision
    return label


def _runtime_text(message: str) -> str:
    translated = _t(message)
    if translated != message:
        return translated
    if message.startswith("Connection failed: "):
        return _t("Connection failed: {error}", error=message.removeprefix("Connection failed: "))
    suffix = " papers require human decisions."
    if message.endswith(suffix) and message.removesuffix(suffix).isdigit():
        return _t(
            "{count} papers require human decisions.",
            count=message.removesuffix(suffix),
        )
    return message


def _result_column_config() -> dict[str, Any]:
    return {
        "title": st.column_config.TextColumn(_t("Title"), width="large"),
        "authors": st.column_config.TextColumn(_t("Authors"), width="large"),
        "year": st.column_config.TextColumn(_t("Year")),
        "venue": st.column_config.TextColumn(_t("Venue")),
        "venue_type": st.column_config.TextColumn(_t("Venue type")),
        "core_rank": st.column_config.TextColumn(_t("CORE rank")),
        "impact_factor": st.column_config.TextColumn(_t("Impact Factor")),
        "manual_decision": st.column_config.TextColumn(_t("Manual decision")),
    }


def _groups_from_frame(frame: pd.DataFrame) -> list[KeywordGroup]:
    groups: list[KeywordGroup] = []
    for row in frame.fillna("").to_dict(orient="records"):
        name = str(row.get("Group") or "").strip()
        terms = _split_terms(str(row.get("Terms") or ""))
        if name and terms:
            groups.append(KeywordGroup(name=name, terms=terms))
    return groups


def _split_lines(value: str) -> list[str]:
    return [line.strip(" -\t") for line in value.splitlines() if line.strip(" -\t")]


def _split_terms(value: str) -> list[str]:
    normalized = value.replace(",", "\n").replace(";", "\n")
    return _split_lines(normalized)


def _split_semicolon_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _boolean_expression(groups: list[KeywordGroup]) -> str:
    parts = []
    for group in groups:
        terms = " OR ".join(f'"{term}"' if " " in term else term for term in group.terms)
        parts.append(f"({terms})")
    return "\nAND\n".join(parts)


def _none_if_zero(value: int | float) -> int | None:
    return None if int(value) == 0 else int(value)


def _filter_audit_frame(frame: pd.DataFrame, search: str, decision: str) -> pd.DataFrame:
    filtered = frame.copy()
    if search.strip():
        needle = search.strip().lower()
        title = filtered.get("title", pd.Series("", index=filtered.index)).astype(str).str.lower()
        abstract = (
            filtered.get("abstract", pd.Series("", index=filtered.index)).astype(str).str.lower()
        )
        filtered = filtered[
            title.str.contains(needle, regex=False) | abstract.str.contains(needle, regex=False)
        ]
    if decision == "Unreviewed":
        filtered = filtered[filtered.get("manual_decision", "").isin(["", "later"])]
    elif decision != "All":
        filtered = filtered[filtered.get("manual_decision", "") == decision]
    return filtered


def _merge_editor_identity(original: pd.DataFrame, edited: pd.DataFrame) -> list[dict[str, str]]:
    updates: list[dict[str, str]] = []
    for position, (_, edited_row) in enumerate(edited.iterrows()):
        original_row = original.iloc[position].to_dict()
        original_row["manual_decision"] = _decision_code(edited_row.get("manual_decision"))
        original_row["manual_notes"] = str(edited_row.get("manual_notes") or "")
        updates.append({str(key): str(value or "") for key, value in original_row.items()})
    return updates


def _paper_metadata(row: dict[str, str]) -> str:
    values = [
        row.get("authors", ""),
        row.get("year", ""),
        row.get("venue", ""),
        f"CORE {row.get('core_rank')}" if row.get("core_rank") else "",
        f"IF {row.get('impact_factor')}" if row.get("impact_factor") else "",
    ]
    return " · ".join(value for value in values if value)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {border: 1px solid #d8dee8; border-radius: 6px; padding: 14px;}
        [data-testid="stSidebar"] {border-right: 1px solid #d8dee8;}
        [data-testid="stAppDeployButton"] {display: none;}
        h1, h2, h3, p, label, button {letter-spacing: 0 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
