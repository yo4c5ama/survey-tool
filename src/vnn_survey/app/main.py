from __future__ import annotations

import json
import os
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from vnn_survey.ai_research import (
    OpenAIResearchClient,
    PaperWorkspace,
    estimate_corpus_requests,
)
from vnn_survey.app.audit import AuditSummary, load_audit
from vnn_survey.app.i18n import LANGUAGE_NAMES, language_name, translate
from vnn_survey.app.manual_papers import ManualPaperStore, create_manual_record
from vnn_survey.app.pipeline_service import (
    PipelineService,
    list_openai_models,
    test_openai_connection,
)
from vnn_survey.app.project_store import KeywordGroup, ProjectSettings, ProjectStore
from vnn_survey.app.project_transfer import (
    ProjectTransferError,
    create_projects_backup,
    import_projects_backup,
    save_uploaded_backup,
)
from vnn_survey.app.run_flow import build_flow_svg, flow_summary_payload, round_flow_stages
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
    "manual_review": "Manual review",
    "snowball": "Snowball",
    "results": "Results",
    "ai_research": "AI research",
}
PAGE_DESCRIPTIONS = {
    "scope": "Research scope and search logic",
    "ai_settings": (
        "Choose a model independently for each AI task. Human reviewers retain final authority."
    ),
    "run_center": (
        "Run discovery first. AI screening is a separate, explicitly confirmed stage."
    ),
    "manual_review": (
        "Review candidate papers, record evidence, and make the final inclusion decisions."
    ),
    "snowball": (
        "Only newly included or related papers from the latest review round become seeds. "
        "Each round collects references and citations, then sends only never-reviewed "
        "papers to a new review queue."
    ),
    "results": (
        "Inspect the reviewed corpus and export the evidence needed for a reproducible survey."
    ),
    "ai_research": (
        "Study individual papers with their PDFs or classify the final reviewed corpus. "
        "AI output remains an analytical aid and should be checked by the researcher."
    ),
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
ABSTRACT_PROVIDER_OPTIONS = [
    "arxiv",
    "pubmed",
    "crossref",
    "semantic_scholar",
    "openalex",
]
ABSTRACT_PROVIDER_LABELS = {
    "arxiv": "arXiv",
    "pubmed": "PubMed",
    "crossref": "Crossref",
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "__disabled__": "Disabled",
}
SNOWBALL_PROVIDER_OPTIONS = ["semantic_scholar", "opencitations", "openalex"]
SNOWBALL_PROVIDER_LABELS = {
    "semantic_scholar": "Semantic Scholar",
    "opencitations": "OpenCitations",
    "openalex": "OpenAlex",
    "__disabled__": "Disabled",
}


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=None, layout="wide")
    _apply_styles()
    store = _store()
    service = PipelineService(store)

    projects = store.list_projects()
    _render_sidebar_header()
    if not projects:
        _render_language_selector()
        _render_backup_restore(store, projects)
        st.sidebar.info(_t("Create your first survey project to begin."))
        _render_create_project(store)
        _render_app_footer()
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
    if st.sidebar.button(
        _t("New project"),
        width="stretch",
        icon=":material/add:",
        type="tertiary",
    ):
        st.session_state["create_project"] = True
    if st.session_state.get("create_project"):
        _render_sidebar_utilities(store, projects)
        _render_create_project(store, can_cancel=True)
        _render_app_footer()
        return

    project = store.load_project(selected_slug)
    saved_openalex_key = store.read_openalex_api_key(project.slug)
    if saved_openalex_key:
        os.environ["OPENALEX_API_KEY"] = saved_openalex_key
    saved_semantic_scholar_key = store.read_semantic_scholar_api_key(project.slug)
    if saved_semantic_scholar_key:
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = saved_semantic_scholar_key
    saved_ncbi_key = store.read_ncbi_api_key(project.slug)
    if saved_ncbi_key:
        os.environ["NCBI_API_KEY"] = saved_ncbi_key
    if project.scholarly_api_email:
        os.environ["CROSSREF_EMAIL"] = project.scholarly_api_email
        os.environ["OPENALEX_EMAIL"] = project.scholarly_api_email
        os.environ["NCBI_EMAIL"] = project.scholarly_api_email

    st.sidebar.markdown(
        f'<div class="sf-nav-label">{escape(_t("Workspace"))}</div>',
        unsafe_allow_html=True,
    )
    page_ids = list(PAGE_LABELS)
    page = st.sidebar.radio(
        _t("Workspace"),
        page_ids,
        format_func=lambda page_id: (
            f"{page_ids.index(page_id) + 1:02d}   {_t(PAGE_LABELS[page_id])}"
        ),
        label_visibility="collapsed",
        key="workspace_page",
    )
    st.sidebar.divider()
    state = service.current_state_or_none(project.slug)
    _render_sidebar_status(project, state)
    _render_sidebar_utilities(store, projects)
    _render_page_header(page, project)

    if page == "scope":
        _render_scope_page(store, project)
    elif page == "ai_settings":
        _render_ai_settings(store, project)
    elif page == "run_center":
        _render_run_center(store, service, project)
    elif page == "manual_review":
        _render_manual_review(store, service, project)
    elif page == "snowball":
        _render_snowball(service, project)
    elif page == "results":
        _render_results(service, project)
    elif page == "ai_research":
        _render_ai_research(store, service, project)
    _render_app_footer()


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
    caption = escape(_t("Systematic literature review workspace"))
    st.sidebar.markdown(
        f"""
        <div class="sf-brand">
            <div class="sf-brand-name">{APP_NAME}</div>
            <div class="sf-brand-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_utilities(
    store: ProjectStore,
    projects: list[ProjectSettings],
) -> None:
    st.sidebar.divider()
    _render_language_selector()
    _render_backup_restore(store, projects)


def _render_language_selector() -> None:
    st.sidebar.selectbox(
        _t("Interface language"),
        list(LANGUAGE_NAMES),
        format_func=language_name,
        key="ui_language",
    )


def _render_backup_restore(
    store: ProjectStore,
    projects: list[ProjectSettings],
) -> None:
    running = any(_task_manager().is_running(project.slug) for project in projects)
    with st.sidebar.expander(_t("Backup and restore")):
        feedback = st.session_state.pop("project_import_feedback", None)
        if feedback:
            st.success(
                _t(
                    "Imported {imported} project(s); skipped {skipped} existing project(s).",
                    imported=feedback["imported"],
                    skipped=feedback["skipped"],
                )
            )
            if feedback["secrets_restored"]:
                st.caption(
                    _t(
                        "Saved API keys were restored for {count} project(s).",
                        count=feedback["secrets_restored"],
                    )
                )
            elif feedback["backup_includes_secrets"]:
                st.warning(_t("The backup contains API keys, but they were not restored."))

        st.caption(
            _t("Export every project, run checkpoint, review, PDF, conversation, and analysis.")
        )
        include_caches = st.checkbox(
            _t("Include rebuildable API caches"),
            value=False,
            key="backup_include_caches",
        )
        include_secrets = st.checkbox(
            _t("Include saved API keys"),
            value=False,
            key="backup_include_secrets",
        )
        if include_secrets:
            st.warning(
                _t(
                    "API keys are stored as readable files inside the ZIP. "
                    "Keep this backup private."
                )
            )
        if running:
            st.info(_t("Stop all running project tasks before exporting or importing."))
        if st.button(
            _t("Create backup"),
            icon=":material/archive:",
            width="stretch",
            disabled=not projects or running,
        ):
            try:
                result = create_projects_backup(
                    store,
                    include_secrets=include_secrets,
                    include_caches=include_caches,
                )
            except (OSError, ProjectTransferError) as exc:
                st.error(_runtime_text(str(exc)))
            else:
                st.session_state["project_backup_path"] = str(result.path)
                st.session_state["project_backup_summary"] = {
                    "projects": result.project_count,
                    "files": result.file_count,
                    "source_bytes": result.source_bytes,
                }

        backup_value = st.session_state.get("project_backup_path")
        backup_path = Path(backup_value) if backup_value else None
        if backup_path and backup_path.exists():
            summary = st.session_state.get("project_backup_summary", {})
            st.caption(
                _t(
                    "Ready: {projects} project(s), {files} files, {size} before compression.",
                    projects=summary.get("projects", 0),
                    files=summary.get("files", 0),
                    size=_format_bytes(summary.get("source_bytes", 0)),
                )
            )
            with backup_path.open("rb") as handle:
                st.download_button(
                    _t("Download backup"),
                    data=handle,
                    file_name=backup_path.name,
                    mime="application/zip",
                    icon=":material/download:",
                    width="stretch",
                )

        st.divider()
        uploaded = st.file_uploader(
            _t("Import SurveyFlow backup"),
            type=["zip"],
            key="project_backup_upload",
        )
        replace_existing = st.checkbox(
            _t("Replace matching projects with the backup"),
            value=False,
            key="project_import_replace",
        )
        conflict = "replace" if replace_existing else "skip"
        restore_secrets = st.checkbox(
            _t("Restore saved API keys when present"),
            value=False,
            key="project_import_secrets",
        )
        replace_confirmed = True
        if replace_existing:
            replace_confirmed = st.checkbox(
                _t("I understand that matching project data will be replaced"),
                value=False,
                key="project_import_replace_confirmed",
            )
        if st.button(
            _t("Import backup"),
            icon=":material/upload:",
            width="stretch",
            disabled=(uploaded is None or running or (replace_existing and not replace_confirmed)),
        ):
            try:
                upload_path = save_uploaded_backup(store, uploaded.getvalue())
                result = import_projects_backup(
                    store,
                    upload_path,
                    conflict=conflict,
                    restore_secrets=restore_secrets,
                )
            except (OSError, ProjectTransferError, ValueError) as exc:
                st.error(_runtime_text(str(exc)))
            else:
                st.session_state["project_import_feedback"] = {
                    "imported": len(result.imported),
                    "skipped": len(result.skipped),
                    "secrets_restored": len(result.restored_secret_projects),
                    "backup_includes_secrets": result.backup_includes_secrets,
                }
                st.rerun()


def _render_sidebar_status(project: ProjectSettings, state: dict[str, Any] | None) -> None:
    st.sidebar.markdown(
        f'<div class="sf-nav-label">{escape(_t("Status"))}</div>',
        unsafe_allow_html=True,
    )
    if not state:
        st.sidebar.markdown(
            f'<div class="sf-sidebar-status"><span></span>{escape(_t("Not started"))}</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(_t("Updated {value}", value=project.updated_at or _t("not yet")))
        return
    status = _state_label(state.get("status", "unknown"))
    st.sidebar.markdown(
        f'<div class="sf-sidebar-status sf-sidebar-status-active"><span></span>'
        f"{escape(status)}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        _t("Run {run_id}", run_id=state.get("run_id", ""))
        + "  ·  "
        + _t("Updated {value}", value=project.updated_at or _t("not yet"))
    )


def _render_page_header(
    page: str,
    project: ProjectSettings,
) -> None:
    page_ids = list(PAGE_LABELS)
    context = _t(
        "Step {current} of {total}",
        current=page_ids.index(page) + 1,
        total=len(page_ids),
    )
    project_context = f"{context}  ·  {project.name}"
    st.markdown(
        f'<div class="sf-page-context">{escape(project_context)}</div>',
        unsafe_allow_html=True,
    )
    st.title(_t(PAGE_LABELS[page]))
    st.markdown(
        f'<p class="sf-page-description">{escape(_t(PAGE_DESCRIPTIONS[page]))}</p>',
        unsafe_allow_html=True,
    )


def _render_app_footer() -> None:
    st.markdown(
        f"""
        <div class="sf-footer">
            <span>&copy; {datetime.now().year} yoac</span>
            <span>{escape(_t("Built with assistance from Codex"))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            placeholder=_t("Describe the models, methods, properties, and application boundaries."),
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
    research_domain, discovery_sources = _render_domain_source_selector(
        prefix=f"scope_{project.slug}",
        current_domain=project.research_domain,
        current_sources=project.discovery_sources,
    )
    groups_frame = pd.DataFrame(
        [{"Group": group.name, "Terms": "\n".join(group.terms)} for group in project.keyword_groups]
    )
    with st.form("scope_form"):
        question_tab, keywords_tab, eligibility_tab = st.tabs(
            [_t("Question"), _t("Keywords"), _t("Criteria")]
        )
        with question_tab:
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
        with keywords_tab:
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
        with eligibility_tab:
            criteria_columns = st.columns(2)
            with criteria_columns[0]:
                inclusion = st.text_area(
                    _t("Inclusion criteria, one per line"),
                    value="\n".join(project.inclusion_criteria),
                    height=180,
                    key=f"scope_inclusion_{project.slug}",
                )
            with criteria_columns[1]:
                exclusion = st.text_area(
                    _t("Exclusion criteria, one per line"),
                    value="\n".join(project.exclusion_criteria),
                    height=180,
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
    models_tab, enrichment_tab, credentials_tab, screening_prompt_tab = st.tabs(
        [_t("Models"), _t("Abstract enrichment"), _t("API key"), _t("Screening prompt")]
    )
    with models_tab:
        _render_ai_model_settings(store, project)
    with enrichment_tab:
        _render_abstract_enrichment_settings(store, project)
    with credentials_tab:
        _render_api_key_settings(store, project)
    with screening_prompt_tab:
        _render_screening_prompt_settings(store, project)


def _render_ai_model_settings(store: ProjectStore, project: ProjectSettings) -> None:
    base_url = st.text_input(
        _t("Base URL"), value=project.llm_base_url, key=f"ai_base_url_{project.slug}"
    )
    fetched_models = st.session_state.get(f"available_models_{project.slug}", [])
    screening_tab, prompt_tab, research_tab = st.tabs(
        [_t("Screening"), _t("Prompt learning"), _t("Research analysis")]
    )
    with screening_tab:
        model_columns = st.columns(2)
        with model_columns[0]:
            title_model = _render_model_selector(
                _t("Title screening model"),
                project.title_screening_model,
                key=f"ai_title_screening_model_{project.slug}",
                fetched_models=fetched_models,
                help_text=_t("Used only for high-recall title prescreening."),
            )
        with model_columns[1]:
            model = _render_model_selector(
                _t("Abstract screening model"),
                project.llm_model,
                key=f"ai_abstract_screening_model_{project.slug}",
                fetched_models=fetched_models,
                help_text=_t("Used for the main abstract-level screening stage."),
            )
    with prompt_tab:
        model_columns = st.columns(2)
        with model_columns[0]:
            prompt_model = _render_model_selector(
                _t("Prompt refinement model"),
                project.prompt_refinement_model,
                key=f"ai_prompt_refinement_model_{project.slug}",
                fetched_models=fetched_models,
                help_text=_t("Learns a proposed prompt from cumulative human decisions."),
            )
        with model_columns[1]:
            replay_model = _render_model_selector(
                _t("Historical replay model"),
                project.prompt_replay_model,
                key=f"ai_prompt_replay_model_{project.slug}",
                fetched_models=fetched_models,
                help_text=_t("Used for the one-time replay of initial AI exclusions."),
            )
    with research_tab:
        model_columns = st.columns(2)
        with model_columns[0]:
            paper_model = _render_model_selector(
                _t("Paper Q&A model"),
                project.paper_qa_model,
                key=f"ai_paper_model_{project.slug}",
                fetched_models=fetched_models,
                help_text=_t("Used for questions about an uploaded paper PDF."),
            )
        with model_columns[1]:
            corpus_model = _render_model_selector(
                _t("Corpus analysis model"),
                project.corpus_analysis_model,
                key=f"ai_corpus_model_{project.slug}",
                fetched_models=fetched_models,
                help_text=_t("Used to design a taxonomy and classify the final corpus."),
            )
    screen_batch_size = st.number_input(
        _t("Abstracts per AI screening batch"),
        min_value=1,
        max_value=50,
        value=min(max(project.llm_screen_batch_size, 1), 50),
        help=_t(
            "Each request screens this many abstracts when possible. Failed batches are "
            "automatically divided so successful papers are preserved."
        ),
        key=f"llm_screen_batch_size_{project.slug}",
    )
    if st.button(_t("Save model settings"), type="primary"):
        if not all(
            [
                title_model.strip(),
                model.strip(),
                prompt_model.strip(),
                replay_model.strip(),
                paper_model.strip(),
                corpus_model.strip(),
            ]
        ):
            st.error(_t("Every AI task requires a model."))
        else:
            project.title_screening_model = title_model.strip()
            project.llm_model = model.strip()
            project.prompt_refinement_model = prompt_model.strip()
            project.prompt_replay_model = replay_model.strip()
            project.paper_qa_model = paper_model.strip()
            project.corpus_analysis_model = corpus_model.strip()
            project.llm_base_url = base_url.strip()
            project.llm_screen_batch_size = int(screen_batch_size)
            store.save_project(project)
            st.success(_t("Model settings saved."))


def _render_abstract_enrichment_settings(
    store: ProjectStore,
    project: ProjectSettings,
) -> None:
    st.caption(
        _t(
            "Abstracts already returned by discovery are always used first. Configure the "
            "fallback providers below; each paper stops after the first successful match."
        )
    )
    provider_options = ["__disabled__", *ABSTRACT_PROVIDER_OPTIONS]
    provider_values: list[str] = []
    provider_columns = st.columns(2)
    for index in range(len(ABSTRACT_PROVIDER_OPTIONS)):
        current = (
            project.abstract_providers[index]
            if index < len(project.abstract_providers)
            and project.abstract_providers[index] in ABSTRACT_PROVIDER_OPTIONS
            else "__disabled__"
        )
        with provider_columns[index % 2]:
            provider_values.append(
                st.selectbox(
                    _t("Fallback priority {index}", index=index + 1),
                    provider_options,
                    index=provider_options.index(current),
                    format_func=lambda value: _t(ABSTRACT_PROVIDER_LABELS[value]),
                    key=f"abstract_provider_{index}_{project.slug}",
                )
            )
    batch_size = st.number_input(
        _t("Maximum identifier batch size"),
        min_value=1,
        max_value=500,
        value=min(max(project.abstract_batch_size, 1), 500),
        help=_t(
            "Provider limits are applied automatically: OpenAlex and arXiv use at most 100, "
            "PubMed 200, and Semantic Scholar 500 identifiers per request."
        ),
        key=f"abstract_batch_size_{project.slug}",
    )
    scholarly_email = st.text_input(
        _t("Scholarly API contact email (optional)"),
        value=project.scholarly_api_email,
        help=_t("Enables the Crossref polite pool and identifies requests to scholarly APIs."),
        key=f"scholarly_api_email_{project.slug}",
    )
    if st.button(
        _t("Save abstract settings"),
        type="primary",
        key=f"save_abstract_settings_{project.slug}",
    ):
        selected_providers = [
            provider for provider in provider_values if provider != "__disabled__"
        ]
        if len(selected_providers) != len(set(selected_providers)):
            st.error(_t("Each abstract provider can appear only once."))
        elif not selected_providers:
            st.error(_t("Select at least one abstract provider."))
        else:
            project.abstract_providers = selected_providers
            project.abstract_batch_size = int(batch_size)
            project.scholarly_api_email = scholarly_email.strip()
            store.save_project(project)
            if project.scholarly_api_email:
                os.environ["CROSSREF_EMAIL"] = project.scholarly_api_email
                os.environ["OPENALEX_EMAIL"] = project.scholarly_api_email
                os.environ["NCBI_EMAIL"] = project.scholarly_api_email
            st.success(_t("Abstract settings saved."))


def _render_api_key_settings(store: ProjectStore, project: ProjectSettings) -> None:
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
            ok, message = test_openai_connection(project.llm_base_url, key_for_action)
            (st.success if ok else st.error)(_runtime_text(message))
    with second:
        if st.button(_t("Apply API key"), width="stretch"):
            if not api_key.strip():
                st.error(_t("Enter a new API key before applying it."))
            else:
                os.environ["OPENAI_API_KEY"] = api_key.strip()
                if remember:
                    store.save_api_key(project.slug, api_key)
                st.success(_t("The API key is ready. It is never written to project YAML or logs."))
    with third:
        if st.button(_t("Refresh model list"), width="stretch"):
            models, message = list_openai_models(project.llm_base_url, key_for_action)
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

    with st.expander(_t("Semantic Scholar API key")):
        saved_semantic_scholar_key = store.read_semantic_scholar_api_key(project.slug)
        semantic_scholar_key = st.text_input(
            _t("Semantic Scholar API key"),
            type="password",
            placeholder=(
                _t("A key is already saved")
                if saved_semantic_scholar_key
                else _t("Optional, but recommended for a dedicated rate limit")
            ),
            key=f"semantic_scholar_api_key_{project.slug}",
            label_visibility="collapsed",
        )
        semantic_columns = st.columns(2)
        with semantic_columns[0]:
            if st.button(
                _t("Apply Semantic Scholar key"),
                width="stretch",
                key=f"apply_semantic_scholar_key_{project.slug}",
            ):
                if not semantic_scholar_key.strip():
                    st.error(_t("Enter a new Semantic Scholar API key before applying it."))
                else:
                    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = semantic_scholar_key.strip()
                    store.save_semantic_scholar_api_key(project.slug, semantic_scholar_key)
                    st.success(_t("The Semantic Scholar API key is ready."))
        with semantic_columns[1]:
            st.link_button(
                _t("Request a Semantic Scholar key"),
                "https://www.semanticscholar.org/product/api",
                width="stretch",
            )

    with st.expander(_t("NCBI API key")):
        saved_ncbi_key = store.read_ncbi_api_key(project.slug)
        ncbi_key = st.text_input(
            _t("NCBI API key"),
            type="password",
            placeholder=(
                _t("A key is already saved")
                if saved_ncbi_key
                else _t("Optional; PubMed batching works without a key")
            ),
            key=f"ncbi_api_key_{project.slug}",
            label_visibility="collapsed",
        )
        if st.button(
            _t("Apply NCBI key"),
            width="stretch",
            key=f"apply_ncbi_key_{project.slug}",
        ):
            if not ncbi_key.strip():
                st.error(_t("Enter a new NCBI API key before applying it."))
            else:
                os.environ["NCBI_API_KEY"] = ncbi_key.strip()
                store.save_ncbi_api_key(project.slug, ncbi_key)
                st.success(_t("The NCBI API key is ready."))

    with st.expander(_t("OpenAlex API key")):
        saved_openalex_key = store.read_openalex_api_key(project.slug)
        openalex_key = st.text_input(
            _t("OpenAlex API key"),
            type="password",
            placeholder=(
                _t("A key is already saved")
                if saved_openalex_key
                else _t("Enter your OpenAlex API key")
            ),
            key=f"openalex_api_key_{project.slug}",
            label_visibility="collapsed",
        )
        openalex_remember = st.checkbox(
            _t("Save this OpenAlex key on this computer"),
            value=True,
            key=f"openalex_remember_{project.slug}",
        )
        openalex_columns = st.columns(2)
        with openalex_columns[0]:
            if st.button(
                _t("Apply OpenAlex key"),
                width="stretch",
                key=f"apply_openalex_key_{project.slug}",
            ):
                if not openalex_key.strip():
                    st.error(_t("Enter a new OpenAlex API key before applying it."))
                else:
                    os.environ["OPENALEX_API_KEY"] = openalex_key.strip()
                    if openalex_remember:
                        store.save_openalex_api_key(project.slug, openalex_key)
                    st.success(_t("The OpenAlex API key is ready."))
        with openalex_columns[1]:
            st.link_button(
                _t("Get a free OpenAlex key"),
                "https://openalex.org/settings/api",
                width="stretch",
            )
        st.caption(
            _t(
                "OpenAlex now requires a free API key for sustained discovery and abstract "
                "enrichment. The key is stored with the same local protection as the OpenAI key."
            )
        )


def _render_screening_prompt_settings(store: ProjectStore, project: ProjectSettings) -> None:
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
    state = service.current_state_or_none(project.slug)
    task = _task_manager().snapshot(project.slug)
    if state or task:
        _render_current_run_progress(
            project.slug,
            service,
            widget_scope="run_center",
        )
    if state:
        _render_literature_flow(state)
        with st.expander(_t("Show round history")):
            _render_round_overview(state)

    if not state:
        if task and task.running:
            st.info(_t("The pipeline is starting. You can leave this page and return later."))
            return
        _initial_discovery_form(service, project)
        return

    if task and task.running:
        return

    latest = state["rounds"][-1]
    status = latest.get("status")
    citation_warnings = bool(
        latest.get("kind") == "snowball"
        and (
            _citation_failure_count(latest)
            or int(latest.get("counts", {}).get("coverage_issue_seeds", 0) or 0)
        )
    )
    if citation_warnings or status == "citation_incomplete":
        st.warning(
            _t(
                "Some seed papers have citation coverage warnings. Available candidates "
                "are saved and can continue to AI screening and manual review."
            )
        )
    if status in {"discovery_complete", "citation_incomplete"}:
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
    has_api_key = service.store.has_api_key(project.slug) or bool(os.environ.get("OPENAI_API_KEY"))
    title_columns = st.columns([2, 1])
    with title_columns[0]:
        use_title_llm = st.toggle(
            _t("Use AI title prescreen before abstract enrichment"),
            value=has_api_key,
            disabled=not has_api_key,
            key=f"title_llm_{project.slug}_{compact}",
            help=_t(
                "Titles are screened in high-recall batches. Ambiguous papers are kept for "
                "abstract enrichment."
            ),
        )
    with title_columns[1]:
        title_batch_size = st.number_input(
            _t("Titles per AI batch"),
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            disabled=not use_title_llm,
            key=f"title_batch_{project.slug}_{compact}",
        )
    if not has_api_key:
        st.caption(_t("Add an API key on AI settings to enable title prescreening."))
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
            use_title_llm=use_title_llm,
            title_batch_size=int(title_batch_size),
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))


def _render_manual_additions(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
    *,
    embedded: bool = False,
    target_round_index: int | None = None,
) -> None:
    if embedded:
        with st.expander(_t("Add papers")):
            _render_manual_additions_content(
                store,
                service,
                project,
                target_round_index=target_round_index,
            )
        return
    _render_manual_additions_content(
        store,
        service,
        project,
        target_round_index=target_round_index,
    )


def _render_manual_additions_content(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
    *,
    target_round_index: int | None = None,
) -> None:
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
    if task and task.running and task.operation in {"manual_additions", "manual_enrichment"}:
        _render_current_run_progress(
            project.slug,
            service,
            widget_scope="manual_additions",
        )

    feedback = st.session_state.pop(f"manual_addition_feedback_{project.slug}", None)
    if feedback:
        _render_manual_addition_feedback(feedback)

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
            result = service.remove_manual_paper(project.slug, records[remove_index])
            st.session_state[f"manual_addition_feedback_{project.slug}"] = result
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
            disabled=bool(task and task.running),
        ):
            result = service.add_manual_paper(
                project.slug,
                selected,
                note,
                round_index=target_round_index,
            )
            st.session_state[f"manual_addition_feedback_{project.slug}"] = result
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
            submitted = st.form_submit_button(
                _t("Add manual record"),
                type="primary",
                disabled=bool(task and task.running),
            )
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
                result = service.add_manual_paper(
                    project.slug,
                    record,
                    manual_note,
                    round_index=target_round_index,
                )
                st.session_state[f"manual_addition_feedback_{project.slug}"] = result
                st.rerun()
            except ValueError as exc:
                st.error(_runtime_text(str(exc)))

    st.divider()
    if not state:
        st.subheader(_t("Apply to discovery"))
        st.info(_t("Saved papers will be included automatically in the next initial discovery."))
        return
    initial_round = next(
        (item for item in state.get("rounds", []) if int(item.get("index", -1)) == 0),
        None,
    )
    if not initial_round:
        st.subheader(_t("Apply to discovery"))
        st.info(_t("Saved papers will be included automatically in the next initial discovery."))
        return
    if any(item.get("files", {}).get("audit") for item in state.get("rounds", [])):
        active_round_index = (
            target_round_index
            if target_round_index is not None
            else int(
                next(
                    item for item in reversed(state["rounds"]) if item.get("files", {}).get("audit")
                )["index"]
            )
        )
        enrichment_status = service.manual_enrichment_status(
            project.slug,
            active_round_index,
        )
        st.subheader(_t("Manual enrichment loop"))
        st.write(
            _t(
                "Start venue enrichment, abstract enrichment, and AI abstract screening for "
                "pending researcher additions. Every AI result returns to the selected manual "
                "review round."
            )
        )
        status_columns = st.columns(2)
        status_columns[0].metric(
            _t("Pending enrichment"),
            enrichment_status["pending"],
        )
        status_columns[1].metric(
            _t("Enriched manual papers"),
            enrichment_status["enriched"],
        )
        has_ai_key = store.has_api_key(project.slug) or bool(os.environ.get("OPENAI_API_KEY"))
        if not has_ai_key:
            st.warning(_t("Add an API key on the AI settings page before continuing."))
        if st.button(
            _t("Start enrichment and AI screening"),
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=(
                not enrichment_status["pending"] or bool(task and task.running) or not has_ai_key
            ),
            key=f"manual_enrichment_start_{project.slug}_{active_round_index}",
        ):
            started = _task_manager().start(
                project.slug,
                "manual_enrichment",
                service.enrich_manual_additions,
                project.slug,
                active_round_index,
            )
            if started:
                st.rerun()
            else:
                st.warning(_t("A pipeline task is already running for this project."))
        return
    st.subheader(_t("Apply to discovery"))
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
    st.subheader(_t("Prepare round {round_index} for review", round_index=round_index))
    has_ai_key = store.has_api_key(project.slug) or bool(os.environ.get("OPENAI_API_KEY"))
    controls = st.columns([2, 1])
    with controls[0]:
        use_llm = st.toggle(
            _t("Use AI abstract screening"),
            value=has_ai_key,
            key=f"use_llm_{round_state.get('index')}_{project.slug}",
        )
    with controls[1]:
        llm_limit = st.number_input(
            _t("AI paper limit"),
            min_value=0,
            value=0,
            help=_t("Use 0 for every eligible paper. A small limit is useful for testing."),
            key=f"llm_limit_{round_state.get('index')}_{project.slug}",
        )

    replay_pending = service.prompt_replay_pending_for_round(project.slug, round_index)
    if replay_pending:
        st.info(
            _t(
                "The approved prompt will re-screen eligible historical exclusions before "
                "this review queue is created."
            )
        )

    refresh_error = ""
    if st.button(
        _t("Refresh model price"),
        icon=":material/refresh:",
        key=f"refresh_model_price_{round_index}_{project.slug}",
    ):
        try:
            service.refresh_llm_model_price(project.slug)
            st.toast(_t("The model price was refreshed from the official OpenAI model page."))
        except RuntimeError as exc:
            refresh_error = str(exc)
    if refresh_error:
        st.warning(
            _t(
                "The live price could not be refreshed. The local fallback remains active: "
                "{error}",
                error=_runtime_text(refresh_error),
            )
        )

    estimate = service.estimate_llm_usage(
        project.slug,
        round_index,
        llm_limit=_none_if_zero(llm_limit),
    )
    metrics = st.columns(4)
    metrics[0].metric(_t("Papers eligible for AI"), f"{estimate['papers']:,}")
    metrics[1].metric(_t("Estimated input tokens"), f"{estimate['estimated_input_tokens']:,}")
    metrics[2].metric(_t("Estimated output tokens"), f"{estimate['estimated_output_tokens']:,}")
    metrics[3].metric(
        _t("Estimated cost"),
        _format_usd_estimate(estimate["estimated_cost_usd"]),
    )
    price = estimate.get("price")
    if price:
        source_label = (
            _t("Official OpenAI price")
            if price.get("source") == "openai_official"
            else _t("Local fallback price")
        )
        st.caption(
            _t(
                "{model}: ${input_price} input and ${output_price} output per 1M tokens. "
                "{source}, updated {updated_at}.",
                model=estimate["model"],
                input_price=f"{price['input_per_million']:g}",
                output_price=f"{price['output_per_million']:g}",
                source=source_label,
                updated_at=price.get("updated_at") or _t("not yet"),
            )
        )
        st.caption(
            _t(
                "Estimated upper bound at the configured output limit: {cost}.",
                cost=_format_usd_estimate(estimate["maximum_cost_usd"]),
            )
        )
    else:
        st.caption(
            _t(
                "No price is available for {model}. Add it to configs/model_pricing.yaml.",
                model=estimate["model"],
            )
        )
    st.caption(
        _t(
            "AI abstract screening will use batches of up to {batch_size} papers "
            "(about {requests} requests before retries).",
            batch_size=estimate["batch_size"],
            requests=estimate["estimated_requests"],
        )
    )
    st.caption(
        _t(
            "The estimate covers abstract screening only and excludes retries, cached-input "
            "discounts, and optional historical replay."
        )
    )
    if (use_llm or replay_pending) and not has_ai_key:
        st.warning(_t("Add an API key on the AI settings page before continuing."))
    button_label = (
        _t("Run AI screening and create review queue")
        if use_llm
        else _t("Run historical replay and create review queue")
        if replay_pending
        else _t("Create human-only review queue")
    )
    if st.button(
        button_label,
        type="primary",
        disabled=(use_llm or replay_pending) and not has_ai_key,
        key=f"prepare_review_{round_index}_{project.slug}",
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


def _render_manual_review(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
) -> None:
    state = service.current_state_or_none(project.slug)
    if not state:
        st.info(_t("Run initial discovery before opening the review workspace."))
        _render_manual_additions(store, service, project, embedded=True)
        return
    task = _task_manager().snapshot(project.slug)
    review_preparation_running = bool(
        task and task.running and task.operation == "review_preparation"
    )
    if review_preparation_running:
        _render_current_run_progress(
            project.slug,
            service,
            widget_scope="manual_review",
        )
        st.divider()
    latest_round = state["rounds"][-1]
    pending_review_round = bool(
        latest_round.get("kind") == "snowball"
        and latest_round.get("status") in {"discovery_complete", "citation_incomplete"}
        and latest_round.get("files", {}).get("enriched")
        and not latest_round.get("files", {}).get("audit")
    )
    if pending_review_round and not review_preparation_running:
        st.warning(
            _t(
                "Snowball round {round_index} has candidates that are not yet in a manual "
                "review round. Prepare the queue below.",
                round_index=latest_round.get("index", ""),
            )
        )
        _review_preparation_panel(store, service, project, latest_round)
        st.divider()
    review_rounds = [item for item in state["rounds"] if item.get("files", {}).get("audit")]
    if not review_rounds:
        if not pending_review_round and not review_preparation_running:
            st.info(_t("Prepare the current round for review in the Run center."))
        _render_manual_additions(store, service, project, embedded=True)
        return
    indexes = [int(item["index"]) for item in review_rounds]
    selection_key = f"audit_round_{state['run_id']}"
    latest_marker_key = f"latest_audit_round_{state['run_id']}"
    latest_audit_round = indexes[-1]
    if st.session_state.get(latest_marker_key) != latest_audit_round:
        st.session_state.pop(selection_key, None)
        st.session_state[latest_marker_key] = latest_audit_round
    round_index = st.selectbox(
        _t("Audit round"),
        indexes,
        index=len(indexes) - 1,
        key=selection_key,
    )
    round_state = next(item for item in review_rounds if int(item["index"]) == round_index)
    removed_duplicates = service.reconcile_snowball_audit(project.slug, round_index)
    if removed_duplicates:
        st.info(
            _t(
                "Removed {count} papers already reviewed in earlier rounds. A backup of the "
                "previous audit file was saved.",
                count=removed_duplicates,
            )
        )
    audit_path = Path(round_state["files"]["audit"])
    _, rows, summary = load_audit(audit_path)
    metric_slots = [column.empty() for column in st.columns(5)]
    _render_audit_metrics(metric_slots, summary)
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
            "snowball_seed_titles",
            "llm_decision",
            "llm_confidence",
            "llm_reason",
            "manual_decision",
            "manual_notes",
        ]
        if column in filtered.columns
        and (column != "impact_factor" or _has_known_impact_factor(filtered))
    ]
    editor_frame = filtered[visible_columns].copy()
    if "manual_decision" in editor_frame:
        editor_frame["manual_decision"] = editor_frame["manual_decision"].map(_decision_label)
    if "llm_decision" in editor_frame:
        editor_frame["llm_decision"] = editor_frame["llm_decision"].map(_decision_label)
    if "venue_type" in editor_frame:
        editor_frame["venue_type"] = editor_frame["venue_type"].map(_value_label)
    if "impact_factor" in editor_frame:
        editor_frame["impact_factor"] = editor_frame["impact_factor"].map(_impact_factor_text)
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
            "snowball_seed_titles": st.column_config.TextColumn(
                _t("Snowball seeds"),
                width="large",
            ),
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
    st.caption(_t("Review decisions are saved automatically after each committed cell edit."))
    updated_rows = _merge_editor_identity(filtered, edited)
    if _audit_rows_changed(filtered, updated_rows):
        saved_summary = service.update_audit(project.slug, round_index, updated_rows)
        _render_audit_metrics(metric_slots, saved_summary)
        st.toast(
            _t(
                "Saved {reviewed} reviewed papers; {unreviewed} still require a decision.",
                reviewed=saved_summary.reviewed,
                unreviewed=saved_summary.unreviewed,
            ),
            icon=":material/check:",
        )

    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            _t("Download this audit CSV"),
            data=audit_path.read_bytes(),
            file_name=audit_path.name,
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with download_columns[1]:
        _render_run_log_download(
            service,
            project.slug,
            key=f"audit_run_log_{state['run_id']}_{round_index}",
        )

    _render_prompt_refinement(store, service, project)

    with st.expander(_t("Paper reader")):
        _render_review_paper_reader(rows, frame, state, round_index)

    _render_manual_additions(
        store,
        service,
        project,
        embedded=True,
        target_round_index=round_index,
    )


def _render_review_paper_reader(
    rows: list[dict[str, str]],
    frame: pd.DataFrame,
    state: dict[str, Any],
    round_index: int,
) -> None:
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


def _render_audit_metrics(metric_slots: list[Any], summary: AuditSummary) -> None:
    metric_slots[0].metric(_t("Candidates"), summary.total)
    metric_slots[1].metric(_t("Reviewed"), summary.reviewed)
    metric_slots[2].metric(_t("Include"), summary.by_decision.get("include", 0))
    metric_slots[3].metric(_t("Related"), summary.by_decision.get("include_related", 0))
    metric_slots[4].metric(_t("Exclude"), summary.by_decision.get("exclude", 0))


def _render_prompt_refinement(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
    *,
    embedded: bool = False,
) -> None:
    if not embedded:
        with st.expander(_t("Refine screening prompt from human decisions")):
            _render_prompt_refinement_content(store, service, project, show_heading=False)
        return
    _render_prompt_refinement_content(store, service, project, show_heading=True)


def _render_prompt_refinement_content(
    store: ProjectStore,
    service: PipelineService,
    project: ProjectSettings,
    *,
    show_heading: bool,
) -> None:
    if show_heading:
        st.subheader(_t("Refine screening prompt from human decisions"))
    st.write(
        _t(
            "After each completed audit round, AI can learn from all human decisions and "
            "reviewer notes collected so far. The active prompt changes only after human "
            "approval."
        )
    )
    overview = service.prompt_refinement_overview(project.slug)
    metrics = st.columns(3)
    metrics[0].metric(_t("Audited papers"), overview["audit_total"])
    metrics[1].metric(_t("Audit rounds"), overview["audit_rounds"])
    metrics[2].metric(_t("Decisions remaining"), overview["unreviewed"])
    st.caption(
        _t(
            "Historical replay pool: {source} initial AI exclusions - {reviewed} already "
            "human-reviewed = {eligible} eligible papers.",
            source=(overview["initial_ai_exclusions"] + overview["reviewed_removed_before_replay"]),
            reviewed=overview["reviewed_removed_before_replay"],
            eligible=overview["initial_ai_exclusions"],
        )
    )
    st.caption(
        _t(
            "Prompt refinement model: {model}",
            model=project.prompt_refinement_model,
        )
    )
    if not overview["available"]:
        st.info(_t("Create and complete a manual review queue first."))
        return
    if overview["unreviewed"]:
        st.info(
            _t(
                "Finish all {count} remaining manual decisions before generating a prompt "
                "proposal.",
                count=overview["unreviewed"],
            )
        )
        return

    task = _task_manager().snapshot(project.slug)
    if task and task.running and task.operation == "prompt_refinement":
        _render_current_run_progress(
            project.slug,
            service,
            widget_scope="prompt_refinement",
        )
        return
    if task and task.error and task.operation == "prompt_refinement":
        st.error(_t("Prompt refinement failed: {error}", error=_runtime_text(task.error)))

    refinement = overview.get("refinement", {})
    status = str(refinement.get("status") or "not_generated")
    has_api_key = store.has_api_key(project.slug) or bool(os.environ.get("OPENAI_API_KEY"))
    if status == "proposed":
        try:
            proposal = service.load_prompt_refinement_proposal(project.slug)
            baseline_prompt = Path(proposal["baseline_prompt_path"]).read_text(encoding="utf-8")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(_runtime_text(str(exc)))
            return
        st.warning(_t("This proposal is waiting for human approval."))
        st.markdown(f"**{_t('Change summary')}**")
        st.write(proposal.get("change_summary") or _t("No summary was provided."))
        detail_columns = st.columns(3)
        _render_prompt_change_list(
            detail_columns[0],
            _t("Retained principles"),
            proposal.get("retained_principles", []),
        )
        _render_prompt_change_list(
            detail_columns[1],
            _t("New rules"),
            proposal.get("new_rules", []),
        )
        _render_prompt_change_list(
            detail_columns[2],
            _t("Risks to inspect"),
            proposal.get("risks", []),
        )
        st.caption(
            _t(
                "The proposal used {used} of {total} audited rows.",
                used=proposal.get("rows_used", 0),
                total=proposal.get("rows_total", 0),
            )
        )
        feedback_csv = refinement.get("feedback_csv_path")
        if feedback_csv and Path(feedback_csv).exists():
            st.download_button(
                _t("Download prompt feedback CSV"),
                data=Path(feedback_csv).read_bytes(),
                file_name=Path(feedback_csv).name,
                mime="text/csv",
                icon=":material/download:",
            )
        prompt_columns = st.columns(2)
        with prompt_columns[0]:
            st.text_area(
                _t("Current prompt used as baseline"),
                value=baseline_prompt,
                height=420,
                disabled=True,
                key=f"prompt_baseline_{refinement.get('refinement_id', '')}",
            )
        with prompt_columns[1]:
            proposed_prompt = st.text_area(
                _t("Proposed prompt (editable before approval)"),
                value=str(proposal.get("revised_prompt") or ""),
                height=420,
                key=f"prompt_proposal_{refinement.get('refinement_id', '')}",
            )
        approve_col, reject_col = st.columns(2)
        with approve_col:
            if st.button(
                _t("Approve and use this prompt"),
                type="primary",
                icon=":material/check:",
                width="stretch",
            ):
                try:
                    service.approve_prompt_refinement(project.slug, proposed_prompt)
                    st.rerun()
                except (OSError, RuntimeError, ValueError) as exc:
                    st.error(_runtime_text(str(exc)))
        with reject_col:
            if st.button(
                _t("Reject proposal"),
                icon=":material/close:",
                width="stretch",
            ):
                try:
                    service.reject_prompt_refinement(project.slug)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(_runtime_text(str(exc)))
        return

    if status == "approved":
        replay_status = str(refinement.get("replay_status") or "not_available")
        if replay_status == "completed":
            st.success(
                _t(
                    "The revised prompt is active. Historical AI exclusions were already "
                    "replayed once; later prompt updates apply only to newly discovered papers."
                )
            )
        else:
            st.success(
                _t(
                    "The revised prompt is active. The one-time historical replay can be "
                    "started during snowballing."
                )
            )
        st.caption(
            _t(
                "Replay status: {status}",
                status=_state_label(replay_status),
            )
        )
        approved_value = refinement.get("approved_prompt_path")
        if approved_value and Path(approved_value).exists():
            with st.expander(_t("View approved prompt")):
                st.code(Path(approved_value).read_text(encoding="utf-8"), language="text")

    if status == "rejected":
        st.info(_t("The previous prompt proposal was rejected."))
    if not has_api_key:
        st.warning(_t("Add an API key on the AI settings page before continuing."))
    if st.button(
        _t("Generate updated prompt proposal")
        if status == "approved"
        else _t("Generate prompt proposal"),
        type="primary",
        icon=":material/auto_awesome:",
        disabled=not has_api_key or bool(task and task.running),
    ):
        started = _task_manager().start(
            project.slug,
            "prompt_refinement",
            service.generate_prompt_refinement,
            project.slug,
        )
        if started:
            st.rerun()
        else:
            st.warning(_t("A pipeline task is already running for this project."))


def _render_prompt_change_list(container, title: str, values: object) -> None:
    with container:
        st.markdown(f"**{title}**")
        items = values if isinstance(values, list) else []
        if not items:
            st.caption(_t("None reported."))
            return
        for item in items:
            st.markdown(f"- {item}")


def _render_snowball(service: PipelineService, project: ProjectSettings) -> None:
    state = service.current_state_or_none(project.slug)
    if not state:
        st.info(_t("Complete the initial discovery and review first."))
        return
    st.subheader(_t("Current snowball state"))
    latest_state_round = state["rounds"][-1]
    state_metrics = st.columns(3)
    state_metrics[0].metric(_t("Round"), latest_state_round.get("index", 0))
    state_metrics[1].metric(
        _t("Status"), _state_label(str(latest_state_round.get("status") or "unknown"))
    )
    state_metrics[2].metric(
        _t("Review queue"), latest_state_round.get("counts", {}).get("audit_queue", 0)
    )
    with st.expander(_t("Show round history")):
        _render_round_overview(state)
    _render_run_log_download(
        service,
        project.slug,
        key=f"snowball_run_log_{state['run_id']}",
    )
    latest_snowball = next(
        (item for item in reversed(state.get("rounds", [])) if item.get("kind") == "snowball"),
        None,
    )
    if latest_snowball:
        provider_successes = latest_snowball.get("counts", {}).get(
            "provider_successes",
            {},
        )
        provider_failures = latest_snowball.get("counts", {}).get(
            "provider_failures",
            {},
        )
        providers = latest_snowball.get("counts", {}).get(
            "citation_providers",
            [],
        )
        if isinstance(providers, list) and providers:
            activity = "; ".join(
                _t(
                    "{provider}: {successes} successful calls, {failures} failures",
                    provider=SNOWBALL_PROVIDER_LABELS.get(provider, provider),
                    successes=(
                        provider_successes.get(provider, 0)
                        if isinstance(provider_successes, dict)
                        else 0
                    ),
                    failures=(
                        provider_failures.get(provider, 0)
                        if isinstance(provider_failures, dict)
                        else 0
                    ),
                )
                for provider in providers
            )
            st.caption(_t("Provider activity: {details}", details=activity))
            if isinstance(provider_failures, dict) and sum(
                int(value or 0) for value in provider_failures.values()
            ):
                st.warning(
                    _t(
                        "Some citation-provider requests failed. The successful results remain "
                        "valid, affected seed papers are marked, and this round can continue "
                        "to screening and manual review."
                    )
                )
        coverage_path_value = latest_snowball.get("files", {}).get("seed_coverage")
        if coverage_path_value and Path(coverage_path_value).exists():
            coverage_frame = pd.read_csv(coverage_path_value, keep_default_na=False)
            issue_frame = (
                coverage_frame[coverage_frame["coverage_status"] != "complete"]
                if "coverage_status" in coverage_frame
                else coverage_frame.iloc[0:0]
            )
            if not issue_frame.empty:
                with st.expander(
                    _t(
                        "Citation coverage warnings ({count} seed papers)",
                        count=len(issue_frame),
                    ),
                    expanded=True,
                ):
                    display_columns = [
                        column
                        for column in [
                            "seed_title",
                            "coverage_status",
                            "missing_providers",
                            "provider_errors",
                            "references_fetched",
                            "citations_fetched",
                        ]
                        if column in issue_frame.columns
                    ]
                    st.dataframe(
                        issue_frame[display_columns],
                        hide_index=True,
                        width="stretch",
                    )
                    st.download_button(
                        _t("Download seed coverage report"),
                        data=Path(coverage_path_value).read_bytes(),
                        file_name=Path(coverage_path_value).name,
                        mime="text/csv",
                        icon=":material/download:",
                    )
    task = _task_manager().snapshot(project.slug)
    if task and task.running and task.operation in {
        "snowball_discovery",
        "targeted_snowball_discovery",
        "review_preparation",
    }:
        _render_current_run_progress(
            project.slug,
            service,
            widget_scope="snowball",
        )
        return
    latest = state["rounds"][-1]
    if latest.get("status") == "converged":
        st.success(
            _t(
                "The incremental snowball has converged because no new papers entered the "
                "review queue. You can still run a single known paper below."
            )
        )
    failed_snowball = bool(
        latest.get("kind") == "snowball" and latest.get("status") in {"failed", "cancelled"}
    )
    retry_snowball = failed_snowball
    if failed_snowball:
        st.error(
            _t(
                "Snowball round {round_index} failed: {error}",
                round_index=latest.get("index", ""),
                error=_runtime_text(str(latest.get("error") or "Unknown error")),
            )
        )
        st.caption(
            _t(
                "Successful citation-provider responses are cached. Retrying reuses them "
                "and keeps the same snowball round."
            )
        )
        if "OpenAlex" in str(latest.get("error") or ""):
            st.link_button(
                _t("OpenAlex usage dashboard"),
                "https://openalex.org/settings/usage",
                icon=":material/open_in_new:",
            )
    if (
        latest.get("kind") == "snowball"
        and latest.get("status") in {"discovery_complete", "citation_incomplete"}
        and latest.get("files", {}).get("enriched")
        and not latest.get("files", {}).get("audit")
    ):
        st.info(
            _t(
                "Snowball discovery is complete. Create its review queue before starting "
                "another round."
            )
        )
        _review_preparation_panel(service.store, service, project, latest)
        return
    review_round = next(
        (
            item
            for item in reversed(state["rounds"])
            if item.get("files", {}).get("audit") and not (retry_snowball and item is latest)
        ),
        None,
    )
    if review_round is None:
        st.info(_t("Prepare the current round in the Run center before continuing."))
        return
    _, _, summary = load_audit(Path(review_round["files"]["audit"]))
    if summary.unreviewed:
        st.warning(
            _t(
                "Finish {count} remaining manual decisions before starting the next round.",
                count=summary.unreviewed,
            )
        )
    st.divider()
    st.subheader(_t("Next citation run"))
    with st.expander(_t("Citation and screening settings"), expanded=True):
        run_options = _render_snowball_run_settings(service, project, state)
    provider_selection_valid = run_options["provider_selection_valid"]
    has_api_key = run_options["has_api_key"]
    replay_overview = service.prompt_replay_overview(project.slug)
    replay_initial_exclusions = False
    if replay_overview["replay_status"] == "completed":
        st.success(
            _t(
                "Initial AI exclusions were replayed once in snowball round {round_index}. "
                "Later prompt updates will not re-run them.",
                round_index=replay_overview.get("replay_round", ""),
            )
        )
    elif replay_overview["approved"] and replay_overview["eligible"]:
        replay_initial_exclusions = st.toggle(
            _t("Re-screen initial AI exclusions with the approved prompt"),
            value=True,
            disabled=not has_api_key,
            key=f"snowball_prompt_replay_{state['run_id']}",
            help=_t(
                "Papers recovered as include, maybe, or failed enter the next manual review. "
                "Papers excluded again remain outside the review queue."
            ),
        )
        st.caption(
            _t(
                "Replay pool: {source} initial AI exclusions - {reviewed} human-reviewed "
                "papers = {eligible} papers sent to AI.",
                source=replay_overview["source_exclusions"],
                reviewed=replay_overview["reviewed_removed"],
                eligible=replay_overview["eligible"],
            )
        )
    elif replay_overview["eligible"]:
        st.info(
            _t(
                "Approve a refined prompt in Manual Review to re-screen "
                "{count} initial AI exclusions.",
                count=replay_overview["eligible"],
            )
        )
    standard_tab, targeted_tab, prompt_tab = st.tabs(
        [
            _t("Latest reviewed seeds"),
            _t("Single-paper snowball"),
            _t("Update AI prompt"),
        ]
    )
    common_disabled = (
        not provider_selection_valid
        or summary.unreviewed > 0
        or (replay_initial_exclusions and not has_api_key)
    )
    with standard_tab:
        st.write(
            _t(
                "Expand only the newly included or related papers from the latest completed "
                "manual review round."
            )
        )
        standard_disabled = common_disabled or (
            latest.get("status") == "converged" and not failed_snowball
        )
        if st.button(
            _t("Retry failed snowball round")
            if failed_snowball
            else _t("Start next snowball round"),
            type="primary",
            disabled=standard_disabled,
            icon=":material/account_tree:",
            key=f"start_incremental_snowball_{state['run_id']}",
        ):
            _start_snowball_task(
                service,
                project,
                run_options,
                replay_initial_exclusions=replay_initial_exclusions,
            )
        if latest.get("status") == "converged" and not failed_snowball:
            st.caption(
                _t(
                    "No unused seeds remain in the latest round. Use the single-paper tab "
                    "for a targeted recovery run."
                )
            )
    with targeted_tab:
        _render_single_paper_snowball(
            service,
            project,
            state,
            run_options,
            replay_initial_exclusions=replay_initial_exclusions,
            disabled=common_disabled or failed_snowball,
        )
        if failed_snowball:
            st.warning(_t("Retry or finish the failed round before starting a new target."))
    with prompt_tab:
        _render_prompt_refinement(
            service.store,
            service,
            project,
            embedded=True,
        )


def _render_snowball_run_settings(
    service: PipelineService,
    project: ProjectSettings,
    state: dict[str, Any],
) -> dict[str, Any]:
    st.markdown(f"**{_t('Citation providers')}**")
    st.caption(
        _t(
            "Providers are tried from left to right for each seed paper. A failed request is "
            "recorded on that seed, while later seed papers are still attempted."
        )
    )
    configured_snowball = load_config(service.store.config_path(project.slug)).snowballing
    saved_providers = state.get("options", {}).get(
        "snowball_citation_providers",
        configured_snowball.providers,
    )
    if not isinstance(saved_providers, list):
        saved_providers = configured_snowball.providers
    saved_slots = [
        *[provider for provider in saved_providers if provider in SNOWBALL_PROVIDER_OPTIONS][:3],
        "__disabled__",
        "__disabled__",
        "__disabled__",
    ][:3]
    provider_choices = ["__disabled__", *SNOWBALL_PROVIDER_OPTIONS]
    provider_columns = st.columns(3)
    selected_slots: list[str] = []
    for index, label in enumerate(
        [_t("Primary provider"), _t("Secondary provider"), _t("Tertiary provider")]
    ):
        current = saved_slots[index]
        with provider_columns[index]:
            selected_slots.append(
                st.selectbox(
                    label,
                    provider_choices,
                    index=provider_choices.index(current),
                    format_func=lambda value: _t(SNOWBALL_PROVIDER_LABELS[value]),
                    key=f"snowball_provider_{index}_{state['run_id']}",
                )
            )
    selected_providers = [provider for provider in selected_slots if provider != "__disabled__"]
    provider_selection_valid = bool(selected_providers) and len(selected_providers) == len(
        set(selected_providers)
    )
    if not selected_providers:
        st.error(_t("Select at least one citation provider."))
    elif len(selected_providers) != len(set(selected_providers)):
        st.error(_t("Each citation provider can appear only once."))

    saved_strategy = str(
        state.get("options", {}).get(
            "snowball_provider_strategy",
            configured_snowball.provider_strategy,
        )
    )
    if saved_strategy not in {"merge", "failover"}:
        saved_strategy = "merge"
    provider_strategy = st.radio(
        _t("Provider mode"),
        ["merge", "failover"],
        index=["merge", "failover"].index(saved_strategy),
        format_func=lambda value: _t("Merge coverage" if value == "merge" else "Failover only"),
        horizontal=True,
        key=f"snowball_provider_strategy_{state['run_id']}",
        help=_t(
            "Merge coverage queries every selected provider and unions the results. "
            "Failover only stops after the first successful provider."
        ),
    )
    if (
        "semantic_scholar" in selected_providers
        and not service.store.read_semantic_scholar_api_key(project.slug)
        and not os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    ):
        st.info(
            _t(
                "Semantic Scholar works without a key, but a personal key gives a more "
                "predictable rate limit."
            )
        )
    if (
        "openalex" in selected_providers
        and not service.store.read_openalex_api_key(project.slug)
        and not os.environ.get("OPENALEX_API_KEY")
    ):
        st.warning(_t("OpenAlex is selected but no OpenAlex API key is available."))

    st.markdown(f"**{_t('Collection limits')}**")
    complete_citations = st.toggle(
        _t("Retrieve all references and citing papers"),
        value=bool(state.get("options", {}).get("snowball_complete_citations", True)),
        key=f"snowball_complete_citations_{state['run_id']}",
        help=_t("Recommended for systematic reviews. Disable only to set per-seed safety limits."),
    )
    columns = st.columns(3)
    with columns[0]:
        backward = st.number_input(
            _t("Reference safety limit per seed"),
            min_value=1,
            max_value=100_000,
            value=max(
                int(state.get("options", {}).get("snowball_backward_limit", 500)),
                1,
            ),
            disabled=complete_citations,
            key=f"snowball_backward_{state['run_id']}",
        )
    with columns[1]:
        forward = st.number_input(
            _t("Citation safety limit per seed"),
            min_value=1,
            max_value=100_000,
            value=max(
                int(state.get("options", {}).get("snowball_forward_limit", 500)),
                1,
            ),
            disabled=complete_citations,
            key=f"snowball_forward_{state['run_id']}",
        )
    with columns[2]:
        abstract_limit = st.number_input(
            _t("Abstract limit"),
            min_value=0,
            value=int(state.get("options", {}).get("snowball_enrich_limit") or 0),
            help=_t("Use 0 for all new candidates."),
            key=f"snowball_abstract_limit_{state['run_id']}",
        )
    core_online = st.checkbox(
        _t("Look up CORE ranks online"),
        value=bool(state.get("options", {}).get("snowball_core_online", True)),
        key=f"snowball_core_online_{state['run_id']}",
    )

    st.markdown(f"**{_t('AI title screening')}**")
    has_api_key = service.store.has_api_key(project.slug) or bool(os.environ.get("OPENAI_API_KEY"))
    title_default = bool(state.get("options", {}).get("title_llm_enabled", has_api_key))
    title_columns = st.columns([2, 1])
    with title_columns[0]:
        use_title_llm = st.toggle(
            _t("Use AI title prescreen before abstract enrichment"),
            value=title_default and has_api_key,
            disabled=not has_api_key,
            key=f"snowball_title_llm_{state['run_id']}",
        )
    with title_columns[1]:
        title_batch_size = st.number_input(
            _t("Titles per AI batch"),
            min_value=10,
            max_value=200,
            value=int(state.get("options", {}).get("title_llm_batch_size", 100)),
            step=10,
            disabled=not use_title_llm,
            key=f"snowball_title_batch_{state['run_id']}",
        )
    return {
        "selected_providers": selected_providers,
        "provider_selection_valid": provider_selection_valid,
        "provider_strategy": provider_strategy,
        "complete_citations": complete_citations,
        "backward": int(backward),
        "forward": int(forward),
        "abstract_limit": int(abstract_limit),
        "core_online": bool(core_online),
        "has_api_key": has_api_key,
        "use_title_llm": bool(use_title_llm),
        "title_batch_size": int(title_batch_size),
    }


def _start_snowball_task(
    service: PipelineService,
    project: ProjectSettings,
    run_options: dict[str, Any],
    *,
    replay_initial_exclusions: bool,
    target_seed: dict[str, Any] | None = None,
) -> None:
    started = _task_manager().start(
        project.slug,
        "targeted_snowball_discovery" if target_seed else "snowball_discovery",
        service.start_snowball_discovery,
        project.slug,
        citation_providers=run_options["selected_providers"],
        provider_strategy=run_options["provider_strategy"],
        max_backward_per_seed=(0 if run_options["complete_citations"] else run_options["backward"]),
        max_forward_per_seed=(0 if run_options["complete_citations"] else run_options["forward"]),
        enrich_limit=_none_if_zero(run_options["abstract_limit"]),
        core_online=run_options["core_online"],
        use_title_llm=run_options["use_title_llm"],
        title_batch_size=run_options["title_batch_size"],
        replay_initial_exclusions=replay_initial_exclusions,
        target_seed=target_seed,
    )
    if started:
        st.rerun()
    else:
        st.warning(_t("A pipeline task is already running for this project."))


def _render_single_paper_snowball(
    service: PipelineService,
    project: ProjectSettings,
    state: dict[str, Any],
    run_options: dict[str, Any],
    *,
    replay_initial_exclusions: bool,
    disabled: bool,
) -> None:
    st.write(
        _t(
            "Resolve one known paper by title and collect only its references and citing "
            "papers. Existing audit decisions are removed before downstream screening."
        )
    )
    run_id = str(state["run_id"])
    title = st.text_input(
        _t("Paper title"),
        placeholder=_t("Enter the full or approximate title"),
        key=f"targeted_snowball_title_{run_id}",
    )
    catalog = _source_catalog()
    available_sources = catalog.available_source_ids()
    default_sources = [
        source for source in project.discovery_sources if source in available_sources
    ]
    lookup_sources = st.multiselect(
        _t("Metadata search sources"),
        available_sources,
        default=default_sources or available_sources[:1],
        format_func=lambda source_id: catalog.sources[source_id].label,
        key=f"targeted_snowball_sources_{run_id}",
    )
    matches_key = f"targeted_snowball_matches_{run_id}"
    query_key = f"targeted_snowball_query_{run_id}"
    errors_key = f"targeted_snowball_errors_{run_id}"
    if st.button(
        _t("Find paper"),
        icon=":material/search:",
        disabled=disabled or not title.strip() or not lookup_sources,
        key=f"targeted_snowball_find_{run_id}",
    ):
        with st.spinner(_t("Searching selected sources...")):
            matches, errors = search_title_candidates(
                title,
                lookup_sources,
                load_config(service.store.config_path(project.slug)),
                limit_per_source=5,
            )
        st.session_state[matches_key] = [record.to_dict(include_raw=True) for record in matches]
        st.session_state[query_key] = title.strip()
        st.session_state[errors_key] = errors

    current_query_matches = st.session_state.get(query_key) == title.strip()
    matches = (
        [PaperRecord.from_dict(value) for value in st.session_state.get(matches_key, [])]
        if current_query_matches
        else []
    )
    if current_query_matches:
        for source_id, error in st.session_state.get(errors_key, {}).items():
            source = catalog.sources.get(source_id)
            st.warning(
                _t(
                    "{source} could not be searched: {error}",
                    source=source.label if source else source_id,
                    error=error,
                )
            )

    target_seed: dict[str, Any] = {"title": title.strip()}
    if matches:
        match_index = st.selectbox(
            _t("Resolved paper"),
            range(len(matches)),
            format_func=lambda index: _manual_match_label(matches[index]),
            key=f"targeted_snowball_match_{run_id}",
        )
        selected = matches[match_index]
        target_seed = selected.to_row()
        st.caption(_paper_metadata(target_seed))
    elif current_query_matches:
        st.info(
            _t(
                "No metadata match was found. The citation providers can still resolve the "
                "exact title when the run starts."
            )
        )

    st.caption(
        _t(
            "If this paper was already reviewed, it will remain excluded from the new review "
            "queue; only previously unseen neighboring papers continue."
        )
    )
    if st.button(
        _t("Start single-paper snowball"),
        type="primary",
        icon=":material/manage_search:",
        disabled=disabled or not title.strip(),
        key=f"targeted_snowball_start_{run_id}",
    ):
        _start_snowball_task(
            service,
            project,
            run_options,
            replay_initial_exclusions=replay_initial_exclusions,
            target_seed=target_seed,
        )


def _render_results(service: PipelineService, project: ProjectSettings) -> None:
    state = service.current_state_or_none(project.slug)
    if not state:
        st.info(_t("No run is available yet."))
        return
    with st.expander(_t("Show round history")):
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
    summary_tab, papers_tab, downloads_tab = st.tabs(
        [_t("Summary"), _t("Included papers"), _t("Downloads")]
    )
    with summary_tab:
        _render_results_summary(audit_rounds, included, audited)
    with papers_tab:
        _render_included_papers(included)
    with downloads_tab:
        _render_result_downloads(included_path, audit_path, report_path)


def _render_results_summary(
    audit_rounds: list[dict[str, Any]],
    included: pd.DataFrame,
    audited: pd.DataFrame,
) -> None:
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


def _render_included_papers(included: pd.DataFrame) -> None:
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
        and (column != "impact_factor" or _has_known_impact_factor(included))
    ]
    display_frame = included[display_columns].copy()
    if "venue_type" in display_frame:
        display_frame["venue_type"] = display_frame["venue_type"].map(_value_label)
    if "manual_decision" in display_frame:
        display_frame["manual_decision"] = display_frame["manual_decision"].map(_decision_label)
    if "impact_factor" in display_frame:
        display_frame["impact_factor"] = display_frame["impact_factor"].map(
            _impact_factor_text
        )
    st.dataframe(
        display_frame,
        hide_index=True,
        width="stretch",
        height=420,
        column_config=_result_column_config(),
    )


def _render_result_downloads(
    included_path: Path,
    audit_path: Path,
    report_path: Path,
) -> None:
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
    progress_state = (service.current_state_or_none(project.slug) or state).get("progress", {})
    corpus_task_running = bool(
        task and task.running and task.operation == "corpus_analysis"
    )
    corpus_progress_visible = bool(
        progress_state.get("operation") == "Corpus analysis"
        and progress_state.get("status") in {"running", "failed", "cancelled"}
    )
    if corpus_task_running or corpus_progress_visible:
        _render_current_run_progress(
            project.slug,
            service,
            widget_scope="corpus_analysis",
        )
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
        classifications[["title", "year", "venue", "primary_category", "rationale"]],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "title": st.column_config.TextColumn(_t("Title"), width="large"),
            "year": st.column_config.TextColumn(_t("Year")),
            "venue": st.column_config.TextColumn(_t("Venue")),
            "primary_category": st.column_config.TextColumn(_t("Primary category")),
            "rationale": st.column_config.TextColumn(_t("Classification rationale"), width="large"),
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
                _t("Mode"): _state_label(item.get("snowball_mode", "")),
                _t("Status"): _state_label(item.get("status", "")),
                _t("Target paper"): counts.get("target_seed_title", ""),
                _t("Pool"): counts.get("pool_rows", counts.get("deduped_records", "")),
                _t("Added"): counts.get("added_rows", ""),
                _t("Manual added"): counts.get("manual_review_additions", ""),
                _t("Rule excluded"): counts.get("rule_excluded", ""),
                _t("Title kept"): counts.get("title_kept", ""),
                _t("Title excluded"): counts.get("title_excluded", ""),
                _t("Abstract lookups"): counts.get("abstracts_attempted", ""),
                _t("Coverage warnings"): counts.get("coverage_issue_seeds", ""),
                _t("Review queue"): counts.get("audit_queue", ""),
                _t("Reviewed"): counts.get("reviewed", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows, dtype=str), hide_index=True, width="stretch")


def _render_literature_flow(state: dict[str, Any]) -> None:
    rounds = [
        (round_state, round_flow_stages(round_state)) for round_state in state.get("rounds", [])
    ]
    rounds = [(round_state, stages) for round_state, stages in rounds if stages]
    if not rounds:
        return

    st.subheader(_t("Literature flow"))
    st.caption(
        _t(
            "The diagram shows key discovery, screening, manual-addition, and audit stages. "
            "Detailed stage counts remain available in the JSON download."
        )
    )
    for round_state, stages in rounds:
        round_index = int(round_state.get("index", 0))
        if len(rounds) > 1:
            st.markdown(
                f"**{_t('Round')} {round_index}: {_state_label(round_state.get('kind', ''))}**"
            )
        nodes: list[str] = []
        for index, stage in enumerate(stages):
            stage_type = str(stage.get("type") or "filter")
            retained = max(int(stage.get("retained") or 0), 0)
            excluded = max(int(stage.get("excluded") or 0), 0)
            details = stage.get("details") or {}
            detail_text = " · ".join(
                f"{escape(_t(str(key)))}: {escape(str(value))}"
                for key, value in details.items()
                if value not in (None, "")
            )
            if stage_type == "discovery":
                change_text = _t("identified")
            elif stage_type == "enrichment":
                change_text = _t("no papers removed")
            else:
                change_text = _t("{count} excluded", count=f"{excluded:,}")
            nodes.append(
                '<div class="survey-flow-node survey-flow-'
                f'{escape(stage_type)}">'
                f'<div class="survey-flow-label">{escape(_t(str(stage.get("label") or "")))}</div>'
                f'<div class="survey-flow-count">{retained:,}</div>'
                f'<div class="survey-flow-change">{escape(change_text)}</div>'
                + (f'<div class="survey-flow-detail">{detail_text}</div>' if detail_text else "")
                + "</div>"
            )
            loop_to = str(stage.get("loop_to") or "")
            if loop_to:
                target = next(
                    (candidate for candidate in stages if candidate.get("key") == loop_to),
                    {},
                )
                nodes.append(
                    '<div class="survey-flow-loop">&#8634; '
                    + escape(
                        _t(
                            "Returns to {stage}",
                            stage=_t(str(target.get("label") or loop_to)),
                        )
                    )
                    + "</div>"
                )
            elif index < len(stages) - 1:
                nodes.append('<div class="survey-flow-arrow" aria-hidden="true">&#8594;</div>')
        st.markdown(f'<div class="survey-flow">{"".join(nodes)}</div>', unsafe_allow_html=True)

    rate_limit_retries = sum(
        int(round_state.get("counts", {}).get("abstract_rate_limit_retries") or 0)
        for round_state, _ in rounds
    )
    rate_limit_wait = sum(
        float(round_state.get("counts", {}).get("abstract_rate_limit_wait_seconds") or 0)
        for round_state, _ in rounds
    )
    if rate_limit_retries:
        st.warning(
            _t(
                "Abstract enrichment was rate-limited: {retries} retries and {wait} seconds "
                "waiting. Completed results were preserved.",
                retries=f"{rate_limit_retries:,}",
                wait=f"{rate_limit_wait:.1f}",
            )
        )

    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            _t("Download flow diagram"),
            data=build_flow_svg(state),
            file_name=f"{state.get('run_id', 'run')}_flow_diagram.svg",
            mime="image/svg+xml",
            icon=":material/image:",
            width="stretch",
        )
    with download_columns[1]:
        st.download_button(
            _t("Download flow counts"),
            data=json.dumps(flow_summary_payload(state), ensure_ascii=False, indent=2),
            file_name=f"{state.get('run_id', 'run')}_flow_summary.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )


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
    selector_columns = st.columns([1, 2])
    with selector_columns[0]:
        domain_id = st.selectbox(
            _t("Research field"),
            list(catalog.profiles),
            format_func=lambda profile_id: catalog.profiles[profile_id].localized_label(language),
            key=domain_key,
            on_change=_reset_sources_for_domain,
            args=(domain_key, sources_key),
        )
    profile = catalog.profiles[domain_id]
    available_sources = catalog.available_source_ids()
    recommended = set(catalog.recommended_sources(domain_id))
    with selector_columns[0]:
        st.caption(profile.localized_description(language))
    with selector_columns[1]:
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
        dict.fromkeys([*selected_sources, *profile.recommended_sources, *profile.optional_sources])
    )
    with st.expander(_t("Coverage")):
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


def _render_manual_addition_feedback(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status == "added_to_review":
        st.success(
            _t(
                "Paper added directly to manual review round {round_index}.",
                round_index=result.get("round_index", ""),
            )
        )
    elif status == "queued_for_enrichment":
        st.success(
            _t(
                "Paper saved. Start the enrichment loop below to add it to review round "
                "{round_index}.",
                round_index=result.get("round_index", ""),
            )
        )
    elif status == "already_in_review":
        st.info(
            _t(
                "This paper is already present in manual review round {round_index}; no "
                "duplicate was created.",
                round_index=result.get("round_index", ""),
            )
        )
    elif status == "removed_from_review":
        st.success(
            _t(
                "The manual paper was removed from the saved collection and manual review; "
                "counts and the literature flow were updated."
            )
        )
    elif status == "removed_from_collection":
        st.success(_t("The manual paper was removed."))
    elif result.get("collection_added"):
        st.success(_t("Paper added to the manual collection."))
    else:
        st.info(_t("The existing manual record was updated instead of duplicated."))


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
        format_func=lambda value: _t("Custom model...") if value == CUSTOM_MODEL_OPTION else value,
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
def _render_current_run_progress(
    project_slug: str,
    service: PipelineService,
    *,
    widget_scope: str,
) -> None:
    with st.container(
        border=True,
        key=f"current_run_panel_{widget_scope}_{project_slug}",
    ):
        _render_current_run_progress_content(
            project_slug,
            service,
            widget_scope=widget_scope,
        )


def _render_current_run_progress_content(
    project_slug: str,
    service: PipelineService,
    *,
    widget_scope: str,
) -> None:
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
    _render_run_log_download(
        service,
        project_slug,
        key=f"run_log_{widget_scope}_{project_slug}",
    )

    if task and task.running:
        if task.cancel_requested:
            st.button(
                _t("Stopping..."),
                icon=":material/progress_activity:",
                disabled=True,
                key=f"stop_run_{widget_scope}_{project_slug}",
            )
            st.caption(_t("The current item will finish safely before the run stops."))
        elif st.button(
            _t("Stop run"),
            icon=":material/stop_circle:",
            type="secondary",
            key=f"stop_run_{widget_scope}_{project_slug}",
        ):
            _task_manager().cancel(project_slug)
            st.rerun()
    elif task and task.can_restart and (task.cancelled or task.error):
        can_resume_initial = operation in {"Initial discovery", "Resume initial discovery"}
        if st.button(
            _t("Resume run") if can_resume_initial else _t("Run again"),
            icon=":material/replay:",
            type="primary",
            key=f"restart_run_{widget_scope}_{project_slug}",
        ):
            restarted = (
                _task_manager().start(
                    project_slug,
                    "resume_initial_discovery",
                    service.resume_initial_discovery,
                    project_slug,
                )
                if can_resume_initial
                else _task_manager().restart(project_slug)
            )
            if restarted:
                st.rerun()
            else:
                st.warning(_t("The previous task is still stopping. Please wait."))
    elif progress_status == "cancelled" and operation in {
        "Initial discovery",
        "Resume initial discovery",
    }:
        if st.button(
            _t("Resume run"),
            icon=":material/replay:",
            type="primary",
            key=f"resume_saved_run_{widget_scope}_{project_slug}",
        ):
            if _task_manager().start(
                project_slug,
                "resume_initial_discovery",
                service.resume_initial_discovery,
                project_slug,
            ):
                st.rerun()

    if not (task and task.running):
        if progress_status == "running":
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

    marker_key = f"progress_status_{widget_scope}_{project_slug}_{state.get('run_id', '')}"
    previous_status = st.session_state.get(marker_key)
    st.session_state[marker_key] = progress_status
    if previous_status == "running" and progress_status != "running":
        st.rerun()


def _render_run_log_download(
    service: PipelineService,
    project_slug: str,
    *,
    key: str,
) -> None:
    try:
        log_path = service.run_log_path(project_slug)
    except (FileNotFoundError, OSError, ValueError):
        return
    st.download_button(
        _t("Export run log"),
        data=log_path.read_bytes(),
        file_name=f"{log_path.parent.name}_run_log.json",
        mime="application/json",
        icon=":material/download:",
        key=key,
        width="content",
    )


def _current_paper_count(state: dict[str, Any]) -> int:
    progress = state.get("progress", {})
    stage = str(progress.get("stage") or "")
    stage_total = progress.get("total")
    if (
        progress.get("status") == "running"
        and stage not in {"Literature search", "Citation snowballing"}
        and stage_total not in (None, "")
    ):
        return max(int(stage_total), 0)
    progress_count = progress.get("paper_count")
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
    if translated != normalized:
        return translated
    label = normalized[:1].upper() + normalized[1:]
    return " ".join(
        word.upper() if word.lower() in {"ai", "llm", "api"} else word
        for word in label.split()
    )


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


def _format_usd_estimate(value: float | None) -> str:
    if value is None:
        return _t("Not available")
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _citation_failure_count(round_state: dict[str, Any]) -> int:
    values = round_state.get("counts", {}).get("provider_failures", {}) or {}
    if not isinstance(values, dict):
        return 0
    return sum(max(int(value or 0), 0) for value in values.values())


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


def _audit_rows_changed(
    original: pd.DataFrame,
    updates: list[dict[str, str]],
) -> bool:
    for position, update in enumerate(updates):
        original_row = original.iloc[position]
        if (
            str(original_row.get("manual_decision") or "").strip().lower()
            != str(update.get("manual_decision") or "").strip().lower()
        ):
            return True
        if (
            str(original_row.get("manual_notes") or "").strip()
            != str(update.get("manual_notes") or "").strip()
        ):
            return True
    return False


def _paper_metadata(row: dict[str, Any]) -> str:
    authors = _display_text(row.get("authors"))
    year = _display_text(row.get("year"))
    venue = _display_text(row.get("venue"))
    core_rank = _display_text(row.get("core_rank"))
    impact_factor = _impact_factor_text(row.get("impact_factor"))
    values = [
        authors,
        year,
        venue,
        f"CORE {core_rank}" if core_rank else "",
        f"IF {impact_factor}" if impact_factor else "",
    ]
    return " · ".join(value for value in values if value)


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _impact_factor_text(value: Any) -> str:
    text = _display_text(value)
    if text.casefold() in {
        "",
        "-",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
        "unknown",
        "not found",
        "not available",
        "unavailable",
    }:
        return ""
    try:
        if float(text) <= 0:
            return ""
    except ValueError:
        pass
    return text


def _has_known_impact_factor(frame: pd.DataFrame) -> bool:
    return "impact_factor" in frame and any(
        _impact_factor_text(value) for value in frame["impact_factor"]
    )


def _format_bytes(value: Any) -> str:
    size = max(float(value or 0), 0.0)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return "0 B"


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --sf-bg: #fbfbfd;
            --sf-surface: #ffffff;
            --sf-subtle: #f5f5f7;
            --sf-border: #d9d9de;
            --sf-border-soft: #e8e8ed;
            --sf-text: #1d1d1f;
            --sf-muted: #6e6e73;
            --sf-blue: #006edb;
            --sf-blue-hover: #0062c4;
            --sf-green: #16805f;
            --sf-red: #c94b4b;
            --sf-focus: rgba(0, 110, 219, 0.18);
        }
        html, body, [data-testid="stAppViewContainer"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
                sans-serif;
        }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: var(--sf-bg);
            color: var(--sf-text);
        }
        [data-testid="stHeader"] {background: rgba(251, 251, 253, 0.88);}
        .block-container {
            max-width: 1240px;
            padding: 2.75rem 2.75rem 5rem;
        }
        .sf-brand {padding: 0.25rem 0 1.35rem;}
        .sf-brand-name {
            color: var(--sf-text);
            font-size: 1.24rem;
            font-weight: 680;
            line-height: 1.2;
        }
        .sf-brand-caption {
            color: var(--sf-muted);
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.28rem;
        }
        .sf-nav-label, .sf-page-context {
            color: var(--sf-muted);
            font-size: 0.69rem;
            font-weight: 650;
            line-height: 1.3;
            text-transform: uppercase;
        }
        .sf-nav-label {margin: 0.2rem 0 0.45rem;}
        .sf-page-context {
            margin-bottom: 0.45rem;
            overflow-wrap: anywhere;
        }
        .sf-page-description {
            color: var(--sf-muted);
            font-size: 0.98rem;
            line-height: 1.55;
            max-width: 760px;
            margin: -0.45rem 0 2.15rem;
        }
        .sf-sidebar-status {
            align-items: center;
            color: var(--sf-muted);
            display: flex;
            font-size: 0.82rem;
            font-weight: 570;
            gap: 0.52rem;
            line-height: 1.3;
            padding: 0.22rem 0;
        }
        .sf-sidebar-status span {
            background: #9a9aa0;
            border-radius: 50%;
            display: inline-block;
            height: 7px;
            width: 7px;
        }
        .sf-sidebar-status-active span {background: var(--sf-green);}
        .sf-footer {
            align-items: center;
            border-top: 1px solid var(--sf-border-soft);
            color: var(--sf-muted);
            display: flex;
            font-size: 0.72rem;
            gap: 0.55rem;
            justify-content: space-between;
            line-height: 1.4;
            margin-top: 4rem;
            padding-top: 1rem;
        }
        h1, h2, h3, p, label, button {letter-spacing: 0 !important;}
        h1 {
            color: var(--sf-text) !important;
            font-size: 2.05rem !important;
            font-weight: 660 !important;
            line-height: 1.14 !important;
            margin: 0 !important;
        }
        h2 {
            color: var(--sf-text) !important;
            font-size: 1.15rem !important;
            font-weight: 640 !important;
            line-height: 1.32 !important;
            margin-top: 1.8rem !important;
        }
        h3 {
            color: var(--sf-text) !important;
            font-size: 1rem !important;
            font-weight: 630 !important;
            line-height: 1.35 !important;
        }
        p, label {line-height: 1.48;}
        [data-testid="stCaptionContainer"] {color: var(--sf-muted);}
        [data-testid="stSidebar"] {
            background: var(--sf-subtle);
            border-right: 1px solid var(--sf-border-soft);
        }
        [data-testid="stSidebarContent"] {padding-top: 1.5rem;}
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {gap: 0.14rem;}
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border: 1px solid transparent;
            border-radius: 7px;
            color: #4a4a4f;
            margin: 0;
            min-height: 2.25rem;
            padding: 0.48rem 0.62rem;
            transition: background-color 120ms ease, border-color 120ms ease;
            width: 100%;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.68);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: var(--sf-surface);
            border-color: var(--sf-border-soft);
            color: var(--sf-text);
            font-weight: 620;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.035);
        }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]
            > div > div > div:first-child {
            display: none;
        }
        [data-testid="stSidebar"] hr {border-color: var(--sf-border-soft);}
        [data-testid="stMetric"] {
            background: var(--sf-subtle);
            border: 1px solid transparent;
            border-radius: 7px;
            min-height: 78px;
            padding: 0.72rem 0.82rem;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--sf-muted) !important;
            font-size: 0.72rem !important;
            font-weight: 560;
            line-height: 1.2;
        }
        [data-testid="stMetricValue"] {
            color: var(--sf-text) !important;
            font-size: 1.02rem !important;
            font-weight: 640;
            line-height: 1.22;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricValue"] > div {font-size: inherit !important; line-height: inherit;}
        [data-testid="stForm"] {
            background: var(--sf-surface);
            border: 1px solid var(--sf-border-soft);
            border-radius: 8px;
            padding: 1.15rem 1.2rem 1.25rem;
        }
        [data-testid="stExpander"] {
            background: transparent;
            border-color: var(--sf-border-soft);
            border-radius: 7px;
            box-shadow: none;
        }
        [data-testid="stExpander"] summary:hover {background: var(--sf-subtle);}
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border: 1px solid var(--sf-border-soft);
            border-radius: 7px;
            overflow: hidden;
        }
        [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--sf-border-soft);
            gap: 1.25rem;
        }
        [data-baseweb="tab"] {
            color: var(--sf-muted);
            font-size: 0.86rem;
            font-weight: 560;
            min-height: 2.65rem;
            padding-left: 0;
            padding-right: 0;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: var(--sf-text);
            font-weight: 630;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
            background: var(--sf-surface);
            border-color: var(--sf-border);
            border-radius: 7px;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--sf-blue);
            box-shadow: 0 0 0 3px var(--sf-focus);
        }
        [data-testid="stBaseButton-primary"] {
            background: var(--sf-blue);
            border-color: var(--sf-blue);
            border-radius: 7px;
            box-shadow: none;
            font-weight: 590;
        }
        [data-testid="stBaseButton-primary"]:hover {
            background: var(--sf-blue-hover);
            border-color: var(--sf-blue-hover);
        }
        [data-testid="stBaseButton-secondary"],
        [data-testid="stDownloadButton"] button,
        [data-testid="stLinkButton"] a {
            background: var(--sf-surface);
            border-color: var(--sf-border);
            border-radius: 7px;
            box-shadow: none;
            color: var(--sf-text);
            font-weight: 560;
        }
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stLinkButton"] a:hover {
            background: var(--sf-subtle);
            border-color: #bdbdc3;
            color: var(--sf-text);
        }
        [data-testid="stAlert"] {border-radius: 7px;}
        [data-testid="stProgressBar"] > div > div {background: var(--sf-blue);}
        [data-testid="stAppDeployButton"] {display: none;}
        .survey-flow {
            display: flex;
            align-items: center;
            gap: 7px;
            overflow-x: auto;
            padding: 0.25rem 0.1rem 0.8rem;
        }
        .survey-flow-node {
            background: var(--sf-surface);
            border: 1px solid var(--sf-border-soft);
            border-top: 3px solid var(--sf-blue);
            border-radius: 7px;
            flex: 0 0 152px;
            min-height: 112px;
            padding: 0.72rem;
        }
        .survey-flow-filter {border-top-color: var(--sf-red);}
        .survey-flow-enrichment {border-top-color: var(--sf-green);}
        .survey-flow-review {border-top-color: #827027;}
        .survey-flow-label {font-size: 0.76rem; font-weight: 630; line-height: 1.25;}
        .survey-flow-count {
            font-size: 1.28rem;
            font-weight: 660;
            line-height: 1.25;
            margin-top: 0.45rem;
        }
        .survey-flow-change {font-size: 0.7rem; color: var(--sf-muted); margin-top: 0.1rem;}
        .survey-flow-detail {
            color: var(--sf-muted);
            font-size: 0.67rem;
            line-height: 1.3;
            margin-top: 0.45rem;
        }
        .survey-flow-arrow {flex: 0 0 auto; color: #9a9aa0; font-size: 1rem;}
        .survey-flow-loop {
            flex: 0 0 132px;
            color: var(--sf-green);
            font-size: 0.72rem;
            font-weight: 620;
            line-height: 1.3;
        }
        @media (max-width: 800px) {
            .block-container {padding: 3.75rem 1rem 3.5rem;}
            h1 {font-size: 1.75rem !important;}
            .sf-page-description {font-size: 0.92rem; margin-bottom: 1.6rem;}
            [data-testid="stMetric"] {min-height: 72px; padding: 0.62rem 0.68rem;}
            [data-testid="stMetricValue"] {font-size: 0.94rem !important;}
            [data-baseweb="tab-list"] {gap: 0.45rem; overflow-x: auto;}
            [data-baseweb="tab"] {font-size: 0.78rem;}
            .sf-footer {align-items: flex-start; flex-direction: column; margin-top: 3rem;}
            .survey-flow-node {flex-basis: 142px;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
