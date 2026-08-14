from __future__ import annotations

import hashlib
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

import requests

from vnn_survey.app.task_manager import cancellable_sleep, raise_if_cancelled
from vnn_survey.config import DiscoveryConfig, SurveyConfig, YearRange
from vnn_survey.dblp import DblpClient
from vnn_survey.dblp_sparql import DblpSparqlClient
from vnn_survey.models import PaperRecord, normalize_title

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass(frozen=True, slots=True)
class SourceSearchResult:
    records: list[PaperRecord]
    fallback_error: str = ""


class SearchProvider(Protocol):
    source_id: str

    def search(self, query: str, limit: int | None = None) -> SourceSearchResult: ...


class DblpProvider:
    source_id = "dblp"

    def __init__(self, config: SurveyConfig, mode: str = "auto") -> None:
        if mode not in {"auto", "api", "sparql"}:
            raise ValueError("DBLP source must be auto, api, or sparql.")
        self.config = config
        self.mode = mode

    def search(self, query: str, limit: int | None = None) -> SourceSearchResult:
        dblp_config = self.config.dblp
        if limit is not None:
            dblp_config = replace(
                dblp_config,
                hits_per_page=max(1, min(limit, 1000)),
                max_pages_per_query=1,
            )
        api = DblpClient(dblp_config)
        sparql = DblpSparqlClient(dblp_config)
        if self.mode == "api":
            return SourceSearchResult(api.search(query)[:limit] if limit else api.search(query))
        if self.mode == "sparql":
            records = sparql.search(query)
            return SourceSearchResult(records[:limit] if limit else records)
        try:
            records = api.search(query)
            return SourceSearchResult(records[:limit] if limit else records)
        except Exception as exc:  # noqa: BLE001 - DBLP fallback is intentional.
            records = sparql.search(query)
            return SourceSearchResult(
                records[:limit] if limit else records,
                fallback_error=str(exc),
            )


class OpenAlexProvider:
    source_id = "openalex"

    def __init__(self, config: DiscoveryConfig, years: YearRange) -> None:
        self.config = config
        self.years = years
        self.session = _session("SurveyFlow/0.1 OpenAlex literature discovery")
        if not os.environ.get(self.config.openalex_api_key_env, "").strip():
            raise RuntimeError(
                "OpenAlex requires an API key for sustained use. Add a free key on AI settings."
            )

    def search(self, query: str, limit: int | None = None) -> SourceSearchResult:
        records: list[PaperRecord] = []
        api_key = os.environ.get(self.config.openalex_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                "OpenAlex requires an API key for sustained use. Add a free key on AI settings."
            )
        per_page, max_pages = _page_limits(self.config, limit, provider_cap=200)
        for page in range(1, max_pages + 1):
            params: dict[str, Any] = {
                "search": _portable_query(query),
                "per-page": per_page,
                "page": page,
            }
            filters = _date_filters(self.years)
            if filters:
                params["filter"] = ",".join(filters)
            email = os.environ.get(self.config.openalex_email_env, "").strip()
            if api_key:
                params["api_key"] = api_key
            if email:
                params["mailto"] = email
            payload = _request_json(
                self.session,
                OPENALEX_WORKS_URL,
                params,
                self.config,
                self.source_id,
            )
            items = payload.get("results", [])
            if not isinstance(items, list) or not items:
                break
            records.extend(_parse_openalex_work(item, query) for item in items)
            if limit and len(records) >= limit:
                return SourceSearchResult(records[:limit])
            if len(items) < per_page:
                break
            cancellable_sleep(self.config.request_delay_seconds)
        return SourceSearchResult(records)


class CrossrefProvider:
    source_id = "crossref"

    def __init__(self, config: DiscoveryConfig, years: YearRange) -> None:
        self.config = config
        self.years = years
        email = os.environ.get(config.crossref_email_env, "").strip()
        agent = "SurveyFlow/0.1 scholarly literature discovery"
        if email:
            agent = f"{agent} (mailto:{email})"
        self.session = _session(agent)

    def search(self, query: str, limit: int | None = None) -> SourceSearchResult:
        records: list[PaperRecord] = []
        per_page, max_pages = _page_limits(self.config, limit, provider_cap=1000)
        for page in range(max_pages):
            params: dict[str, Any] = {
                "query.bibliographic": _portable_query(query),
                "rows": per_page,
                "offset": page * per_page,
            }
            filters = _crossref_date_filters(self.years)
            if filters:
                params["filter"] = ",".join(filters)
            payload = _request_json(
                self.session,
                CROSSREF_WORKS_URL,
                params,
                self.config,
                self.source_id,
            )
            items = payload.get("message", {}).get("items", [])
            if not isinstance(items, list) or not items:
                break
            records.extend(_parse_crossref_work(item, query) for item in items)
            if limit and len(records) >= limit:
                return SourceSearchResult(records[:limit])
            if len(items) < per_page:
                break
            cancellable_sleep(self.config.request_delay_seconds)
        return SourceSearchResult(records)


class ArxivProvider:
    source_id = "arxiv"

    def __init__(self, config: DiscoveryConfig, years: YearRange) -> None:
        self.config = config
        self.years = years
        self.session = _session("SurveyFlow/0.1 arXiv literature discovery")

    def search(self, query: str, limit: int | None = None) -> SourceSearchResult:
        records: list[PaperRecord] = []
        per_page, max_pages = _page_limits(self.config, limit, provider_cap=2000)
        for page in range(max_pages):
            params = {
                "search_query": f"all:{_arxiv_query(query)}",
                "start": page * per_page,
                "max_results": per_page,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            payload = _request_text(
                self.session,
                ARXIV_API_URL,
                params,
                self.config,
                self.source_id,
                suffix="xml",
            )
            items = _parse_arxiv_feed(payload, query)
            if not items:
                break
            records.extend(record for record in items if self.years.contains(record.year))
            if limit and len(records) >= limit:
                return SourceSearchResult(records[:limit])
            if len(items) < per_page:
                break
            cancellable_sleep(self.config.arxiv_request_delay_seconds)
        return SourceSearchResult(records)


class PubmedProvider:
    source_id = "pubmed"

    def __init__(self, config: DiscoveryConfig, years: YearRange) -> None:
        self.config = config
        self.years = years
        self.session = _session("SurveyFlow/0.1 PubMed literature discovery")

    def search(self, query: str, limit: int | None = None) -> SourceSearchResult:
        records: list[PaperRecord] = []
        per_page, max_pages = _page_limits(self.config, limit, provider_cap=200)
        term = _pubmed_query(query, self.years)
        common = {"db": "pubmed", "retmode": "json"}
        api_key = os.environ.get(self.config.pubmed_api_key_env, "").strip()
        email = os.environ.get(self.config.pubmed_email_env, "").strip()
        if api_key:
            common["api_key"] = api_key
        if email:
            common["email"] = email
        common["tool"] = "surveyflow"

        for page in range(max_pages):
            search_params = {
                **common,
                "term": term,
                "retstart": page * per_page,
                "retmax": per_page,
            }
            payload = _request_json(
                self.session,
                PUBMED_ESEARCH_URL,
                search_params,
                self.config,
                f"{self.source_id}-search",
            )
            identifiers = payload.get("esearchresult", {}).get("idlist", [])
            if not identifiers:
                break
            fetch_params = {
                **common,
                "id": ",".join(str(item) for item in identifiers),
                "retmode": "xml",
            }
            fetched = _request_text(
                self.session,
                PUBMED_EFETCH_URL,
                fetch_params,
                self.config,
                f"{self.source_id}-fetch",
                suffix="xml",
            )
            records.extend(_parse_pubmed_xml(fetched, query))
            if limit and len(records) >= limit:
                return SourceSearchResult(records[:limit])
            if len(identifiers) < per_page:
                break
            cancellable_sleep(_pubmed_delay(self.config, bool(api_key)))
        return SourceSearchResult(records)


def create_provider(
    source_id: str,
    config: SurveyConfig,
    *,
    dblp_mode: str = "auto",
) -> SearchProvider:
    providers: dict[str, type[Any]] = {
        "openalex": OpenAlexProvider,
        "crossref": CrossrefProvider,
        "arxiv": ArxivProvider,
        "pubmed": PubmedProvider,
    }
    if source_id == "dblp":
        return DblpProvider(config, dblp_mode)
    provider_type = providers.get(source_id)
    if provider_type is None:
        raise ValueError(f"Unsupported discovery source: {source_id}")
    return provider_type(config.discovery, config.years)


def search_title_candidates(
    title: str,
    source_ids: Iterable[str],
    config: SurveyConfig,
    *,
    dblp_mode: str = "auto",
    limit_per_source: int = 5,
) -> tuple[list[PaperRecord], dict[str, str]]:
    target = title.strip()
    if not target:
        raise ValueError("Enter a paper title before searching.")
    selected_sources = list(source_ids)
    lookup_config = replace(config, years=YearRange())
    candidates: list[PaperRecord] = []
    errors: dict[str, str] = {}
    for source_id in selected_sources:
        try:
            lookup_limit = max(limit_per_source * 4, 20)
            result = create_provider(source_id, lookup_config, dblp_mode=dblp_mode).search(
                target,
                limit=lookup_limit,
            )
            source_candidates = [
                record
                for record in result.records
                if _title_similarity(target, record.title) >= 0.55
            ]
            source_candidates.sort(
                key=lambda record: _title_similarity(target, record.title),
                reverse=True,
            )
            candidates.extend(source_candidates[:limit_per_source])
        except Exception as exc:  # noqa: BLE001 - show partial provider results.
            errors[source_id] = str(exc)
    candidates = _dedupe_title_candidates(candidates)
    candidates.sort(key=lambda record: _title_similarity(target, record.title), reverse=True)
    return candidates[: max(limit_per_source * max(len(selected_sources), 1), 1)], errors


def _parse_openalex_work(item: dict[str, Any], query: str) -> PaperRecord:
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    doi = _clean_doi(item.get("doi"))
    return PaperRecord(
        title=str(item.get("display_name") or item.get("title") or "").strip(),
        source="openalex",
        query=query,
        year=_to_int(item.get("publication_year")),
        authors=[
            str(authorship.get("author", {}).get("display_name") or "").strip()
            for authorship in item.get("authorships", [])
            if isinstance(authorship, dict)
            and str(authorship.get("author", {}).get("display_name") or "").strip()
        ],
        venue=str(source.get("display_name") or "").strip() or None,
        doi=doi,
        url=str(
            primary_location.get("landing_page_url")
            or item.get("doi")
            or item.get("id")
            or ""
        ).strip()
        or None,
        publication_type=str(item.get("type_crossref") or item.get("type") or "").strip()
        or None,
        provider_id=_provider_id(item.get("id")),
        abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")) or None,
        abstract_source="openalex"
        if item.get("abstract_inverted_index")
        else None,
        raw=item,
    )


def _parse_crossref_work(item: dict[str, Any], query: str) -> PaperRecord:
    authors = []
    for author in item.get("author", []):
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part
            for part in [
                str(author.get("given") or "").strip(),
                str(author.get("family") or "").strip(),
            ]
            if part
        )
        if name:
            authors.append(name)
    doi = _clean_doi(item.get("DOI"))
    return PaperRecord(
        title=_first_text(item.get("title")),
        source="crossref",
        query=query,
        year=_crossref_year(item),
        authors=authors,
        venue=_first_text(item.get("container-title")) or None,
        doi=doi,
        url=str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
        or None,
        publication_type=str(item.get("subtype") or item.get("type") or "").strip() or None,
        provider_id=doi,
        abstract=_crossref_abstract(item.get("abstract")) or None,
        abstract_source="crossref" if item.get("abstract") else None,
        raw=item,
    )


def _parse_arxiv_feed(payload: str, query: str) -> list[PaperRecord]:
    root = ET.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", namespace):
        identifier = _element_text(entry.find("atom:id", namespace))
        provider_id = identifier.rstrip("/").rsplit("/", 1)[-1]
        published = _element_text(entry.find("atom:published", namespace))
        abstract = _collapse(_element_text(entry.find("atom:summary", namespace)))
        links = {
            str(link.attrib.get("rel") or ""): str(link.attrib.get("href") or "")
            for link in entry.findall("atom:link", namespace)
        }
        records.append(
            PaperRecord(
                title=_collapse(_element_text(entry.find("atom:title", namespace))),
                source="arxiv",
                query=query,
                year=_to_int(published[:4]),
                authors=[
                    _collapse(_element_text(author.find("atom:name", namespace)))
                    for author in entry.findall("atom:author", namespace)
                ],
                venue="arXiv",
                doi=_clean_doi(_element_text(entry.find("arxiv:doi", namespace))),
                url=links.get("alternate") or identifier or None,
                publication_type="preprint",
                provider_id=provider_id or None,
                abstract=abstract or None,
                abstract_source="arxiv" if abstract else None,
                raw={"id": identifier, "published": published},
            )
        )
    return records


def _parse_pubmed_xml(payload: str, query: str) -> list[PaperRecord]:
    root = ET.fromstring(payload)
    records: list[PaperRecord] = []
    for article_node in root.findall(".//PubmedArticle"):
        citation = article_node.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        if citation is None or article is None:
            continue
        pmid = _element_text(citation.find("PMID"))
        journal = article.find("Journal")
        venue = _element_text(journal.find("Title")) if journal is not None else ""
        abstract = _pubmed_abstract(article)
        records.append(
            PaperRecord(
                title=_collapse(_element_text(article.find("ArticleTitle"))),
                source="pubmed",
                query=query,
                year=_pubmed_year(article),
                authors=_pubmed_authors(article),
                venue=venue or None,
                doi=_pubmed_doi(article_node),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                publication_type=_element_text(
                    article.find("PublicationTypeList/PublicationType")
                )
                or "journal article",
                provider_id=pmid or None,
                abstract=abstract or None,
                abstract_source="pubmed" if abstract else None,
                raw={"pmid": pmid},
            )
        )
    return records


def _reconstruct_openalex_abstract(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            try:
                words.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue
    return " ".join(word for _, word in sorted(words)).strip()


def _crossref_abstract(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        root = ET.fromstring(f"<root>{text}</root>")
        return _collapse(" ".join(root.itertext()))
    except ET.ParseError:
        return _collapse(html.unescape(re.sub(r"<[^>]+>", " ", text)))


def _pubmed_abstract(article: ET.Element) -> str:
    sections: list[str] = []
    for node in article.findall("Abstract/AbstractText"):
        text = _collapse(" ".join(node.itertext()))
        if not text:
            continue
        label = str(node.attrib.get("Label") or "").strip()
        sections.append(f"{label}: {text}" if label else text)
    return " ".join(sections).strip()


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    config: DiscoveryConfig,
    namespace: str,
) -> dict[str, Any]:
    cache_path = _cache_path(config.cache_dir, namespace, url, params, "json")
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            raise_if_cancelled()
            response = session.get(url, params=params, timeout=config.timeout_seconds)
            raise_if_cancelled()
            response.raise_for_status()
            payload = response.json()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < config.retries:
                cancellable_sleep(config.request_delay_seconds * attempt)
    raise RuntimeError(f"{namespace} request failed") from last_error


def _request_text(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    config: DiscoveryConfig,
    namespace: str,
    *,
    suffix: str,
) -> str:
    cache_path = _cache_path(config.cache_dir, namespace, url, params, suffix)
    if cache_path and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            raise_if_cancelled()
            response = session.get(url, params=params, timeout=config.timeout_seconds)
            raise_if_cancelled()
            response.raise_for_status()
            payload = response.text
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(payload, encoding="utf-8")
            return payload
        except requests.RequestException as exc:
            last_error = exc
            if attempt < config.retries:
                cancellable_sleep(config.request_delay_seconds * attempt)
    raise RuntimeError(f"{namespace} request failed") from last_error


def _session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": user_agent, "Accept": "application/json, application/xml"}
    )
    return session


def _cache_path(
    cache_dir: Path | None,
    namespace: str,
    url: str,
    params: dict[str, Any],
    suffix: str,
) -> Path | None:
    if cache_dir is None:
        return None
    serialized = json.dumps([url, sorted(params.items())], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return cache_dir / namespace / f"{digest}.{suffix}"


def _page_limits(
    config: DiscoveryConfig,
    limit: int | None,
    *,
    provider_cap: int,
) -> tuple[int, int]:
    per_page = min(max(config.results_per_page, 1), provider_cap)
    max_pages = max(config.max_pages_per_query, 1)
    if limit is not None:
        per_page = min(max(limit, 1), provider_cap)
        max_pages = 1
    return per_page, max_pages


def _portable_query(query: str) -> str:
    return _collapse(query.replace("|", " OR "))


def _arxiv_query(query: str) -> str:
    portable = _portable_query(query)
    return f'"{portable}"' if " OR " not in portable else f"({portable})"


def _pubmed_query(query: str, years: YearRange) -> str:
    portable = _portable_query(query)
    if years.start is None and years.end is None:
        return portable
    start = years.start or 1000
    end = years.end or 3000
    return f'({portable}) AND ("{start}"[Date - Publication] : "{end}"[Date - Publication])'


def _date_filters(years: YearRange) -> list[str]:
    filters = []
    if years.start is not None:
        filters.append(f"from_publication_date:{years.start}-01-01")
    if years.end is not None:
        filters.append(f"to_publication_date:{years.end}-12-31")
    return filters


def _crossref_date_filters(years: YearRange) -> list[str]:
    filters = []
    if years.start is not None:
        filters.append(f"from-pub-date:{years.start}-01-01")
    if years.end is not None:
        filters.append(f"until-pub-date:{years.end}-12-31")
    return filters


def _crossref_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and isinstance(parts[0], list) and parts[0]:
            return _to_int(parts[0][0])
    return None


def _pubmed_year(article: ET.Element) -> int | None:
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is None:
        return None
    year = _element_text(pub_date.find("Year"))
    if year:
        return _to_int(year)
    match = re.search(r"\b(18|19|20|21)\d{2}\b", _element_text(pub_date.find("MedlineDate")))
    return _to_int(match.group(0)) if match else None


def _pubmed_authors(article: ET.Element) -> list[str]:
    authors = []
    for author in article.findall("AuthorList/Author"):
        collective = _element_text(author.find("CollectiveName"))
        name = collective or " ".join(
            value
            for value in [
                _element_text(author.find("ForeName")),
                _element_text(author.find("LastName")),
            ]
            if value
        )
        if name:
            authors.append(name)
    return authors


def _pubmed_doi(article_node: ET.Element) -> str | None:
    for article_id in article_node.findall(".//ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            return _clean_doi(_element_text(article_id))
    for location in article_node.findall(".//ELocationID"):
        if location.attrib.get("EIdType") == "doi":
            return _clean_doi(_element_text(location))
    return None


def _pubmed_delay(config: DiscoveryConfig, has_api_key: bool) -> float:
    return max(config.request_delay_seconds, 0.11 if has_api_key else 0.34)


def _title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def _dedupe_title_candidates(records: list[PaperRecord]) -> list[PaperRecord]:
    deduped: dict[str, PaperRecord] = {}
    for record in records:
        key = f"{normalize_title(record.title)}:{record.year or ''}"
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
            continue
        primary = record if record.doi and not existing.doi else existing
        deduped[key] = replace(
            primary,
            discovery_sources=list(
                dict.fromkeys(
                    [
                        *existing.discovery_sources,
                        *record.discovery_sources,
                        existing.source,
                        record.source,
                    ]
                )
            ),
            discovery_queries=list(
                dict.fromkeys(
                    [*existing.discovery_queries, *record.discovery_queries]
                )
            ),
        )
    return list(deduped.values())


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0] if value else "").strip()
    return str(value or "").strip()


def _element_text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def _clean_doi(value: Any) -> str | None:
    doi = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi or None


def _provider_id(value: Any) -> str | None:
    identifier = str(value or "").strip().rstrip("/")
    return identifier.rsplit("/", 1)[-1] if identifier else None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collapse(value: str) -> str:
    return " ".join(str(value).split())
