#!/usr/bin/env python3
"""fetch_pmc_publications.py
Fetch and prepare metadata for a PMC Down Syndrome corpus.

Workflow:
  1. Parse a MEDLINE .txt export  -> Journal Article (PMID, PMCID) pairs
  2. Cross-reference the INCLUDE Publications Excel to mark is_include
  3. Fetch full metadata for each PMID (OpenAlex + NCBI + CrossRef + PMC OAI)
  4. Inject INCLUDE provenance fields into each saved JSON
  5. Emit one JSON per PMID to the output directory (resume-safe)

Output JSON contains all standard fields from notebook 1 PLUS:
  is_include      bool   True if the paper is in the INCLUDE Publications corpus
  condition       str    INCLUDE Condition column (semicolon-separated if multi)
  grant_number    str    INCLUDE Grant Number
  rfa_pa_number   str    INCLUDE RFA/PA Number
  fiscal_year     str    INCLUDE Fiscal Year

Usage
-----
  python scripts/fetch_pmc_publications.py \
      --medline srv/input/PMC_DownSynd_05_18_2026.txt \
      --include-excel srv/input/INCLUDE_Publications_by_2025-11-14.xlsx \
      --out-dir srv/output/DS_PMC_05_2026 \
      [--overwrite] [--limit N] [--sleep 0.5] [--sleep-pmid 1.0]

Requirements
------------
  pip install requests tenacity openpyxl pandas tqdm python-dotenv
  (these are already installed in the dschat conda env)
"""
# =============================================================================
# Standard library
# =============================================================================
import io, os, re, gzip, time, json, requests, functools, builtins
import argparse, sys
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError as ETParseError

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def _tqdm(x, **kw): return x  # type: ignore

# Force line-buffered output for tailing logs
builtins.print = functools.partial(builtins.print, flush=True)

# =============================================================================
# Config — read from environment / .env file
# =============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

def _require_env(var: str) -> str:
    val = os.getenv(var, "").strip()
    if not val:
        raise EnvironmentError(
            f"Environment variable \'{var}\' is not set.\n"
            f"Set it before running:  export {var}=your@email.edu"
        )
    return val

NCBI_EMAIL     = _require_env("NCBI_EMAIL")
CROSSREF_EMAIL = _require_env("CROSSREF_EMAIL")

_raw_key     = os.getenv("NCBI_API_KEY", "").strip()
NCBI_API_KEY = _raw_key if re.match(r"^[a-f0-9]{36}$", _raw_key) else ""

MISSING = "N/A"

# =============================================================================
# Fetch functions — copied verbatim from srv/nbs/1_fetch_publication_metadata.ipynb
# =============================================================================



# -- 2. OpenAlex metadata ------------------------------------------------------

def _decode_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted-index format."""
    if not inverted_index:
        return MISSING
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def _openalex_get(url: str) -> dict:
    """GET an OpenAlex URL with tenacity retry (429 / 5xx) if available."""
    if _HAS_TENACITY:
        @retry(
            retry=retry_if_exception_type(requests.exceptions.HTTPError),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _get():
            r = requests.get(url, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                r.raise_for_status()   # triggers retry
            r.raise_for_status()
            return r.json()
        return _get()
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()


def _fetch_openalex(pmid: str) -> dict:
    """
    Fetch metadata from https://api.openalex.org/works/pmid:{pmid}

    Returns
    -------
    dict with keys: pmid, pmcid, doi, openalex_id, title, year, type,
                    source, language, topic, subfield, field, domain,
                    sdg, authors, institutions, funders_openalex
    """
    url = f"https://api.openalex.org/works/pmid:{pmid}"
    d   = _openalex_get(url)

    # Basic fields
    title       = d.get("title",            MISSING)
    year        = d.get("publication_year", MISSING)
    work_type   = d.get("type",             MISSING)
    language    = d.get("language",         MISSING)
    doi         = d.get("doi",              MISSING)
    openalex_id = d.get("id",               MISSING)

    # Source
    source = MISSING
    primary_location = d.get("primary_location") or {}
    if primary_location.get("source"):
        source = primary_location["source"].get("display_name", MISSING)

    # Topic / Subfield / Field / Domain
    topic = subfield = field = domain = MISSING
    topics = d.get("topics", [])
    if topics:
        t0       = topics[0]
        topic    = t0.get("display_name", MISSING)
        subfield = (t0.get("subfield") or {}).get("display_name", MISSING)
        field    = (t0.get("field")    or {}).get("display_name", MISSING)
        domain   = (t0.get("domain")   or {}).get("display_name", MISSING)

    # Sustainable Development Goals
    sdgs    = [s.get("display_name", "") for s in d.get("sustainable_development_goals", [])]
    sdg_str = ", ".join(sdgs) if sdgs else MISSING

    # Authors & Institutions
    authors, institutions_list = [], []
    for authorship in d.get("authorships", []):
        name = authorship.get("author", {}).get("display_name", "Unknown")
        authors.append(name)
        for inst in authorship.get("institutions", []):
            inst_name = inst.get("display_name", "")
            if inst_name and inst_name not in institutions_list:
                institutions_list.append(inst_name)

    # Funders
    funder_names = []
    for funder in d.get("funders", []):
        name = funder.get("display_name", "")
        if name and name not in funder_names:
            funder_names.append(name)
    for grant in d.get("grants", []):
        name = grant.get("funder_display_name", "")
        if name and name not in funder_names:
            funder_names.append(name)

    # PMCID -- search every location using multiple patterns:
    #   1. PMC\d+ anywhere in landing_page_url or pdf_url
    #   2. /pmc/articles/<digits> in landing_page_url
    #   3. OAI id field: pmh:oai:pubmedcentral.nih.gov:<digits>
    def _extract_pmcid(loc: dict):
        landing = loc.get("landing_page_url", "") or ""
        pdf     = loc.get("pdf_url",          "") or ""
        loc_id  = loc.get("id",               "") or ""
        for text in (landing, pdf):
            m = re.search(r"PMC(\d+)", text, re.IGNORECASE)
            if m: return f"PMC{m.group(1)}"
            m = re.search(r"/pmc/articles/(\d+)", text, re.IGNORECASE)
            if m: return f"PMC{m.group(1)}"
        m = re.match(r"pmh:oai:pubmedcentral\.nih\.gov:(\d+)", loc_id, re.IGNORECASE)
        if m: return f"PMC{m.group(1)}"
        return None

    pmcid = MISSING
    for loc in d.get("locations", []):
        found = _extract_pmcid(loc)
        if found:
            pmcid = found
            break

    return {
        "pmid":             pmid,
        "pmcid":            pmcid,
        "doi":              doi,
        "openalex_id":      openalex_id,
        "title":            title,
        "year":             year,
        "type":             work_type,
        "source":           source,
        "language":         language,
        "topic":            topic,
        "subfield":         subfield,
        "field":            field,
        "domain":           domain,
        "sdg":              sdg_str,
        "authors":          authors,
        "institutions":     institutions_list,
        "funders_openalex": funder_names,
    }





# -- 3. NCBI efetch -- Keywords, MeSH Terms, Grants & Funding -----------------

# NCBI rate limits: 3 req/s without key, 10 req/s with key
_NCBI_SLEEP = 0.34 if not NCBI_API_KEY else 0.11   # seconds between NCBI calls


def _ncbi_get(params: dict) -> bytes:
    """GET NCBI efetch with retry on transient errors."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}

    if _HAS_TENACITY:
        @retry(
            retry=retry_if_exception_type(requests.exceptions.HTTPError),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _get():
            r = requests.get(url, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                r.raise_for_status()
            r.raise_for_status()
            return r.content
        return _get()
    else:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.content


def _fetch_ncbi(pmid: str) -> dict:
    """
    Fetch Keywords, MeSH Terms and Grants from NCBI efetch.

    Returns
    -------
    dict with keys: keywords (list), mesh_terms (list), grants (list of dicts)
    """
    time.sleep(_NCBI_SLEEP)   # respect per-call rate limit
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "email": NCBI_EMAIL}
    content = _ncbi_get(params)
    root = ET.fromstring(content)

    # Author Keywords
    keywords = [kw.text.strip() for kw in root.findall(".//Keyword") if kw.text]

    # MeSH Terms (descriptor + qualifiers)
    mesh_terms = []
    for mesh in root.findall(".//MeshHeading"):
        descriptor = mesh.find("DescriptorName")
        if descriptor is not None:
            term       = descriptor.text or ""
            qualifiers = [q.text for q in mesh.findall("QualifierName") if q.text]
            if qualifiers:
                term += f" [{', '.join(qualifiers)}]"
            mesh_terms.append(term)

    # Grants & Funding
    grants = []
    for grant in root.findall(".//Grant"):
        grants.append({
            "grant_id": grant.findtext("GrantID", default=MISSING),
            "acronym":  grant.findtext("Acronym",  default=MISSING),
            "agency":   grant.findtext("Agency",   default=MISSING),
            "country":  grant.findtext("Country",  default=MISSING),
        })

    return {
        "keywords":   keywords,
        "mesh_terms": mesh_terms,
        "grants":     grants,
    }





# -- 4. CrossRef -- Citation count ---------------------------------------------

def _fetch_crossref(doi: str) -> dict:
    """
    Fetch citation count from https://api.crossref.org/works/{doi}

    Returns
    -------
    dict with keys: cited_by_count, crossref_type, publisher
    """
    if not doi or doi == MISSING:
        return {"cited_by_count": MISSING, "crossref_type": MISSING,
                "publisher": MISSING, "crossref_error": "No DOI available"}

    doi_str = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    url     = f"https://api.crossref.org/works/{doi_str}"
    headers = {"User-Agent": f"PublicationFetcher/1.0 (mailto:{CROSSREF_EMAIL})"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("message", {})
        return {
            "cited_by_count": data.get("is-referenced-by-count", MISSING),
            "crossref_type":  data.get("type",      MISSING),
            "publisher":      data.get("publisher",  MISSING),
        }
    except Exception as exc:
        return {"cited_by_count": MISSING, "crossref_type": MISSING,
                "publisher": MISSING, "crossref_error": str(exc)}


# -- 4b. Eligibility gate for incremental PMID ingestion -----------------------
# Used by scripts/ingest_pmids.py. Runs BEFORE the expensive multi-source
# enrichment chain (OpenAlex/CrossRef/PMC OAI), so ineligible PMIDs are
# rejected with a single cheap PubMed call rather than a full fetch.

# Publication Type values that disqualify a paper from the corpus, matching
# the audit performed during the Scientific Reports revision (see
# analysis/nbs/11_corpus_reproducibility_manifest.ipynb, Cell 3): case
# reports, retracted publications, and non-journal-article content types.
EXCLUDED_PUBLICATION_TYPES = {
    "Case Reports", "Retracted Publication", "Conference Proceedings",
    "Comment", "Editorial", "Letter", "Personal Narrative", "Biography",
    "Historical Article", "Introductory Journal Article",
}


def _fetch_pubmed_eligibility_info(pmid: str) -> dict:
    """
    Single PubMed efetch call returning everything the eligibility gate needs:
    Publication Type list, PMCID (if the article has one), DOI, and title.
    Deliberately separate from _fetch_ncbi() (which returns Keywords/MeSH/
    Grants) so a PMID can be rejected on Publication Type or missing-PMCID
    grounds without waiting on the fuller multi-field parse.
    """
    time.sleep(_NCBI_SLEEP)
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "email": NCBI_EMAIL}
    content = _ncbi_get(params)
    root = ET.fromstring(content)

    pub_types = [pt.text.strip() for pt in root.findall(".//PublicationType") if pt.text]

    pmcid = MISSING
    doi = MISSING
    # Scope to the article's OWN id list. A bare ".//ArticleId" also matches every
    # <ReferenceList><Reference><ArticleIdList> entry, so on a paper with PMC-indexed
    # references (PMID 11178230 has 54) the loop ends up assigning a cited reference's
    # PMCID to this article -- which then also makes the "is it in PMC" gate below pass
    # for papers that are not in PMC at all.
    for aid in root.findall(".//PubmedArticle/PubmedData/ArticleIdList/ArticleId"):
        id_type = aid.get("IdType", "")
        if id_type == "pmc" and aid.text:
            pmcid = aid.text.strip()
            if not pmcid.upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"
        elif id_type == "doi" and aid.text:
            doi = aid.text.strip()

    title = MISSING
    t = root.find(".//ArticleTitle")
    if t is not None and t.text:
        title = t.text.strip()

    return {"pmid": pmid, "pub_types": pub_types, "pmcid": pmcid, "doi": doi, "title": title}


def _check_retraction_crossref(doi: str, email: str) -> dict:
    """
    CrossRef retraction check. Ported from analysis/nbs/6_retraction_check.ipynb
    with the relation-key fix applied during this revision: different
    publishers register the retraction signal under different keys inside
    CrossRef's `relation` object -- Springer/BMC and Wiley/Hindawi were
    confirmed (via live testing) to use "retraction", not the "is-retracted-by"
    key CrossRef's own documentation names, so both are checked. This is a
    SECOND, independent signal from _fetch_pubmed_eligibility_info()'s
    Publication Type check above -- neither source is complete alone (see
    notebook 6's summary: one paper in this corpus had no CrossRef signal
    under any key but was tagged "Retracted Publication" in MEDLINE, while two
    others had the reverse -- a MEDLINE PT tag would have caught them anyway,
    but were only detectable via CrossRef's "retraction" key once the bug was
    fixed). Both checks running together is the point.

    Returns
    -------
    dict with keys: is_retracted (bool), retraction_doi (str|None), error (str|None)
    """
    result = {"is_retracted": False, "retraction_doi": None, "error": None}
    if not doi or doi == MISSING:
        result["error"] = "no_doi"
        return result

    doi_clean = str(doi).replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    url = f"https://api.crossref.org/works/{doi_clean}"
    headers = {"User-Agent": f"T21IngestPipeline/1.0 (mailto:{email})"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            result["error"] = "doi_not_found_in_crossref"
            return result
        resp.raise_for_status()
        msg = resp.json().get("message", {})

        for update in msg.get("update-to", []):
            if str(update.get("type", "")).lower() == "retraction":
                result["is_retracted"] = True
                result["retraction_doi"] = update.get("DOI")
                return result

        relation = msg.get("relation") or {}
        retracted_by = relation.get("is-retracted-by") or relation.get("retraction") or []
        if retracted_by:
            result["is_retracted"] = True
            result["retraction_doi"] = (
                retracted_by[0].get("id") if isinstance(retracted_by, list) and retracted_by else None
            )
    except Exception as exc:
        result["error"] = str(exc)[:200]

    return result


def check_pmid_eligibility(pmid: str, crossref_email: str = None) -> dict:
    """
    Full eligibility gate for a single PMID: Publication Type filter, PMC
    open-access availability (has a PMCID at all), and a CrossRef retraction
    cross-check independent of the Publication Type tag. Used by
    scripts/ingest_pmids.py before running the expensive enrichment chain.

    Returns
    -------
    dict with keys: pmid, pmcid, doi, title, eligible (bool), reason (str|None)
    reason is None when eligible=True; otherwise one of:
      "publication_type:<comma-separated excluded types>"
      "no_pmcid_not_in_pmc"
      "retracted_crossref:<retraction notice DOI>"
      "pubmed_fetch_error:<message>"
    """
    email = crossref_email or CROSSREF_EMAIL
    try:
        info = _fetch_pubmed_eligibility_info(pmid)
    except Exception as exc:
        return {"pmid": pmid, "pmcid": MISSING, "doi": MISSING, "title": MISSING,
                "eligible": False, "reason": f"pubmed_fetch_error:{str(exc)[:150]}"}

    flagged = EXCLUDED_PUBLICATION_TYPES & set(info["pub_types"])
    if flagged:
        return {**info, "eligible": False,
                "reason": f"publication_type:{','.join(sorted(flagged))}"}

    if info["pmcid"] == MISSING:
        return {**info, "eligible": False, "reason": "no_pmcid_not_in_pmc"}

    cr = _check_retraction_crossref(info["doi"], email)
    if cr.get("is_retracted"):
        return {**info, "eligible": False,
                "reason": f"retracted_crossref:{cr.get('retraction_doi')}"}

    return {**info, "eligible": True, "reason": None}



# -- 5. PMC full-text section parser ------------------------------------------

SECTION_CANON = {
    # ── Abstract ────────────────────────────────────────────────────────────
    "abstract":               "Abstract",
    "summary":                "Abstract",          # eLife, Nature
    "article summary":        "Abstract",
    "lay summary":            "Abstract",
    "plain language summary": "Abstract",
    "graphical abstract":     "Abstract",
    "structured abstract":    "Abstract",
    "mini abstract":          "Abstract",

    # ── Introduction / Background ────────────────────────────────────────────
    "background":                    "Introduction",
    "intro":                         "Introduction",
    "introduction":                  "Introduction",
    "background and aims":           "Introduction",   # Gut, GI journals
    "background and objectives":     "Introduction",
    "background and rationale":      "Introduction",
    "background and purpose":        "Introduction",   # Stroke / neurology
    "aims":                          "Introduction",
    "objectives":                    "Introduction",
    "rationale":                     "Introduction",
    "overview":                      "Introduction",
    "context":                       "Introduction",
    "motivation":                    "Introduction",

    # ── Methods ─────────────────────────────────────────────────────────────
    "methods":                                        "Methods",
    "materials and methods":                          "Methods",
    "material and methods":                           "Methods",
    "materials & methods":                            "Methods",
    "patients and methods":                           "Methods",
    "patients and study design":                      "Methods",
    "subjects and methods":                           "Methods",
    "participants and methods":                       "Methods",
    "methodology":                                    "Methods",
    "study design":                                   "Methods",
    "study design and methods":                       "Methods",
    "experimental procedure":                         "Methods",
    "experimental procedures":                        "Methods",
    "experimental design":                            "Methods",
    "experimental section":                           "Methods",  # Chemistry
    "methods and materials":                          "Methods",
    "methods and analysis":                           "Methods",
    "design and methods":                             "Methods",
    "design and setting":                             "Methods",
    "design":                                         "Methods",
    "approach":                                       "Methods",
    "protocol":                                       "Methods",
    "analytic methods":                               "Methods",
    "analytical methods":                             "Methods",
    "statistical methods":                            "Methods",
    "statistical analysis":                           "Methods",
    "statistical analyses":                           "Methods",
    "data collection":                                "Methods",
    "data collection and analysis":                   "Methods",
    "data analysis":                                  "Methods",
    "data processing":                                "Methods",
    "computational methods":                          "Methods",
    "model":                                          "Methods",
    "models":                                         "Methods",
    "clinical methods":                               "Methods",
    "clinical procedures":                            "Methods",
    "settings and participants":                      "Methods",
    "setting":                                        "Methods",
    "setting and participants":                       "Methods",
    "participants":                                   "Methods",
    "study population":                               "Methods",
    "patient population":                             "Methods",
    "cohort description":                             "Methods",
    "inclusion and exclusion criteria":               "Methods",
    "endpoints":                                      "Methods",
    "primary endpoint":                               "Methods",
    "secondary endpoints":                            "Methods",
    "outcome measures":                               "Methods",
    # NOTE: "outcomes" resolves to Methods here; if a paper uses it for
    # reported findings, the _canonical fallback will title-case it as-is.
    "outcomes":                                       "Methods",
    # STAR Methods (Cell Press: Cell, Cell Reports, Immunity, etc.)
    "star methods":                                   "Methods",
    "star★methods":                                   "Methods",
    "star☆methods":                                   "Methods",
    "star⋆methods":                                   "Methods",
    "key resources table":                            "Methods",
    "experimental model and study participant details":"Methods",
    "experimental model and subject details":         "Methods",
    "experimental models and study participants":     "Methods",
    "method details":                                 "Methods",
    "quantification and statistical analysis":        "Methods",
    "resource availability":                          "Methods",
    "contact for reagent and resource sharing":       "Methods",
    "lead contact":                                   "Methods",
    "materials availability":                         "Methods",
    "data and code availability":                     "Methods",
    "experimental model":                             "Methods",
    "reagent details":                                "Methods",
    # eLife-style
    "materials":                                      "Methods",
    "reagents and resources":                         "Methods",
    # Bioinformatics / CS
    "implementation":                                 "Methods",
    "algorithms":                                     "Methods",
    "system design":                                  "Methods",
    "architecture":                                   "Methods",
    "network architecture":                           "Methods",
    "pipeline":                                       "Methods",
    "workflow":                                       "Methods",

    # ── Results ──────────────────────────────────────────────────────────────
    "results":                "Results",
    "findings":               "Results",
    "main findings":          "Results",
    "key findings":           "Results",
    "observations":           "Results",
    "experiments":            "Results",
    "data":                   "Results",
    "clinical results":       "Results",
    "laboratory results":     "Results",
    "experimental results":   "Results",
    "in vitro results":       "Results",
    "in vivo results":        "Results",
    "computational results":  "Results",
    "case presentation":      "Results",
    "case description":       "Results",
    "clinical case":          "Results",

    # ── Results + Discussion (combined) ─────────────────────────────────────
    "results and discussion":  "Results and Discussion",
    "results & discussion":    "Results and Discussion",
    "results and discussions": "Results and Discussion",

    # ── Discussion ───────────────────────────────────────────────────────────
    "discussion":                    "Discussion",
    "limitations":                   "Discussion",
    "strengths and limitations":     "Discussion",
    "limitations and future directions": "Discussion",
    "limitations and strengths":     "Discussion",
    "future directions":             "Discussion",
    "future perspectives":           "Discussion",
    "perspectives":                  "Discussion",
    "interpretation":                "Discussion",
    "clinical implications":         "Discussion",
    "implications":                  "Discussion",
    "comment":                       "Discussion",   # short reports / letters
    "commentary":                    "Discussion",

    # ── Conclusion ───────────────────────────────────────────────────────────
    "conclusion":                    "Conclusion",
    "conclusions":                   "Conclusion",
    "concluding remarks":            "Conclusion",
    "concluding thoughts":           "Conclusion",
    "summary and conclusions":       "Conclusion",
    "conclusion and future directions": "Conclusion",
    "conclusions and perspectives":  "Conclusion",
    "conclusions and future work":   "Conclusion",
    "closing remarks":               "Conclusion",
    "final remarks":                 "Conclusion",
    "take-home messages":            "Conclusion",   # review articles
    "take home messages":            "Conclusion",
    "clinical bottom line":          "Conclusion",   # BMJ

    # ── Acknowledgments ──────────────────────────────────────────────────────
    "acknowledgments":   "Acknowledgments",
    "acknowledgements":  "Acknowledgments",
    "acknowledgment":    "Acknowledgments",
    "acknowledgement":   "Acknowledgments",

    # ── Ethics ───────────────────────────────────────────────────────────────
    "ethics":                                          "Ethics",
    "ethical approval":                                "Ethics",
    "ethics approval":                                 "Ethics",
    "ethics approval and consent to participate":      "Ethics",
    "ethical approval and consent to participate":     "Ethics",
    "consent for publication":                         "Ethics",
    "ethical statement":                               "Ethics",
    "ethics statement":                                "Ethics",
    "ethical considerations":                          "Ethics",
    "institutional review board":                      "Ethics",
    "irb":                                             "Ethics",
    "declaration of helsinki":                         "Ethics",
    "human subjects":                                  "Ethics",
    "animal care and use":                             "Ethics",
    "animal ethics":                                   "Ethics",
    "animal welfare":                                  "Ethics",

    # ── Data Availability ────────────────────────────────────────────────────
    "data availability":               "Data Availability",
    "data availability statement":     "Data Availability",
    "availability of data and materials": "Data Availability",
    "availability of data":            "Data Availability",
    "data and materials availability": "Data Availability",
    "data sharing":                    "Data Availability",
    "data sharing statement":          "Data Availability",
    # NOTE: "source data" is kept under Data Availability (not Supplementary)
    # because in PMC back-matter it most commonly refers to underlying datasets.
    "source data":                     "Data Availability",
    "supporting data":                 "Data Availability",
    "data deposition":                 "Data Availability",
    "data access":                     "Data Availability",
    "data resources":                  "Data Availability",
    "open data":                       "Data Availability",

    # ── Code Availability ────────────────────────────────────────────────────
    "code availability":           "Code Availability",
    "code availability statement": "Code Availability",
    "source code availability":    "Code Availability",
    "software availability":       "Code Availability",
    "availability of code":        "Code Availability",
    "custom code":                 "Code Availability",
    "computer code":               "Code Availability",
    "software":                    "Code Availability",
    "code":                        "Code Availability",
    "open source code":            "Code Availability",

    # ── Author Contributions ─────────────────────────────────────────────────
    "author contributions":        "Author Contributions",
    "authors’ contributions": "Author Contributions",   # curly apostrophe
    "authors' contributions":      "Author Contributions",   # straight apostrophe
    "contribution of authors":     "Author Contributions",
    "author contribution":         "Author Contributions",
    "contributions":               "Author Contributions",
    "contributor roles":           "Author Contributions",   # CRediT taxonomy
    "credit author statement":     "Author Contributions",

    # ── Competing Interests ──────────────────────────────────────────────────
    "conflict of interest":             "Competing Interests",
    "conflicts of interest":            "Competing Interests",
    "competing interest":               "Competing Interests",
    "competing interests":              "Competing Interests",
    "declaration of competing interest":"Competing Interests",
    "declaration of interests":         "Competing Interests",
    "disclosures":                      "Competing Interests",
    "financial disclosures":            "Competing Interests",
    "financial disclosure":             "Competing Interests",
    "potential conflicts of interest":  "Competing Interests",
    "declarations":                     "Competing Interests",   # Springer
    "competing financial interests":    "Competing Interests",

    # ── Supplementary ────────────────────────────────────────────────────────
    "supplementary information":  "Supplementary",
    "supplementary material":     "Supplementary",
    "supplementary materials":    "Supplementary",
    "supplemental information":   "Supplementary",
    "supplementary data":         "Supplementary",
    "supplemental data":          "Supplementary",
    "supporting information":     "Supplementary",
    "additional information":     "Supplementary",
    "appendix":                   "Supplementary",
    "online appendix":            "Supplementary",
    "extended data":              "Supplementary",   # Nature journals
    "expanded view":              "Supplementary",   # EMBO journals

    # ── Highlights ───────────────────────────────────────────────────────────
    "highlights":         "Highlights",
    "article highlights": "Highlights",
    "key points":         "Highlights",
    "key messages":       "Highlights",
    "key takeaways":      "Highlights",
    "clinical relevance": "Highlights",   # clinical journals

    # ── References ───────────────────────────────────────────────────────────
    "references":      "References",
    "bibliography":    "References",
    "literature cited":"References",
    "citations":       "References",
    "reference list":  "References",

    # ── Preprint / Peer-review specific ──────────────────────────────────────
    "author response":      "FullText",   # eLife decision letters
    "decision letter":      "FullText",
    "public review":        "FullText",   # eLife, PCI
    "peer review":          "FullText",
    "reviewer comments":    "FullText",
    "response to reviewers":"FullText",

    # ── Other / Uncategorized-but-common ─────────────────────────────────────
    "glossary":             "FullText",
    "abbreviations":        "FullText",
    "list of abbreviations":"FullText",
    "definitions":          "FullText",
    "nomenclature":         "FullText",
    "note":                 "FullText",
    "notes":                "FullText",
    "editor's note":       "FullText",
    "author's note":       "FullText",
    "case report":          "FullText",   # case study journals
}

# Curated exact-match table for headings that legitimately name MORE THAN ONE
# main section at once (e.g. "Results and Discussion", a combined-section
# convention some journals use throughout, not a one-off). Checked ahead of
# the generic keyword fallback in pick_bucket() so these well-known, common
# patterns resolve via an explicit, auditable mapping rather than a fuzzy
# scan; the keyword fallback (_keyword_sections_in_title) remains the safety
# net for any combined heading not listed here. Keys are pre-normalized the
# same way as SECTION_CANON's (outline prefix + trailing punctuation
# stripped, lowercased, whitespace-collapsed — see _combo_sections_for_title).
SECTION_COMBO_CANON = {
    "results and discussion":               ("Results", "Discussion"),
    "results & discussion":                 ("Results", "Discussion"),
    "results, discussion":                  ("Results", "Discussion"),
    "methods and results":                  ("Methods", "Results"),
    "methods & results":                    ("Methods", "Results"),
    "discussion and conclusion":            ("Discussion", "Conclusion"),
    "discussion and conclusions":           ("Discussion", "Conclusion"),
    "discussion & conclusion":              ("Discussion", "Conclusion"),
    "discussion & conclusions":             ("Discussion", "Conclusion"),
    "discussion, conclusion":               ("Discussion", "Conclusion"),
    "discussion, conclusions":              ("Discussion", "Conclusion"),
    "general discussion and conclusion":    ("Discussion", "Conclusion"),
    "general discussion and conclusions":   ("Discussion", "Conclusion"),
    "results, discussion and conclusion":       ("Results", "Discussion", "Conclusion"),
    "results, discussion and conclusions":      ("Results", "Discussion", "Conclusion"),
    "results, discussion, and conclusion":      ("Results", "Discussion", "Conclusion"),
    "results, discussion, and conclusions":     ("Results", "Discussion", "Conclusion"),
    "results and discussion and conclusion":    ("Results", "Discussion", "Conclusion"),
    "results and discussion and conclusions":   ("Results", "Discussion", "Conclusion"),
}


def _combo_sections_for_title(title: str) -> tuple:
    """Look up `title` in SECTION_COMBO_CANON using the same normalization
    _canonical() applies (outline-prefix strip, trailing-punctuation strip,
    lowercase, whitespace-collapse). Returns () if no combo entry matches."""
    base = _strip_outline_prefix(title or "")
    base = re.sub(r"[\s:.\-–—]+$", "", base)
    key  = re.sub(r"\s+", " ", base.strip().lower())
    return SECTION_COMBO_CANON.get(key, ())


COMMON_SECTIONS = [
    "Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion",
    "Acknowledgments", "Ethics", "Data Availability", "Code Availability",
    "Author Contributions", "Competing Interests", "Keywords", "Highlights",
    "Supplementary", "References", "FullText",
]

# Lookup: canonical name -> set of lowercase title/attribute aliases
BACK_LOOKUP = {
    "Acknowledgments":      {"acknowledgments", "acknowledgements", "acknowledgment",
                             "acknowledgement"},
    "Ethics":               {"ethics", "ethical approval", "ethics approval",
                             "ethics approval and consent to participate",
                             "ethical approval and consent to participate",
                             "consent for publication", "ethical statement",
                             "ethics statement", "ethical considerations",
                             "institutional review board", "irb",
                             "animal care and use", "animal ethics"},
    "Data Availability":    {"data availability", "data availability statement",
                             "availability of data and materials",
                             "availability of data", "data and materials availability",
                             "data sharing", "data sharing statement",
                             "source data", "supporting data",
                             "data deposition", "data access",
                             "data resources", "open data"},
    "Code Availability":    {"code availability", "code availability statement",
                             "source code availability", "software availability",
                             "availability of code", "custom code", "computer code",
                             "software", "open source code"},
    "Author Contributions": {"author contributions", "authors' contributions",
                             "authors’ contributions",
                             "contribution of authors", "author contribution",
                             "contributions", "contributor roles",
                             "credit author statement"},
    "Competing Interests":  {"conflict of interest", "conflicts of interest",
                             "competing interest", "competing interests",
                             "declaration of competing interest",
                             "declaration of interests", "disclosures",
                             "financial disclosures", "financial disclosure",
                             "potential conflicts of interest", "declarations",
                             "competing financial interests"},
    "Highlights":           {"highlights", "article highlights", "key points",
                             "key messages", "key takeaways", "clinical relevance"},
    "Supplementary":        {"supplementary information", "supplementary material",
                             "supplementary materials", "supplemental information",
                             "supplementary data", "supplemental data",
                             "supporting information", "additional information",
                             "appendix", "online appendix", "extended data",
                             "expanded view"},
}


def _strip_outline_prefix(s: str) -> str:
    if not s: return ""
    s = s.strip()
    s = re.sub(r'^(?:section|sec\.)\s*\d+(?:\.\d+)*\s*[:.)-\u2013\u2014]?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^\(?(?:\d+(?:\.\d+)*|[ivxlcdm]+|[IVXLCDM]+|[A-Z])\)?\s*(?:[.)-\u2013\u2014:]\s+)', '', s)
    s = re.sub(r'^\d+(?:\.\d+)*\s+', '', s)
    return s.strip()


def _canonical(name: str) -> str:
    base = _strip_outline_prefix(name or "")
    # Strip trailing heading punctuation (colons, periods, dashes) that some
    # publishers' JATS XML appends to top-level headings, e.g. "Methods:",
    # "Discussion:" (observed in Wiley/Genetic Epidemiology XML — no sec-type
    # attribute, titles rely on this colon-suffixed convention instead).
    # Without this, "Methods:" fails the exact-string match against SPECIAL/
    # COMMON_SECTIONS below (they hold the colon-free "Methods"), so the
    # section — and everything nested inside it — silently falls through to
    # the generic "FullText" bucket, which TARGET_SECTIONS in
    # embed_and_insert.py never embeds: the content is not merely excluded
    # from retrieval (like Abstract/Introduction), it is dropped entirely.
    base = re.sub(r"[\s:.\-–—]+$", "", base)
    key  = re.sub(r"\s+", " ", base.strip().lower())
    return SECTION_CANON.get(key, (base or "").strip().title())


# Whole-word keyword signals for the five body sections that TARGET_SECTIONS
# actually embeds (everything but Abstract, handled separately above). This
# is a LAST-RESORT fallback for pick_bucket(), used only after the curated
# SECTION_CANON alias table has already failed to match a heading — it exists
# because publishers vary heading phrasing far more than any fixed alias list
# can enumerate (numbered subheadings like "3.2 Statistical Methods",
# study-specific labels like "Study Results", or combined headings like
# "Discussion and Conclusion" that name more than one target section at
# once). Patterns are deliberately narrow (whole-word, no generic terms like
# "summary" or "findings" that would false-positive on unrelated headings)
# since a false MATCH here only duplicates a paragraph into an extra bucket
# (harmless — a minor retrieval redundancy) whereas a false MISS means the
# paragraph is silently dropped into "FullText" and never embedded at all —
# the asymmetry favors recall.
_TARGET_KEYWORD_PATTERNS = [
    (re.compile(r"\bintroduction\b", re.IGNORECASE), "Introduction"),
    (re.compile(r"\bmethods?\b", re.IGNORECASE), "Methods"),
    (re.compile(r"\bresults?\b", re.IGNORECASE), "Results"),
    (re.compile(r"\bdiscussions?\b", re.IGNORECASE), "Discussion"),
    (re.compile(r"\bconclusions?\b|\bconcluding\s+remarks?\b", re.IGNORECASE), "Conclusion"),
]


def _keyword_sections_in_title(title: str) -> set:
    """Return every canonical body-section name whose keyword appears as a
    whole word in `title`, order-independent. A heading naming exactly one
    section ("3.2 Statistical Methods") yields a single-element set; a
    combined heading ("Discussion and Conclusion") yields two."""
    return {canon for pat, canon in _TARGET_KEYWORD_PATTERNS if pat.search(title or "")}


def _title_to_back_canon(title_text: str):
    """Map a title/attribute string to a canonical back-matter section name."""
    key = re.sub(r"\s+", " ", (title_text or "").strip().lower())
    for canon, aliases in BACK_LOOKUP.items():
        if key in aliases:
            return canon
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _text(elem) -> str:
    if elem is None: return ""
    return _norm("".join(elem.itertext()))


def _direct_paragraphs(elem) -> list:
    paras = []
    for p in elem.findall("./{*}p"):
        t = _text(p)
        if t: paras.append(t)
    for lst in elem.findall("./{*}list"):
        bullets = [_text(li) for li in lst.findall("./{*}list-item")]
        if bullets:
            paras.append("\n".join(f"- {b}" for b in bullets if b))
    for box in elem.findall("./{*}boxed-text"):
        bt = _text(box)
        if bt: paras.append(bt)
    return paras


def _join(paras: list) -> str:
    return "\n\n".join(p for p in paras if _norm(p))


def _first_title(elem) -> str:
    t = elem.find("./{*}title")
    if t is not None: return _text(t)
    return (elem.get("sec-type") or elem.get("notes-type") or "").strip() or ""


def _elem_text_body(elem) -> list:
    """Get paragraph text from an element, stripping the title line."""
    paras = _direct_paragraphs(elem)
    if not paras:
        full  = _text(elem)
        title = _first_title(elem)
        body  = full[len(title):].strip() if title and full.startswith(title) else full
        if body:
            paras = [body]
    return paras


def _load_root(xml_bytes: bytes):
    raw = xml_bytes
    if raw[:2] == b"\x1f\x8b":
        try: raw = gzip.decompress(raw)
        except Exception: pass
    raw = re.sub(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]", b"", raw)
    try:
        return ET.parse(io.BytesIO(raw)).getroot()
    except ETParseError:
        pass
    try:
        import lxml.etree as LET
        parser = LET.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
        return LET.parse(io.BytesIO(raw), parser=parser).getroot()
    except Exception as exc:
        raise RuntimeError(f"Cannot parse PMC XML: {exc}")


def _to_common(raw: dict) -> dict:
    """Normalise raw section dict to COMMON_SECTIONS, using MISSING as sentinel."""
    tmp: dict = {}
    for k, v in (raw or {}).items():
        name = _canonical(k)
        text = (v or "").strip()
        if not text: continue
        if name in ("Results and Discussion", "Results And Discussion"):
            tmp["Results"]    = (tmp.get("Results",    "") + ("\n\n" if "Results"    in tmp else "") + text).strip()
            tmp["Discussion"] = (tmp.get("Discussion", "") + ("\n\n" if "Discussion" in tmp else "") + text).strip()
            continue
        if name in COMMON_SECTIONS:
            tmp[name] = (tmp.get(name, "") + ("\n\n" if name in tmp else "") + text).strip()
        else:
            tmp["FullText"] = (tmp.get("FullText", "") + ("\n\n" if "FullText" in tmp else "") + text).strip()
    # Use MISSING (not the bare string "NA") as the uniform sentinel
    return {sec: (tmp.get(sec, MISSING) or MISSING) for sec in COMMON_SECTIONS}


def _extract_back_sections(back, add_fn):
    """
    Extract all back-matter sections from a PMC <back> element.

    PMC XML uses several different tag types for back-matter content:
      <sec>             - generic sections (Data Availability, Code Availability, etc.)
      <ack>             - Acknowledgments
      <notes>           - Author Contributions, Competing Interests, Ethics, etc.
      <app-group>/<app> - Supplementary material
      <ref-list>        - References

    Uses direct children only for <sec> to avoid double-counting nested sections.
    """
    if back is None:
        return

    # 1. <sec> direct children only (non-recursive) to prevent double-counting
    for sec in back.findall("./{*}sec"):
        title_text = _first_title(sec)
        canon = _title_to_back_canon(title_text) if title_text else None
        if not canon:
            canon = _canonical(title_text) if title_text else None
            if canon not in COMMON_SECTIONS:
                canon = None
        if canon:
            paras = _elem_text_body(sec)
            if paras:
                add_fn(canon, _join(paras))

    # 2. <ack> (Acknowledgments -- dedicated tag, not a <sec>)
    for ack in back.findall(".//{*}ack"):
        paras = _elem_text_body(ack)
        if paras:
            add_fn("Acknowledgments", _join(paras))

    # 3. <notes> (Author Contributions, Competing Interests, Ethics, etc.)
    for notes in back.findall(".//{*}notes"):
        notes_type = (notes.get("notes-type") or "").strip().lower().replace("-", " ")
        canon = _title_to_back_canon(notes_type) if notes_type else None
        if not canon:
            title_text = _first_title(notes)
            canon = _title_to_back_canon(title_text) if title_text else None
        if canon:
            paras = _elem_text_body(notes)
            if paras:
                add_fn(canon, _join(paras))

    # 4. <app-group>/<app> (Supplementary material)
    for app_group in back.findall(".//{*}app-group"):
        for app in app_group.findall("./{*}app"):
            title_text = _first_title(app)
            canon = _title_to_back_canon(title_text) if title_text else "Supplementary"
            paras = _elem_text_body(app)
            if paras:
                add_fn(canon or "Supplementary", _join(paras))

    # 5. <ref-list> (References)
    ref_list = back.find(".//{*}ref-list")
    if ref_list is not None:
        refs = [_text(ref) for ref in ref_list.findall(".//{*}ref") if _text(ref)]
        if refs:
            add_fn("References", "\n".join(refs))


def _parse_pmc_xml(xml_bytes: bytes) -> dict:
    """Parse PMC OAI XML bytes and return a dict of COMMON_SECTIONS."""
    root0 = _load_root(xml_bytes)
    root  = root0
    if not re.search(r"(^|})article$", str(getattr(root, "tag", ""))):
        art = root.find(".//{*}article")
        if art is not None:
            root = art

    sections_raw = defaultdict(str)

    def add(sec_name: str, text: str):
        sec_name = _canonical(sec_name)
        t = (text or "").strip()
        if not t: return
        sections_raw[sec_name] = (sections_raw[sec_name] + ("\n\n" if sections_raw[sec_name] else "") + t)

    # Abstract
    for ab in root.findall(".//{*}abstract"):
        labeled_sections = ab.findall("./{*}sec")
        if labeled_sections:
            all_paras = []
            for sec in labeled_sections:
                label = _first_title(sec)
                paras = [_text(p) for p in sec.findall("./{*}p")]
                if label and paras:
                    all_paras.append(f"{label}: " + " ".join(paras))
                elif paras:
                    all_paras.extend(paras)
            if all_paras:
                add("Abstract", _join(all_paras))
                continue
        paras = [_text(p) for p in ab.findall("./{*}p")]
        t = _join(paras) if paras else _text(ab)
        if t: add("Abstract", t)

    # Keywords from XML <kwd-group>
    # NOTE: these may overlap with NCBI keywords; consumers should be aware
    # that result["keywords"] (from NCBI) and result["sections"]["Keywords"]
    # (from PMC XML) can contain the same terms.
    kw = []
    for kg in root.findall(".//{*}kwd-group"):
        for k in kg.findall(".//{*}kwd"):
            tt = _text(k)
            if tt: kw.append(tt)
    if kw:
        seen, uniq = set(), []
        for k in kw:
            if k.lower() not in seen:
                seen.add(k.lower()); uniq.append(k)
        add("Keywords", ", ".join(uniq))

    # Body sections
    # SPECIAL controls two things: (1) which canonical names a top-level
    # section header can resolve to, and (2) parent-bucket PROPAGATION — once
    # a section is tagged with one of these buckets, every nested <sec> child
    # inherits that same bucket regardless of its own title (pick_bucket's
    # `if parent_bucket in SPECIAL: return parent_bucket` short-circuit).
    # This must cover every TARGET_SECTIONS entry that can legitimately carry
    # sub-headings (i.e. every body section except Abstract, which is parsed
    # separately above): Methods and Results were the only members present
    # historically, so a Discussion/Introduction/Conclusion whose real content
    # lived in nested sub-sections (e.g. "Discussion > Strengths and
    # Limitations") rather than direct <p> children had no propagation path
    # and silently fell through to the generic "FullText" bucket instead —
    # content that TARGET_SECTIONS in embed_and_insert.py never embeds.
    # Combined-heading strings ("Results and Discussion") are NOT listed here
    # — they're now handled generically by SECTION_COMBO_CANON /
    # _combo_sections_for_title() in pick_bucket(), which returns a frozenset
    # of every named section rather than one hardcoded pair. _to_common()
    # still contains the original special-case for these two literal strings
    # as a harmless legacy fallback; it's unreachable via this path now that
    # the combo table resolves them earlier, but is kept in case some other
    # caller ever feeds it a pre-merged section dict directly.
    SPECIAL = {"Results", "Methods", "Discussion", "Introduction", "Conclusion"}

    def pick_bucket(sec, parent_bucket):
        # A section's OWN title/sec-type is always checked before falling
        # back to inherited parent_bucket propagation — a subsection with a
        # clear identity of its own (e.g. a literal "Results" heading nested
        # under some unrelated, unmatched parent) must win over whatever
        # bucket an ancestor happened to be assigned, including a bucket the
        # positional fallback in walk_sec assigned to that ancestor. Parent
        # propagation exists only for genuinely unlabeled/generic
        # subsections ("Discovery Cohort:", "Study Population") that carry
        # no independent section-identifying signal at all.
        title_ = _first_title(sec)
        sec_type = (sec.get("sec-type") or "").strip()
        for candidate in (_canonical(title_), _canonical(sec_type)):
            if candidate in SPECIAL: return candidate
        if sec_type.lower() == "supplementary-material":
            if re.search(r"method", title_, re.IGNORECASE):
                return "Methods"
        # Curated combined-heading table (e.g. "Results and Discussion") —
        # checked before the fuzzy keyword fallback so well-known patterns
        # resolve via an explicit mapping. See SECTION_COMBO_CANON.
        combo_hits = _combo_sections_for_title(title_)
        if combo_hits:
            return combo_hits[0] if len(combo_hits) == 1 else frozenset(combo_hits)
        # Last-resort generic fallback — see _keyword_sections_in_title's docstring.
        kw_hits = _keyword_sections_in_title(title_)
        if len(kw_hits) == 1:
            return next(iter(kw_hits))
        if len(kw_hits) > 1:
            return frozenset(kw_hits)
        # Nothing in this section's own title/sec-type gave a signal —
        # now fall back to whatever bucket (single or combo) the parent
        # already resolved to, if any.
        if isinstance(parent_bucket, frozenset): return parent_bucket
        if parent_bucket in SPECIAL: return parent_bucket
        return None

    def _is_star_methods(sec) -> bool:
        title_   = _first_title(sec)
        sec_type = (sec.get("sec-type") or "").strip().lower()
        title_key = re.sub(r"\s+", " ", (title_ or "").strip().lower())
        if re.search(r"star[\s\u2605\u2606]*methods", title_key, re.IGNORECASE):
            return True
        if sec_type == "supplementary-material" and re.search(r"method", title_, re.IGNORECASE):
            return True
        return False

    def _sec_names_bucket(sec, target: str) -> bool:
        """True if `sec` alone (ignoring parent-bucket propagation) resolves
        to `target` — used only by the has_*_anywhere pre-scans below."""
        if target == "Methods" and _is_star_methods(sec):
            return True
        title_ = _first_title(sec)
        sec_type = (sec.get("sec-type") or "").strip()
        if _canonical(title_) == target or _canonical(sec_type) == target:
            return True
        if target == "Methods" and sec_type.lower() == "supplementary-material" \
                and re.search(r"method", title_, re.IGNORECASE):
            return True
        return (target in _combo_sections_for_title(title_)
                or target in _keyword_sections_in_title(title_))

    def _has_bucket_anywhere(sec, target: str) -> bool:
        if _sec_names_bucket(sec, target):
            return True
        return any(_has_bucket_anywhere(c, target) for c in sec.findall("./{*}sec"))

    def walk_sec(sec, parent_bucket=None):
        if _is_star_methods(sec):
            for child in sec.findall("./{*}sec"):
                walk_sec(child, "Methods")
            paras = _direct_paragraphs(sec)
            if paras: add("Methods", _join(paras))
            return
        bucket = pick_bucket(sec, parent_bucket)
        title_ = _first_title(sec)
        # Positional fallback for top-level sections pick_bucket couldn't
        # place at all: some journals (Nature/Cell-family in particular)
        # never use a literal "Results" heading — findings are spread across
        # several thematically-titled top-level sections instead (e.g. "Cell
        # composition differs in Ts21"). Others (biostatistics/computational
        # methodology papers especially) do the same for Methods — the
        # method IS the paper, developed under sections like "Bias Formulae"
        # or "Model Overview" with no section ever literally titled
        # "Methods". Either way, no text-matching mechanism above can
        # recognize an individual thematic section on its own.
        #
        # Gated on which of Methods/Results/Discussion is already confirmed
        # present ELSEWHERE in the paper, to avoid misfiring on opinion/
        # ethics/review articles that use the same "several thematic
        # top-level sections" shape but have neither a Methods nor a
        # Results-equivalent (verified directly against three such flagged
        # corpus papers — none had a Methods section, so the Results branch
        # correctly never fires for them):
        #   • has Methods, missing Results  -> orphaned sections = Results
        #   • has Discussion or Results, missing Methods -> orphaned = Methods
        #     (Discussion/Results without identifiable Methods is still a
        #     real technical/analytical signal — e.g. a paper that discusses
        #     findings must have built on some methodological content
        #     earlier in the body, even if no heading ever says so; and a
        #     genuine review's synthesis content becoming retrievable under
        #     an imprecise "Methods" label is strictly better than it being
        #     silently dropped — all four retrieved sections carry equal
        #     rerank priority, so the label affects the citation tag shown
        #     to users, not retrieval eligibility)
        # A section is only eligible if it isn't Supplementary Material and
        # doesn't independently name some other recognized bucket
        # (Acknowledgments, etc. — some publishers place these in <body>
        # rather than <back>).
        if (bucket is None and parent_bucket is None
                and (sec.get("sec-type") or "").strip().lower() != "supplementary-material"
                and _canonical(title_) not in COMMON_SECTIONS):
            if has_methods_anywhere:
                bucket = "Results"
            elif has_discussion_or_results_anywhere:
                bucket = "Methods"
        paras  = _direct_paragraphs(sec)
        if paras:
            text = _join(paras)
            if isinstance(bucket, frozenset):
                # Combined heading — the paragraph genuinely belongs to every
                # named section, so write it to each rather than picking one.
                for b in bucket:
                    add(b, text)
            else:
                add(bucket or title_, text)
        for child in sec.findall("./{*}sec"):
            walk_sec(child, bucket)

    body = root.find(".//{*}body")
    if body is not None:
        top_secs = body.findall("./{*}sec")
        has_methods_anywhere = any(_has_bucket_anywhere(s, "Methods") for s in top_secs)
        has_discussion_or_results_anywhere = any(
            _has_bucket_anywhere(s, "Discussion") or _has_bucket_anywhere(s, "Results")
            for s in top_secs
        )
        for top_sec in top_secs:
            walk_sec(top_sec, None)
        # Body-level content that is NOT wrapped in any <sec> -- e.g. a <boxed-text>
        # sibling of the <sec> elements (a real, recurring pattern: BMJ Open's
        # "Strengths and limitations of this study" box is a direct <body> child,
        # not nested in a <sec>). The <sec>-only walk above never visits these, so
        # without this fallback they are silently dropped rather than captured
        # anywhere. Route to FullText (the existing generic catch-all for content
        # that doesn't map to one of the 6 target buckets) rather than guessing a
        # specific bucket -- this is deliberately generic, not BMJ Open-specific.
        direct_body_paras = _direct_paragraphs(body)
        if direct_body_paras:
            add("FullText", _join(direct_body_paras))
        if not sections_raw:
            ft = _text(body)
            if ft: add("FullText", ft)
    else:
        for sub in root.findall(".//{*}sub-article"):
            sub_body = sub.find(".//{*}body")
            if sub_body is not None:
                ft = _text(sub_body)
                if ft: add("FullText", ft)

    back = root.find(".//{*}back")
    _extract_back_sections(back, add)

    return _to_common(dict(sections_raw))





# -- 6. PMC OAI XML downloader ------------------------------------------------

def _normalize_pmcid(ident: str) -> str:
    """Normalize any PMCID variant to 'PMC<digits>'."""
    s      = str(ident).strip().upper()
    s      = re.sub(r"\.0$", "", s)
    digits = re.findall(r"\d+", s)
    if digits:
        return f"PMC{max(digits, key=len)}"
    raise ValueError(f"Invalid PMCID: {ident!r}")


def _fetch_pmc_sections(pmcid: str) -> dict:
    """
    Download PMC XML via OAI and parse it into COMMON_SECTIONS.

    Returns
    -------
    dict[str, str] -- one entry per section; missing sections are MISSING.
    """
    if not pmcid or pmcid == MISSING:
        return {sec: MISSING for sec in COMMON_SECTIONS}

    try:
        pmcid     = _normalize_pmcid(pmcid)
        m         = re.search(r"(\d+)$", pmcid)
        if m is None:
            raise ValueError(f"Could not extract numeric ID from PMCID: {pmcid!r}")
        pmcid_num = m.group(1)

        url = (
            "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord"
            f"&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}"
            "&metadataPrefix=pmc"
        )
        r = requests.get(
            url,
            headers={"Accept-Encoding": "gzip, deflate"},
            allow_redirects=True,
            timeout=60,
        )
        r.raise_for_status()

        xml_bytes = r.content
        if not xml_bytes.strip().startswith(b"<"):
            try: xml_bytes = gzip.decompress(xml_bytes)
            except Exception: pass

        try:
            ET.fromstring(xml_bytes)
        except ET.ParseError:
            xml_bytes = xml_bytes.decode("utf-8", errors="replace").encode("utf-8")

        return _parse_pmc_xml(xml_bytes)

    except Exception as exc:
        print(f"    [WARNING] PMC fetch failed for {pmcid}: {exc}")
        return {sec: MISSING for sec in COMMON_SECTIONS}





# -- 7. MAIN FUNCTION: fetch_and_save_metadata ---------------------------------

def fetch_and_save_metadata(
    pmid: str,
    out_dir: str | Path = None,
    fetch_fulltext: bool = True,
    overwrite: bool = False,
    sleep_between: float = 0.5,
) -> dict:
    """
    Fetch all metadata for a single PMID and save the result to a JSON file.

    Parameters
    ----------
    pmid : str
        PubMed ID, e.g. "39120971".
    out_dir : str or Path
        Directory where the JSON file will be saved.
        File is named  publication_metadata_PMID_<pmid>.json
        Default: ./publication_metadata_json/
    fetch_fulltext : bool
        If True (default), fetch full-text sections from PMC OAI.
        Set to False to skip PMC when full text is not needed.
    overwrite : bool
        If False (default), skip fetch when the JSON file already exists.
    sleep_between : float
        Seconds to pause between each API call.

    Returns
    -------
    dict -- the full metadata (also saved to disk as JSON).

    Output JSON structure
    ---------------------
    {
        "pmid":             str,
        "pmcid":            str,
        "doi":              str,
        "openalex_id":      str,
        "title":            str,
        "year":             int,
        "type":             str,
        "source":           str,          # journal / venue name
        "language":         str,
        "topic":            str,
        "subfield":         str,
        "field":            str,
        "domain":           str,
        "sdg":              str,
        "authors":          [str, ...],
        "institutions":     [str, ...],
        "funders_openalex": [str, ...],
        "keywords":         [str, ...],   # from NCBI efetch
        "mesh_terms":       [str, ...],   # from NCBI efetch
        "grants":           [{grant_id, acronym, agency, country}, ...],
        "citations":        int,          # from CrossRef
        "crossref":         {cited_by_count, crossref_type, publisher},
        "sections": {
            "Abstract":             str,
            "Introduction":         str,
            "Methods":              str,
            "Results":              str,
            "Discussion":           str,
            "Conclusion":           str,
            "Acknowledgments":      str,
            "Ethics":               str,
            "Data Availability":    str,
            "Code Availability":    str,
            "Author Contributions": str,
            "Competing Interests":  str,
            "Keywords":             str,
            "Highlights":           str,
            "Supplementary":        str,
            "References":           str,
            "FullText":             str,
        }
    }
    """
    pmid     = str(pmid).strip()
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "output" / "DS_PMC_05_2026"
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"publication_metadata_PMID_{pmid}.json"

    if out_path.exists() and not overwrite:
        print(f"[SKIP] File already exists: {out_path.name}  (set overwrite=True to re-fetch)")
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh)

    print(f"{'='*60}")
    print(f"Fetching metadata for PMID: {pmid}")
    print(f"{'='*60}")

    result = {"pmid": pmid}

    # -- Step 1: OpenAlex --
    print(f"[1/4] OpenAlex  -> api.openalex.org")
    try:
        oa = _fetch_openalex(pmid)
        result.update(oa)
        print(f"      Title     : {result.get('title', MISSING)[:80]}")
        print(f"      PMCID     : {result.get('pmcid', MISSING)}")
    except Exception as exc:
        print(f"      [ERROR] OpenAlex: {exc}")
        result["openalex_error"] = str(exc)
    time.sleep(sleep_between)

    # -- Step 2: NCBI efetch --
    print(f"[2/4] NCBI      -> eutils.ncbi.nlm.nih.gov/efetch")
    try:
        ncbi = _fetch_ncbi(pmid)
        result["keywords"]   = ncbi["keywords"]
        result["mesh_terms"] = ncbi["mesh_terms"]
        result["grants"]     = ncbi["grants"]
        print(f"      Keywords  : {len(result['keywords'])} terms")
        print(f"      MeSH      : {len(result['mesh_terms'])} terms")
        print(f"      Grants    : {len(result['grants'])} grants")
    except Exception as exc:
        print(f"      [ERROR] NCBI: {exc}")
        result["ncbi_error"] = str(exc)
    time.sleep(sleep_between)

    # -- Step 3: CrossRef --
    print(f"[3/4] CrossRef  -> api.crossref.org")
    doi = result.get("doi", MISSING)
    cr  = _fetch_crossref(doi)
    result["citations"] = cr.get("cited_by_count", MISSING)
    result["crossref"]  = cr
    print(f"      DOI       : {doi}")
    print(f"      Citations : {result['citations']}")
    time.sleep(sleep_between)

    # -- Step 4: PMC full-text --
    if fetch_fulltext:
        pmcid = result.get("pmcid", MISSING)
        print(f"[4/4] PMC OAI   -> pmc.ncbi.nlm.nih.gov/oai  (PMCID={pmcid})")
        sections = _fetch_pmc_sections(pmcid)
        result["sections"] = sections
        fetched = [s for s, v in sections.items() if v and v != MISSING]
        print(f"      Sections  : {fetched}")
    else:
        print(f"[4/4] PMC OAI   -> skipped (fetch_fulltext=False)")
        result["sections"] = {sec: MISSING for sec in COMMON_SECTIONS}

    # -- Save to JSON --
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"")
    print(f"Saved -> {out_path.resolve()}")
    print(f"{'='*60}")
    return result





# =============================================================================
# MEDLINE text file parser
# =============================================================================

def parse_medline_journal_articles(medline_path: str | Path) -> list[dict]:
    """Parse a PubMed/MEDLINE .txt export and return Journal Article records.

    Returns a list of dicts with keys:
        pmid  str  PubMed ID
        pmcid str  PMC ID (e.g. "PMC12345678"), or "" if not present
    """
    txt = Path(medline_path).read_text(encoding="utf-8", errors="replace")
    raw_records = [r.strip() for r in txt.split("\n\n") if r.strip()]

    results = []
    for rec in raw_records:
        # Only process Journal Articles
        if "PT  - Journal Article" not in rec:
            continue

        pmid_m  = re.search(r"^PMID- (\d+)", rec, re.MULTILINE)
        pmcid_m = re.search(r"^PMC - (PMC\d+)", rec, re.MULTILINE)

        pmid  = pmid_m.group(1)  if pmid_m  else ""
        pmcid = pmcid_m.group(1) if pmcid_m else ""

        if pmid:
            results.append({"pmid": pmid, "pmcid": pmcid})

    return results


# =============================================================================
# INCLUDE Excel loader
# =============================================================================

def load_include_lookup(excel_path: str | Path) -> dict[str, dict]:
    """Load INCLUDE Publications Excel → lookup by PMID string.

    Returns
    -------
    dict mapping PMID string to:
        {condition, grant_number, rfa_pa_number, fiscal_year}
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required: pip install pandas openpyxl")

    df = pd.read_excel(Path(excel_path))
    lookup: dict[str, dict] = {}

    def _safe_str(val) -> str:
        if val is None:
            return ""
        s = str(val).strip()
        return "" if s.lower() in ("nan", "none", "n/a", "") else s

    for _, row in df.iterrows():
        pmid = _safe_str(row.get("PMID"))
        if not pmid:
            continue
        lookup[pmid] = {
            "condition":    _safe_str(row.get("Condition")),
            "grant_number": _safe_str(row.get("Grant Number")),
            "rfa_pa_number":_safe_str(row.get("RFA/PA Number")),
            "fiscal_year":  _safe_str(row.get("Fiscal Year")),
        }

    return lookup


# =============================================================================
# INCLUDE-only batch processor (for papers not in the MEDLINE export)
# =============================================================================

def fetch_include_only_publications(
    include_excel_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
    limit: int | None = None,
    sleep_between: float = 0.5,
    sleep_between_pmids: float = 1.0,
) -> dict:
    """Fetch metadata for INCLUDE-funded papers NOT already present in out_dir.

    Use this after running fetch_pmc_publications() for the MEDLINE corpus.
    It reads the INCLUDE Excel, finds PMIDs whose JSON is missing from out_dir,
    fetches them through the same pipeline, and saves with is_include=True.
    No --mark-include workaround needed — provenance is embedded in the JSON.

    Parameters
    ----------
    include_excel_path : Path to INCLUDE Publications .xlsx
    out_dir            : Same output directory used for the MEDLINE fetch
    overwrite          : Re-fetch even if JSON already has is_include field
    limit              : Process at most this many papers (for testing)
    sleep_between      : Seconds between API calls per PMID
    sleep_between_pmids: Seconds between successive PMIDs

    Returns
    -------
    dict with keys ok / skipped / failed (lists)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FETCH INCLUDE-ONLY PUBLICATIONS (not in MEDLINE export)")
    print("=" * 70)

    # Load INCLUDE Excel
    print(f"[1] Loading INCLUDE Excel: {include_excel_path}")
    include_lookup = load_include_lookup(include_excel_path)
    print(f"    INCLUDE PMIDs loaded  : {len(include_lookup)}")

    # Find which PMIDs are already present in out_dir
    already_fetched: set[str] = set()
    for f in out_dir.glob("publication_metadata_PMID_*.json"):
        pmid_from_name = f.stem.replace("publication_metadata_PMID_", "").strip()
        if pmid_from_name:
            already_fetched.add(pmid_from_name)

    # Papers in INCLUDE Excel but NOT already fetched
    missing_pmids = [
        pmid for pmid in include_lookup
        if pmid not in already_fetched
    ]
    if overwrite:
        missing_pmids = list(include_lookup.keys())

    print(f"    Already in out_dir    : {len(already_fetched)}")
    print(f"    Missing INCLUDE papers : {len(missing_pmids)}")

    if limit:
        missing_pmids = missing_pmids[:limit]
        print(f"    (Limited to first {limit} records)")

    print(f"[2] Output directory: {out_dir.resolve()}")
    print()

    summary: dict = {"ok": [], "skipped": [], "failed": []}
    total = len(missing_pmids)

    if total == 0:
        print("All INCLUDE papers already present — nothing to fetch.")
        return summary

    for idx, pmid in enumerate(_tqdm(missing_pmids, desc="Fetching INCLUDE-only papers"), start=1):
        out_path = out_dir / f"publication_metadata_PMID_{pmid}.json"

        # Resume check
        if out_path.exists() and not overwrite:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if "is_include" in existing:
                    summary["skipped"].append({"pmid": pmid, "path": str(out_path)})
                    continue
            except Exception:
                pass

        print(f"\n[{idx}/{total}] PMID {pmid}  [INCLUDE-only]")
        print("─" * 50)

        try:
            result = fetch_and_save_metadata(
                pmid,
                out_dir=out_dir,
                fetch_fulltext=True,
                overwrite=True,
                sleep_between=sleep_between,
            )

            # Inject INCLUDE provenance
            inc_meta = include_lookup[pmid]
            result["is_include"]    = True
            result["condition"]     = inc_meta.get("condition",    "")
            result["grant_number"]  = inc_meta.get("grant_number", "")
            result["rfa_pa_number"] = inc_meta.get("rfa_pa_number","")
            result["fiscal_year"]   = inc_meta.get("fiscal_year",  "")

            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)

            print(f"    is_include : True  [INCLUDE]")
            summary["ok"].append({"pmid": pmid, "path": str(out_path)})

        except Exception as exc:
            print(f"    [ERROR] {exc}")
            summary["failed"].append({"pmid": pmid, "error": str(exc)})

        if idx < total and sleep_between_pmids > 0:
            time.sleep(sleep_between_pmids)

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"  OK      : {len(summary['ok'])}")
    print(f"  Skipped : {len(summary['skipped'])}")
    print(f"  Failed  : {len(summary['failed'])}")
    if summary["failed"]:
        print("  Failed PMIDs:")
        for f in summary["failed"]:
            print(f"    {f['pmid']} -> {f['error']}")
    print(f"  Output  : {out_dir.resolve()}")
    print("=" * 70)
    return summary


# =============================================================================
# Main batch processor
# =============================================================================

def fetch_pmc_publications(
    medline_path: str | Path,
    include_excel_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
    limit: int | None = None,
    sleep_between: float = 0.5,
    sleep_between_pmids: float = 1.0,
) -> dict:
    """Fetch metadata for all Journal Article PMIDs in a MEDLINE export.

    Parameters
    ----------
    medline_path        : Path to MEDLINE .txt file
    include_excel_path  : Path to INCLUDE Publications .xlsx
    out_dir             : Directory for output JSON files
    overwrite           : Re-fetch even if JSON already has is_include field
    limit               : Process at most this many papers (for testing)
    sleep_between       : Seconds between API calls per PMID
    sleep_between_pmids : Seconds between successive PMIDs

    Returns
    -------
    dict with keys ok / skipped / failed (lists)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FETCH PMC DOWN SYNDROME CORPUS")
    print("=" * 70)

    # Parse MEDLINE
    print(f"[1] Parsing MEDLINE file: {medline_path}")
    records = parse_medline_journal_articles(medline_path)
    print(f"    Journal Article records: {len(records)}")

    # Load INCLUDE lookup
    print(f"[2] Loading INCLUDE Excel: {include_excel_path}")
    include_lookup = load_include_lookup(include_excel_path)
    include_pmids  = set(include_lookup.keys())
    print(f"    INCLUDE PMIDs loaded  : {len(include_pmids)}")

    if limit:
        records = records[:limit]
        print(f"    (Limited to first {limit} records)")

    print(f"[3] Output directory: {out_dir.resolve()}")
    print()

    summary: dict = {"ok": [], "skipped": [], "failed": []}
    total = len(records)

    for idx, rec in enumerate(_tqdm(records, desc="Fetching papers"), start=1):
        pmid  = rec["pmid"]
        pmcid = rec["pmcid"]

        out_path = out_dir / f"publication_metadata_PMID_{pmid}.json"

        # Resume check — skip if already fetched and has is_include
        if out_path.exists() and not overwrite:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if "is_include" in existing:
                    summary["skipped"].append({"pmid": pmid, "path": str(out_path)})
                    continue
            except Exception:
                pass   # JSON corrupted — re-fetch

        print(f"\n[{idx}/{total}] PMID {pmid}  PMC {pmcid or 'unknown'}")
        print("─" * 50)

        try:
            # ── Fetch via standard pipeline ──────────────────────────────
            result = fetch_and_save_metadata(
                pmid,
                out_dir=out_dir,
                fetch_fulltext=True,
                overwrite=True,          # always overwrite within this loop
                sleep_between=sleep_between,
            )

            # ── Inject INCLUDE provenance ─────────────────────────────────
            is_include = pmid in include_pmids
            inc_meta   = include_lookup.get(pmid, {})

            result["is_include"]    = is_include
            result["condition"]     = inc_meta.get("condition",    "")
            result["grant_number"]  = inc_meta.get("grant_number", "")
            result["rfa_pa_number"] = inc_meta.get("rfa_pa_number","")
            result["fiscal_year"]   = inc_meta.get("fiscal_year",  "")

            # Re-save with INCLUDE fields
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)

            include_tag = " [INCLUDE]" if is_include else ""
            print(f"    is_include : {is_include}{include_tag}")
            summary["ok"].append({"pmid": pmid, "path": str(out_path)})

        except Exception as exc:
            print(f"    [ERROR] {exc}")
            summary["failed"].append({"pmid": pmid, "error": str(exc)})

        # pause between PMIDs
        if idx < total and sleep_between_pmids > 0:
            time.sleep(sleep_between_pmids)

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"  OK      : {len(summary['ok'])}")
    print(f"  Skipped : {len(summary['skipped'])}")
    print(f"  Failed  : {len(summary['failed'])}")
    if summary["failed"]:
        print("  Failed PMIDs:")
        for f in summary["failed"]:
            print(f"    {f['pmid']} -> {f['error']}")
    print(f"  Output  : {out_dir.resolve()}")
    print("=" * 70)
    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    base_dir = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Fetch metadata for all Journal Article PMIDs in a MEDLINE export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_pmc_publications.py \
      --medline input/PMC_DownSynd_05_18_2026.txt \
      --include-excel input/INCLUDE_Publications_by_2025-11-14.xlsx \
      --out-dir output/DS_PMC_05_2026

  # Test run (first 10 papers only):
  python scripts/fetch_pmc_publications.py \
      --medline input/PMC_DownSynd_05_18_2026.txt \
      --include-excel input/INCLUDE_Publications_by_2025-11-14.xlsx \
      --out-dir output/DS_PMC_05_2026_test \
      --limit 10

  # Re-fetch everything (ignore cached JSONs):
  python scripts/fetch_pmc_publications.py ... --overwrite
        """,
    )
    parser.add_argument(
        "--medline",
        type=Path,
        default=None,
        help="MEDLINE .txt export file  (default: input/PMC_DownSynd_05_18_2026.txt). "
             "Not required when --include-only is set.",
    )
    parser.add_argument(
        "--include-excel",
        type=Path,
        default=base_dir / "input" / "INCLUDE_Publications_by_2025-11-14.xlsx",
        help="INCLUDE Publications .xlsx (default: input/INCLUDE_Publications_by_2025-11-14.xlsx)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "output" / "DS_PMC_05_2026",
        help="Output directory for JSON files (default: output/DS_PMC_05_2026)",
    )
    parser.add_argument(
        "--include-only",
        action="store_true",
        help="Fetch only INCLUDE papers not already present in --out-dir "
             "(use after the main MEDLINE fetch to top-up INCLUDE-only papers). "
             "--medline is not needed in this mode.",
    )
    parser.add_argument("--overwrite",  action="store_true", help="Re-fetch even if JSON already exists")
    parser.add_argument("--limit",      type=int,   default=None,  help="Process at most N papers (for testing)")
    parser.add_argument("--sleep",      type=float, default=0.5,   help="Seconds between API calls within each PMID")
    parser.add_argument("--sleep-pmid", type=float, default=1.0,   help="Seconds between successive PMIDs")

    args = parser.parse_args()

    # Validate
    if not args.include_excel.exists():
        parser.error(f"INCLUDE Excel not found: {args.include_excel}")

    if args.include_only:
        # Generic top-up mode: fetch INCLUDE-only papers (not in MEDLINE export)
        fetch_include_only_publications(
            include_excel_path=args.include_excel,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            limit=args.limit,
            sleep_between=args.sleep,
            sleep_between_pmids=args.sleep_pmid,
        )
    else:
        # Full MEDLINE corpus mode
        medline = args.medline or (base_dir / "input" / "PMC_DownSynd_05_18_2026.txt")
        if not medline.exists():
            parser.error(f"MEDLINE file not found: {medline}  (pass --medline <path> or use --include-only)")
        fetch_pmc_publications(
            medline_path=medline,
            include_excel_path=args.include_excel,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            limit=args.limit,
            sleep_between=args.sleep,
            sleep_between_pmids=args.sleep_pmid,
        )


if __name__ == "__main__":
    main()
