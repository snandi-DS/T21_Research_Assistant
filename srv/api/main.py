# api/main.py
"""
T21 Research Assistant v7 - RAG API
LangChain/LangGraph Multi-Agent Architecture

Collection: DS_INCLUDE_2026_09_02_DEEPINFRA_2048
Sections indexed: Results, Discussion, Methods, Conclusion
"""
from __future__ import annotations
import asyncio
import os
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

# Load .env from project root (one level above api/)
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain-based Multi-Agent RAG
from api.synthesizer_langchain import (
    RAGConfig, LangChainMultiAgentRAG as MultiAgentRAG,
    _normalize_pmc_citation_format, _split_semicolon_citations,
    _hyperlink_inline_pmc, _inject_include_tags, _dedupe_summary_sentences,
    _summary_word_count, _pad_summary_from_ke_final, _truncate_summary_to_150_words,
    _dedup_inline_list_items, _strip_comma_list_sentences,
    _strip_broken_risk_intro_sentences, _fix_citations_after_period,
    _strip_raw_data_sentences, _strip_further_research_sentences,
    _strip_uncited_summary_sentences,
    _repair_mutated_pmc_ids, _strip_hallucinated_citations,
    _strip_dangling_stat_parens, _append_summary_citation,
)
print("[OK] Using LangChain/LangGraph Multi-Agent RAG")

# --------- Config & RAG init (no hard-coded secrets) ----------
cfg = RAGConfig().override_with_env()

# Initialize RAG system (singleton)
try:
    rag = MultiAgentRAG(cfg)
    print(f"[OK] RAG initialized: chat={cfg.provider_chat}, embed={cfg.provider_embed}")
except Exception as e:
    print(f"[ERROR] RAG initialization failed: {e}")
    rag = None

# --------- Query cache (deterministic responses for repeated questions) -----
_query_cache: Dict[str, dict] = {}

def _cache_key(question: str, top_k: Optional[int], rerank_alpha: Optional[float] = None,
               section_mode: Optional[str] = None, sections: Optional[list] = None) -> str:
    """Normalize question and top_k into a stable cache key."""
    norm = question.lower().strip()
    sections_str = ','.join(sorted(sections)) if sections else ''
    return hashlib.md5(f"{norm}|{top_k}|{rerank_alpha}|{section_mode}|{sections_str}".encode()).hexdigest()

# --------- FastAPI app ----------
# ROOT_PATH allows Swagger UI /docs to work correctly when proxied under a sub-path
# (e.g. Apache ProxyPass /t21-api → localhost:8698). Set ROOT_PATH=/t21-api in .env.
_root_path = os.getenv("ROOT_PATH", "")

app = FastAPI(
    root_path=_root_path,
    title="T21 Research Assistant v7 API",
    version="7.0.0",
    description=(
        "## T21 Research Assistant — Evidence-Based Q&A for Down Syndrome Research\n\n"
        "Multi-agent Retrieval-Augmented Generation (RAG) system over the **T21 Down Syndrome PMC Corpus** "
        "(open-access, peer-reviewed Down Syndrome / Trisomy 21 literature — see `/health` for live paper/vector counts).\n\n"
        "### Pipeline\n"
        "1. **Retrieve** — Milvus IVF_FLAT vector search (5× oversampling, per-section retrieval)\n"
        "2. **Rerank** — DeepInfra cross-encoder blended score (`alpha·vector + (1-alpha)·cross_encoder`, weighted by `section_priority`)\n"
        "3. **Synthesize** — Context-Content-Conclusion (C-C-C) schema, ≤150 words, inline PMC citations\n"
        "4. **Safety guard** — Filters prescription / directive language\n"
        "5. **Finalize** — Citation hyperlinking, hallucination filter, length floor\n\n"
        "### Key Notes\n"
        "- Responses are **cached** in-memory by MD5(question + top_k). Clear with `POST /cache/clear` after code changes.\n"
        "- `POST /embed` and `POST /embed_batch` expose the active embedder — useful for notebooks and benchmarking.\n"
        "- This API is for **research and educational use only**. It does not provide medical advice."
    ),
    contact={
        "name": "Sutanu Nandi / Costello Lab",
        "email": "james.costello@cuanschutz.edu",
    },
    license_info={
        "name": "Research Use Only — Not for Clinical Use",
    },
    openapi_tags=[
        {"name": "RAG", "description": "Main retrieval-augmented generation endpoints"},
        {"name": "Search", "description": "Direct vector search (debug / inspection)"},
        {"name": "Embeddings", "description": "Expose the active embedder for external use (notebooks, benchmarking)"},
        {"name": "Cache", "description": "In-memory query cache management"},
        {"name": "Health", "description": "Service health and configuration introspection"},
    ],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Request/Response Schemas ----------
class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Research question about Down Syndrome / Trisomy 21")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of chunks to retrieve. Defaults to `TOP_K_DEFAULT` (.env). Auto-boosted for complex or long questions.")
    include_hits: bool = Field(False, description="If true, includes `hits` (raw Milvus hits) and `re_ranked_hits` (after cross-encoder reranking) in the response")
    rerank_alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Per-request override for RERANK_ALPHA (vector vs cross-encoder blend; 0=CE-only, 1=vector-only). Defaults to server `RERANK_ALPHA` env var.")
    section_mode: Optional[str] = Field(None, description="Section retrieval mode: `standard` (Results/Discussion/Methods/Conclusion, default) or `with_abstract` (adds Abstract+Introduction to retrieval pool).")
    sections: Optional[List[str]] = Field(None, description="Explicit section list override. When provided, takes priority over `section_mode`. Valid values: Results, Discussion, Methods, Conclusion, Abstract, Introduction.")
    enable_judge: Optional[bool] = Field(None, description="Per-request override for LLM-as-judge. Set false to skip judge_improve (faster; useful for parameter sweeps). Defaults to server ENABLE_JUDGE env var.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the prevalence of autism in Down syndrome?",
                "top_k": 8,
                "include_hits": False,
            }
        }
    }

class AnswerResponse(BaseModel):
    markdown: str = Field(..., description="Full synthesized answer in markdown — contains `# Summary` (≤150 words), `# Key Evidence` (bullet list with PMC citations), and `# Reasoning` (4 bullet plain-prose explanation)")
    structured: Optional[Dict[str, Any]] = Field(None, description="Structured answer data returned when the structured synthesis path produces a valid structured output")
    citations: Optional[List[str]] = Field(None, description="Deduplicated list of cited PMC IDs with section label, e.g. `['PMC12345678:Results', 'PMC87654321:Discussion']`")
    trace: Optional[List[Dict[str, Any]]] = Field(None, description="Per-node LangGraph execution trace for debugging pipeline behaviour")
    raw_evidence: Optional[List[Dict[str, Any]]] = Field(None, description="Top-K evidence chunks used for synthesis — each entry includes `pmcid`, `section`, `title`, `text`, `score`, `pub_year`, `mesh_terms`, `keywords`")
    hits: Optional[List[Dict[str, Any]]] = Field(None, description="Raw Milvus retrieval hits before reranking (only present when `include_hits=true`)")
    re_ranked_hits: Optional[List[Dict[str, Any]]] = Field(None, description="Hits after cross-encoder reranking (only present when `include_hits=true`)")
    judge_scores: Optional[Dict[str, Any]] = Field(None, description="LLM-as-judge scores (accuracy, informativeness, completeness 1-5 + justification)")
    improved_summary: Optional[str] = Field(None, description="Judge-guided improved summary with citation hyperlinks")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Free-text search query — embedded via the active embedder and searched against the Milvus collection")
    top_k: Optional[int] = Field(None, ge=1, le=100, description="Number of results to return. Defaults to `TOP_K_DEFAULT` (.env).")

    model_config = {
        "json_schema_extra": {
            "example": {"query": "cardiac defects Trisomy 21", "top_k": 10}
        }
    }

class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]] = Field(..., description="List of matching chunks. Each hit includes `pmcid`, `section`, `title`, `text`, `score`, `pub_year`, `pub_type`, `mesh_terms`, `keywords`.")
    count: int = Field(..., description="Number of hits returned")

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")
    mode: str = Field("query", description="`query` vs `document` embedding mode (use `query` for retrieval queries, `document` for passages/model responses/ground truth). Under the default DeepInfra embedder both modes encode symmetrically with no instruction prefix; the `nemotron` provider adds an instruction prefix for `query` mode only.")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "What cardiac conditions are common in Down syndrome?", "mode": "query"}
        }
    }

class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector (2048-dim under the default DeepInfra embedder; 4096-dim under the `nemotron` provider)")
    dim: int = Field(..., description="Vector dimensionality (matches the active embedder — 2048 for DeepInfra, 4096 for nemotron)")

class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to embed (order preserved in response)")
    mode: str = Field("document", description="`document` — no prefix (default; for passages / model responses / ground truth); `query` — instruction prefix added to each text")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Batch document embedding",
                    "value": {"texts": ["Down syndrome is caused by trisomy 21.", "Cardiac defects are common in T21."], "mode": "document"},
                }
            ]
        }
    }

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors (2048-dim under the default DeepInfra embedder; 4096-dim under the `nemotron` provider), one per input text")
    dim: int = Field(..., description="Vector dimensionality (matches the active embedder)")
    count: int = Field(..., description="Number of embeddings returned")

# --------- Endpoints ----------
@app.get("/", tags=["Health"], summary="Service info", description="Returns service name, version, status, and a summary of available endpoints.")
def root():
    return {
        "name": "T21 Research Assistant v7 API",
        "version": "7.0.0",
        "status": "ready" if rag else "initialization_failed",
        "collection": cfg.milvus_collection,
        "docs": "http://<host>:8698/docs",
        "endpoints": {
            "GET  /health":        "Service health, model names, Milvus vector/document counts",
            "POST /answer":        "Full RAG pipeline — retrieve, rerank, synthesize, finalize",
            "POST /search":        "Vector search only (debug — no LLM synthesis)",
            "POST /cache/clear":   "Invalidate in-memory MD5 query cache",
            "POST /embed":         "Embed a single text via the active embedder (DeepInfra, 2048-dim)",
            "POST /embed_batch":   "Batch-embed multiple texts via the active embedder (DeepInfra, 2048-dim)",
        }
    }

@app.get(
    "/health",
    tags=["Health"],
    summary="Service health and configuration",
    description=(
        "Returns the current health status, active LLM/embed/rerank model names, "
        "Milvus collection info, and vector/document counts.\n\n"
        "| Field | Description |\n"
        "|-------|-------------|\n"
        "| `architecture` | Always `LangChain/LangGraph` |\n"
        "| `provider_chat` | Active chat LLM provider (`openai` / `anthropic` / `google` / `ollama_remote`) |\n"
        "| `chat_model` | Resolved model name for the active chat provider |\n"
        "| `embed_model` | Active embed model (default: DeepInfra `nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim) |\n"
        "| `rerank_model` | Active reranker (default: `DeepInfraReranker`; `nemotron` provider uses a local cross-encoder) |\n"
        "| `milvus_collection` | Active vector collection name |\n"
        "| `total_vectors` | Total indexed section-level vectors |\n"
        "| `estimated_documents` | Unique PMCIDs in the collection |\n"
        "| `include_count` | Unique PMCIDs with `is_include=1` (NIH INCLUDE-funded papers) |"
    ),
)
def health():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    # Get the actual model names based on provider
    chat_model_name = "N/A"
    embed_model_name = "N/A"

    if cfg.provider_chat == "openai":
        chat_model_name = cfg.chat_model
    elif cfg.provider_chat == "anthropic":
        chat_model_name = cfg.anthropic_chat_model
    elif cfg.provider_chat == "google":
        chat_model_name = cfg.google_chat_model
    elif cfg.provider_chat == "ollama":
        chat_model_name = cfg.ollama_chat_model
    elif cfg.provider_chat == "ollama_remote":
        chat_model_name = cfg.nemotron_chat_model
    elif cfg.provider_chat == "deepinfra":
        chat_model_name = cfg.deepinfra_chat_model

    if cfg.provider_embed == "openai":
        embed_model_name = cfg.embed_model
    elif cfg.provider_embed == "google":
        embed_model_name = cfg.google_embed_model
    elif cfg.provider_embed == "ollama":
        embed_model_name = cfg.ollama_embed_model
    elif cfg.provider_embed == "nemotron":
        embed_model_name = "nvidia/llama-embed-nemotron-8b"
    elif cfg.provider_embed == "deepinfra":
        embed_model_name = cfg.deepinfra_embed_model

    # Reranker model name — provider-aware
    if cfg.provider_rerank == "deepinfra":
        # Derive the model id from the DeepInfra inference URL's last path segment
        rerank_model_name = cfg.deepinfra_rerank_url.rstrip("/").split("/v1/inference/")[-1]
    else:
        rerank_model_name = "nvidia/llama-nemotron-rerank-1b-v2"

    # Get database statistics
    db_stats = {
        "total_vectors": "N/A",
        "estimated_documents": "N/A",
        "include_count": "N/A"
    }

    try:
        # Try to get collection stats from Milvus
        from pymilvus import connections, Collection, utility
        # Ensure connection
        milvus_uri = os.getenv("MILVUS_URI") or f"http://{cfg.milvus_host}:{cfg.milvus_port}"
        try:
            connections.connect(alias="default", uri=milvus_uri)
        except Exception:
            pass  # Already connected

        if utility.has_collection(cfg.milvus_collection):
            collection = Collection(cfg.milvus_collection)
            collection.load()
            # num_entities only counts sealed segments; use query-based count instead
            # to include growing (recently inserted, not yet flushed) segments.
            try:
                count_res = collection.query(expr="pk >= 0", output_fields=["count(*)"])
                num_entities = count_res[0].get("count(*)", 0) if count_res else 0
            except Exception:
                num_entities = collection.num_entities
            db_stats["total_vectors"] = f"{num_entities:,}"
            # Count unique PMCIDs via pk-cursor pagination (avoids offset+limit ≤ 16384 limit)
            # Only count articles (excludes reviews, preprints, letters)
            try:
                unique_pmcids: set = set()
                include_pmcids: set = set()
                last_pk = -1
                _page = 16000
                while True:
                    results: List[Dict[str, Any]] = collection.query(
                        expr=f'pk > {last_pk} && pub_type == "article"',
                        output_fields=['pk', 'pmcid', 'is_include'],
                        limit=_page,
                    )  # type: ignore[assignment]
                    if not results:
                        break
                    for r in results:
                        pmcid = r.get('pmcid')
                        if not pmcid:
                            continue
                        unique_pmcids.add(pmcid)
                        if r.get('is_include'):
                            include_pmcids.add(pmcid)
                    last_pk = max(r['pk'] for r in results)
                    if len(results) < _page:
                        break
                db_stats["estimated_documents"] = f"{len(unique_pmcids):,}"
                db_stats["include_count"] = f"{len(include_pmcids):,}"
            except Exception as qe:
                print(f"[WARN] Query for unique PMCIDs failed: {qe}")
                db_stats["estimated_documents"] = f"{num_entities // 10:,}"  # Fallback estimate
        else:
            print(f"[WARN] Collection {cfg.milvus_collection} not found")
    except Exception as e:
        # If we can't get stats, just show N/A
        print(f"[WARN] Could not get Milvus stats: {e}")

    return {
        "status": "healthy",
        "config": {
            "architecture": "LangChain/LangGraph",
            "provider_chat": cfg.provider_chat,
            "provider_embed": cfg.provider_embed,
            "chat_model": chat_model_name,
            "embed_model": embed_model_name,
            "rerank_model": rerank_model_name,
            "milvus_collection": cfg.milvus_collection,
            "top_k_default": cfg.top_k_default,
            "rerank_alpha": cfg.rerank_alpha,
            "enable_rerank": cfg.enable_rerank,
        },
        "database": db_stats
    }

@app.post(
    "/answer",
    response_model=AnswerResponse,
    tags=["RAG"],
    summary="Ask a Down Syndrome research question",
    description=(
        "Runs the full 6-node LangGraph RAG pipeline and returns a structured, citation-backed answer.\n\n"
        "**Pipeline steps:** retrieve → rerank → synthesize → raw_evidence → safety_guard → finalize\n\n"
        "**Response fields:**\n"
        "- `markdown` — Full synthesized answer with `# Summary`, `# Key Evidence`, `# Reasoning` sections "
        "and inline `[PMCxxxxxxx:Section](url)` hyperlinks\n"
        "- `citations` — List of cited PMC IDs with section labels, e.g. `['PMC12345678:Results', ...]`\n"
        "- `raw_evidence` — Top-K evidence chunks used for synthesis (title, section, text snippet, score)\n"
        "- `trace` — Per-node execution trace for debugging\n"
        "- `hits` / `re_ranked_hits` — Raw and reranked Milvus hits (only when `include_hits=true`)\n\n"
        "**Caching:** responses are cached in-memory by MD5(question + top_k). "
        "Insufficient-evidence responses are never cached. Clear cache with `POST /cache/clear`."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Simple question",
                            "value": {"question": "What is the prevalence of autism in Down syndrome?", "top_k": 8},
                        },
                        "with_hits": {
                            "summary": "Include raw retrieval hits",
                            "value": {"question": "What cardiac defects are common in Trisomy 21?", "top_k": 10, "include_hits": True},
                        },
                    }
                }
            }
        }
    },
)
def answer(req: AnswerRequest):
    """
    Main RAG endpoint — retrieves, reranks, and synthesizes answer.
    Returns cached response if the same question was asked before.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    # Resolve section list: explicit `sections` > section_mode > default (4 sections)
    _sections_override: Optional[list] = None
    if req.sections:
        _sections_override = req.sections
    elif req.section_mode == "with_abstract":
        _sections_override = ["Results", "Discussion", "Methods", "Conclusion", "Abstract", "Introduction"]
    # "standard" or None → leave as None (uses default 4-section list)

    key = _cache_key(req.question, req.top_k, req.rerank_alpha, req.section_mode, req.sections)
    if key in _query_cache:
        return _query_cache[key]

    # Per-request judge toggle — temporarily override cfg, restore after the call
    _orig_judge = rag.cfg.enable_judge
    if req.enable_judge is not None:
        rag.cfg.enable_judge = req.enable_judge

    try:
        result = rag.answer(
            req.question,
            top_k=req.top_k,
            rerank_alpha_override=req.rerank_alpha,
            sections_override=_sections_override,
        )

        # Bake improved_summary into the markdown answer so the improved version IS the answer.
        import re as _re
        _md_answer = result.get("markdown", "")
        _improved = (result.get("improved_summary") or "").strip()
        if _improved:
            # Normalize citations in the improved summary before splicing —
            # the improve LLM bypasses the finalize pipeline so compound
            # semicolon brackets and space-INCLUDE variants must be fixed here.
            _hits = result.get("re_ranked_hits") or result.get("hits") or []
            _improved = _normalize_pmc_citation_format(_improved)
            _improved = _split_semicolon_citations(_improved)
            _improved = _inject_include_tags(_improved, _hits)
            # NOTE: _hyperlink_inline_pmc requires section headers to find its
            # processing spans, so calling it on bare prose (_improved has no
            # headers) is a no-op.  We splice first, then re-hyperlink the full
            # document below so the new summary section gets URLs attached.
            # Strip outer-bracket wrapping that the improve model can introduce:
            # [[PMC:Section]] -> [PMC:Section]  (before hyperlinking)
            _improved = _normalize_pmc_citation_format(_improved)
            # Strip standalone [INCLUDE](url) links the improve model can hallucinate
            # (it sometimes adds [INCLUDE](https://www.nih.gov/include-project) as a
            # separate badge instead of using the ', INCLUDE' suffix in the citation).
            _improved = _re.sub(r'\s*\[INCLUDE\]\([^)]+\)', '', _improved)
            # Quality gate: discard improved_summary if it regresses on citation
            # diversity (fewer unique papers than the original finalize summary) or
            # degenerates into a semicolon-chained run-on (≥3 [cite]; patterns).
            _orig_sum_m = _re.search(
                r'#{1,2}\s*Summary(?:\s*\(.*?\))?\s*\n(.*?)(?=\n#{1,2}\s|\Z)',
                _md_answer, _re.DOTALL | _re.IGNORECASE,
            )
            _orig_sum_body = _orig_sum_m.group(1).strip() if _orig_sum_m else ""
            _orig_pmc_ids  = set(_re.findall(r'PMC\d{4,}', _orig_sum_body))
            _imp_pmc_ids   = set(_re.findall(r'PMC\d{4,}', _improved))
            _orig_cite_n   = len(_re.findall(r'\[PMC\d{4,}:[A-Za-z0-9_-]+', _orig_sum_body))
            _imp_cite_n    = len(_re.findall(r'\[PMC\d{4,}:[A-Za-z0-9_-]+', _improved))
            _diversity_regressed = len(_orig_pmc_ids) >= 2 and len(_imp_pmc_ids) < len(_orig_pmc_ids)
            _citation_stripped   = _orig_cite_n >= 3 and _imp_cite_n < max(1, _orig_cite_n - 1)
            _semicolon_runon     = len(_re.findall(r'\[PMC[^\]]+\]\s*;', _improved)) >= 3
            if _diversity_regressed or _citation_stripped or _semicolon_runon:
                _reason = ("diversity_regressed" if _diversity_regressed
                           else "citation_stripped" if _citation_stripped
                           else "semicolon_runon")
                print(f"[IMPROVE_QG] discarded improved_summary ({_reason}): "
                      f"orig={len(_orig_pmc_ids)} papers/{_orig_cite_n} cites "
                      f"imp={len(_imp_pmc_ids)} papers/{_imp_cite_n} cites")
                _improved = ""
            if _improved:
                _splice_pat = _re.compile(
                    r'(#{1,2}\s*Summary(?:\s*\(.*?\))?\s*\n)(.*?)(?=\n#{1,2}\s|\Z)',
                    _re.DOTALL | _re.IGNORECASE,
                )
                _sm = _splice_pat.search(_md_answer)
                if _sm:
                    _md_answer = _md_answer[:_sm.start(2)] + _improved + "\n" + _md_answer[_sm.end(2):]
                # Strip verbatim methods sentences the improve model can re-introduce
                # from KE context. finalize already ran _strip_raw_data_sentences on
                # the original summary, but the improve model rewrites from KE bullets.
                _md_answer = _strip_raw_data_sentences(_md_answer)
                # Move citations placed AFTER a period to before it.
                _md_answer = _fix_citations_after_period(_md_answer)
                # Hallucination guard (the improve model bypasses finalize, so this
                # must be re-applied here): first repair digit-mutated IDs back to
                # the real corpus ID (e.g. PMC979796221 -> PMC9796221), then strip
                # any remaining citations whose IDs are not in the evidence corpus.
                # Must run BEFORE _hyperlink_inline_pmc.
                _md_answer = _strip_dangling_stat_parens(_md_answer)
                _valid_cites = set(result.get("citations") or [])
                if _valid_cites:
                    _md_answer = _repair_mutated_pmc_ids(_md_answer, _valid_cites)
                    _md_answer = _strip_hallucinated_citations(_md_answer, _valid_cites)
                # Post-strip quality gate: the pre-splice gate counted the improve
                # model's citation tags at face value, but fabricated IDs are only
                # removed by the hallucination strip above.  If the SURVIVING
                # citations vanished or regressed, revert the Summary to the
                # original finalize output, which is guaranteed to carry valid
                # citations (finalize appends a fallback citation itself).
                _post_m = _splice_pat.search(_md_answer)
                if _post_m and _orig_sum_body:
                    _post_body   = _post_m.group(2)
                    _post_ids    = set(_re.findall(r'PMC\d{4,}', _post_body))
                    _post_cite_n = len(_re.findall(r'\[PMC\d{4,}:[A-Za-z0-9_-]+', _post_body))
                    _post_regressed = (
                        _post_cite_n == 0
                        or (len(_orig_pmc_ids) >= 2 and len(_post_ids) < len(_orig_pmc_ids))
                        or (_orig_cite_n >= 3 and _post_cite_n < max(1, _orig_cite_n - 1))
                    )
                    if _post_regressed:
                        print(f"[IMPROVE_QG] reverted improved_summary post-strip: "
                              f"cites {_post_cite_n}/{_orig_cite_n}, "
                              f"papers {len(_post_ids)}/{len(_orig_pmc_ids)}")
                        _md_answer = (_md_answer[:_post_m.start(2)]
                                      + _orig_sum_body + "\n"
                                      + _md_answer[_post_m.end(2):])
                # Re-hyperlink all PMC citations in the full document.
                _md_answer = _hyperlink_inline_pmc(_md_answer)
                # Strip outer-bracket wrapping: [[PMC:Section](url)] -> [PMC:Section](url)
                _md_answer = _re.sub(
                    r'\[(\[PMC\d{4,}:[^\]]+\]\(https?://[^)]+\))\]',
                    r'\1',
                    _md_answer,
                )
                # Remove duplicate consecutive list items and duplicate sentences
                _md_answer = _dedup_inline_list_items(_md_answer)
                _md_answer = _dedupe_summary_sentences(_md_answer)
                # Strip garbled / banned sentences the improve model can introduce
                _md_answer = _strip_comma_list_sentences(_md_answer)
                _md_answer = _strip_broken_risk_intro_sentences(_md_answer)
                _md_answer = _strip_further_research_sentences(_md_answer)
                # Strip uncited summary sentences (catches [PMCxxxxxxx] placeholders
                # — the improve model sometimes copies exemplar IDs verbatim).
                _md_answer = _strip_uncited_summary_sentences(_md_answer)
                # Hard-cap: improved summary may exceed 150 words.
                _md_answer = _truncate_summary_to_150_words(_md_answer)

        # Enforce minimum word count (130 words, citations excluded) after any
        # improve-model splice.  Also serves as safety net when judge skips rewrite
        # (improved_summary=None) but finalize still produced a short summary due
        # to truncation or sentence-boundary rollback.
        if _summary_word_count(_md_answer) < 60:
            _md_answer = _pad_summary_from_ke_final(_md_answer, min_wc=60, target_wc=120)
            # The pad step draws sentences from Key Evidence bullets; dedupe and cap.
            _md_answer = _dedupe_summary_sentences(_md_answer)
            _md_answer = _truncate_summary_to_150_words(_md_answer)

        # Final citation guarantee: the Summary must never reach the user with
        # zero citations while valid evidence citations exist (tolerant check —
        # matches bare, hyperlinked, and INCLUDE-tagged citation forms).
        _sum_final_m = _re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)",
                                  _md_answer, _re.S)
        if (result.get("citations") and _sum_final_m
                and "[EVIDENCE_GAP]" not in _md_answer
                and "Insufficient Evidence" not in _md_answer
                and not _re.search(r"\[PMC\d{4,}:", _sum_final_m.group(1))):
            print("[CITE_GUARANTEE] Summary had no citations; appending fallback")
            _md_answer = _append_summary_citation(_md_answer, result["citations"][0])
            _md_answer = _hyperlink_inline_pmc(_md_answer)

        response = {
            "markdown": _md_answer,
            "structured": result.get("structured"),
            "citations": result.get("citations"),
            "trace": result.get("trace"),
            "raw_evidence": result.get("raw_evidence"),
            "judge_scores": result.get("judge_scores"),
            "improved_summary": result.get("improved_summary"),
        }

        if req.include_hits:
            response["hits"] = result.get("hits")
            response["re_ranked_hits"] = result.get("re_ranked_hits")

        # Do not cache evidence gap responses so a rephrased query gets a fresh attempt.
        # Also do not cache responses missing the # Reasoning section — they are
        # incomplete (generated before the Reasoning fix) and must not persist.
        _md_for_cache = _md_answer
        _reason_cache_m = _re.search(r"(?im)^#\s+Reasoning\s*\n(.*?)(?=^#\s+|\Z)", _md_for_cache, _re.S | _re.M)
        _cacheable = (
            "Insufficient Evidence" not in _md_for_cache
            and bool(_reason_cache_m and _reason_cache_m.group(1).strip())
        )
        if _cacheable:
            _query_cache[key] = response
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")
    finally:
        rag.cfg.enable_judge = _orig_judge


@app.post(
    "/cache/clear",
    tags=["Cache"],
    summary="Clear in-memory query cache",
    description=(
        "Invalidates all cached RAG responses. **Always run this after restarting the API or "
        "making code changes** — the cache persists across requests within a single process lifetime."
    ),
)
def clear_cache():
    """Clear the query response cache."""
    _query_cache.clear()
    return {"status": "ok", "message": "Query cache cleared"}

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Vector search only (no synthesis)",
    description=(
        "Runs only the retrieval step — embeds the query via the active embedder and searches Milvus. "
        "No reranking or LLM synthesis is performed.\n\n"
        "Useful for:\n"
        "- Inspecting which chunks the retriever returns for a given query\n"
        "- Debugging retrieval quality before running the full pipeline\n"
        "- Verifying Milvus collection contents\n\n"
        "Each hit includes: `pmcid`, `section`, `title`, `text`, `score`, `pub_year`, `pub_type`, `mesh_terms`, `keywords`."
    ),
)
def search(req: SearchRequest):
    """
    Debug endpoint — vector search without synthesis.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        hits = rag.retriever.search(req.query, top_k=req.top_k or cfg.top_k_default)
        return {"hits": hits, "count": len(hits)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post(
    "/embed",
    response_model=EmbedResponse,
    tags=["Embeddings"],
    summary="Embed a single text via the active embedder (DeepInfra, 2048-dim)",
    description=(
        "Embeds a single text using the active embedder (default: DeepInfra "
        "`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim; the `nemotron` provider uses a "
        "locally-loaded 4096-dim model instead).\n\n"
        "| `mode` | Instruction prefix | Use for |\n"
        "|--------|-------------------|---------|\n"
        "| `query` | Added under `nemotron` (`Instruct: Retrieve relevant biomedical passages\\nQuery: `); no-op under DeepInfra (symmetric encoding) | Retrieval queries |\n"
        "| `document` | None | Passages, model responses, ground truth |\n\n"
        "Used by Notebook 6 (UMAP) to embed query and model responses via the production API "
        "rather than loading the model a second time."
    ),
)
def embed(req: EmbedRequest):
    """
    Embed a single text using the active embedder (default: DeepInfra, symmetric encoding).
    mode='query'    -> adds instruction prefix (nemotron provider only; no-op on DeepInfra)
    mode='document' -> no prefix
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        if req.mode == "query":
            vec = rag.embeddings.embed_query(req.text)
        else:
            vec = rag.embeddings.embed_documents([req.text])[0]
        return {"embedding": vec, "dim": len(vec)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed error: {str(e)}")

@app.post(
    "/embed_batch",
    response_model=EmbedBatchResponse,
    tags=["Embeddings"],
    summary="Batch-embed multiple texts via the active embedder (DeepInfra, 2048-dim)",
    description=(
        "Batch-embeds multiple texts using the active embedder (default: DeepInfra "
        "`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim; the `nemotron` provider uses a "
        "locally-loaded 4096-dim model instead).\n\n"
        "| `mode` | Instruction prefix | Use for |\n"
        "|--------|-------------------|---------|\n"
        "| `document` | None (default) | Passages, model responses, ground truth — used by BERTScore in Notebook 5 |\n"
        "| `query` | Added to each text under `nemotron`; no-op under DeepInfra | Batch retrieval query embedding |\n\n"
        "Returns a list of embedding vectors (dimensionality matches the active embedder) in the same "
        "order as `texts`. Internally calls `embed_documents()` (batch-optimized) for `document` mode "
        "and `embed_query()` per text for `query` mode."
    ),
)
def embed_batch(req: EmbedBatchRequest):
    """
    Batch-embed multiple texts using the active embedder (default: DeepInfra, symmetric encoding).
    mode='document' -> no prefix (default; for passages, responses, GT — used by BERTScore)
    mode='query'    -> instruction prefix added to each text (for retrieval query batches)
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        if req.mode == "query":
            vecs = [rag.embeddings.embed_query(t) for t in req.texts]
        else:
            vecs = rag.embeddings.embed_documents(req.texts)
        dim = len(vecs[0]) if vecs else 0
        return {"embeddings": vecs, "dim": dim, "count": len(vecs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed batch error: {str(e)}")

# ─── Auto-insert embeddings at startup if Milvus collection is empty ────────
async def _auto_insert_if_empty() -> None:
    """
    Background startup task: waits for Milvus to become healthy, checks the
    vector collection, and auto-inserts from pickle files if it is empty.
    Runs concurrently — the API is fully responsive (healthcheck, /health)
    while the insert is in progress. The Streamlit UI will show a yellow
    "Vector Database Empty" banner until the insert finishes.
    """
    import urllib.request
    from pymilvus import connections, Collection, utility

    milvus_uri = os.getenv("MILVUS_URI") or f"http://{cfg.milvus_host}:{cfg.milvus_port}"
    milvus_host = os.getenv("MILVUS_HOST", cfg.milvus_host)
    milvus_health_url = f"http://{milvus_host}:9091/healthz"
    pickle_dir_env = os.getenv("PICKLE_DIR")
    if not pickle_dir_env:
        # No hardcoded fallback: a silent wrong-path default here previously pointed at a
        # 2-generations-stale pickle dir (INCLUDE_PUB_by_2025_11_14_nemotron_4096) that, if it
        # ever happened to exist, would auto-insert stale data into an empty production
        # collection with no warning. Fail loud-but-safe instead — skip auto-insert entirely
        # and require PICKLE_DIR to be set explicitly (docker-compose.yml sets it).
        print("[AutoInsert] WARNING: PICKLE_DIR not set — skipping auto-insert (set it in "
              "docker-compose.yml / .env to enable auto-populate-if-empty).")
        return
    pickle_dir = Path(pickle_dir_env)

    # ── 1. Wait for Milvus healthz (non-blocking, up to 3 min) ───────────────
    print("[AutoInsert] Waiting for Milvus to become healthy...")
    milvus_ready = False
    for attempt in range(36):          # 36 × 5 s = 180 s max
        try:
            with urllib.request.urlopen(milvus_health_url, timeout=3) as resp:
                if resp.status == 200:
                    milvus_ready = True
                    print(f"[AutoInsert] Milvus healthy (attempt {attempt + 1})")
                    break
        except Exception:
            pass
        await asyncio.sleep(5)

    if not milvus_ready:
        print("[AutoInsert] WARNING: Milvus not healthy after 3 min — skipping auto-insert.")
        return

    # ── 2. Check collection entity count (blocking I/O in executor) ──────────
    loop = asyncio.get_event_loop()

    def _check_count() -> int:
        try:
            connections.connect(alias="default", uri=milvus_uri)
            if not utility.has_collection(cfg.milvus_collection):
                return 0
            col = Collection(cfg.milvus_collection)
            col.load()
            return col.num_entities
        except Exception as exc:
            print(f"[AutoInsert] Could not check collection: {exc}")
            return 0

    count = await loop.run_in_executor(None, _check_count)
    print(f"[AutoInsert] Collection '{cfg.milvus_collection}': {count:,} entities")

    if count > 0:
        print("[AutoInsert] Collection has data — skipping auto-insert.")
        return

    # ── 3. Auto-insert from pickle files (blocking subprocess in executor) ────
    if not pickle_dir.exists():
        print(f"[AutoInsert] Pickle directory not found: {pickle_dir} — skipping.")
        return

    pkl_count = len(list(pickle_dir.glob("*.pkl")))
    if pkl_count == 0:
        print(f"[AutoInsert] No pickle files in {pickle_dir} — skipping.")
        return

    print(f"[AutoInsert] Empty collection — inserting {pkl_count} pickle files...")

    def _run_insert() -> int:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/embed_and_insert.py",
                "--pickle-dir", str(pickle_dir),
                "--insert",
            ],
            timeout=600,   # 10-min guard
        )
        return result.returncode

    try:
        rc = await loop.run_in_executor(None, _run_insert)
        if rc == 0:
            print("[AutoInsert] Insert completed successfully.")
        else:
            print(f"[AutoInsert] Insert exited with code {rc}.")
    except subprocess.TimeoutExpired:
        print("[AutoInsert] Insert timed out after 10 minutes.")
    except Exception as exc:
        print(f"[AutoInsert] Insert error: {exc}")


# ─── FastAPI lifecycle events ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("T21 Research Assistant RAG API Starting")
    print(f"Chat Provider: {cfg.provider_chat}")
    print(f"Embed Provider: {cfg.provider_embed}")
    print(f"Milvus: {cfg.milvus_host}:{cfg.milvus_port}/{cfg.milvus_collection}")
    print("=" * 50)
    # Kick off Milvus check + auto-insert as a background coroutine.
    # The API starts serving immediately; insert runs concurrently.
    asyncio.ensure_future(_auto_insert_if_empty())

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down RAG API...")
