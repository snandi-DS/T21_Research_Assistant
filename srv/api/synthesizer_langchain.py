#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain-based Multi-Agent RAG Synthesizer for DS-Chat
WITH SECTION-AWARE RERANKING using LangGraph for agent orchestration

This implementation uses:
- LangChain for LLM integration (OpenAI, Anthropic, Google, Ollama)
- LangChain Milvus for vector retrieval
- LangGraph for multi-agent orchestration

Usage (Python):
    from synthesizer_langchain import RAGConfig, LangChainMultiAgentRAG
    cfg = RAGConfig().override_with_env()
    rag = LangChainMultiAgentRAG(cfg)
    out = rag.answer("How is sleep apnea related to Down syndrome?", top_k=6)
    print(out["markdown"])
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated
from operator import add

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel

# LangGraph imports
from langgraph.graph import StateGraph, END
try:
    from langgraph.prebuilt import ToolNode  # type: ignore[import-untyped]
except ImportError:
    ToolNode = None  # type: ignore[assignment]  # Not used in current implementation

# LangChain LLM providers
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None


class OpenAIEmbeddingsDirect:
    """Minimal OpenAI embeddings wrapper that avoids tiktoken downloads.

    Some environments block access to `openaipublic.blob.core.windows.net`, which
    `tiktoken` may need at runtime. This class calls the OpenAI embeddings API
    directly via the official `openai` client and does not use tokenization.
    """

    def __init__(self, model: str, api_key: str):
        from openai import OpenAI

        base_url = os.getenv("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        # OpenAI returns data entries with an index field.
        data_sorted = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data_sorted]

# =============================================================================
# Nemotron Model Classes
# =============================================================================

class NemotronEmbeddings:
    """Embeddings using nvidia/llama-embed-nemotron-8b (HuggingFace).

    Asymmetric retrieval:
    - Queries receive an instruction prefix
    - Documents/passages have no prefix

    Output dimension: 4096 (requires MILVUS_DIM=4096 in .env and a matching collection).
    """

    def __init__(
        self,
        model_name: str = "nvidia/llama-embed-nemotron-8b",
        query_instruction: str = "Instruct: Retrieve relevant biomedical passages\nQuery: ",
        max_length: int = 512,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch and transformers are required for NemotronEmbeddings. "
                "Run: pip install torch transformers"
            )
        from transformers import AutoTokenizer, AutoModel  # type: ignore[import-untyped]

        self.query_instruction = query_instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[NemotronEmbed] Loading {model_name} on {self._device} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(self._device)
        self._model.eval()
        print("[NemotronEmbed] Model loaded.")

    def _encode(self, texts: List[str]) -> "torch.Tensor":
        """Encode texts with mean-pooling + L2 normalisation."""
        all_embeddings: List["torch.Tensor"] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            attn = encoded["attention_mask"]
            tok_emb = outputs.last_hidden_state
            mask_exp = attn.unsqueeze(-1).expand(tok_emb.size()).float()
            emb = torch.sum(tok_emb * mask_exp, dim=1)
            emb = emb / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with the instruction prefix."""
        full_text = f"{self.query_instruction}{text}"
        return self._encode([full_text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (no instruction prefix)."""
        return self._encode(texts).tolist()


class DeepInfraEmbeddings:
    """Embeddings via DeepInfra's OpenAI-compatible endpoint.

    Model: nvidia/llama-nemotron-embed-vl-1b-v2 (output dim: 2048).
    Query and documents are encoded symmetrically (no instruction prefix) —
    the OpenAI-compatible endpoint does not expose a query/passage input_type,
    so consistent encoding on both sides guarantees retrieval alignment.

    Requires MILVUS_DIM=2048 and a matching 2048-dim collection.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
        base_url: str = "https://api.deepinfra.com/v1/openai",
        batch_size: int = 32,
    ):
        import requests as _requests
        self._requests = _requests
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._batch_size = max(1, int(batch_size))
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"[DeepInfraEmbed] Configured model={model} endpoint={self._url}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = {"input": batch, "model": self._model, "encoding_format": "float"}
            resp = self._requests.post(self._url, json=payload, headers=self._headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend([r["embedding"] for r in rows])
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)


class DeepInfraReranker:
    """Cross-encoder reranker using nvidia/llama-nemotron-rerank-vl-1b-v2 via DeepInfra REST API.

    Calls POST https://api.deepinfra.com/v1/inference/<model> with Bearer auth.
    Returns a relevance score in [0, 1] for each document.
    """

    def __init__(
        self,
        api_key: str,
        url: str = "https://api.deepinfra.com/v1/inference/nvidia/llama-nemotron-rerank-vl-1b-v2",
    ):
        import requests as _requests  # local import — already in stdlib-adjacent stdlib
        self._requests = _requests
        self._url = url
        self._headers = {
            "Authorization": f"bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"[DeepInfraReranker] Configured endpoint: {url}")

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return relevance scores in [0, 1] for each (query, document) pair."""
        payload = {"queries": [query], "documents": documents}
        resp = self._requests.post(self._url, json=payload, headers=self._headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # DeepInfra returns: {"rankings": [{"document": {...}, "relevance_score": float, "index": int}, ...]}
        rankings = data.get("rankings") or data.get("results") or []
        if rankings:
            indexed = sorted(rankings, key=lambda r: r.get("index", 0))
            return [float(r.get("relevance_score", r.get("score", 0.0))) for r in indexed]
        # Fallback: if the shape is a flat list of scores
        if isinstance(data, list):
            return [float(x) for x in data]
        return [0.0] * len(documents)


class NemotronCrossEncoderReranker:
    """Cross-encoder reranker using nvidia/llama-nemotron-rerank-1b-v2 (HuggingFace).

    Scores (query, document) pairs directly — no LLM prompt required.
    Returns a relevance probability in [0, 1] for each document.
    """

    def __init__(
        self,
        model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
        max_length: int = 512,
        batch_size: int = 4,
        device: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch and transformers are required for NemotronCrossEncoderReranker. "
                "Run: pip install torch transformers"
            )
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[import-untyped]

        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[NemotronRerank] Loading {model_name} on {self._device} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(self._device)
        self._model.eval()
        print("[NemotronRerank] Model loaded.")

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return relevance scores in [0, 1] for each (query, document) pair."""
        scores: List[float] = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            queries = [query] * len(batch_docs)
            inputs = self._tokenizer(
                queries,
                batch_docs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            for j in range(logits.shape[0]):
                if logits.shape[-1] == 2:
                    sc = torch.softmax(logits[j], dim=-1)[1].item()
                else:
                    sc = torch.sigmoid(logits[j][0]).item()
                scores.append(float(sc))
        return scores


def _ollama_call_settings() -> dict:
    """Read per-call latency-bounding settings from the environment (no RAGConfig
    threading needed). All values have safe defaults.

    - OLLAMA_TIMEOUT   : per-call httpx timeout (s). For a streaming call this bounds
                         the gap BETWEEN chunks (and the initial connect/first-token),
                         not the total — that is what _stream_chat_bounded's deadline is for.
    - OLLAMA_DEADLINE  : hard wall-clock cap (s) for a single streaming attempt.
    - OLLAMA_FALLBACK_MODEL : DeepInfra model used when Ollama times out/fails for
                         synthesis/improve (default a reliable gpt-oss, NOT the
                         engine_overloaded-prone Nemotron-120B).
    """
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, "") or default)
        except (TypeError, ValueError):
            return default
    return {
        "timeout": _f("OLLAMA_TIMEOUT", 20.0),
        "deadline": _f("OLLAMA_DEADLINE", 25.0),
        "fallback_model": os.environ.get("OLLAMA_FALLBACK_MODEL", "openai/gpt-oss-120b"),
        "di_base": os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),
        "di_key": os.environ.get("DEEPINFRA_API_KEY", ""),
        "di_effort": os.environ.get("DEEPINFRA_REASONING_EFFORT", "none"),
    }


def _stream_chat_bounded(client, *, model, messages, deadline, label,
                         think=False, options=None, fmt=None) -> str:
    """Accumulate streamed Ollama chunks, raising TimeoutError once total elapsed
    exceeds `deadline`. The httpx Client `timeout` bounds inter-chunk gaps / a dead
    socket; this loop bounds TOTAL wall-clock (a slow-trickling stream). Prints the
    wall-clock duration for observability."""
    t0 = time.time()
    out = []
    kwargs = {"messages": messages, "stream": True, "think": think, "options": options or {}}
    if fmt is not None:
        kwargs["format"] = fmt
    for part in client.chat(model, **kwargs):
        out.append(part.message.content or "")
        if time.time() - t0 > deadline:
            raise TimeoutError(f"{label}: exceeded wall-clock deadline {deadline:.0f}s")
    print(f"  [OLLAMA_TIMING] {label}: {time.time() - t0:.1f}s")
    return "".join(out)


def _deepinfra_fallback_chat(messages, *, model, base_url, api_key, seed=42,
                             reasoning_effort="none", schema=None, timeout=180.0) -> str:
    """One-shot DeepInfra OpenAI-compatible chat used as a failover when Ollama is
    slow/unavailable. Flattens LangChain messages into system+user. If `schema` is
    given, appends a JSON-only instruction (best-effort, not grammar-constrained)."""
    import requests as _requests
    _sys = "\n\n".join(m.content for m in messages if getattr(m, "type", "") == "system")
    _usr = "\n\n".join(m.content for m in messages if getattr(m, "type", "") != "system")
    if schema is not None:
        _sys += ("\n\nReturn ONLY a JSON object matching this schema — no prose, no "
                 "code fences:\n" + json.dumps(schema))
    payload = {
        "model": model, "temperature": 0.0, "seed": seed, "max_tokens": 4096,
        "messages": [{"role": "system", "content": _sys}, {"role": "user", "content": _usr}],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    t0 = time.time()
    resp = _requests.post(
        base_url.rstrip("/") + "/chat/completions", json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    print(f"  [OLLAMA_FALLBACK] DeepInfra {model}: {time.time() - t0:.1f}s")
    return resp.json()["choices"][0]["message"]["content"].strip()


class NemotronOllamaLLM(BaseChatModel):
    """LangChain-compatible wrapper for nemotron-3-super via Ollama remote API.

    Uses the `ollama` Python client with a custom host and Bearer token,
    exactly as shown in the reference notebook.
    """

    model_name: str = "nemotron-3-super"
    base_url: str = "https://ollama.com"
    api_key: str = ""
    temperature: float = 0.0
    seed: int = 42

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            from ollama import Client  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("ollama package not installed. Run: pip install ollama")

        _cs = _ollama_call_settings()
        client = Client(
            host=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=_cs["timeout"],   # bound dead-socket / inter-chunk gaps
        )

        _ROLE_MAP = {
            "human": "user",
            "system": "system",
            "ai": "assistant",
            "chat": "user",
        }
        ollama_messages = [
            {"role": _ROLE_MAP.get(getattr(m, "type", "human"), "user"), "content": m.content}
            for m in messages
        ]
        _opts = {
            "temperature": self.temperature,
            "seed": self.seed,
            "top_k": 1,        # int (not 1.0) — greedy decoding for reproducibility
            "top_p": 1.0,
        }

        # Bounded primary: Ollama stream with a wall-clock deadline, retried once.
        # think=False keeps the hidden chain-of-thought off (major latency win).
        response = None
        for _attempt in (1, 2):
            try:
                response = _stream_chat_bounded(
                    client, model=self.model_name, messages=ollama_messages,
                    deadline=_cs["deadline"], label=f"_generate try{_attempt}",
                    think=False, options=_opts,
                )
                break
            except Exception as e:  # TimeoutError, connection errors, etc.
                print(f"  [OLLAMA_WARN] _generate try{_attempt} failed: {type(e).__name__}: {e}")

        # Failover: DeepInfra (reliable, bounded 180s) so we never hang for minutes.
        if response is None:
            if not _cs["di_key"]:
                raise RuntimeError("Ollama remote timed out and no DEEPINFRA_API_KEY for fallback.")
            response = _deepinfra_fallback_chat(
                messages, model=_cs["fallback_model"], base_url=_cs["di_base"],
                api_key=_cs["di_key"], seed=self.seed, reasoning_effort=_cs["di_effort"],
            )

        ai_msg = AIMessage(content=response.strip())
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def invoke_structured(
        self,
        messages: List[BaseMessage],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke with JSON schema structured output via Ollama ``format`` parameter.

        Ollama passes the schema to the model as a grammar constraint, so the
        response is always valid JSON matching the schema.  Returns the parsed dict.
        """
        try:
            from ollama import Client  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("ollama package not installed. Run: pip install ollama")

        _cs = _ollama_call_settings()
        client = Client(
            host=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=_cs["timeout"],
        )
        _ROLE_MAP = {"human": "user", "system": "system", "ai": "assistant", "chat": "user"}
        ollama_messages = [
            {"role": _ROLE_MAP.get(getattr(m, "type", "human"), "user"), "content": m.content}
            for m in messages
        ]
        _opts = {
            "temperature": self.temperature,
            "seed": self.seed,
            "top_k": 1,        # int (not 1.0) — greedy decoding for reproducibility
            "top_p": 1.0,
            "num_predict": 4096,   # enough headroom for full structured JSON
        }
        # Bounded primary: stream with format=schema, wall-clock deadline, retried once.
        response_text = None
        for _attempt in (1, 2):
            try:
                response_text = _stream_chat_bounded(
                    client, model=self.model_name, messages=ollama_messages,
                    deadline=_cs["deadline"], label=f"invoke_structured try{_attempt}",
                    think=False, options=_opts, fmt=schema,
                )
                if response_text.strip():
                    break
            except Exception as e:
                print(f"  [OLLAMA_WARN] invoke_structured try{_attempt} failed: {type(e).__name__}: {e}")
                response_text = None
        # Failover: DeepInfra with schema-in-prompt (best-effort JSON). If this also
        # fails or returns non-JSON, the caller (_synthesize_structured) catches the
        # exception and falls back to the now-bounded markdown chain (_generate).
        if not response_text or not response_text.strip():
            if not _cs["di_key"]:
                raise ValueError("Ollama structured call timed out and no DEEPINFRA_API_KEY for fallback.")
            response_text = _deepinfra_fallback_chat(
                messages, model=_cs["fallback_model"], base_url=_cs["di_base"],
                api_key=_cs["di_key"], seed=self.seed, reasoning_effort=_cs["di_effort"],
                schema=schema,
            )
        stripped = response_text.strip()
        if not stripped:
            raise ValueError(
                "Ollama remote + DeepInfra fallback both returned empty structured response."
            )
        return json.loads(stripped)

    @property
    def _llm_type(self) -> str:
        return "nemotron-ollama-remote"


try:
    from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
except ImportError:
    ChatAnthropic = None  # type: ignore[assignment]

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # type: ignore[import-untyped]
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]
    GoogleGenerativeAIEmbeddings = None  # type: ignore[assignment]

try:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    ChatOllama = None
    OllamaEmbeddings = None

# PyTorch / HuggingFace — needed for Nemotron local models
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# Milvus integration
try:
    from langchain_milvus import Milvus
except ImportError:
    try:
        from langchain_community.vectorstores import Milvus
    except ImportError:
        Milvus = None

try:
    from pymilvus import connections, Collection, utility
except ImportError:
    connections = None
    Collection = None
    utility = None


# =============================================================================
# Section Priority Weights
# =============================================================================
# NOTE: All four active sections are intentionally equal (1.0).
# Section priority multiplier applied directly to the blended score:
#   final = combined * priority
#
# Production / standard mode: only Results/Discussion/Methods/Conclusion are
# retrieved, so Abstract/Introduction/FullText/NA are never scored.
# Their 0.0 entries are a safety net — if a stray chunk appears it is zeroed.
#
# with_abstract mode (parameter sweep): sections_override is set to include
# Abstract and Introduction. The reranker detects this and gives ALL sections
# in the override list priority=1.0 so Abstract/Intro chunks compete on equal
# footing. SECTION_PRIORITY is NOT consulted for those sections in that mode.
SECTION_PRIORITY = {
    "Results":      1.0,
    "Discussion":   1.0,
    "Methods":      1.0,
    "Conclusion":   1.0,
    "Abstract":     0.0,
    "Introduction": 0.0,
    "FullText":     0.0,
    "NA":           0.0,
}


# =============================================================================
# Utility Functions (reused from original)
# =============================================================================
def _now_ms() -> int:
    return int(time.time() * 1000)

def _unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _short_hash(s: str, n: int = 10) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def _normalize_section(section: str) -> str:
    if not section:
        return "NA"
    s = section.strip().title()
    aliases = {
        "Result": "Results",
        "Method": "Methods",
        "Conclusions": "Conclusion",
        "Intro": "Introduction",
        "Disc": "Discussion",
        "Full Text": "FullText",
        "Fulltext": "FullText",
    }
    return aliases.get(s, s)

def _compose_citation_tag(h: Dict[str, Any]) -> str:
    pmcid = str(h.get("pmcid") or "").strip()
    pmid = str(h.get("pmid") or "").strip()
    doi = str(h.get("doi") or "").strip()
    section = str(h.get("section") or "").strip() or "NA"

    if pmcid:
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        return f"{pmcid}:{section}"
    if pmid:
        return f"PMID{pmid}:{section}"
    if doi:
        return f"{doi}:{section}"
    return f"NA:{section}"

def _get_dynamic_top_k(question: str, base_k: int = 12, max_k: int = 30) -> int:
    q_lower = question.lower()
    complex_keywords = [
        'compare', 'versus', 'vs', 'difference', 'similar', 'contrast',
        'all', 'comprehensive', 'complete', 'overview', 'summarize',
        'how many', 'list', 'what are the', 'types of', 'categories',
        'mechanisms', 'pathways', 'treatments', 'therapies',
        'subgroups', 'subtypes', 'classification',
        # list-type question starters
        'which', 'what conditions', 'what factors', 'what genes',
        'lower risk', 'higher risk', 'protective', 'risk factors',
        'associated with', 'related to', 'linked to',
    ]
    boost = 0
    for kw in complex_keywords:
        if kw in q_lower:
            boost = 5
            break
    if len(question) > 100:
        boost = max(boost, 3)
    if ' and ' in q_lower and len(question) > 50:
        boost = max(boost, 4)
    # Broad enumeration questions ("which conditions", "higher/lower risk") need
    # wider retrieval coverage to avoid missing key comorbidities (e.g. Alzheimer's).
    _broad_list_re = re.compile(
        r'\b(which|what)\b.{0,80}\b(conditions?|diseases?|comorbid|complications?|risks?)\b'
        r'|\b(higher risk|lower risk|at risk for|risk factors?)\b',
        re.IGNORECASE,
    )
    if _broad_list_re.search(question):
        boost = max(boost, 8)
    return min(base_k + boost, max_k)


# ── Question-type classifier ──────────────────────────────────────────────────
# Returns one of: "enumeration", "mechanism", "epidemiology", "general"
# Used by the retrieval node to bias section diversity toward the most
# informative section for each question type.
_ENUM_PATTERNS = re.compile(
    r'\b(which|what are the|what is the list|list (all|the)|name (all|the)|'
    r'identify (all|the)|how many (types|kinds|forms|genes|proteins|receptors|'
    r'subunits|subtypes|mutations|variants|pathways|factors|conditions|features)|'
    r'enumerate|what genes|what proteins|what receptors|what factors|what conditions|'
    r'what subunits|what mutations|what variants|what types|what kinds)\b',
    re.IGNORECASE,
)
_MECH_PATTERNS = re.compile(
    r'\b(how does|why does|why do|how do|what causes|what drives|what underlies|'
    r'mechanism|pathway|pathways|signaling|downstream|upstream|cascade|'
    r'what is the role of|what is the function of|how is .{1,40} regulated|'
    r'what leads to|what results in|by what)\b',
    re.IGNORECASE,
)
_EPI_PATTERNS = re.compile(
    r'\b(prevalence|incidence|how common|what percentage|what proportion|'
    r'how frequent|frequency|what fraction|what rate|what is the rate|'
    r'risk of|odds of|how often)\b',
    re.IGNORECASE,
)


def _detect_question_type(question: str) -> str:
    """Classify question into enumeration / mechanism / epidemiology / general.

    Used to adaptively bias section diversity in retrieval:
    - enumeration: Discussion/Conclusion list all class members  → boost Discussion
    - mechanism:   Discussion synthesises pathway details        → boost Discussion
    - epidemiology: Results sections hold statistics            → boost Results
    - general:     use env-var defaults
    """
    if _ENUM_PATTERNS.search(question):
        return "enumeration"
    if _MECH_PATTERNS.search(question):
        return "mechanism"
    if _EPI_PATTERNS.search(question):
        return "epidemiology"
    return "general"


def _fix_unbracketed_citations(md: str) -> str:
    pattern = r'(?<!\[)\b(PMC\d{4,}:[A-Za-z0-9_-]+)\b'
    return re.sub(pattern, r'[\1]', md)


def _normalize_pmc_citation_format(md: str) -> str:
    """Normalize non-standard PMC citation tags to the canonical [PMCxxxxxxxx:Section] form.

    The LLM occasionally wraps extra prose inside citation brackets, e.g.:
      [high-relevance source (PMC10306300):Discussion]
      [(PMC10306300):Discussion]
      [[PMC10306300:Discussion]]   ← double-bracket (model error)
    This function extracts the PMC ID and section label and rewrites the tag
    as [PMC10306300:Discussion] so that all downstream processing works correctly.
    Standard tags [PMCxxxxxxxx:Section] pass through unchanged.
    """
    # ── Pre-step A: Unwrap outer double-brackets around MULTIPLE inner citations ─
    # [[PMCxx:Sec1][PMCyy:Sec2]]          → [PMCxx:Sec1] [PMCyy:Sec2]
    # [[PMCxx:Sec1][PMCyy:Sec2][PMCzz:Sec3]] → [PMCxx:Sec1] [PMCyy:Sec2] [PMCzz:Sec3]
    _multi_dbl = re.compile(r'\[\[([^\[\]]+(?:\]\[[^\[\]]+)+)\]\]')
    md = _multi_dbl.sub(
        lambda m: ' '.join(f'[{p}]' for p in re.split(r'\]\[', m.group(1))),
        md,
    )

    # ── Pre-step B: Split fused PMC citations jammed into one bracket ──────────
    # [PMCxx:SectionPMCyy:Section]         → [PMCxx:Section] [PMCyy:Section]
    # [PMCxx:SectionPMCyy:Section, INCLUDE] → [PMCxx:Section] [PMCyy:Section, INCLUDE]
    # Triggered when the LLM concatenates two citations without any separator.
    def _split_fused(m: re.Match) -> str:
        inner = m.group(1)
        # Insert ] [ boundary wherever an alphabetic section label runs directly
        # into the start of the next PMC\d+ token (no space or punctuation between).
        split = re.sub(r'([A-Za-z][A-Za-z0-9_-]*)(?=PMC\d{4,})', r'\1] [', inner)
        if split != inner:
            return f'[{split}]'
        return m.group(0)
    md = re.sub(r'\[([^\[\]]+PMC\d{4,}[^\[\]]+)\]', _split_fused, md)

    # ── Pre-step C: Ensure a space between directly adjacent citation tokens ────
    # [PMCxx:Section][PMCyy:Section] → [PMCxx:Section] [PMCyy:Section]
    md = re.sub(r'(\])\s*(\[PMC\d)', r'\1 \2', md)

    # Strip double-bracket citations: [[PMCxx:Section]] → [PMCxx:Section]
    # The LLM sometimes emits [[PMC:Section]] which after hyperlinking becomes
    # [[PMC:Section](url)] — an invalid markdown link displayed as raw text.
    md = re.sub(
        r'\[\[(PMC\d{4,}:[A-Za-z0-9_-]+(?:,\s*INCLUDE)?)\]\]',
        r'[\1]',
        md,
    )
    # Normalize space-INCLUDE to comma-INCLUDE anywhere inside brackets (single and compound forms)
    # Handles: [PMCxx:Section INCLUDE] and [...; PMCxx:Section INCLUDE; ...] and INCLUDE](url)
    md = re.sub(
        r'(PMC\d{4,}:[A-Za-z0-9_-]+)\s+INCLUDE(?=[;,\]\(])',
        r'\1, INCLUDE',
        md,
    )
    return re.sub(
        r'\[(?:[^\[\]]*?\()?(PMC\d{4,})(?:\))?:([A-Za-z0-9_-]+)\]',
        r'[\1:\2]',
        md,
    )


def _fix_misplaced_bracket_citations(md: str) -> str:
    """Fix citations where the LLM closes the bracket before the section label.

    Transforms: [PMC10306300]:Results]  ->  [PMC10306300:Results]
    Also handles: ([PMC10306300]:Results]) -> ([PMC10306300:Results])
    These malformed tags are invisible to _hyperlink_inline_pmc and render
    as raw text (e.g. '([PMC10306300]):Results]') in the Streamlit UI.
    """
    return re.sub(
        r'\[PMC(\d{4,})\]:([A-Za-z][A-Za-z0-9 ]*)\]',
        r'[PMC\1:\2]',
        md,
    )


def _fix_citations_after_period(md: str) -> str:
    """Move inline citations that were placed AFTER a sentence-ending period to
    BEFORE the period, so that the citation ends the sentence correctly.

    Transforms: "...gene. [PMC123:Results] Next sentence"
    Into:        "...gene [PMC123:Results]. Next sentence"

    Only moves consecutive [PMC:Section] tags that directly follow a period
    (with optional whitespace) and are themselves followed by a space/newline
    or end of string — i.e., they are standalone post-period tags, not part of
    a hyperlinked reference that already has the right structure.
    """
    # Match: a sentence-ending period (the misplaced one), then OPTIONAL horizontal
    # whitespace, then one or more bracketed [PMCxxx:Sec] tags in plain [tag] form
    # (not yet hyperlinked), then an OPTIONAL trailing sentence punctuation.
    # Two leak shapes are handled:
    #   "lesions. [PMC:Results] Next"   (period, space, citation)
    #   "lesions.[PMC:Results]."        (period glued to citation, plus trailing period)
    # Only [ \t]* (horizontal whitespace, never \n) sits between the period and the
    # citation, so a citation at the very end of the Summary body can NEVER reach
    # across the newline into '# Key Evidence' and merge it inline.
    _AFTER_PERIOD_RE = re.compile(
        r'([.!?])[ \t]*((?:\[PMC\d+:[^\]]+\][ \t]*)+)([.!?])?',
        re.MULTILINE,
    )
    def _move_tags(m: re.Match) -> str:
        lead = m.group(1)                       # the misplaced leading punctuation
        tags = m.group(2).strip()
        tags = re.sub(r'\]\s*\[', '] [', tags)  # space between adjacent tags
        tags = re.sub(r'[ \t]+', ' ', tags)
        trail = m.group(3) or lead              # keep original sentence punctuation
        return f' {tags}{trail} '

    def _cleanup(body: str) -> str:
        body = re.sub(r'[ \t]{2,}', ' ', body)            # collapse double spaces
        body = re.sub(r'[ \t]+([.!?,;:])', r'\1', body)   # no space before punctuation
        body = re.sub(r'[ \t]+(\n|\Z)', r'\1', body)      # no trailing space at line/string end
        return body

    # Only apply inside the Summary and Key Evidence sections (not Reasoning)
    for header in ('Summary', 'Key Evidence'):
        sec_m = re.search(
            rf'(?im)^#\s+{header}\s*\n(.*?)(?=^#\s+|\Z)', md, re.S
        )
        if not sec_m:
            continue
        new_body = _cleanup(_AFTER_PERIOD_RE.sub(_move_tags, sec_m.group(1)))
        if new_body != sec_m.group(1):
            md = md[: sec_m.start(1)] + new_body + md[sec_m.end(1):]
    return md


def _split_semicolon_citations(md: str) -> str:
    """Expand semicolon-separated citation lists into individual bracketed tokens.

    Models sometimes output [PMC123:Results; PMC456:Discussion] as a single
    bracket.  This function rewrites such expressions as separate tokens:
    [PMC123:Results] [PMC456:Discussion]
    so that _hyperlink_inline_pmc can link each one individually.
    """
    def _expand(m: re.Match) -> str:
        inner = m.group(1)          # everything between [ and ]
        # Normalize space-INCLUDE → comma-INCLUDE per part before validating
        # e.g. "PMC123:Discussion INCLUDE" → "PMC123:Discussion, INCLUDE"
        parts = [
            re.sub(r'\s+INCLUDE$', ', INCLUDE', p.strip(), flags=re.IGNORECASE)
            for p in inner.split(";")
        ]
        # Each part may look like PMCxxxxxxxx:Section or PMCxxxxxxxx:Section, INCLUDE
        valid = re.compile(r'^PMC\d{4,}:[A-Za-z0-9_-]+(?:,\s*INCLUDE)?$')
        # Deduplicate while preserving order (e.g. same PMC repeated twice)
        seen: set = set()
        unique: list = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        tokens = [f"[{p}]" for p in unique if valid.match(p)]
        if not tokens:
            return m.group(0)       # leave unchanged if nothing valid matched
        return " ".join(tokens)    # works for 1 token (deduped) or multiple

    # Match a bracket containing ≥1 semicolon where every segment is PMC:Section
    return re.sub(r'\[([^\]\[]+;[^\]\[]+)\]', _expand, md)


def _add_missing_citation_periods(md: str) -> str:
    """Add a trailing period when a sentence-ending [PMC:Section] citation has none.

    Handles two cases:
    1. End-of-line: "...text [PMC123:Results]\\n"  ->  "...text [PMC123:Results].\\n"
    2. Mid-paragraph: "...text [PMC123:Results] Next sentence" ->
                      "...text [PMC123:Results]. Next sentence"

    Lines that already end with a period (or other punctuation) after the tag
    are left unchanged.  Applied only within Summary and Key Evidence sections.
    """
    # Case 1: citation at end of line with no following punctuation
    _eol_pattern = re.compile(r'(\[PMC\d+:[^\]]+\])([ \t]*)$', re.MULTILINE)
    # Case 2: citation mid-paragraph followed by space + uppercase (next sentence)
    _mid_pattern = re.compile(r'(\[PMC\d+:[^\]]+\])( +)(?=[A-Z])')

    for header in ('Summary', 'Key Evidence'):
        sec_m = re.search(
            rf'(?im)^#\s+{header}\s*\n(.*?)(?=^#\s+|\Z)', md, re.S
        )
        if not sec_m:
            continue
        new_body = sec_m.group(1)
        # Regex to detect lowercase conjunction continuation on next line
        _conj_re = re.compile(
            r'[ \t]*[\r\n]+[ \t]*(?:and|or|but|which|that|,)\b', re.IGNORECASE
        )
        def _eol_sub(m, _body=new_body):
            # Don't add period if next line starts with a lowercase conjunction
            # (citation was mid-sentence; line just wrapped)
            if _conj_re.match(_body[m.end():]):
                return m.group(0)
            # Don't add period if immediately preceded by a comma
            before = _body[:m.start()]
            if before.rstrip()[-1:] == ',':
                return m.group(0)
            return m.group(1) + '.' + m.group(2)
        new_body = _eol_pattern.sub(_eol_sub, new_body)
        new_body = _mid_pattern.sub(lambda m: m.group(1) + '. ', new_body)
        if new_body != sec_m.group(1):
            md = md[:sec_m.start(1)] + new_body + md[sec_m.end(1):]
    return md


def _inject_include_tags(md: str, hits: list) -> str:
    """Append ', INCLUDE' to citation tags from INCLUDE-funded papers.

    Transforms  [PMCxxxxxxx:Results]  →  [PMCxxxxxxx:Results, INCLUDE]
    for papers where is_include=True in the retrieved hit list.

    Runs AFTER all normalisation / stripping passes (so the canonical
    [PMCxx:Section] format is stable) and BEFORE _hyperlink_inline_pmc.
    """
    include_pmcids: set[str] = set()
    for h in hits:
        if not h.get("is_include"):
            continue
        pmcid = str(h.get("pmcid") or "").strip()
        if not pmcid:
            continue
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        include_pmcids.add(pmcid.upper())

    if not include_pmcids:
        return md

    def _repl(m: re.Match) -> str:
        pmc_full = m.group(1)   # e.g. PMC12345678
        section  = m.group(2)   # e.g. Results
        if pmc_full.upper() in include_pmcids:
            return f"[{pmc_full}:{section}, INCLUDE]"
        return m.group(0)

    # Only modify plain [PMCxx:Section] tags — skip already-tagged ones
    return re.sub(
        r'\[(PMC\d{4,}):([A-Za-z0-9_-]+)\](?!, INCLUDE)(?!\()',
        _repl,
        md,
    )


def _hyperlink_inline_pmc(md: str) -> str:
    """Convert bare inline PMCID:Section citations into markdown links.

    Processes: Summary, Key Evidence, Reasoning.
    """
    from typing import Tuple

    def _section_span(text: str, header: str, level: int = 1) -> Tuple[int,int]:
        """Find span for a section header. level=1 for #, level=2 for ##"""
        prefix = "#" * level
        m = re.search(rf"^{prefix}\s+{re.escape(header)}\s*$", text, flags=re.MULTILINE)
        if not m: return -1,-1
        start = m.end()
        # Find next header at same or higher level
        m2 = re.search(rf"^#{{{1},{level}}}\s+[A-Za-z].*$", text[start:], flags=re.MULTILINE)
        end = start + (m2.start() if m2 else len(text) - start)
        return start,end

    # Identify sections to hyperlink citations
    spans = []
    for header in ("Summary", "Key Evidence", "Reasoning"):
        s,e = _section_span(md, header)
        if s != -1 and e != -1:
            spans.append((s,e))
    if not spans:
        return md

    # Rebuild document applying replacements only inside spans
    parts = []
    last = 0
    for s,e in spans:
        parts.append(md[last:s])
        segment = md[s:e]
        # 1) Link bracketed [PMCxxxxxxx:Section] or [PMCxxxxxxx:Section, INCLUDE] citations
        def _repl_bracketed(m):
            token   = m.group(1)   # full bracket token including optional ", INCLUDE"
            pmc_num = m.group(2)
            return f"{token}(https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_num}/)"
        segment2 = re.sub(
            r"(\[PMC(\d{4,}):[A-Za-z0-9_-]+(?:,\s*INCLUDE)?\])(?!\()",
            _repl_bracketed,
            segment,
        )
        # 2) Link bare PMCxxxxxxx IDs (Reasoning prose style) — not already inside a link
        #    Lookbehind (?<!\[) avoids re-linking already-bracketed [PMCxxxxxx...] tokens.
        #    Lookahead (?![\d:/\]]) avoids matching inside URLs (.../articles/PMC12345678/).
        def _repl_bare(m):
            pmc_id = m.group(0)  # e.g. PMC10306300
            pmc_num = m.group(1)
            return f"[{pmc_id}](https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/)"
        segment2 = re.sub(r"(?<!\[)PMC(\d{4,})(?![\d:/\]])", _repl_bare, segment2)
        parts.append(segment2)
        last = e
    parts.append(md[last:])
    _result = "".join(parts)

    # ── Post-hyperlink cleanup: remove residual citation artifacts ──────────────
    # Fixes: [[PMCxx:Section](url)] (outer lone bracket wrapping a hyperlink)
    _result = re.sub(
        r'\[(\[PMC[^\[\]]+\]\(https?://[^)]+\))\](?!\()',
        r'\1',
        _result,
    )
    # Fixes: [PMCxx:Section](url)]:SectionPMCyy:Section, INCLUDE  (trailing artifact)
    _result = re.sub(
        r'(\(https?://[^)]+\))\]?:?[A-Za-z0-9_]*PMC\d{4,}[A-Za-z0-9_:]*(?:,\s*INCLUDE)?',
        r'\1',
        _result,
    )
    return _result


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class RAGConfig:
    # Provider switches
    # Default to Nemotron stack if env vars are not provided.
    provider_chat: str = os.environ.get("PROVIDER_CHAT", "ollama_remote").lower()
    provider_embed: str = os.environ.get("PROVIDER_EMBED", "nemotron").lower()

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "").strip())
    embed_model: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5.4")
    rerank_model: str = os.environ.get("OPENAI_RERANK_MODEL", "gpt-5.4")

    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "").strip())
    anthropic_chat_model: str = os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
    anthropic_rerank_model: str = os.environ.get("ANTHROPIC_RERANK_MODEL", "claude-sonnet-4-6")

    # Google
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "").strip())
    google_chat_model: str = os.environ.get("GOOGLE_CHAT_MODEL", "gemini-1.5-pro")
    google_embed_model: str = os.environ.get("GOOGLE_EMBED_MODEL", "models/embedding-001")
    google_rerank_model: str = os.environ.get("GOOGLE_RERANK_MODEL", "gemini-1.5-pro")

    # Ollama (local)
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_chat_model: str = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1")
    ollama_embed_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_rerank_model: str = os.environ.get("OLLAMA_RERANK_MODEL", "llama3.1")

    # Nemotron — Ollama remote (provider_chat="ollama_remote") + HuggingFace local models
    # NOTE: NemotronEmbeddings outputs 4096-dim vectors.
    #       If switching from a 1536-dim collection, set MILVUS_DIM=4096 and re-ingest.
    provider_rerank: str = os.environ.get("PROVIDER_RERANK", "nemotron").lower()
    deepinfra_api_key: str = field(default_factory=lambda: os.environ.get("DEEPINFRA_API_KEY", "").strip())
    deepinfra_rerank_url: str = os.environ.get(
        "DEEPINFRA_RERANK_URL",
        "https://api.deepinfra.com/v1/inference/nvidia/llama-nemotron-rerank-vl-1b-v2",
    )
    # DeepInfra OpenAI-compatible chat + embeddings (provider_chat="deepinfra" / provider_embed="deepinfra")
    deepinfra_base_url: str = os.environ.get(
        "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
    )
    deepinfra_chat_model: str = os.environ.get(
        "DEEPINFRA_CHAT_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
    )
    deepinfra_embed_model: str = os.environ.get(
        "DEEPINFRA_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    deepinfra_embed_batch_size: int = int(os.environ.get("DEEPINFRA_EMBED_BATCH_SIZE", "32"))
    deepinfra_judge_model: str = os.environ.get("DEEPINFRA_JUDGE_MODEL", "openai/gpt-oss-120b")
    # Nemotron-3-Super is a reasoning model; "none" suppresses the chain-of-thought it would
    # otherwise emit inline into the answer (also ~9x less output → much faster synthesis).
    deepinfra_reasoning_effort: str = os.environ.get("DEEPINFRA_REASONING_EFFORT", "none")
    nemotron_embed_model: str = os.environ.get("NEMOTRON_EMBED_MODEL", "nvidia/llama-embed-nemotron-8b")
    nemotron_rerank_model: str = os.environ.get("NEMOTRON_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
    nemotron_chat_model: str = os.environ.get("NEMOTRON_CHAT_MODEL", "nemotron-3-super")
    nemotron_ollama_base_url: str = os.environ.get("NEMOTRON_OLLAMA_BASE_URL", "https://ollama.com")
    nemotron_ollama_api_key: str = field(default_factory=lambda: os.environ.get("OLLAMA_API_KEY", "").strip())
    nemotron_embed_instruction: str = os.environ.get(
        "NEMOTRON_EMBED_INSTRUCTION",
        "Instruct: Retrieve relevant biomedical passages\nQuery: "
    )
    nemotron_embed_batch_size: int = 8
    nemotron_embed_max_length: int = 512
    nemotron_rerank_batch_size: int = 4
    nemotron_device: str = os.environ.get("NEMOTRON_DEVICE", "")  # empty = auto-detect

    # Milvus
    milvus_host: str = os.environ.get("MILVUS_HOST", "127.0.0.1")
    milvus_port: str = os.environ.get("MILVUS_PORT", "19530")
    milvus_collection: str = os.environ.get("MILVUS_COLLECTION", "DS_INCLUDE_2026_09_02_DEEPINFRA_2048")
    milvus_search_params_json: str = os.environ.get("MILVUS_SEARCH_PARAMS", '{"metric_type":"IP","params":{"nprobe":64}}')

    # Behavior
    strict_mode: bool = True
    min_sources: int = 2
    min_hits: int = 1
    min_avg_sim: float = 0.40  # Lowered from 0.50: broad/generic questions have lower top-hit vector scores
    no_answer_text: str = "No relevant information found in the available literature to answer this question."
    top_k_default: int = 8
    max_context_chars: int = 12000
    evidence_only_summary: bool = os.environ.get("EVIDENCE_ONLY_SUMMARY", "false").strip().lower() in ("1","true","yes","y","on")
    focus_question_terms: bool = os.environ.get("FOCUS_QUESTION_TERMS", "false").strip().lower() in ("1","true","yes","y","on")

    # Agent toggles
    enable_rerank: bool = True
    enable_safety_guard: bool = True
    enable_judge: bool = True

    # Reranker settings
    rerank_alpha: float = 0.6

    # Per-section retrieval
    chunks_per_section: int = int(os.environ.get("CHUNKS_PER_SECTION", "12"))

    def require(self):
        if (self.provider_chat == "openai" or self.provider_embed == "openai") and not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set while OpenAI provider in use")
        if self.provider_chat == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set while Anthropic provider in use")
        if (self.provider_chat == "google" or self.provider_embed == "google") and not self.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY not set while Google provider in use")
        if self.provider_chat == "ollama_remote" and not self.nemotron_ollama_api_key:
            raise RuntimeError("OLLAMA_API_KEY not set while ollama_remote provider in use")
        if (self.provider_chat == "deepinfra" or self.provider_embed == "deepinfra") and not self.deepinfra_api_key:
            raise RuntimeError("DEEPINFRA_API_KEY not set while DeepInfra provider in use")

    def override_with_env(self) -> "RAGConfig":
        """Override config with environment variables."""
        env_map = {
            "PROVIDER_CHAT": "provider_chat",
            "PROVIDER_EMBED": "provider_embed",
            "OPENAI_API_KEY": "openai_api_key",
            "OPENAI_EMBED_MODEL": "embed_model",
            "OPENAI_CHAT_MODEL": "chat_model",
            "OPENAI_RERANK_MODEL": "rerank_model",
            "MILVUS_HOST": "milvus_host",
            "MILVUS_PORT": "milvus_port",
            "MILVUS_COLLECTION": "milvus_collection",
            "MILVUS_SEARCH_PARAMS": "milvus_search_params_json",
        }
        bools = {
            "STRICT_MODE": "strict_mode",
            "ENABLE_RERANK": "enable_rerank",
            "ENABLE_SAFETY_GUARD": "enable_safety_guard",
            "EVIDENCE_ONLY_SUMMARY": "evidence_only_summary",
            "FOCUS_QUESTION_TERMS": "focus_question_terms",
            "ENABLE_JUDGE": "enable_judge",
        }
        ints = {
            "MIN_SOURCES": "min_sources",
            "TOP_K_DEFAULT": "top_k_default",
            "MAX_CONTEXT_CHARS": "max_context_chars",
            "NEMOTRON_EMBED_BATCH_SIZE": "nemotron_embed_batch_size",
            "NEMOTRON_EMBED_MAX_LENGTH": "nemotron_embed_max_length",
            "NEMOTRON_RERANK_BATCH_SIZE": "nemotron_rerank_batch_size",
            "DEEPINFRA_EMBED_BATCH_SIZE": "deepinfra_embed_batch_size",
        }
        floats = {"RERANK_ALPHA": "rerank_alpha", "MIN_AVG_SIM": "min_avg_sim"}

        # Nemotron-specific string fields handled by env_map
        env_map.update({
            "PROVIDER_RERANK": "provider_rerank",
            "DEEPINFRA_API_KEY": "deepinfra_api_key",
            "DEEPINFRA_RERANK_URL": "deepinfra_rerank_url",
            "DEEPINFRA_BASE_URL": "deepinfra_base_url",
            "DEEPINFRA_CHAT_MODEL": "deepinfra_chat_model",
            "DEEPINFRA_EMBED_MODEL": "deepinfra_embed_model",
            "DEEPINFRA_JUDGE_MODEL": "deepinfra_judge_model",
            "DEEPINFRA_REASONING_EFFORT": "deepinfra_reasoning_effort",
            "NEMOTRON_EMBED_MODEL": "nemotron_embed_model",
            "NEMOTRON_RERANK_MODEL": "nemotron_rerank_model",
            "NEMOTRON_CHAT_MODEL": "nemotron_chat_model",
            "NEMOTRON_OLLAMA_BASE_URL": "nemotron_ollama_base_url",
            "OLLAMA_API_KEY": "nemotron_ollama_api_key",
            "NEMOTRON_EMBED_INSTRUCTION": "nemotron_embed_instruction",
            "NEMOTRON_DEVICE": "nemotron_device",
        })
        for e, attr in env_map.items():
            val = os.environ.get(e)
            if val:
                setattr(self, attr, val)
        for e, attr in bools.items():
            val = os.environ.get(e)
            if val is not None:
                setattr(self, attr, val.strip().lower() in ("1", "true", "yes", "y", "on"))
        for e, attr in ints.items():
            val = os.environ.get(e)
            if val is not None:
                try:
                    setattr(self, attr, int(val))
                except:
                    pass
        for e, attr in floats.items():
            val = os.environ.get(e)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except:
                    pass
        return self


# =============================================================================
# LangChain LLM Factory
# =============================================================================
def make_langchain_llm(cfg: RAGConfig, role: str = "chat"):
    """Create LangChain chat model based on provider configuration."""
    provider = cfg.provider_chat.lower()

    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
        model = {
            "chat": cfg.anthropic_chat_model,
            "rerank": cfg.anthropic_rerank_model,
        }.get(role, cfg.anthropic_chat_model)
        return ChatAnthropic(
            model=model,
            api_key=cfg.anthropic_api_key,
            temperature=0.0,
            max_tokens=2048,
        )

    elif provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
        model = {
            "chat": cfg.google_chat_model,
            "rerank": cfg.google_rerank_model,
        }.get(role, cfg.google_chat_model)
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=cfg.google_api_key,
            temperature=0.0,
            max_output_tokens=2048,
        )

    elif provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("langchain-community not installed. Run: pip install langchain-community")
        model = {
            "chat": cfg.ollama_chat_model,
            "rerank": cfg.ollama_rerank_model,
        }.get(role, cfg.ollama_chat_model)
        return ChatOllama(
            model=model,
            base_url=cfg.ollama_base_url,
            temperature=0.0,
        )

    elif provider == "ollama_remote":
        # nemotron-3-super via Ollama remote API with Bearer auth
        return NemotronOllamaLLM(
            model_name=cfg.nemotron_chat_model,
            base_url=cfg.nemotron_ollama_base_url,
            api_key=cfg.nemotron_ollama_api_key,
            temperature=0.0,
            seed=42,
        )

    elif provider == "deepinfra":
        # NVIDIA Nemotron-3-Super via DeepInfra's OpenAI-compatible chat completions API
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed. Run: pip install langchain-openai")
        model = {
            "chat": cfg.deepinfra_chat_model,
            "rerank": cfg.deepinfra_chat_model,
        }.get(role, cfg.deepinfra_chat_model)
        # reasoning_effort=none suppresses Nemotron-3-Super's inline chain-of-thought.
        _di_extra = {}
        if (cfg.deepinfra_reasoning_effort or "").strip():
            _di_extra["reasoning_effort"] = cfg.deepinfra_reasoning_effort.strip()
        return ChatOpenAI(  # type: ignore[call-arg]
            model=model,
            api_key=cfg.deepinfra_api_key,  # type: ignore[arg-type]
            base_url=cfg.deepinfra_base_url,
            temperature=0.0,
            seed=42,
            max_tokens=4096,  # type: ignore[call-arg]
            extra_body=_di_extra or None,  # type: ignore[call-arg]
        )

    else:  # OpenAI (default)
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed. Run: pip install langchain-openai")
        model = {
            "chat": cfg.chat_model,
            "rerank": cfg.rerank_model,
        }.get(role, cfg.chat_model)
        # GPT-5 series models require max_completion_tokens instead of max_tokens
        is_gpt5_series = model.startswith("gpt-5") or model.startswith("gpt-4.1")
        if is_gpt5_series:
            return ChatOpenAI(  # type: ignore[call-arg]
                model=model,
                api_key=cfg.openai_api_key,  # type: ignore[arg-type]
                temperature=0.0,
                seed=42,
                max_completion_tokens=4096,
            )
        return ChatOpenAI(  # type: ignore[call-arg]
            model=model,
            api_key=cfg.openai_api_key,  # type: ignore[arg-type]
            temperature=0.0,
            seed=42,
            max_tokens=2048,  # type: ignore[call-arg]
        )


def make_langchain_embeddings(cfg: RAGConfig):
    """Create LangChain embeddings based on provider configuration."""
    provider = cfg.provider_embed.lower()

    if provider == "google":
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError("langchain-google-genai not installed")
        return GoogleGenerativeAIEmbeddings(
            model=cfg.google_embed_model,
            google_api_key=cfg.google_api_key,
        )

    elif provider == "ollama":
        if OllamaEmbeddings is None:
            raise RuntimeError("langchain-community not installed")
        return OllamaEmbeddings(
            model=cfg.ollama_embed_model,
            base_url=cfg.ollama_base_url,
        )

    elif provider == "deepinfra":
        # nvidia/llama-nemotron-embed-vl-1b-v2 via DeepInfra OpenAI-compatible API
        # Output dim: 2048 — ensure MILVUS_DIM=2048 and a matching collection.
        return DeepInfraEmbeddings(
            api_key=cfg.deepinfra_api_key,
            model=cfg.deepinfra_embed_model,
            base_url=cfg.deepinfra_base_url,
            batch_size=cfg.deepinfra_embed_batch_size,
        )

    elif provider == "nemotron":
        # nvidia/llama-embed-nemotron-8b loaded locally via HuggingFace
        # Output dim: 4096 — ensure MILVUS_DIM=4096 and re-ingest if switching from text-embedding-3-small
        device = cfg.nemotron_device if cfg.nemotron_device else None
        return NemotronEmbeddings(
            model_name=cfg.nemotron_embed_model,
            query_instruction=cfg.nemotron_embed_instruction,
            max_length=cfg.nemotron_embed_max_length,
            batch_size=cfg.nemotron_embed_batch_size,
            device=device,
        )

    else:  # OpenAI (default)
        # Prefer direct OpenAI embeddings to avoid tiktoken runtime downloads.
        # Set USE_LANGCHAIN_OPENAI_EMBEDDINGS=1 to force the LangChain wrapper.
        use_langchain_wrapper = os.getenv("USE_LANGCHAIN_OPENAI_EMBEDDINGS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )

        if use_langchain_wrapper:
            if OpenAIEmbeddings is None:
                raise RuntimeError("langchain-openai not installed")
            return OpenAIEmbeddings(
                model=cfg.embed_model,
                api_key=cfg.openai_api_key,  # type: ignore[arg-type]
            )

        return OpenAIEmbeddingsDirect(
            model=cfg.embed_model,
            api_key=cfg.openai_api_key,
        )


# =============================================================================
# LangGraph State Definition
# =============================================================================
class RAGState(TypedDict):
    """State for the RAG pipeline graph."""
    question: str
    top_k: int
    hits: List[Dict[str, Any]]
    reranked_hits: List[Dict[str, Any]]
    deduped_hits: List[Dict[str, Any]]  # Added for section deduplication
    markdown: str
    structured: Optional[Dict[str, Any]]
    citations: List[str]
    raw_evidence: List[Dict[str, Any]]
    safety_flagged: bool
    safety_patterns: List[str]
    missing_terms: List[str]
    unmatched_entities: List[str]  # Gene symbols / named entities in Q absent from all evidence
    question_verdict: str          # "VALID" | "INVALID:<reason>" — set by question_judge node
    trace: Annotated[List[Dict[str, Any]], add]
    error: Optional[str]
    judge_scores: Optional[Dict[str, Any]]   # {accuracy,informativeness,completeness 1-5 + justification}
    improved_summary: Optional[str]          # judge-guided rewrite with citation hyperlinks
    rerank_alpha_override: Optional[float]   # per-request alpha override for parameter sweep
    sections_override: Optional[List[str]]   # per-request section list override for parameter sweep


# =============================================================================
# LangChain Milvus Retriever
# =============================================================================
class LangChainMilvusRetriever:
    """Custom Milvus retriever using LangChain embeddings."""

    def __init__(self, cfg: RAGConfig, embeddings):
        self.cfg = cfg
        self.embeddings = embeddings
        self._coll: Any = None

    def _ensure_connection(self):
        if self._coll is not None:
            return
        if connections is None or Collection is None:
            raise RuntimeError("pymilvus is not installed")
        try:
            uri = os.environ.get("MILVUS_URI")
            token = os.environ.get("MILVUS_TOKEN")
            if uri:
                connections.connect(alias="default", uri=uri, token=token or "")  # type: ignore[arg-type]
            else:
                connections.connect(alias="default", host=self.cfg.milvus_host, port=self.cfg.milvus_port)
        except Exception:
            pass
        self._coll = Collection(self.cfg.milvus_collection)
        try:
            self._coll.load()
        except Exception:
            pass

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search Milvus using LangChain embeddings."""
        self._ensure_connection()

        # Get query embedding using LangChain
        query_vec = self.embeddings.embed_query(query)

        # Search Milvus
        try:
            params = json.loads(self.cfg.milvus_search_params_json)
        except json.JSONDecodeError:
            params = {"metric_type": "IP", "params": {"nprobe": 64}}

        res: Any = self._coll.search(
            data=[query_vec],
            anns_field="vector",
            param=params,
            limit=top_k,
            expr='pub_type == "article" and section in ["Results", "Discussion", "Methods", "Conclusion"]',
            output_fields=[
                "pk", "doi", "pmid", "pmcid", "openalex_id", "title",
                "year", "pub_type", "source", "language",
                "topic", "subfield", "pub_field", "domain", "sdg", "citations",
                "keywords", "mesh_terms", "authors", "institutions",
                "funders_openalex", "grants",
                "section", "chunk_id", "text",
                "is_include",
            ],
            timeout=60,
        )

        hits = []
        for hit in res[0]:
            ent = {
                "doi":              self._get_field(hit, "doi"),
                "pmid":             self._get_field(hit, "pmid"),
                "pmcid":            self._get_field(hit, "pmcid"),
                "openalex_id":      self._get_field(hit, "openalex_id"),
                "title":            self._get_field(hit, "title"),
                "year":             self._get_field(hit, "year"),
                "pub_type":         self._get_field(hit, "pub_type"),
                "source":           self._get_field(hit, "source"),
                "language":         self._get_field(hit, "language"),
                "topic":            self._get_field(hit, "topic"),
                "subfield":         self._get_field(hit, "subfield"),
                "pub_field":        self._get_field(hit, "pub_field"),
                "domain":           self._get_field(hit, "domain"),
                "sdg":              self._get_field(hit, "sdg"),
                "citations":        self._get_field(hit, "citations"),
                "keywords":         self._get_field(hit, "keywords"),
                "mesh_terms":       self._get_field(hit, "mesh_terms"),
                "authors":          self._get_field(hit, "authors"),
                "institutions":     self._get_field(hit, "institutions"),
                "funders_openalex": self._get_field(hit, "funders_openalex"),
                "grants":           self._get_field(hit, "grants"),
                "section":          self._get_field(hit, "section"),
                "chunk_id":         self._get_field(hit, "chunk_id"),
                "text":             self._get_field(hit, "text"),
                "is_include":       bool(self._get_field(hit, "is_include") or False),
                "score":            float(getattr(hit, "distance", 0.0)),
                "_pk":              self._get_field(hit, "pk") or getattr(hit, "id", None),
            }
            hits.append(ent)
        return hits

    def search_per_section(
        self,
        query: str,
        chunks_per_section: int = 12,
        sections: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-N chunks independently from each section, then combine.

        Unlike the single-query ``search()`` which retrieves across all sections at
        once (and can be dominated by whichever section has the highest vector
        similarity), this method fires one Milvus query per section so every section
        contributes equally to the candidate pool.  The reranker then scores the
        combined pool and surfaces the most relevant chunks regardless of section.

        Args:
            query:              The user question.
            chunks_per_section: Number of chunks to fetch from each section
                                (env var ``CHUNKS_PER_SECTION``, default 12).
            sections:           Sections to query; defaults to the four active
                                retrieval sections.
        Returns:
            Deduplicated list of hit dicts (combined across all sections).
        """
        self._ensure_connection()
        if sections is None:
            sections = ["Results", "Discussion", "Methods", "Conclusion"]

        # Embed once — reused for all section queries.
        query_vec = self.embeddings.embed_query(query)

        try:
            params = json.loads(self.cfg.milvus_search_params_json)
        except json.JSONDecodeError:
            params = {"metric_type": "IP", "params": {"nprobe": 64}}

        _OUTPUT_FIELDS = [
            "pk", "doi", "pmid", "pmcid", "openalex_id", "title",
            "year", "pub_type", "source", "language",
            "topic", "subfield", "pub_field", "domain", "sdg", "citations",
            "keywords", "mesh_terms", "authors", "institutions",
            "funders_openalex", "grants",
            "section", "chunk_id", "text",
            "is_include",
        ]

        seen_pks: set = set()
        all_hits: List[Dict[str, Any]] = []

        for section in sections:
            try:
                res: Any = self._coll.search(
                    data=[query_vec],
                    anns_field="vector",
                    param=params,
                    limit=chunks_per_section,
                    expr=f'pub_type == "article" and section == "{section}"',
                    output_fields=_OUTPUT_FIELDS,
                    timeout=60,
                )
            except Exception as exc:
                print(f"  [RETRIEVE_SECTION_WARN] section={section} failed: {exc}")
                continue

            for hit in res[0]:
                pk = self._get_field(hit, "pk") or getattr(hit, "id", None)
                if pk is not None and pk in seen_pks:
                    continue
                if pk is not None:
                    seen_pks.add(pk)
                ent = {
                    "doi":              self._get_field(hit, "doi"),
                    "pmid":             self._get_field(hit, "pmid"),
                    "pmcid":            self._get_field(hit, "pmcid"),
                    "openalex_id":      self._get_field(hit, "openalex_id"),
                    "title":            self._get_field(hit, "title"),
                    "year":             self._get_field(hit, "year"),
                    "pub_type":         self._get_field(hit, "pub_type"),
                    "source":           self._get_field(hit, "source"),
                    "language":         self._get_field(hit, "language"),
                    "topic":            self._get_field(hit, "topic"),
                    "subfield":         self._get_field(hit, "subfield"),
                    "pub_field":        self._get_field(hit, "pub_field"),
                    "domain":           self._get_field(hit, "domain"),
                    "sdg":              self._get_field(hit, "sdg"),
                    "citations":        self._get_field(hit, "citations"),
                    "keywords":         self._get_field(hit, "keywords"),
                    "mesh_terms":       self._get_field(hit, "mesh_terms"),
                    "authors":          self._get_field(hit, "authors"),
                    "institutions":     self._get_field(hit, "institutions"),
                    "funders_openalex": self._get_field(hit, "funders_openalex"),
                    "grants":           self._get_field(hit, "grants"),
                    "section":          self._get_field(hit, "section"),
                    "chunk_id":         self._get_field(hit, "chunk_id"),
                    "text":             self._get_field(hit, "text"),
                    "is_include":       bool(self._get_field(hit, "is_include") or False),
                    "score":            float(getattr(hit, "distance", 0.0)),
                    "_pk":              pk,
                }
                all_hits.append(ent)

        # Sort by vector score descending so callers get a consistent ordering
        # before reranking.
        all_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return all_hits

    @staticmethod
    def _get_field(hit, field: str):
        try:
            return hit.entity.get(field, None)
        except Exception:
            try:
                return getattr(hit, field)
            except Exception:
                return None



# =============================================================================
# Section Diversity Enforcement (Phase 2)
# =============================================================================
def apply_section_diversity(
    hits: List[Dict[str, Any]], 
    target_k: int = 20,
    min_results: int = 6,      # Minimum Results chunks
    min_discussion: int = 4,    # Minimum Discussion chunks  
    min_methods: int = 3        # Minimum Methods chunks
) -> List[Dict[str, Any]]:
    """
    Ensure diverse section representation in retrieval results.
    
    Strategy:
    1. Group hits by section
    2. Reserve slots for priority sections (Results, Discussion, Methods)
    3. Fill remaining slots with highest-scoring chunks
    
    This solves the problem where vector search favors Abstracts/Introductions
    (general language) over Results sections (specific technical data).
    """
    from collections import defaultdict
    
    if not hits:
        return hits
    
    # Group by section
    by_section = defaultdict(list)
    for hit in hits:
        section = _normalize_section(hit.get("section", "NA"))
        by_section[section].append(hit)
    
    # Sort each section by vector score (descending)
    for section in by_section:
        by_section[section].sort(
            key=lambda h: h.get("score", 0), 
            reverse=True
        )
    
    # Allocate slots for priority sections
    diverse = []
    
    # Priority 1: Results (highest value for research questions)
    diverse.extend(by_section.get("Results", [])[:min_results])
    
    # Priority 2: Discussion (interpretations and conclusions)
    diverse.extend(by_section.get("Discussion", [])[:min_discussion])
    
    # Priority 3: Methods (study design and protocols)
    diverse.extend(by_section.get("Methods", [])[:min_methods])
    
    # Fill remaining slots with best remaining chunks (maintain diversity)
    used_ids = {id(h) for h in diverse}
    remaining = [h for h in hits if id(h) not in used_ids]
    
    slots_left = target_k - len(diverse)
    if slots_left > 0:
        diverse.extend(remaining[:slots_left])
    
    # Maintain original vector score order for consistency
    diverse.sort(key=lambda h: h.get("score", 0), reverse=True)
    
    return diverse[:target_k]


def apply_paper_diversity(
    hits: List[Dict[str, Any]],
    target_k: int = 20,
    max_chunks_per_paper: int = 2,
) -> List[Dict[str, Any]]:
    """
    Cap the number of chunks contributed by any single paper (PMCID).
    Ensures answers draw from multiple papers rather than one dominant paper.
    Iterates in score order, accepting a chunk only if its paper has not yet
    reached max_chunks_per_paper.
    """
    counts: Dict[str, int] = {}
    selected = []

    for hit in hits:                          # already sorted by score descending
        pmcid = hit.get("pmcid") or hit.get("pmid") or "unknown"
        counts[pmcid] = counts.get(pmcid, 0)
        if counts[pmcid] < max_chunks_per_paper:
            selected.append(hit)
            counts[pmcid] += 1
        if len(selected) >= target_k:
            break

    return selected


# =============================================================================
# LangGraph Question Judge Node
# =============================================================================
_JUDGE_SYSTEM = """\
You are a strict topic-relevance classifier for a biomedical Q&A system specialising in \
Down syndrome (Trisomy 21) research.

Respond with EXACTLY ONE of these two outputs — no extra text, no preamble:

  VALID — the question is about DS/Trisomy 21 directly, OR about a condition, gene, \
mechanism, or comorbidity studied in the context of DS (e.g. Alzheimer's disease, \
leukemia, hypothyroidism, DYRK1A, APP, Ts65Dn/Dp16 models, interferon signalling, \
congenital heart defects, autism in DS, etc.), OR is a valid DS research question even \
if "Down syndrome" is not explicitly stated.

  INVALID: <one concise sentence explaining why> — use when:
  · The question has NO Down syndrome connection, even if otherwise biomedically valid.
  · The question is wholly off-topic (cryptocurrency, astrophysics, cooking, etc.).
  · The premise depends on a gene/entity that does not exist in any known database.
  · The question is a personal/biographical identity question about a named individual.

Examples:
  "What is the prevalence of autism in Down syndrome?" → VALID
  "Do Dp16 mice have diaphragmatic hernias?" → VALID
  "What causes type 2 diabetes in Down syndrome?" → VALID
  "What causes type 2 diabetes in the general population?" → INVALID: No Down syndrome connection.
  "How does chloroplast dysfunction cause DS?" → INVALID: Chloroplasts are plant organelles irrelevant to human biology.
  "How does gene FX84A9 drive co-occurring conditions in Down syndrome?" → INVALID: FX84A9 is not a known biological entity.
  "What are cryptocurrency trends in 2025?" → INVALID: Unrelated to Down syndrome research.
  "Who is Sutanu Nandi?" → INVALID: Biographical identity question, not a DS research question.
"""

# Personal treatment-advice-seeking phrasing ("drugs that I can use to treat X",
# "should I take Y", "what medication for my child"). Such questions are NOT
# answered at all — the question_judge pre-flight blocks them before retrieval
# and the API returns a decline + a suggested research-framed rephrase. Research
# questions about the same drugs ("what drugs have been studied to treat X in
# DS?") remain fully answerable. Shared with safety_guard as a backstop.
_DRUG_NOUNS = r"(?:drugs?|medications?|medicines?|treatments?|dose|dosage|therap(?:y|ies))"

_ADVICE_SEEKING_PATTERNS = [
    r"\b(?:can|could|should|shall|may|might|must|do)\s+i\b",          # "should I take", "can I use"
    # "drugs I can use" / "drug I should used" — both word orders, inflected verbs
    r"\b(?:i|we)\s+(?:can|could|should|shall|may|might|must)\s+"
    r"(?:use|used|take|taken|took|give|given|gave|start|started|stop|stopped|try|tried|get|receive)\b",
    r"\bmy\s+(?:child(?:ren)?|son|daughter|kid|baby|patients?|brother|sister|mother|father|mom|dad|wife|husband)\b",
    r"\b(?:for|to|help)\s+me\b",
    r"\bwhat\s+should\s+(?:i|we)\b",
    # "suggest/recommend which drug", "recommend a treatment", "prescribe some medication"
    r"\b(?:suggest|recommend|advise|prescribe)\s+(?:a\s+|an\s+|some\s+|the\s+|which\s+|what\s+)?" + _DRUG_NOUNS + r"\b",
    # "can you suggest ... drug", "could you recommend a good treatment" — drug noun anywhere later
    r"\b(?:can|could|would|will)\s+you\s+(?:suggest|recommend|advise|prescribe)\b(?=[^?]*\b" + _DRUG_NOUNS + r"\b)",
    # "which drug ... I ..." — drug-noun question with first-person reference
    r"\b(?:which|what)\s+" + _DRUG_NOUNS + r"\b[^?]*\b(?:i|me|my)\b",
]


def _is_advice_seeking(question: str) -> bool:
    """True when the question asks for personal treatment advice rather than research findings."""
    return any(re.search(p, question or "", flags=re.IGNORECASE) for p in _ADVICE_SEEKING_PATTERNS)


def _research_rephrase_suggestion(question: str) -> str:
    """Build a research-framed rephrase of an advice-seeking drug/treatment question."""
    q = (question or "").strip().rstrip("?").strip()

    def _clean(topic: str) -> str:
        return re.sub(r"^(?:my|our)\s+", "", topic.strip(), flags=re.IGNORECASE)

    m = re.search(r"\btreat(?:ing)?\s+(.+)$", q, flags=re.IGNORECASE)
    if m:
        return f"What drugs have been studied to treat {_clean(m.group(1))}?"
    m = re.search(r"\b(?:for|in|with)\s+(.+)$", q, flags=re.IGNORECASE)
    if m:
        return f"What drugs have been studied for {_clean(m.group(1))}?"
    return "What drugs have been studied for this condition in Down syndrome?"


_JUDGE_CACHE: Dict[str, str] = {}  # simple process-lifetime cache to avoid re-judging repeated Qs


def create_question_judge_node(llm, cfg: RAGConfig):
    """Create a question-judge node that runs before retrieval.

    The node makes a fast LLM call to classify the question as VALID or INVALID.
    INVALID questions short-circuit the pipeline via a conditional edge to END,
    populated with question_verdict = "INVALID:<reason>".

    The judge is deliberately biased toward VALID — it only blocks questions that
    are definitively off-topic or depend on non-existent biological entities.
    """

    def judge(state: RAGState) -> RAGState:
        t0 = _now_ms()
        question = state["question"]

        # Deterministic pre-check (before the LLM and before the cache): personal
        # treatment-advice-seeking questions are declined outright — the tool
        # reports published research, it does not recommend or prescribe.
        # answer() detects the INVALID_ADVICE marker and returns a dedicated
        # decline + rephrase-suggestion response instead of the evidence-gap text.
        if _is_advice_seeking(question):
            t1 = _now_ms()
            return {
                **state,
                "question_verdict": "INVALID_ADVICE",
                "trace": [{"agent": "question_judge", "timing_ms": t1 - t0,
                           "note": "Blocked: personal treatment-advice-seeking question"}],
            }

        # Fast path: already judged (e.g. warm cache)
        if question in _JUDGE_CACHE:
            verdict = _JUDGE_CACHE[question]
            t1 = _now_ms()
            return {
                **state,
                "question_verdict": verdict,
                "trace": [{"agent": "question_judge", "timing_ms": t1 - t0,
                           "note": f"Cached verdict: {verdict[:60]}"}],
            }

        verdict = "VALID"  # default — fail open
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            _sys = SystemMessage(content=_JUDGE_SYSTEM)
            _human = HumanMessage(content=f"Question: {question}")
            _resp = llm.invoke([_sys, _human])
            raw = (_resp.content if hasattr(_resp, "content") else str(_resp)).strip()
            # Accept only lines starting with VALID or INVALID
            first_line = raw.splitlines()[0].strip() if raw else "VALID"
            if first_line.upper().startswith("INVALID"):
                verdict = first_line  # keep full "INVALID: <reason>"
            else:
                verdict = "VALID"
        except Exception as exc:
            # On any LLM error, default to VALID so we don't block valid questions
            verdict = "VALID"
            _logger.warning(f"[JUDGE] LLM call failed ({exc}), defaulting to VALID")

        _JUDGE_CACHE[question] = verdict
        t1 = _now_ms()
        return {
            **state,
            "question_verdict": verdict,
            "trace": [{"agent": "question_judge", "timing_ms": t1 - t0,
                       "note": f"Verdict: {verdict[:80]}"}],
        }

    return judge


def _judge_route(state: RAGState) -> str:
    """Conditional edge: route to 'retrieve' or 'end_invalid' based on judge verdict."""
    verdict = state.get("question_verdict", "VALID")
    if verdict.upper().startswith("INVALID"):
        return "end_invalid"
    return "retrieve"


def create_invalid_end_node():
    """Stub node for INVALID questions — just passes state through; answer() handles output."""

    def end_invalid(state: RAGState) -> RAGState:
        return state

    return end_invalid


# =============================================================================
# LangGraph Retrieval Node
# =============================================================================
def create_retrieval_node(retriever: LangChainMilvusRetriever, cfg: RAGConfig):
    """Create retrieval node for LangGraph."""

    def retrieve(state: RAGState) -> RAGState:
        t0 = _now_ms()
        question = state["question"]
        base_k = state.get("top_k") or cfg.top_k_default
        top_k = _get_dynamic_top_k(question, base_k=base_k, max_k=max(base_k + 10, 30))

        # Per-section retrieval: fetch CHUNKS_PER_SECTION candidates from each of
        # the four active sections independently, then let the reranker decide which
        # are most relevant.  This guarantees equal representation from every section
        # before reranking, eliminating the need for apply_section_diversity.
        _chunks_per_section = cfg.chunks_per_section
        _sections = state.get("sections_override") or None  # None → default 4 sections
        all_hits = retriever.search_per_section(question, chunks_per_section=_chunks_per_section, sections=_sections)
        _sec_label = ",".join(_sections) if _sections else "default-4"
        total_retrieved = len(all_hits)
        print(f"[RETRIEVE] per-section mode: {_chunks_per_section}×{_sec_label} → {total_retrieved} unique chunks")

        # Evidence-first missing-term check using full retrieval pool (no truncation)
        missing_terms = _missing_terms_in_hits(question, all_hits)

        # Named-entity check: gene symbols / protein IDs in the question absent from
        # all evidence are flagged so the synthesis node can address them explicitly.
        unmatched_entities = _unmatched_named_entities(question, all_hits)

        # Apply a loose pre-reranker paper diversity cap: allow up to 4 chunks per
        # paper so the reranker has enough candidates from each paper to score.  The
        # strict cap (MAX_CHUNKS_PER_PAPER, default 2) is enforced post-reranking
        # inside the reranker node where the reranked order is preserved.
        diverse_hits = apply_paper_diversity(
            all_hits,
            target_k=min(top_k * 4, total_retrieved),
            max_chunks_per_paper=max(4, int(os.environ.get("MAX_CHUNKS_PER_PAPER", "2"))),
        )

        t1 = _now_ms()

        return {
            **state,
            "hits": diverse_hits,
            "top_k": top_k,
            "missing_terms": missing_terms,
            "unmatched_entities": unmatched_entities,
            "trace": [{
                "agent": "retrieval",
                "timing_ms": t1 - t0,
                "note": f"Per-section {_chunks_per_section}×4={total_retrieved} → {len(diverse_hits)} after paper-diversity cap"
            }],
        }

    return retrieve


def create_synthesis_node(llm, cfg: RAGConfig):
    """Create synthesis node using LangChain."""

    if cfg.strict_mode:
        evidence_gap_block = (
            "EVIDENCE GAP: If the corpus does not directly address the question, output ONLY "
            "\"[EVIDENCE_GAP]\" on its own line. Do NOT describe adjacent or tangential evidence; stop."
        )
    else:
        evidence_gap_block = (
            "EVIDENCE GAP: Only output \"[EVIDENCE_GAP]\" when the corpus is empty or clearly "
            "unrelated. Otherwise synthesize with explicit caveats about indirect evidence or "
            "limited coverage."
        )
    no_evidence_block = ""  # consolidated into evidence_gap_block above

    summary_style_block = ""
    if cfg.evidence_only_summary:
        summary_style_block = (
            "\n[EVIDENCE-ONLY MODE]\n"
            "The Summary must contain ONLY direct statements supported by the evidence snippets.\n"
            "Do NOT add interpretation, implications, or background beyond what is stated.\n"
            "Do NOT use phrases like \"Taken together\" or broader synthesis language.\n"
            "If the question asks \"which\", answer with a direct comma-separated list backed by citations.\n"
        )

    SYNTH_PROMPT = ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior biomedical researcher specializing in Down syndrome (Trisomy 21) genomics, neurodevelopment, and clinical comorbidities. Synthesize the retrieved evidence at the standard of a high-impact journal abstract (Lancet, JAMA, Nature Medicine): precise, data-anchored, and appropriately hedged.

[OUTPUT DISCIPLINE] Begin your reply with "# Summary" on the first line. No preamble, no thinking aloud ("We need to...", "Let me...", "Here is..."), no code fences. Output ONLY the three sections below as finished text: # Summary, # Key Evidence, # Reasoning.

[INVIOLABLE CONSTRAINTS]
1. EVIDENCE-LOCKED: use ONLY the provided snippets; insert zero external facts.
2. CITE ONLY PMC IDs that appear in the EVIDENCE CORPUS below. Never invent, guess, or fabricate a PMC ID; this is a CRITICAL FAILURE. Omit any claim you cannot cite.
3. NO CLINICAL DIRECTIVES: report drug/treatment findings as published ("researchers found metformin improved X"); never issue dosing or personal advice.
4. NUMERICAL INTEGRITY: preserve exact statistics; never round or approximate.
5. EPISTEMIC HONESTY: "evidence is insufficient" is valid; never fabricate certainty.
{summary_style_block}

[EVIDENCE WEIGHTING] Lead with Results (direct findings, statistics) and Methods; then Discussion and Conclusion (interpretation). Abstract and Introduction are contextual only. Prefer concordant findings across independent studies. When the same finding appears in multiple sections, cite Results over Discussion. Cite at least 3 different PMC IDs across the response wherever the evidence supports it.

[QUESTION-TYPE ADAPTATION]
- MECHANISM (why, how, what role, what pathway): state the molecular chain (gene to protein to phenotype) FIRST from Discussion/Conclusion; statistics confirm, not replace it.
- EPIDEMIOLOGY (how common, what percentage, what risk, how many): lead with Results-section quantitative data (prevalence, OR, HR, RR, CI, p-value).
- LIST ("which X are..."): if 6 or fewer members, name the FULL set explicitly EXACTLY ONCE, in the first sentence (e.g. "Four chromosome-21 interferon receptors are triplicated in DS: IFNAR1, IFNAR2, IFNGR2, and IL10RB [PMC:Results]."). In EVERY later sentence do NOT re-list the whole set; instead add a distinct per-member detail or a new finding (e.g. tissue specificity, magnitude, mechanism), referring to individual members by name. Repeating the same full enumeration across sentences is a quality failure. If more than 6 members, group into 3-4 categories, one sentence each with 2-3 named examples, never a single run-on comma dump. Put exhaustive enumeration in Key Evidence.
- COMORBIDITY ("which conditions, at higher risk"): answer with named chronic diagnoses (congenital heart defects, Alzheimer's disease, hypothyroidism, leukemia, autism, sleep apnea, epilepsy). Include condition prevalence/incidence; EXCLUDE acute-care severity metrics (ICU admission, ventilation, intubation, length of stay) unless the question asks about outcomes, prognosis, or mortality.
- TREATMENT: name specific agents, doses, and outcomes as reported; no prescriptive language.

# Summary  (target 130-150 words; HARD CAP 150, citations excluded from the count; do not annotate the heading with a word count)
Context-Content-Conclusion, a single flowing prose paragraph (no bullets):
1. CONTEXT: ONE orienting sentence on the SPECIFIC topic's significance (no citation). Never open with generic "Down syndrome is a chromosomal condition..." boilerplate; lead with the specific finding.
2. CONTENT: 3-5 plain past-tense sentences, inverted pyramid (most important quantitative or mechanistic finding FIRST; never repeat it). ONE primary statistic per finding. EACH sentence MUST place its [PMC:Section] citation INSIDE the sentence, immediately BEFORE the closing period: write "STAT1 was upregulated [PMC123:Results]." and NEVER "STAT1 was upregulated. [PMC123:Results]." or "STAT1 was upregulated.[PMC123:Results].". A finding sentence without a citation is a CRITICAL FAILURE. CONNECTED PROSE (REQUIRED): the Summary must read as one flowing argument, not a list of isolated facts. Open most content sentences (sentences 3 onward) with a causal or contrastive bridge that links it to the previous sentence, e.g. "This triplication drives...", "As a result...", "Consistent with this...", "In mouse models, ...", "At the tissue level, ...", "Mechanistically, ...", "In contrast to euploid controls, ...". Do NOT use the banned additive heralds ("Notably,", "Importantly,", "Furthermore,", "Interestingly,", "Taken together,"), but DO use meaningful connectors — a Summary of bald, unconnected sentences is a quality failure. Paraphrase; never copy a sentence verbatim or include sample sizes, group composition, or strain labels (those are Methods, not synthesis).
   WORKED EXAMPLE (connected prose):
     DISCONNECTED (bad): "Trisomy 21 triplicates IFNAR1, IFNAR2, IFNGR2 and IL10RB [PMC1:Discussion]. RNA-seq confirmed upregulation of Ifnar and Ifngr in Dp mice [PMC2:Results]. Ifngr was upregulated in the cerebellum [PMC3:Discussion]."
     CONNECTED (good): "Trisomy 21 triplicates IFNAR1, IFNAR2, IFNGR2 and IL10RB [PMC1:Discussion]. This added gene dosage raises receptor expression: RNA-seq confirmed upregulation of Ifnar and Ifngr in Dp mouse progenitors [PMC2:Results]. The effect is tissue-specific, with Ifngr induction confined to the cerebellum in the Ts1Cje brain [PMC3:Discussion]."
3. CONCLUSION: ONE synthesis sentence that directly answers the question (no citation), naming the specific entity (gene, pathway, condition), never vague stand-ins ("these factors") and never raw data or "further research is needed".

# Key Evidence
4-6 standalone past-tense bullets; each states one specific finding with its quantitative support (effect size, n, p-value or CI where available) and ends in [PMC:Section]. Prefer Results-section sources. For list questions, one item per bullet.

# Reasoning
Exactly 4 bullets, 1-2 plain-prose sentences each, no citations:
- **Evidence selection:** why these findings were prioritized (study quality, section, sample size, replication).
- **Reasoning chain:** how the evidence connects to the Summary conclusion.
- **Confidence & gaps:** where the evidence is strongest and where it is limited.
- **Reader caveats:** what to bear in mind (study design, single-population data, confounders).

[STYLE]
- Past tense for findings ("was elevated", "demonstrated"); calibrate verbs to evidence strength (Results: "demonstrated/revealed/identified"; Discussion: "suggested/indicated"; single study: "may/appears to").
- Distinguish association from causation; use causal language only when the evidence does.
- Name specific genes and proteins; define abbreviations on first use ("amyloid precursor protein (APP)"). Use person-first language ("individuals with DS"). For eponyms, lead with the English name.
- NEVER use em dashes or en dashes: use commas, semicolons, colons, parentheses, or separate sentences; use a hyphen only for numeric ranges ("5-10%").
- AVOID: meta-commentary ("based on the evidence..."), DS boilerplate openers, weak heralds ("Notably,", "Importantly,", "Interestingly,", "Taken together,", "These results highlight..."), vague hedges, redundant attribution ("studies indicate"), hedging stacks ("may potentially suggest"), non-specific superlatives, and pipeline language ("corpus", "retrieved passages"; write "published literature").
- NO WORD-LEVEL REPETITION: do not lean on a filler word across sentences. In particular, never use "overall" as a verbal tic, and do not repeat the same adjective, adverb, or framing phrase (e.g. "global", "significant", "clear") in more than one sentence. Each sentence should use fresh wording; if a modifier adds no information, delete it.
- NO REPEATED ENUMERATION: state any list of genes/conditions/items ONCE (in the first sentence). NEVER re-list the same full set in a later sentence; later sentences must add a NEW finding per member or be cut. A summary that names the same gene list (e.g. "IFNAR1, IFNAR2, IFNGR2 and IL10RB") in more than one sentence is a FAILURE. When a later sentence reports a finding that applies to the WHOLE set, refer back with "these genes", "these receptors", or "their overexpression" instead of re-typing the member names (BAD: "Overexpression of IFNAR1, IFNAR2, IFNGR2 and IL10RB amplifies..."; GOOD: "Their overexpression amplifies...").
- NO ADJACENT DUPLICATE TOKEN: never write the same word twice in a row across a conjunction or comma (e.g. "Ifnar and Ifnar", "STAT1, STAT1"). Each item in a list must be distinct.
  WORKED EXAMPLE for the question "Which interferon receptors are triplicated in Down syndrome?":
    BAD (full set re-listed in two sentences; garbled duplicate token):
      "Trisomy 21 triplicates IFNAR1, IFNAR2, IFNGR2 and IL10RB on chromosome 21. The genes IFNAR1, IFNAR2, IFNGR2 and IL10RB are overexpressed in DS [PMC8273564:Discussion]. Ifnar and Ifnar were upregulated in Dp mice [PMC5012864:Results]."
    GOOD (full set named ONCE; each later sentence adds a NEW finding; distinct tokens):
      "Trisomy 21 triplicates four interferon receptor genes on chromosome 21: IFNAR1, IFNAR2, IFNGR2, and IL10RB [PMC8291705:Discussion]. These genes are overexpressed in Down syndrome, amplifying interferon responsiveness [PMC8273564:Discussion]. In Dp mice, Ifnar and Ifngr were upregulated in hematopoietic progenitors [PMC5012864:Results]. Ifngr induction was cerebellum-specific in the Ts1Cje brain [PMC4124147:Discussion]."
{evidence_gap_block}

[FINAL CHECK] Start with "# Summary"; output the three sections only; every Summary finding ends with a real [PMC:Section]; the Summary is 130-150 words and ends on a synthesis sentence; no em dashes.
"""),
        ("human", """QUESTION: {question}

EVIDENCE CORPUS (ranked by relevance score, highest first):
{context}

[!] Cite ONLY PMC IDs from the EVIDENCE CORPUS above. Do NOT invent PMC IDs.
{list_instruction}""")])

    chain = SYNTH_PROMPT | llm | StrOutputParser()
    lenient_chain = None
    if not cfg.strict_mode:
        lenient_system = (
            "You are a senior biomedical researcher specializing in Down syndrome. "
            "Synthesize the evidence in past tense at journal-abstract precision.\n"
            "Rules: (1) Use ONLY the provided evidence — no external facts. "
            "(2) Every finding sentence must end with [PMCxxxxxxx:Section] before the period. "
            "(3) Hedge appropriately when evidence is indirect or from a single study. "
            "(4) Do NOT output [EVIDENCE_GAP]; if evidence is sparse, synthesize with explicit caveats. "
            "(5) 130-150 words; no em dashes; no repeated enumeration."
        )
        if cfg.evidence_only_summary:
            lenient_system += (
                " (6) Evidence-only mode: include ONLY direct statements; no broader implications."
            )
        lenient_system += (
            "\n\nFormat (all three sections REQUIRED):\n"
            "# Summary\n<flowing prose paragraph, 130-150 words>\n\n"
            "# Key Evidence\n- <finding> [PMC:Section]\n\n"
            "# Reasoning\n- **Evidence selection:** ...\n- **Reasoning chain:** ...\n"
            "- **Confidence & gaps:** ...\n- **Reader caveats:** ..."
        )

        LENIENT_PROMPT = ChatPromptTemplate.from_messages([
            ("system", lenient_system),
            ("human", """[QUESTION]
{question}

[EVIDENCE]
{context}"""),
        ])
        lenient_chain = LENIENT_PROMPT | llm | StrOutputParser()

    def synthesize(state: RAGState) -> RAGState:
        t0 = _now_ms()
        hits = state.get("reranked_hits") or state.get("hits", [])
        question = state["question"]

        # Deduplicate by (pmcid, section) - keep only highest-scoring chunk per paper-section
        from collections import defaultdict
        section_map = defaultdict(list)
        for h in hits:
            pmcid = str(h.get("pmcid") or "").strip()
            section = _normalize_section(h.get("section") or "NA")
            key = (pmcid, section)
            section_map[key].append(h)
        
        # Keep highest-scoring chunk for each paper-section
        deduped = []
        for key, chunks in section_map.items():
            # Sort by rerank score (or vector score if rerank not available)
            chunks_sorted = sorted(
                chunks, 
                key=lambda h: h.get("_rerank_final") or h.get("score", 0), 
                reverse=True
            )
            deduped.append(chunks_sorted[0])  # Keep best chunk
        
        # Sort deduplicated chunks by score to maintain quality order
        deduped.sort(key=lambda h: h.get("_rerank_final") or h.get("score", 0), reverse=True)

        # Build context from deduplicated chunks
        blocks = []
        for h in deduped:
            text = _clean_evidence_chunk((h.get("text") or "").strip())
            tag = _compose_citation_tag(h)
            blocks.append(f"[{tag}]\n{text}")

        # Truncate context
        acc = []
        used = 0
        for b in blocks:
            if used + len(b) > cfg.max_context_chars:
                break
            acc.append(b)
            used += len(b)
        context = "\n\n".join(acc)

        # Structured output path (Nemotron / Ollama remote)
        # When using the Nemotron provider, we pass a JSON schema to the Ollama
        # `format` parameter which forces schema-compliant output.  This guarantees
        # that Summary / Key Evidence / Reasoning sections always exist and that
        # citations appear inline — eliminating the need for _reconstruct_list_summary
        # and the regen retry logic below.
        # NOTE: Set SKIP_STRUCTURED_OUTPUT=true to bypass this path (saves ~60-80s
        # when the remote Ollama endpoint does not support the `format` parameter).
        _skip_structured = os.environ.get("SKIP_STRUCTURED_OUTPUT", "").strip().lower() in ("1", "true", "yes", "y", "on")
        if not _skip_structured and cfg.provider_chat == "ollama_remote" and isinstance(llm, NemotronOllamaLLM):
            _struct_md = _synthesize_structured(llm, question, context, cfg)
            if _struct_md is not None:
                # Pre-clean raw-data artifacts before the quality gate so that verbatim
                # Methods table rows (e.g. "Number of animals—WT (n = 25, 14 male and
                # 11 female)...") don't inflate the word count and fool the gate into
                # accepting a structurally broken summary.
                _struct_md = _strip_raw_data_sentences(_struct_md)
                # Quality gate: cleaned Summary must have ≥100 words AND ≥1 inline citation.
                # Nemotron sometimes underfills the summary JSON field (2 plain sentences,
                # no bullets, no citations). If the gate fails, fall through to the markdown chain.
                _sm_m = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', _struct_md, re.S)
                _sm_body = _sm_m.group(1).strip() if _sm_m else ''
                _sm_wc = len([w for w in re.sub(r'\[PMC[^\]]+\](?:\([^)]+\))?', '', _sm_body).split() if w])
                _sm_has_cite = bool(re.search(r'\[PMC\d+:[^\]]+\]', _sm_body))
                _sm_has_artifact = bool(_RAW_DATA_RE.search(_sm_body))
                # Check duplicate ratio: reject if >50% of Summary sentences are
                # verbatim copies of Key Evidence bullets (structured output sometimes
                # just writes KE bullets as prose in the Summary field).
                _ke_m2 = re.search(r'(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)', _struct_md, re.S)
                _cite_re2 = re.compile(r'\[PMC[^\]]+\](?:\([^)]+\))?')
                _dup_ratio = 0.0
                if _ke_m2:
                    _ke_norm2 = {
                        re.sub(r'\s+', ' ', _cite_re2.sub('', b).strip().rstrip('.').lower())
                        for b in re.findall(r'^\s*[-*]\s+(.+)', _ke_m2.group(1), re.MULTILINE)
                        if len(_cite_re2.sub('', b).strip()) > 30
                    }
                    if _ke_norm2:
                        _sm_sents2 = re.split(r'(?<=[.!?])\s+', _sm_body.strip())
                        _dup_count2 = sum(
                            1 for s in _sm_sents2
                            if re.sub(r'\s+', ' ', _cite_re2.sub('', s).strip().rstrip('.').lower()) in _ke_norm2
                        )
                        _dup_ratio = _dup_count2 / max(len(_sm_sents2), 1)
                _is_dup_summary = _dup_ratio >= 0.5
                if _sm_wc >= 100 and _sm_has_cite and not _sm_has_artifact and not _is_dup_summary:
                    _structured = _parse_markdown_to_json(_struct_md)
                    _citations = _unique([_compose_citation_tag(h) for h in deduped[:len(acc)]])
                    t1 = _now_ms()
                    return {
                        **state,
                        "deduped_hits": deduped,
                        "markdown": _struct_md,
                        "structured": _structured,
                        "citations": _citations,
                        "trace": [{"agent": "synthesis", "timing_ms": t1 - t0,
                                   "note": f"Structured synthesis from {len(deduped)} chunks (Nemotron structured output)"}],
                    }
                else:
                    _gate_reasons = []
                    if _sm_wc < 100: _gate_reasons.append(f"WC={_sm_wc}")
                    if not _sm_has_cite: _gate_reasons.append("cite=False")
                    if _sm_has_artifact: _gate_reasons.append("artifact=True")
                    if _is_dup_summary: _gate_reasons.append(f"dup_ratio={_dup_ratio:.2f}")
                    print(f"  [SYNTH] Structured summary rejected ({', '.join(_gate_reasons)}) — falling back to markdown chain")
            # _struct_md is None or failed quality gate → fall through to old markdown chain.

        # Generate synthesis
        # Inject list-question guidance when the question asks for enumeration
        _list_q_pat = re.compile(
            r'\b(which|what)\b.{0,80}\b(conditions?|risks?|diseases?|comorbid|higher risk|lower risk|associated|complications?'
            r'|receptors?|genes?|proteins?|pathways?|subunits?|markers?|biomarkers?|mutations?|variants?|types?|kinds?|forms?'
            r'|drugs?|medications?|treatments?|therapies?|interventions?|factors?|mechanisms?|signs?|symptoms?)\b'
            r'|\b(what are the|list the|name the|enumerate)\b',
            re.IGNORECASE,
        )
        if _list_q_pat.search(question):
            list_instruction = (
                "\n[!] LIST-QUESTION MODE — follow this structure EXACTLY:\n"
                "  1. ONE framing/context sentence (no citation needed).\n"
                "     CRITICAL: Do NOT write circular filler like 'These conditions represent...',\n"
                "     'This population faces...', or 'Understanding these conditions is important...'.\n"
                "     Instead, open with a SPECIFIC orienting fact — name the number of distinct\n"
                "     domains, the mechanism, or the strongest quantitative signal from the evidence.\n"
                "     GOOD: 'Trisomy 21 confers elevated comorbidity burden across at least five\n"
                "     distinct clinical domains, with neurological, immunological, and cardiac\n"
                "     risks predominating.'\n"
                "     BAD: 'These conditions represent significant clinical vulnerabilities\n"
                "     requiring monitoring and intervention.'\n"
                "  2. TWO or THREE sentences naming SPECIFIC items (receptors, genes, conditions, etc.)\n"
                "     WITH individual per-item citations immediately after each item name.\n"
                "     Example format (use ACTUAL names from the evidence, NOT placeholders):\n"
                "     \"The triplicated genes include GENE_A [PMC:Results], GENE_B [PMC:Results],\n"
                "     and GENE_C [PMC:Discussion].\"\n"
                "  3. ONE or TWO sentences with quantitative or mechanistic highlights from Results sections.\n"
                "  4. ONE synthesis/conclusion sentence.\n"
                "Do NOT use generic phrases like 'gene dosage effects' when the evidence names specific\n"
                "entities. Do NOT defer all named items to Key Evidence — they MUST be in the Summary."
            )
            # Lightweight polarity hint — single line, guides focus without hardcoded exclusion lists
            _lower_q = re.compile(
                r'\b(lower risk|less likely|protective|not at (higher )?risk|reduced risk|'
                r'lower(ed)? (rates?|prevalence)|less (common|frequent|prevalent)|'
                r'decreased risk|do not develop|rarely develop)\b', re.IGNORECASE)
            _higher_q = re.compile(
                r'\b(higher risk|elevated risk|increased risk|more likely|greater risk|'
                r'more (common|frequent|prevalent)|over-represented)\b', re.IGNORECASE)
            if _lower_q.search(question) and not _higher_q.search(question):
                list_instruction += (
                    "\n[POLARITY: This question asks about conditions where DS individuals have "
                    "REDUCED or LOWER risk — name only those where risk is decreased compared to "
                    "the general population.]"
                )
            elif _higher_q.search(question) and not _lower_q.search(question):
                list_instruction += (
                    "\n[POLARITY: This question asks about conditions where DS individuals have "
                    "ELEVATED or HIGHER risk — name only those where risk is increased compared to "
                    "the general population.]"
                )
        else:
            list_instruction = ""

        # Named-entity forcing note: if gene symbols / specific IDs from the question
        # are absent from all retrieved evidence, append an instruction to the prompt
        # so the LLM address the absence explicitly instead of silently ignoring it.
        unmatched_entities = state.get("unmatched_entities") or []
        if unmatched_entities:
            entity_list = ", ".join(unmatched_entities)
            list_instruction += (
                f"\n[NAMED ENTITY CHECK] The following specific identifier(s) from the question "
                f"were NOT found in any retrieved evidence chunk: {entity_list}. "
                f"If the question's core premise depends on this specific entity and it is absent "
                f"from all evidence, you MUST output [EVIDENCE_GAP]. "
                f"Do NOT synthesize a generic DS answer while silently ignoring these terms."
            )

        md = chain.invoke({"question": question, "context": context, "list_instruction": list_instruction})
        if lenient_chain is not None and "[EVIDENCE_GAP]" in md:
            md = lenient_chain.invoke({"question": question, "context": context})
        # artifacts (methods table, first-person narration) were detected — verbatim
        # data inflates word count so the < 100 check alone is insufficient.
        _has_raw_artifact = bool(_RAW_DATA_RE.search(md or ""))
        if list_instruction and (_summary_word_count(md) < 100 or _has_raw_artifact) and "[EVIDENCE_GAP]" not in md:
            # Use a polarity-appropriate example so the LLM enumerates the right conditions
            _lower_q_re = re.compile(
                r'\b(lower risk|less likely|protective|not at (higher )?risk|reduced risk|'
                r'lower(ed)? (rates?|prevalence)|less (common|frequent|prevalent)|'
                r'decreased risk|do not develop|rarely develop)\b', re.IGNORECASE)
            _higher_q_re = re.compile(
                r'\b(higher risk|elevated risk|increased risk|more likely|greater risk|'
                r'more (common|frequent|prevalent)|over-represented)\b', re.IGNORECASE)
            if _lower_q_re.search(question) and not _higher_q_re.search(question):
                _sent2_example = (
                    "'Individuals with Down syndrome have reduced rates of solid malignancies "
                    "[PMC:Results], hypertension [PMC:Results], and solid tumors [PMC:Discussion].'"
                )
            elif _higher_q_re.search(question) and not _lower_q_re.search(question):
                _sent2_example = (
                    "'Elevated-risk conditions documented in DS include "
                    "Alzheimer\u2019s disease [PMC:Discussion], congenital heart defects [PMC:Results], "
                    "autoimmune hypothyroidism [PMC:Results], autism spectrum disorder [PMC:Results], "
                    "and acute leukemia [PMC:Results].'"
                )
            else:
                _sent2_example = (
                    "'The triplicated genes include GENE_A [PMC:Results], GENE_B [PMC:Results], "
                    "and GENE_C [PMC:Discussion].'"
                )
            # Detect whether the question is a comorbidity/conditions question to
            # add the clinical exclusion rule to the retry prompt.
            _is_comorbidity_q = bool(re.search(
                r'\b(conditions?|comorbid|diseases?|higher risk|elevated risk|at risk for)\b',
                question, re.IGNORECASE,
            ))
            _clinical_exclusion_note = (
                "\nCLINICAL CONTEXT: This question asks about CONDITIONS / COMORBIDITIES. "
                "Answer with NAMED CHRONIC DIAGNOSES (e.g. congenital heart disease, Alzheimer's disease, "
                "hypothyroidism, leukemia, autism spectrum disorder, obesity, diabetes, sleep apnea). "
                "EXCLUDE from both the framing sentence AND the condition list: ICU admission rates, "
                "mechanical ventilation rates, intubation rates, hospital length of stay — these are "
                "acute-care severity metrics, NOT baseline conditions. "
                "If the evidence reports ICU/ventilation stats for a domain, name only the UNDERLYING "
                "CONDITION (e.g. 'cardiovascular disease') and its prevalence %."
            ) if _is_comorbidity_q else ""
            _regen_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a scientific writer. Write exactly a 130–150 word paragraph. "
                 "Use ONLY the provided evidence. Every item you name MUST have a "
                 "[PMCxxxxxxx:Section] citation directly after it.\n"
                 "FORBIDDEN openers: 'Based on', 'According to', 'The evidence shows', "
                 "'Research has shown', 'Studies indicate'. Start with a clinical fact or finding."
                 + _clinical_exclusion_note
                ),
                ("human", (
                    f"QUESTION: {question}\n\n"
                    f"EVIDENCE:\n{context}\n\n"
                    "Write the paragraph using this structure:\n"
                    "Sentence 1: ONE clinical framing sentence stating the significance — "
                    "start with the clinical entity or population (NOT 'Based on' or 'According to').\n"
                    "Sentence 2: ONE sentence listing ALL specific items "
                    "(conditions/genes/receptors/etc.) found in the evidence that directly answer the question, "
                    "with a citation after EACH item name.\n"
                    f"Example: {_sent2_example}\n"
                    "Sentence 3: ONE mechanistic or quantitative highlight from Results.\n"
                    "Sentence 4: ONE concluding sentence answering the question.\n\n"
                    "Output format:\n# Summary\n<paragraph>"
                )),
            ])
            # Detect malformed retry output — two patterns:
            # 1. Garbled 'at elevated risk for [verbatim evidence text]' — the model
            #    appends evidence prose directly after the opener instead of a clean list.
            #    The tell-tale sign is 'risk for' followed by a word that is NOT a clean
            #    condition name (i.e. followed by a known bad word OR no citation follows).
            # 2. ICU/ventilation content in the framing sentence — excluded for comorbidity Qs.
            _COMMA_LIST_RETRY_RE = re.compile(
                r'at (?:elevated|higher|increased) risk for\s+'
                r'(?:Among|Individuals|With|In|Immune|The|This|A |An |patients|people)\b'
                r'|at (?:elevated|higher|increased) risk for [A-Z][^.!?\[]*(?:prevalence|admission|rate|remain|include|and [A-Z])'
                ,
                re.IGNORECASE,
            )
            _ICU_FRAMING_RE = re.compile(
                r'\b(ICU admission|mechanical ventilation|ventilat(?:ion|ed)|intubat(?:ion|ed)|'
                r'hospital(?:is|iz)ation rates?|hospital stay|length of stay)\b',
                re.IGNORECASE,
            )
            try:
                md2 = (_regen_prompt | llm | StrOutputParser()).invoke({})
                _retry_bad = False
                if _COMMA_LIST_RETRY_RE.search(md2 or ""):
                    print("  [SYNTH_WARN] List-question retry produced comma-list concat — skipping retry")
                    _retry_bad = True
                if _is_comorbidity_q and _ICU_FRAMING_RE.search((md2 or "").split('.')[0]):
                    print("  [SYNTH_WARN] List-question retry framing sentence has ICU/ventilation content — skipping retry")
                    _retry_bad = True
                if _retry_bad:
                    raise ValueError("retry output has disqualifying pattern")
                _md2_wc = _summary_word_count(md2)
                _md1_wc = _summary_word_count(md)
                # Accept retry if it has citations and is substantive — also accept
                # when original had raw-data artifacts even if retry is shorter.
                if _md2_wc >= 60 and _summary_has_citation(md2) and (
                    _md2_wc > _md1_wc or _has_raw_artifact
                ):
                    # IMPORTANT: only transplant the Summary section from the retry.
                    # Replacing md entirely with md2 discards Key Evidence + Reasoning.
                    _summ2_m = re.search(
                        r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md2, re.S | re.M
                    )
                    if _summ2_m:
                        _summ1_m = re.search(
                            r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S | re.M
                        )
                        if _summ1_m:
                            new_summ = "# Summary\n" + _summ2_m.group(1).strip() + "\n"
                            md = md[: _summ1_m.start()] + new_summ + md[_summ1_m.end():]
                        else:
                            # No existing summary — use full retry but graft in any
                            # Key Evidence + Reasoning that were in the original.
                            _ke_orig = re.search(
                                r"(?im)^#\s+Key Evidence\s*\n.*?(?=^#\s+|\Z)", md, re.S | re.M
                            )
                            _re_orig = re.search(
                                r"(?im)^#\s+Reasoning\s*\n.*?(?=^#\s+|\Z)", md, re.S | re.M
                            )
                            md = md2.rstrip("\n")
                            if _ke_orig and not re.search(r"(?im)^#\s+Key Evidence", md2):
                                md += "\n\n" + _ke_orig.group(0).strip()
                            if _re_orig and not re.search(r"(?im)^#\s+Reasoning", md2):
                                md += "\n\n" + _re_orig.group(0).strip()
                            md += "\n"
                    else:
                        # retry didn't produce a proper header — graft non-Summary sections
                        # from the original md to avoid discarding Key Evidence + Reasoning.
                        _ke_orig = re.search(
                            r"(?im)^#\s+Key Evidence\s*\n.*?(?=^#\s+|\Z)", md, re.S | re.M
                        )
                        _re_orig = re.search(
                            r"(?im)^#\s+Reasoning\s*\n.*?(?=^#\s+|\Z)", md, re.S | re.M
                        )
                        md = md2.rstrip("\n")
                        if _ke_orig and not re.search(r"(?im)^#\s+Key Evidence", md2):
                            md += "\n\n" + _ke_orig.group(0).strip()
                        if _re_orig and not re.search(r"(?im)^#\s+Reasoning", md2):
                            md += "\n\n" + _re_orig.group(0).strip()
                        md += "\n"
                    print(f"  [SYNTH] List-question retry {'replaced artifact' if _has_raw_artifact else 'improved'} summary: {_md2_wc} words")
            except Exception as _regen_err:
                print(f"  [SYNTH_WARN] List-question retry failed: {_regen_err}")

        # Parse to structured
        structured = _parse_markdown_to_json(md)

        # Extract citations from deduplicated chunks used in context
        citations = _unique([_compose_citation_tag(h) for h in deduped[:len(acc)]])

        t1 = _now_ms()
        return {
            **state,
            "deduped_hits": deduped,  # Save deduplicated hits for other nodes
            "markdown": md,
            "structured": structured,
            "citations": citations,
            "trace": [{"agent": "synthesis", "timing_ms": t1 - t0, "note": f"Synthesized from {len(deduped)} deduplicated chunks (was {len(hits)})"}],
        }

    return synthesize


# =============================================================================
# Structured Output Support (Nemotron / Ollama remote)
# =============================================================================

# JSON schema passed to Ollama's `format` parameter — constrains model output to
# a fixed structure so no regex post-processing is needed.
_RAG_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["summary", "key_evidence", "reasoning"],
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "130\u2013150 word PROSE paragraph (NO markdown bullet points). "
                "Structure: (1) ONE orienting sentence with NO citation; (2) 3\u20135 complete "
                "prose sentences each ending with [PMCxxxxxxx:Section] BEFORE the period; "
                "(3) ONE synthesis/conclusion sentence with NO citation. "
                "FORBIDDEN: Do NOT write a sentence like 'Individuals with DS are at elevated "
                "risk for X, Y, and Z' as an intro \u2014 that comma-list pattern is forbidden. "
                "Instead write one sentence per clinical domain: 'Cardiovascular conditions, "
                "including congenital heart defects, affected 35.6\u0025 of individuals with DS "
                "[PMCxxxxxxx:Results].' Cite \u22653 different PMC IDs. "
                "Citations excluded from word count."
            ),
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "REQUIRED \u2014 always 4\u20136 strings (never an empty array). "
                "Each string = one specific finding with: cohort/population context + "
                "exact statistics (%, OR, HR, CI, p-value, n) + [PMCxxxxxxx:Section] at end. "
                "One bullet per condition or finding \u2014 do not merge multiple findings into one."
            ),
        },
        "reasoning": {
            "type": "object",
            "required": ["evidence_selection", "reasoning_chain", "confidence_gaps", "reader_caveats"],
            "properties": {
                "evidence_selection": {
                    "type": "string",
                    "description": "Why these specific findings were prioritised (1–2 sentences).",
                },
                "reasoning_chain": {
                    "type": "string",
                    "description": "How the evidence logically connects to the Summary conclusion (1–2 sentences).",
                },
                "confidence_gaps": {
                    "type": "string",
                    "description": "Where the evidence is strongest and where limitations exist (1–2 sentences).",
                },
                "reader_caveats": {
                    "type": "string",
                    "description": "What the reader should bear in mind when interpreting the answer (1–2 sentences).",
                },
            },
        },
    },
}


def _structured_response_to_markdown(data: Dict[str, Any]) -> str:
    """Convert a structured JSON response dict into the expected markdown format.

    Produces the same ``# Summary / # Key Evidence / # Reasoning`` layout that
    the downstream pipeline (finalize, safety_guard, UI) expects.
    """
    lines: List[str] = []

    # Summary — strip any section-header text the LLM may have embedded within
    # the summary JSON field (e.g. "...Discussion]. # Key Evidence\n- bullet" which
    # occurs when the model treats summary as the full response body).
    summary = (data.get("summary") or "").strip()
    _hdr_in_summary = re.search(r'(?:^|\s)(#\s+(?:Key Evidence|Reasoning)\b)', summary)
    if _hdr_in_summary:
        summary = summary[: _hdr_in_summary.start(1)].rstrip()
        print("  [STRUCT] Stripped embedded section header from summary field")

    # Strip malformed comma-list intro sentences that the model emits when constrained
    # by the JSON grammar, e.g.:
    #   "Individuals with DS are at elevated risk for Immune disorders, CHD and Alzheimer's disease."
    # These are a degenerate form of enumeration that violates the prose paragraph requirement.
    # Detection: sentence starts with a subject containing "DS"/"Down syndrome" and contains
    # "at elevated/higher/increased risk for" or "are at risk for" followed by ≥2 comma-separated items.
    _COMMA_LIST_INTRO_RE = re.compile(
        r'(?:^|\n)([^\n.!?]{0,60}(?:Down syndrome|individuals with DS|people with DS)[^\n.!?]{0,40}'
        r'(?:at (?:elevated|higher|increased|greater) risk for|are at risk for|experience[sd]? elevated risk for)'
        r'[^\n.!?]{0,20}(?:[A-Z][^,\n.!?]+,\s*){2,}[^\n.!?]+[.!?])',
        re.IGNORECASE,
    )
    _m = _COMMA_LIST_INTRO_RE.search(summary)
    if _m:
        # Remove only that sentence, preserve everything else
        summary = (summary[: _m.start(1)] + summary[_m.end(1):]).strip()
        print("  [STRUCT] Stripped comma-list intro sentence from summary field")

    lines.append("# Summary")
    lines.append(summary)
    lines.append("")

    # Key Evidence
    lines.append("# Key Evidence")
    for bullet in (data.get("key_evidence") or []):
        b = bullet.strip()
        if not b.startswith("-"):
            b = "- " + b
        lines.append(b)
    lines.append("")

    # Reasoning
    lines.append("# Reasoning")
    reasoning = data.get("reasoning") or {}
    for key, label in [
        ("evidence_selection", "Evidence selection"),
        ("reasoning_chain", "Reasoning chain"),
        ("confidence_gaps", "Confidence & gaps"),
        ("reader_caveats", "Reader caveats"),
    ]:
        val = (reasoning.get(key) if isinstance(reasoning, dict) else "") or ""
        val = val.strip()
        if val:
            lines.append(f"- **{label}:** {val}")
    lines.append("")

    return "\n".join(lines)


def _synthesize_structured(
    llm: "NemotronOllamaLLM",
    question: str,
    context: str,
    cfg: "RAGConfig",
) -> Optional[str]:
    """Run structured-output synthesis via Ollama ``format`` schema.

    Returns a markdown string on success, or ``None`` if the call fails
    (caller should fall back to the old markdown chain in that case).
    """
    # Build a lightweight list/polarity hint if the question is a list question.
    _lq = re.compile(
        r'\b(which|what)\b.{0,80}\b(conditions?|risks?|diseases?|comorbid|higher risk|lower risk|'
        r'associated|complications?|receptors?|genes?|proteins?|pathways?|subunits?|markers?|'
        r'biomarkers?|mutations?|variants?|types?|kinds?|forms?|drugs?|medications?|treatments?|'
        r'therapies?|interventions?|factors?|mechanisms?|signs?|symptoms?)\b'
        r'|\b(what are the|list the|name the|enumerate)\b',
        re.IGNORECASE,
    )
    _lower_re = re.compile(
        r'\b(lower risk|less likely|protective|not at (higher )?risk|reduced risk|'
        r'lower(ed)? (rates?|prevalence)|less (common|frequent|prevalent)|'
        r'decreased risk|do not develop|rarely develop)\b',
        re.IGNORECASE,
    )
    _higher_re = re.compile(
        r'\b(higher risk|elevated risk|increased risk|more likely|greater risk|'
        r'more (common|frequent|prevalent)|over-represented)\b',
        re.IGNORECASE,
    )
    if _lq.search(question):
        if _lower_re.search(question) and not _higher_re.search(question):
            list_hint = (
                "LIST QUESTION — enumerate specific items with a per-item [PMCxxxxxxx:Section] "
                "citation directly after each item name. Focus on conditions/factors where "
                "individuals with DS have LOWER or REDUCED risk compared to the general population."
            )
        elif _higher_re.search(question) and not _lower_re.search(question):
            list_hint = (
                "LIST QUESTION — enumerate specific items with a per-item [PMCxxxxxxx:Section] "
                "citation directly after each item name. Focus on conditions/factors where "
                "individuals with DS have HIGHER or ELEVATED risk compared to the general population.\n"
                "LARGE SET RULE: Group named conditions by clinical domain (e.g. cardiovascular, "
                "neurological, metabolic, immunological) — write ONE sentence per domain naming "
                "2-3 specific conditions with citations; do NOT dump a single run-on comma list."
            )
        else:
            list_hint = (
                "LIST QUESTION — enumerate specific items (conditions, genes, pathways, etc.) "
                "with a per-item [PMCxxxxxxx:Section] citation directly after each item name."
            )
    else:
        list_hint = ""

    system_content = (
        "You are a senior biomedical researcher synthesizing Down syndrome (Trisomy 21) "
        "peer-reviewed literature. Return a JSON object matching the provided schema exactly.\n\n"
        "RULES:\n"
        "1. Cite ONLY PMC IDs present in the EVIDENCE CORPUS — never invent or guess IDs.\n"
        "2. summary field: 130–150 word PROSE paragraph (NO bullet points, NO comma-lists of "
        "items). Structure: ONE orienting sentence (no citation) → 3–5 complete prose sentences "
        "each ending with [PMCxxxxxxx:Section] BEFORE the period → ONE conclusion sentence (no "
        "citation). LIST/COMORBIDITY questions: write ONE sentence per clinical domain "
        "(cardiovascular, neurological, metabolic, immunological) naming 2–3 specific named "
        "conditions with prevalence citations. Never start a sentence with 'Individuals with DS "
        "are at elevated risk for X, Y, Z' — that comma-list is forbidden. "
        "Cite ≥3 different PMC IDs. Citations excluded from word count.\n"
        "3. key_evidence field: REQUIRED — always 4–6 items (never empty). Each item = "
        "cohort/population context + one specific quantitative finding + exact statistics "
        "(%, OR, HR, CI, p-value, n) + [PMCxxxxxxx:Section] at end. These are the detailed "
        "statistics that support the Summary prose — must be populated even if summary also "
        "mentions the finding at a higher level.\n"
        "4. reasoning field: REQUIRED — four plain-prose entries (evidence_selection, "
        "reasoning_chain, confidence_gaps, reader_caveats), 1–2 sentences each, no citations.\n"
        "5. If evidence is clearly insufficient, write \"[EVIDENCE_GAP]: <one sentence reason>\" "
        "as the entire summary value.\n"
        "6. CLINICAL QUESTION TYPES — answer precisely what was asked:\n"
        "   Comorbidity/conditions questions: name CHRONIC DIAGNOSES (congenital heart defects, "
        "Alzheimer's disease, hypothyroidism, leukemia, autism, sleep apnea, epilepsy, etc.). "
        "EXCLUDE from summary: ICU admission rates, mechanical ventilation rates, intubation "
        "rates, hospital LOS — these are acute severity metrics, not baseline conditions. "
        "If evidence only has ICU/ventilation stats, cite the underlying condition prevalence.\n"
        "   Clinical outcomes/prognosis/mortality questions: ICU rate, ventilation, survival "
        "ARE the answer — include fully.\n"
        "   Mechanism/pathway questions: molecular explanation first (gene→protein→phenotype).\n"
        "   Prevalence/incidence questions: lead with quantitative Results-section statistics.\n"
        + (f"7. {list_hint}\n" if list_hint else "")
    )
    human_content = (
        f"QUESTION: {question}\n\n"
        f"EVIDENCE CORPUS (ranked by relevance score, highest first):\n{context}"
    )
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    try:
        data = llm.invoke_structured(messages, _RAG_RESPONSE_SCHEMA)

        # Quality gate: if key_evidence is empty the model crammed everything into
        # the summary field (common with grammar-constrained JSON generation).
        # Fall back to the markdown chain which handles this reliably.
        _ke = data.get("key_evidence") or []
        if not _ke:
            print("  [STRUCT] key_evidence empty — falling back to markdown chain")
            return None

        # Quality gate: if the summary still contains the comma-list intro pattern
        # ("at elevated/higher risk for X, Y, Z"), the grammar constraint is fighting
        # the instructions. Fall back to the markdown chain.
        _summary_raw = (data.get("summary") or "")
        _COMMALIST_QG = re.compile(
            r'(?:at (?:elevated|higher|increased) risk for\s+(?:[A-Z][^.!?]*,\s*){2,})',
            re.IGNORECASE,
        )
        if _COMMALIST_QG.search(_summary_raw):
            print("  [STRUCT] comma-list intro detected in summary — falling back to markdown chain")
            return None

        md = _structured_response_to_markdown(data)

        # If evidence gap triggered and lenient mode is on, retry without gap rule.
        if "[EVIDENCE_GAP]" in (data.get("summary") or "") and not cfg.strict_mode:
            lenient_content = system_content.replace(
                "5. If evidence is clearly insufficient, write \"[EVIDENCE_GAP]: <one sentence reason>\" "
                "as the entire summary value.\n",
                "5. If evidence is sparse, synthesise with explicit caveats rather than refusing.\n",
            )
            messages_lenient = [
                SystemMessage(content=lenient_content),
                HumanMessage(content=human_content),
            ]
            try:
                data2 = llm.invoke_structured(messages_lenient, _RAG_RESPONSE_SCHEMA)
                if "[EVIDENCE_GAP]" not in (data2.get("summary") or ""):
                    md = _structured_response_to_markdown(data2)
            except Exception as _le:
                print(f"  [SYNTH_WARN] Structured lenient retry failed: {_le}")

        return md

    except Exception as e:
        import traceback as _tb
        _err_detail = _tb.format_exc()
        print(f"  [SYNTH_WARN] Structured output failed ({type(e).__name__}: {e}), falling back to markdown chain")
        try:
            with open("/tmp/synth_structured_error.txt", "w") as _ef:
                _ef.write(f"question: {question[:100]}\nexception: {type(e).__name__}: {e}\n\n{_err_detail}\n")
        except Exception:
            pass
        return None


def _parse_markdown_to_json(md: str) -> Dict[str, Any]:
    """Parse markdown sections to structured dict."""
    sections = re.split(r"^#\s+", md, flags=re.MULTILINE)
    out = {}
    for sec in sections:
        lines = sec.strip().splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        body = lines[1:]
        key = header.lower().replace(" ", "_")
        body_stripped = [l for l in body if l.strip()]
        if body_stripped and all(l.strip().startswith("-") for l in body_stripped):
            vals = [l.strip().lstrip("-").strip() for l in body_stripped]
            out[key] = vals
        else:
            out[key] = "\n".join(body).strip()
    return out


_SPELL_NORMALIZE = {
    # Common medical misspellings that would bypass the stoplist
    "disfunction": "dysfunction",
    "disfunctions": "dysfunctions",
    "inflamation": "inflammation",
    "inflamatory": "inflammatory",
    "nuerological": "neurological",
    "nuero": "neuro",
    "diabetis": "diabetes",
    "alzeheimer": "alzheimer",
    "alzheimer": "alzheimer",
    "alzeimers": "alzheimers",
    "cognative": "cognitive",
    "behaviorial": "behavioral",
    "behavioural": "behavioral",
    "gastointestinal": "gastrointestinal",
    "gastrointestinal": "gastrointestinal",
    "hipothyroid": "hypothyroid",
    "hiperthyroid": "hyperthyroid",
    "seziure": "seizure",
    "seziures": "seizures",
    "leukaemia": "leukemia",
    "haematologic": "hematologic",
    "haematopoietic": "hematopoietic",
    "immunodeficency": "immunodeficiency",
    "immunodeficiency": "immunodeficiency",
    "cardivascular": "cardiovascular",
    "catarct": "cataract",
    "cataracts": "cataracts",
}


def _extract_key_terms(question: str) -> List[str]:
    """Extract key terms from the question for evidence-first checks."""
    q = re.sub(r"[^a-zA-Z0-9\s-]", " ", question.lower())
    tokens = [t for t in re.split(r"\s+", q) if t]
    # Normalize common medical misspellings before stoplist check so that
    # e.g. "disfunction" → "dysfunction" is caught by the stoplist and not
    # extracted as an unmatched key term, causing a false evidence gap.
    tokens = [_SPELL_NORMALIZE.get(t, t) for t in tokens]
    stop = {
        "which","what","who","whom","whose","when","where","why","how",
        "does","do","did","is","are","was","were","be","been","being",
        "the","a","an","and","or","but","if","then","than","that","this",
        "these","those","for","to","of","in","on","at","by","with","from",
        "about","as","into","over","after","before","between","among","during",
        "within","without","across","per","via","not","no","yes","higher","lower",
        "risk","risks","conditions","condition","people","persons","individuals",
        "down","syndrome","trisomy","t21","ds",
        # Plain-language question framing — these describe how something behaves
        # but are too generic to appear verbatim in results-section evidence text.
        "prone","predisposed","predisposition","susceptible","susceptibility",
        "increased","decreased","elevated","reduced","altered","altered",
        "prevalence","prevalences","incidence","frequency","rate","rates",
        "compared","comparison","difference","differences","similar","similarly",
        "known","reported","shown","found","observed","noted","documented",
        "experience","experiences","experienced","experiencing",
        "develop","develops","developed","developing",
        "present","presents","presented","presenting",
        "exhibit","exhibits","exhibited","exhibiting",
        "occur","occurs","occurred","occurring",
        "blood","pressure","pressures",
        # Plain-English framing verbs / generic nouns that wrap medical topics
        # but don't appear verbatim in results-section text
        "increase","increases","decrease","decreases",
        "problem","problems","issue","issues",
        "disease","diseases","illness","illnesses",
        "decline","declines","progression","progressions",
        "symptom","symptoms","feature","features","finding","findings",
        "marker","markers","indicator","indicators",
        # Generic anatomical / structural words that describe body parts rather than
        # specific conditions — they don't appear verbatim in evidence but are part
        # of natural question phrasing (e.g. "thyroid glands", "brain cells").
        "gland","glands","cells","cell","organ","organs","tissue","tissues",
        "level","levels","system","systems","function","functions","activity",
        "protein","proteins","gene","genes","pathway","pathways",
        "early","often","common","commonly","severe","typical","typically",
        "normal","abnormal","usually","likely","unlikely","frequently","rarely",
        # Generic medical/scientific framing words that appear in questions but
        # not reliably in evidence body text
        "clinical","clinically","medical","medically","health","healthy",
        "patient","patients","subject","subjects","participant","participants",
        "study","studies","research","researchers","literature","evidence",
        # Question framing / meta-research words — appear in questions but not
        # necessarily verbatim in evidence text; removing them prevents false-positive
        # evidence-gap short-circuits.
        "cause","causes","caused","causing",
        "contribute","contributes","contributed","contributing","contribution","contributions",
        "dysfunction","dysfunctions",
        "disorder","disorders",
        "impairment","impairments",
        "mechanism","mechanisms",
        "underlying","underlie","underlies",
        "affect","affects","affected","affecting",
        "improve","improves","improved","improving",
        "impact","impacts",
        "effect","effects",
        "relationship","relationships",
        "studied","assess","assessing","assess","evaluate","evaluating",
        "involved","involve","involve",
        "contribute","role","roles",
        "associated","association",
    }
    # Keep longer, contentful tokens
    terms = []
    for t in tokens:
        if len(t) < 5:
            continue
        if t in stop:
            continue
        terms.append(t)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _term_variants(term: str) -> List[str]:
    """Generate simple morphological variants for evidence matching."""
    variants = {term}
    suffixes = ["ing", "ed", "es", "s", "ion", "tion", "ation", "ality", "ity", "ment", "ness", "ies"]
    for suf in suffixes:
        if term.endswith(suf) and len(term) - len(suf) >= 5:
            variants.add(term[: -len(suf)])
    # Minimal biomedical abbreviations (avoid aggressive synonym expansion)
    if term == "interferon":
        variants.add("ifn")
    if term == "overactive":
        variants.update(["hyperactive", "hyperthyroid", "hyperthyroidism", "graves"])
    # Common clinical synonym expansions
    if term == "hypertension":
        variants.update(["blood pressure", "hypertensive", "hypotension", "hypotensive"])
    if term == "hypotension":
        variants.update(["blood pressure", "hypotensive", "hypertension", "hypertensive"])
    if term == "seizure":
        variants.update(["epilepsy", "epileptic", "convulsion", "convulsions"])
    if term == "epilepsy":
        variants.update(["seizure", "seizures", "epileptic"])
    if term == "obesity":
        variants.update(["obese", "overweight", "adiposity", "bmi", "body mass"])
    if term == "dementia":
        variants.update(["alzheimer", "cognitive decline", "neurodegeneration"])
    if term == "leukemia":
        variants.update(["aml", "all", "mlds", "trisomy 21", "hematologic"])
    # Anatomical structure ↔ adjective form (corpus often uses shorter root form)
    if term in ("diaphragmatic", "diaphragm"):
        variants.update(["diaphragm", "diaphragmatic", "cdh", "congenital diaphragmatic"])
    if term in ("hernia", "hernias", "herniation"):
        variants.update(["hernia", "herniation", "cdh", "congenital diaphragmatic"])
    return list(variants)


# Terms that definitively place a question outside human biomedical / DS research.
# A question whose key term is (a) in this set AND (b) absent from all retrieved
# evidence is hard-blocked before reaching the LLM.
#
# IMPORTANT — keep this list NARROW. Any scientifically plausible DS / human
# biology term (even niche ones like "diaphragmatic hernia" or "Dp16") must NOT
# appear here, because sparse corpus coverage is not the same as being off-topic.
# The LLM synthesis node (with its own [EVIDENCE_GAP] mechanism) is the right
# judge for anything that could conceivably appear in DS peer-reviewed literature.
_HARD_OFFTOPIC_TERMS: frozenset = frozenset({
    # Plant cell biology / botany
    "chloroplast", "thylakoid", "chlorophyll", "photosynthesis", "plastid",
    "stomata", "xylem", "phloem", "cellulose", "lignin", "chlorosis",
    "photosynth", "mesophyll",
    # Microbiology with no human relevance
    "prokaryote", "prokaryotic", "archaeal", "archaea", "cyanobacterium",
    "cyanobacteria",
    # Clearly non-biological / unrelated domains
    "astrophysics", "cryptocurrency", "blockchain", "thermodynamics",
    "galaxy", "asteroid", "quasar", "supernova",
})


def _missing_terms_in_hits(question: str, hits: List[Dict[str, Any]]) -> List[str]:
    """Return hard-off-topic key terms from 'question' absent from retrieved evidence.

    Only terms in _HARD_OFFTOPIC_TERMS that are ALSO not found in any retrieved
    chunk are returned.  All other missing terms are intentionally ignored here —
    the LLM synthesis node is a much better judge for valid DS biology questions
    whose specific angle is simply not well-covered in the top-K results.

    This two-tier design prevents false evidence-gap rejections for valid biological
    questions (e.g. "diaphragmatic hernia in Dp16 mice", "liver dysfunction",
    anatomical phenotypes) while still hard-blocking genuinely inapplicable topics
    (e.g. chloroplast biology, plant cell biology, astrophysics).
    """
    terms = _extract_key_terms(question)
    # Restrict to terms that are unambiguously off-topic for human/DS biology
    offtopic_terms = [t for t in terms if t in _HARD_OFFTOPIC_TERMS]
    if not offtopic_terms:
        return []  # Nothing is definitively off-topic — let the LLM decide

    min_hits = int(os.environ.get("MIN_TERM_MATCHES", "1"))
    missing = []
    for t in offtopic_terms:
        count = 0
        variants = _term_variants(t)
        pat = r"\b(" + "|".join(re.escape(v) for v in variants) + r")\w*\b"
        pattern = re.compile(pat, re.IGNORECASE)
        for h in hits:
            fields = [
                h.get("text") or "",
                h.get("title") or "",
                h.get("keywords") or "",
                h.get("mesh_terms") or "",
            ]
            blob = " ".join(fields)
            if pattern.search(blob):
                count += 1
                if count >= min_hits:
                    break
        if count < min_hits:
            missing.append(t)
    return missing


# Regex pattern that matches tokens looking like specific gene symbols, protein IDs,
# mouse model names, chemical compound IDs, or other specific named biological entities
# that the LLM must not silently ignore if absent from evidence.
# Matches: ALL-CAPS+digit combos (BRCA1, FX84A9, DYRK1A), mixed-case gene-like tokens
# (Dp16, Ts65Dn), and NNNxNNN-style IDs. Does NOT match plain English words.
_NAMED_ENTITY_PAT = re.compile(
    r'\b('
    r'[A-Z]{2,}[0-9]+[A-Z0-9]*'      # e.g. BRCA1, FX84A9, DYRK1A, APP2
    r'|[A-Z][a-z]+[0-9]+[A-Za-z0-9]*'  # e.g. Dp16, Ts65Dn, Hsa21
    r'|[A-Z]{3,}[a-z]*[0-9]*'          # e.g. STAT1, GATA1, NFATC1
    r')\b'
)
# Known DS-biology tokens that look like gene symbols but are common English/medical words
# — do NOT treat these as named entities requiring presence in evidence.
_NAMED_ENTITY_EXCLUDE = frozenset({
    "DNA", "RNA", "PCR", "RNAi", "mRNA", "siRNA", "ATP", "ADP", "AMP",
    "NADH", "NADPH", "NAD", "FAD", "GTP", "GDP", "cAMP", "cGMP",
    "MRI", "CT", "EEG", "ECG", "EKG", "IQ", "BMI", "CBC", "CNS", "PNS",
    "CHD", "CDH", "ASD", "VSD", "AML", "ALL", "MDS",
    "DS", "T21", "HSA", "Hsa",
    "USA", "UK", "NIH", "FDA", "WHO", "CDC",
})


def _unmatched_named_entities(question: str, hits: List[Dict[str, Any]]) -> List[str]:
    """Return specific named-entity tokens from the question absent from ALL retrieved evidence.

    These are gene symbols / protein IDs / model names (e.g. FX84A9, Dp16, DYRK1A) that
    were explicitly named in the question but do not appear in any retrieved chunk.  Unlike
    plain biology words, a named entity absent from all evidence is a concrete signal the
    LLM should address — either by flagging [EVIDENCE_GAP] or synthesizing with an explicit
    caveat that the specific entity was not found in the published literature.

    Common gene-symbol-like abbreviations (DNA, RNA, AML, etc.) are excluded.
    """
    # Extract gene-like tokens from the question (original case preserved for matching)
    q_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", question)
    candidates = _NAMED_ENTITY_PAT.findall(q_clean)
    # Remove exclusions and very short tokens
    entities = [e for e in candidates if e.upper() not in _NAMED_ENTITY_EXCLUDE and len(e) >= 3]
    if not entities:
        return []

    min_hits = int(os.environ.get("MIN_TERM_MATCHES", "1"))
    unmatched = []
    for entity in entities:
        count = 0
        pattern = re.compile(r'\b' + re.escape(entity) + r'\w*\b', re.IGNORECASE)
        for h in hits:
            blob = " ".join([
                h.get("text") or "",
                h.get("title") or "",
                h.get("keywords") or "",
                h.get("mesh_terms") or "",
            ])
            if pattern.search(blob):
                count += 1
                if count >= min_hits:
                    break
        if count < min_hits:
            unmatched.append(entity)
    return unmatched


def _summary_word_count(md: str) -> int:
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return 0
    body = match.group(1)
    # Strip ALL citation tokens including INCLUDE-tagged forms like
    # [PMCxxxxxxx:Section, INCLUDE](url) — use [^\]]+ to match section
    # regardless of ", INCLUDE" suffix so the URL does not bleed into the count.
    body = re.sub(r'\[PMC[^\]]+\](?:\([^)]*\))?', '', body)
    return len(re.findall(r"\b\w+\b", body))


def _extract_summary_text(md: str) -> str:
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return ""
    return match.group(1).strip()


def _normalize_key_evidence_bullets(md: str) -> str:
    """Ensure each Key Evidence bullet appears on its own line with a '- ' prefix."""
    match = re.search(r"(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    # Split on any bullet marker style: '- ', '• ', '* ' at start of a line
    # This handles both LLM-generated '- ' bullets and '•' bullets
    parts = re.split(r"(?m)^\s*[-•*]\s+", body)
    items = [p.strip() for p in parts if p.strip()]

    # Fallback: if no bullet markers found at all, split on double-newline or
    # treat each non-empty line as a separate item
    if not items or len(items) == 1:
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        if lines:
            items = lines

    if not items:
        return md

    new_body = "\n".join(f"- {item}" for item in items)
    new_section = "# Key Evidence\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]


def create_raw_evidence_node(cfg: RAGConfig):
    """Create raw evidence extraction node with section grouping."""

    # Section display order (priority order)
    SECTION_ORDER = ["Results", "Methods", "Conclusion", "Discussion", "Abstract", "Introduction", "FullText", "NA"]

    def extract_evidence(state: RAGState) -> RAGState:
        t0 = _now_ms()
        # Use deduplicated hits if available, otherwise fall back to reranked/raw hits
        hits = state.get("deduped_hits") or state.get("reranked_hits") or state.get("hits", [])
        top_k = state.get("top_k", cfg.top_k_default)

        items = []
        section_counts = {}

        for h in hits[:top_k]:
            tag = _compose_citation_tag(h)
            score = h.get("_rerank_final") or h.get("score")
            section_bonus = h.get("_section_bonus", 0.0)
            txt = (h.get("text") or "").strip()
            section = _normalize_section(h.get("section") or "NA")

            # Track section counts
            section_counts[section] = section_counts.get(section, 0) + 1

            items.append({
                "pmid":         h.get("pmid"),
                "pmcid":        h.get("pmcid"),
                "title":        h.get("title") or "Untitled",
                "authors":      h.get("authors") or "",
                "year":         h.get("year"),
                "source":       h.get("source") or "",
                "topic":        h.get("topic") or "",
                "subfield":     h.get("subfield") or "",
                "pub_field":    h.get("pub_field") or "",
                "keywords":     h.get("keywords") or "",
                "mesh_terms":   h.get("mesh_terms") or "",
                "citations":    h.get("citations"),
                "section":      section,
                "score":        float(score) if score else None,
                "section_bonus": section_bonus,
                "citation_tag": tag,
                "text":         txt,
                "is_include":   bool(h.get("is_include") or False),
            })

        # Sort items by section order for better display
        def section_sort_key(item):
            section = item.get("section", "NA")
            try:
                return SECTION_ORDER.index(section)
            except ValueError:
                return len(SECTION_ORDER)

        items_sorted = sorted(items, key=section_sort_key)

        # Create section summary
        section_summary = ", ".join([f"{sec}: {cnt}" for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1])])

        t1 = _now_ms()
        return {
            **state,
            "raw_evidence": items_sorted,
            "trace": [{"agent": "raw_evidence", "timing_ms": t1 - t0, "note": f"Extracted {len(items)} items ({section_summary})"}],
        }

    return extract_evidence


def create_safety_guard_node(cfg: RAGConfig):
    """Create safety guard node."""

    # Patterns that flag genuinely prescriptive/directive language the model
    # should never output.  Research reporting of drug names, dosages, routes,
    # adverse events, etc. is intentionally allowed (per user requirement).
    PRESCRIPTION_PATTERNS = [
        r"\bprescrib(e|ing)\b",          # active-voice prescribing instruction
        r"\btake\s+\d+",                  # "take 2 tablets" dosing instruction
        r"\byou\s+should\s+(take|use|avoid|stop)\b",  # direct patient advice
        r"\bcall\s+911\b",               # emergency directive
        r"\bdo\s+not\s+take\b",          # direct contraindication order
    ]
    # DIAGNOSTIC_PATTERNS intentionally empty: \bdiagnos(e|is|tic)\b and
    # \btreat(ment|ing|ed)?\b fire on >85% of DS research answers (e.g.
    # "diagnosed with DS", "treatment group") causing alert fatigue.
    # True prescriptive language is already covered by PRESCRIPTION_PATTERNS.
    DIAGNOSTIC_PATTERNS: List[str] = []

    # Advice-seeking QUESTION backstop: such questions are normally declined
    # before retrieval by the question_judge pre-check (module-level
    # _ADVICE_SEEKING_PATTERNS / _is_advice_seeking). If one ever slips through
    # to synthesis, the guard still flags it and prepends the banner.

    def safety_guard(state: RAGState) -> RAGState:
        if not cfg.enable_safety_guard:
            return {**state, "safety_flagged": False, "safety_patterns": []}

        t0 = _now_ms()
        md = state.get("markdown", "")
        question = state.get("question", "") or ""
        patterns_found = []

        for pat in PRESCRIPTION_PATTERNS + DIAGNOSTIC_PATTERNS:
            if re.search(pat, md, flags=re.IGNORECASE):
                patterns_found.append(pat)

        for pat in _ADVICE_SEEKING_PATTERNS:
            if re.search(pat, question, flags=re.IGNORECASE):
                patterns_found.append(f"advice_seeking_question:{pat}")

        flagged = len(patterns_found) > 0
        md_out = md

        if flagged:
            banner = ("> **[!] WARNING:** This tool cannot recommend or prescribe any drug or treatment. "
                      "The text below summarizes published research findings only; it is not medical advice. "
                      "Consult a qualified clinician for any treatment decision.\n\n")
            if not md_out.lstrip().startswith("> **[!] WARNING:**"):
                md_out = banner + md_out

        t1 = _now_ms()
        return {
            **state,
            "markdown": md_out,
            "safety_flagged": flagged,
            "safety_patterns": patterns_found,
            "trace": [{"agent": "safety_guard", "timing_ms": t1 - t0, "note": f"Flagged: {flagged}"}],
        }

    return safety_guard




def _truncate_summary_to_150_words(md: str) -> str:
    """Hard-truncate the Summary section to 150 readable words.

    Word count is based on visible text only — inline PMC citation links
    [PMCxxxxxxx:Section](url) are excluded from the count so they do not
    consume word slots.
    """
    import re

    pattern = re.compile(
        r'(#{1,2}\s*Summary(?:\s*\(.*?\))?[ \t]*\n)(.*?)(?=\n#{1,2}\s|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(md)
    if not match:
        return md

    header = match.group(1)
    body   = match.group(2).rstrip()

    # Count readable words using \b\w+\b — consistent with _summary_word_count.
    # Using .split() instead would miscount hyphenated words like "interferon‑stimulated"
    # (counts as 1 word with .split() but 2 with \b\w+\b), causing the 120-word
    # rollback guard to allow rollbacks that leave fewer than 120 visible words.
    _citation_re = re.compile(r'\[PMC[^\]]+\](?:\([^)]*\))?')

    def _wc(text: str) -> int:
        return len(re.findall(r'\b\w+\b', _citation_re.sub('', text)))

    if _wc(body) <= 150:
        return md

    # Find the character position of the 150th readable word in the original body
    # by re-scanning the body character by character using the same \b\w+\b metric.
    word_idx  = 0
    char_pos  = 0
    tokens    = re.split(r'(\s+)', body)   # alternating text/whitespace chunks
    cutoff    = len(body)

    for token in tokens:
        if not token.strip():              # whitespace chunk
            char_pos += len(token)
            continue
        # Check how many readable words this token contributes
        clean = _citation_re.sub('', token)
        n     = len(re.findall(r'\b\w+\b', clean))
        if n == 0:                         # pure citation token — skip
            char_pos += len(token)
            continue
        word_idx += n
        char_pos += len(token)
        if word_idx >= 150:
            cutoff = char_pos
            break

    truncated = body[:cutoff].rstrip()

    # Try to end at a sentence boundary — but only if it keeps ≥ 120 readable
    # words.  Without this guard the rollback can drop from 150 to ~88 words
    # when the cutoff falls inside a long sentence, producing summaries far
    # shorter than expected.
    sentence_end = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
    )
    if sentence_end > len(truncated) // 2:
        candidate = truncated[: sentence_end + 1]
        if _wc(candidate) >= 80:
            truncated = candidate
        else:
            # Rolling back would lose too many words — hard-cut at word 150 instead
            truncated = truncated.rstrip(",;:") + "."
    else:
        truncated = truncated.rstrip(",;:") + "."

    # Belt-and-suspenders: if the last sentence ends with a dangling preposition /
    # article (e.g. "...of only 88 of.") — a sign that hard-cut fired mid-phrase —
    # roll back one more sentence boundary.
    _dangling_last_re = re.compile(
        r'\b(?:of|to|the|a|an|and|or|from|by|for|only|with|into|upon|that|such|as'
        r'|its|their|those|these|was|were|is|are|has|have|been|be|at|in|on|via)\s*\.\s*$',
        re.IGNORECASE,
    )
    if _dangling_last_re.search(truncated):
        inner = truncated[:-1].rstrip()   # strip the added period
        last_end = max(inner.rfind(". "), inner.rfind("! "), inner.rfind("? "))
        if last_end > 0 and _wc(inner[:last_end + 1]) >= 60:
            truncated = inner[:last_end + 1]

    new_section = header + truncated + "\n"
    return md[: match.start()] + new_section + md[match.end():]


def _dedup_summary_vs_ke(md: str, floor: int = 70) -> str:
    """Remove Summary sentences that are verbatim copies of Key Evidence bullets.

    The LLM (especially via structured output) sometimes writes the KE bullets
    directly as prose sentences in the Summary, then repeats them as proper
    bullets.  Strip the verbatim duplicates from the Summary so the two sections
    carry distinct content.

    floor: minimum readable-word count that must remain after deduplication.
    Pass floor=0 to strip all duplicates regardless of remaining word count.
    """
    ke_m = re.search(r'(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
    if not ke_m:
        return md

    _cite_re = re.compile(r'\[PMC[^\]]+\](?:\([^)]+\))?')

    ke_norm: set = set()
    for bullet in re.findall(r'^\s*[-*]\s+(.+)', ke_m.group(1), re.MULTILINE):
        norm = _cite_re.sub('', bullet).strip().rstrip('.').lower()
        norm = re.sub(r'\s+', ' ', norm)
        if len(norm) > 30:
            ke_norm.add(norm)

    if not ke_norm:
        return md

    summ_m = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
    if not summ_m:
        return md

    summ_body = summ_m.group(1)
    sentences = re.split(r'(?<=[.!?])\s+', summ_body.strip())

    cleaned = []
    dropped = 0
    for sent in sentences:
        norm_sent = _cite_re.sub('', sent).strip().rstrip('.').lower()
        norm_sent = re.sub(r'\s+', ' ', norm_sent)
        if norm_sent in ke_norm:
            dropped += 1
            continue
        cleaned.append(sent)

    if not dropped:
        return md

    # Safety guard: only strip if the remaining Summary would have ≥ floor readable
    # words.  When the LLM generates a very short Summary (≤ 60 words) where
    # almost every sentence matches a KE bullet, stripping leaves only 1 sentence
    # (~17 words) which is then rebuilt from the same KE content — a circular
    # no-op.  In that case keep the duplicates (they're still valid content)
    # and let _needs_recon / _pad_summary_from_ke_final handle length.
    remaining_wc = len([
        w for w in _cite_re.sub('', ' '.join(cleaned)).split() if w
    ])
    if remaining_wc < floor:
        # Special-case override: when ALL non-first sentences are KE duplicates
        # (i.e. the LLM wrote the Summary AS the KE bullets), strip them anyway
        # so _needs_recon can run a proper reconstruction.  Keeping verbatim copies
        # is strictly worse — the LLM-review pass may have expanded them past the
        # 110-word threshold, permanently bypassing _needs_recon.
        total_non_first = len(sentences) - 1 if len(sentences) > 1 else 0
        if total_non_first > 0 and dropped >= total_non_first:
            new_body = (sentences[0].strip() + '\n') if sentences else ''
            print(f"  [FINAL] All {dropped} finding sentences are KE duplicates "
                  f"— cleared Summary (kept first sentence) for reconstruction")
            return md[:summ_m.start(1)] + new_body + md[summ_m.end(1):]
        return md

    new_body = ' '.join(cleaned).strip() + '\n'
    new_md = md[:summ_m.start(1)] + new_body + md[summ_m.end(1):]
    print(f"  [FINAL] Stripped {dropped} verbatim KE duplicate(s) from Summary (kept {remaining_wc} words)")
    return new_md


def _dedup_inline_list_items(text: str) -> str:
    """Remove duplicate consecutive items in comma-separated enumerations.

    Catches LLM copy-paste artifacts such as:
      'upregulation of Ifnar, Ifnar, and Ifngr'  →  'upregulation of Ifnar and Ifngr'
      'genes APP, APP, and DYRK1A'               →  'genes APP and DYRK1A'

    Handles single-word tokens and short multi-word phrases (up to 4 words).
    Applied to the full markdown (Summary + Key Evidence) in the finalize node.
    """
    # Match a word/phrase repeated immediately after a comma
    _pat = re.compile(
        r'\b([A-Za-z][\w\-/]*(?:\s+[A-Za-z][\w\-/]*){0,3}),\s+\1\b',
        re.IGNORECASE,
    )
    prev = None
    while prev != text:
        prev = text
        text = _pat.sub(r'\1', text)
    # Generic adjacent-duplicate collapse: ANY token immediately repeated across a
    # conjunction or comma is reduced to one occurrence. Entity-agnostic (works for
    # 'Ifnar and Ifnar', 'STAT1 and STAT1', 'hypothyroidism, hypothyroidism', etc.)
    # — no per-term list. Catches a common generation artifact the LLM cannot be
    # reliably prompted out of. Loops until stable.
    _adj_dup = re.compile(
        r'\b([A-Za-z][\w\-/]*)(?:\s+(?:and|or)\s+|,\s+)\1\b', re.IGNORECASE
    )
    prev = None
    while prev != text:
        prev = text
        text = _adj_dup.sub(r'\1', text)
    # Clean up orphaned comma left before a conjunction by the deduplication
    # e.g. 'Ifnar, and Ifngr' (2-item list) → 'Ifnar and Ifngr'
    text = re.sub(r',\s+(and|or)(\s)', r' \1\2', text)
    # Clean up double commas
    text = re.sub(r',\s*,', ',', text)
    return text


def _dedupe_summary_sentences(md: str) -> str:
    """Remove duplicate and near-duplicate sentences in the Summary section.

    Pass 1 — exact dedup: same text after normalising whitespace + case.
    Pass 2 — semantic dedup: two sentences sharing >60% of their content words
    (non-stopword nouns/verbs/adjectives, ignoring citations) are considered
    near-duplicates; the shorter/later one is dropped.
    """
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    _stopwords = {
        "a","an","the","of","in","on","at","to","for","with","by","from","as",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "and","or","but","if","that","this","these","those","it","its",
        "not","no","nor","so","yet","both","either","neither","each",
        "such","than","then","when","where","which","who","whom","how",
        "what","whether","while","although","because","since","until","unless",
        "their","they","them","we","our","us","you","your","i","my","me",
        "individuals","people","patients","syndrome","down","ds","t21",
    }

    def _content_words(s: str) -> set:
        # Strip PMC citations and URLs before computing content word overlap.
        # Token pattern is alphanumeric ([a-z0-9]) so named entities — gene symbols
        # (ifnar1, il10rb, stat1), acronyms, and numbers — are INCLUDED in the
        # overlap, not silently dropped. This makes near-duplicate detection
        # entity-agnostic: it works for any query (genes, conditions, drugs),
        # not just prose words. Generic — no per-term special-casing.
        s = re.sub(r'\[PMC\d+:[^\]]+\](?:\([^)]*\))?', '', s)
        words = re.findall(r'\b[a-z0-9]{3,}\b', s.lower())
        return {w for w in words if w not in _stopwords}

    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # Split into sentences
    parts = re.split(r'(?<=[\.\!\?])\s+', body)
    parts = [s.strip() for s in parts if s.strip()]

    # Pass 1: exact dedup
    seen_exact = set()
    after_exact = []
    for s in parts:
        key = re.sub(r'\s+', ' ', s).lower()
        if key not in seen_exact:
            seen_exact.add(key)
            after_exact.append(s)

    # Pass 2: semantic near-dedup (drop later sentence if Jaccard > 0.60)
    kept = []
    kept_cwords = []
    for s in after_exact:
        cw = _content_words(s)
        is_dup = any(_jaccard(cw, prev) > 0.60 for prev in kept_cwords)
        if not is_dup:
            kept.append(s)
            kept_cwords.append(cw)

    # Pass 3: subset dedup — drop a later sentence whose medical content words
    # are entirely a subset of words already covered by earlier sentences.
    # This catches the pattern where an earlier "grouped" sentence covers
    # conditions A, B, C and later sentences individually re-state each.
    covered_words: set = set()
    final_kept = []
    for i, s in enumerate(kept):
        cw = _content_words(s)
        # Always keep first and last sentence (context and conclusion are citation-free)
        if i == 0 or i == len(kept) - 1:
            final_kept.append(s)
            covered_words |= cw
            continue
        # Drop if all medical content words (≥5 chars, excluding stopwords) are
        # already fully covered by what we've kept — this is a topic re-statement.
        medical_cw = {w for w in cw if len(w) >= 5}
        if medical_cw and medical_cw.issubset(covered_words):
            continue  # redundant sentence — all its topics already covered
        final_kept.append(s)
        covered_words |= cw

    new_body = " ".join(final_kept).strip()
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]


# Regex that identifies the primary verb in a KE bullet sentence.
# Used to extract the subject noun phrase for enumeration summaries.
_KE_VERB_PAT = re.compile(
    r'\b(is|are|was|were|has|have|occurs?|shows?|appear|develop|suggest|'
    r'include|involve|co.occur|affect|contribute|result|demonstrate|exhibit|'
    r'presents?|co.occurs?|manifests?)\b',
    re.IGNORECASE,
)


def _extract_ke_subject(sentence: str) -> str:
    """Extract the subject noun phrase from a KE bullet sentence.

    Returns the words before the first main verb, capped at 7 words.
    Falls back to the first 4 words if no verb is found.
    """
    # Strip trailing citation tags first
    clean = re.sub(r'\s*\[PMC[^\]]+\]\s*$', '', sentence).strip()
    # Also strip any markdown-linked citations
    clean = re.sub(r'\[PMC[^\]]+\]\([^)]+\)', '', clean).strip()
    m = _KE_VERB_PAT.search(clean)
    if m:
        subj = clean[: m.start()].strip().rstrip(',')
        words = subj.split()
        if 1 <= len(words) <= 7:
            return subj
    # Fallback: first 4 words or up to first comma
    before_comma = clean.split(',')[0].strip()
    words = before_comma.split()[:5]
    return " ".join(words)


def _extract_condition_from_ke_bullet(bullet: str) -> str:
    """Extract the key medical condition / entity from a KE bullet string.

    Returns a short phrase like 'dementia' or 'autoimmune hypothyroidism and
    celiac disease', or '' if nothing suitable is found.
    Hyperlinked PMC citations ([PMCxxxx:Section](url)) are stripped first.
    """
    # Strip PMC citation links
    clean = re.sub(r'\[PMC\d+:[^\]]+\]\([^)]+\)', '', bullet)
    clean = re.sub(r'\[PMC\d+:[^\]]+\]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    _med_terms = {
        'disease', 'disorder', 'syndrome', 'leukemia', 'leukaemia', 'carcinoma',
        'cancer', 'tumor', 'infection', 'apnea', 'fibrillation', 'neuropathy',
        'thyroiditis', 'enteropathy', 'malignancy', 'defect', 'failure', 'anemia',
        'megakaryoblastic', 'cardiomyopathy', 'thrombocytopenia', 'tachycardia',
        # Common DS comorbidities not in the original set
        'obesity', 'hypothyroidism', 'hyperthyroidism', 'autism', 'dementia',
        'diabetes', 'hypertension', 'epilepsy', 'seizure', 'constipation',
        'depression', 'anxiety', 'cataract', 'scoliosis',
    }

    # Pattern G (HIGHEST PRIORITY): "at (high|elevated) risk of (others,) (such as) A, B, C"
    # This targets "while being at high risk of others, such as Alzheimer's disease,
    # CHD, autism..." and must run BEFORE Pattern C which would grab just "others".
    # Char class [^.;(\n]: stops at sentence-ending dot, semicolon, open-paren, newline.
    m = re.search(
        r'\bat\s+(?:high|higher|elevated)\s+risk\s+of\s+(?:others?\s*,\s*)?'
        r'(?:such\s+as\s+)?'
        r'([A-Za-z][^.;(\n]+)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip()
        cond = re.sub(r'\s*\(\d+(?:[,\-\u2013]\d+)*\)\s*$', '', cond).strip().rstrip('.,; ')
        if (
            cond.count(',') >= 1
            and len(cond) >= 5
            and not re.match(r'^others?\b', cond, re.I)
        ):
            return cond[:150]

    # Pattern A: "prevalence of X, Y, Z is [elevated|similar|...]"
    m = re.search(
        r'\bprevalence\s+of\s+([A-Za-z][^.;\[({)\n]+?)\s+is\b',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        if len(cond) >= 5:
            return cond[:100]

    # Pattern B: multiple "risk of X" clauses → join them
    risk_hits = re.findall(
        r'\brisk\s+of\s+([A-Za-z][^,;\[({)\n]{3,40}?)(?=[,;\[({)\n]|$)',
        clean, re.IGNORECASE,
    )
    if len(risk_hits) >= 2:
        cleaned_hits = []
        for c in risk_hits:
            c = c.strip().rstrip('.,; ')
            c = re.sub(r'\s+(?:during|in|at)\s+\S.*$', '', c, flags=re.IGNORECASE).strip()
            if len(c) >= 3:
                cleaned_hits.append(c)
        if len(cleaned_hits) >= 2:
            if len(cleaned_hits) == 2:
                return f"{cleaned_hits[0]} and {cleaned_hits[1]}"
            return ", ".join(cleaned_hits[:-1]) + f", and {cleaned_hits[-1]}"

    # Pattern C: single "risk of/for X" — skip bare "others" (handled above by Pattern G)
    m = re.search(
        r'\brisk\s+(?:of|for)\s+([A-Za-z][^,.;\[({)\n]{3,50}?)(?=[,.;\[({)\n]|$)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        cond = re.sub(r'\s+(?:including|such\s+as)\s+.*$', '', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\s+(?:during|at|in)\s+\S.*$', '', cond, flags=re.IGNORECASE).strip()
        if len(cond) >= 3 and not re.match(
            r'^(?:developing|getting|having|both|these|others?|the\s)\b', cond, re.I
        ):
            return cond[:50]

    # Pattern D: "higher incidence of X"
    m = re.search(
        r'\b(?:higher\s+)?incidence\s+of\s+([A-Za-z][^,;\[({)\n]{3,50}?)(?=[,;\[({)\n]|$)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        cond = re.sub(r'\s+(?:including|such\s+as)\s+.*$', '', cond, flags=re.IGNORECASE).strip()
        if len(cond) >= 3:
            return cond[:50]

    # Pattern K: "N-fold higher odds/risk/likelihood of X"
    # Handles bullets like "People with DS had 2.9-fold higher odds of COVID-19 diagnosis"
    m = re.search(
        r'\b\d+(?:\.\d+)?(?:\s*[-\u2013]\s*\d+(?:\.\d+)?)?\s*-?\s*fold\b[^.]*'
        r'\b(?:odds?\s+of|risk\s+of|likelihood\s+of|higher\s+(?:odds?|risk)\s+of)\s+'
        r'([A-Za-z][^,.;\[({)\n]{3,60}?)(?=[,.;\[({)\n]|\s+and\s+\d|$)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        if len(cond) >= 3:
            return cond[:60]

    # Pattern H: "lower/reduced rates/levels/risk of X" — lower-risk conditions
    m = re.search(
        r'\b(?:lower|reduced|decreased)\s+(?:rates?|levels?|risk|prevalence)\s+of\s+'
        r'([A-Za-z][^,.;\[({)\n]{3,60}?)(?=[,.;\[({)\n]|$)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        cond = re.sub(r'\s+compared\s+.*$', '', cond, flags=re.IGNORECASE).strip()
        cond = re.sub(r'\s+(?:in|among|for)\s+.*$', '', cond, flags=re.IGNORECASE).strip()
        if len(cond) >= 3:
            return cond[:60]

    # Pattern I: "X, including A and B, are/is lower" — captures the 'including' sub-list
    # e.g. "Traditional vascular risk factors, including hypertension and diabetes type 2,
    #       are lower in adults with DS"
    m = re.search(
        r'(?:risk factors?|conditions?|diseases?)[^,]*,\s*including\s+'
        r'([A-Za-z][^,;\[({)\n]+?)\s*,\s+(?:are|is|were|was)\s+(?:lower|reduced|less)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        if len(cond) >= 3:
            return cond[:80]

    # Pattern E: "such as X and Y" / "including X, Y, Z" (stop at first comma)
    m = re.search(
        r'\b(?:such\s+as|including)\s+([A-Za-z][^,;\[({)\n]{5,80}?)(?=[,;\[({)\n]|$)',
        clean, re.IGNORECASE,
    )
    if m:
        cond = m.group(1).strip().rstrip('.,; ')
        if len(cond) >= 5:
            return cond[:70]

    # Pattern L: comma-separated list at sentence start followed by "are/were among".
    # Handles "Obesity, hypothyroidism and obstructive sleep apnea (OSA) are among
    # the most frequent comorbidities..."  Must run BEFORE F2 which would only
    # grab the first word before the comma.
    m = re.match(
        r'^([A-Z][^.;!?\[]{5,150}?)\s+(?:are|were|is|was)\s+'
        r'(?:among|the\s+most|frequently|identified\s+as|recognized\s+as|reported\s+as)',
        clean,
    )
    if m:
        cond = m.group(1).strip().rstrip(',; ')
        if len(cond) >= 5:
            return cond[:150]

    # Pattern F1: "ConditionName is/are/was/were/affected/accounted..." at sentence start
    # Extended verb list captures bullets like "Cardiovascular disease affected 35.6%..."
    # and "Respiratory tract infections accounted for nearly 80%..."
    m = re.match(
        r'^([A-Z][a-z]+(?:\s+[a-z]+){1,5})\s+'
        r'(?:is|are|was|were|occurs?\b|has\b|have\b|affected\b|accounted\b|'
        r'occurred\b|showed\b|increased\b|decreased\b|reported\b|identified\b)',
        clean,
    )
    if m:
        subj = m.group(1).strip()
        if any(t in subj.lower() for t in _med_terms):
            return subj

    # Pattern F2: "ConditionName (abbrev)..." or "ConditionName, ..."
    m = re.match(r'^([A-Z][a-zA-Z\s]+?)\s*[,(]', clean)
    if m:
        subj = m.group(1).strip()
        if len(subj.split()) >= 2 and any(t in subj.lower() for t in _med_terms):
            return subj

    return ""


def _is_gene_like_list(s: str) -> bool:
    """Return True when a list string is predominantly gene/molecule names
    rather than disease conditions.

    Gene-like tokens: 2+ uppercase letters optionally followed by digits/uppercase
    (e.g. OAS1, IFNAR1, MX1, STAT1, STAT2, IL10RB, NF-KB).
    When ≥40% of the whitespace-delimited tokens are gene-like, we treat the
    whole list as a molecular list and avoid the 'elevated risk for' template.
    """
    tokens = re.findall(r'\b\w+\b', s)
    if len(tokens) < 2:
        return False
    gene_like = sum(1 for t in tokens if re.match(r'^[A-Z]{2,}[\dA-Z]*$', t))
    return gene_like / len(tokens) >= 0.4


# Discourse/transition markers and fragments that are NEVER valid enumeration
# items. Generic — independent of topic (genes, conditions, drugs). Used to reject
# junk extracted by the reconstruction strategies (e.g. a bullet that yields
# "on the other hand" instead of a real condition name).
_ENUM_JUNK_RE = re.compile(
    r'^(?:on the other hand|on the contrary|by contrast|in contrast|conversely|'
    r'thus|hence|therefore|however|moreover|furthermore|whereas|nonetheless|'
    r'nevertheless|meanwhile|additionally|consequently|accordingly|notably|'
    r'importantly|interestingly|similarly|likewise|also|as such|'
    r'as (?:previously |well )?(?:described|noted|shown|reported|mentioned)|'
    r'ttc|etc|i\.e|e\.g|in (?:addition|particular|general|fact))\b',
    re.IGNORECASE,
)


# Raw verbatim methods / sample-inventory narration that must NEVER be stitched
# into a reconstructed Summary, regardless of where in the sentence it appears
# (the start-only first-person check misses 'Using ... we completed ...' and
# '... mice were selected because ...'). Generic — methods-narration patterns,
# no topic terms.
_RECON_RAW_SENT_RE = re.compile(
    r'\b(?:we|our)\s+(?:completed|performed|conducted|selected|employed|analyz\w*|'
    r'used|examined|investigat\w*|generated|collected|measured|assessed|report\w*)\b'
    r'|\bwere\s+selected\s+because\b'
    r'|\(\s*\d+\s+(?:male|female|men|women)\b'                 # (163 male and 141 female)
    r'|\bdata\s+from\s+\d+\s+(?:individuals?|patients?|participants?|samples?|subjects?)\b'
    r'|\b(?:Extended\s+Data|Supplementary\s+(?:Fig|Table|Data)|Suppl\.|Methods\s*,)\b'
    r'|\bFig\.?\s*\d',
    re.IGNORECASE,
)


def _is_junk_enum_item(s: str) -> bool:
    """True when a reconstruction-extracted 'item' is not a real entity but a
    discourse fragment, stray punctuation, or too short to be a condition/gene
    name. Generic — no topic-specific term list. Prevents garbled output like
    'are at elevated risk for on the other hand, by contrast, thus'."""
    s2 = re.sub(r'\[PMC[^\]]+\]', '', s).strip().strip('.,;:()')
    if len(s2) < 4:
        return True
    if _ENUM_JUNK_RE.match(s2):
        return True
    # must contain at least one substantive (≥4-letter) word to be a real item
    if not re.search(r'[A-Za-z]{4,}', s2):
        return True
    return False


def _reconstruct_list_summary(md: str, question: str = "") -> str:
    """When a list-type question produces a short Summary (< 100 words), rebuild
    it by reformatting the Key Evidence bullets as flowing prose.

    For enumeration questions ("which conditions/genes/receptors..."), creates a
    compact enumeration sentence naming ALL KE items so nothing is omitted.
    For other questions, converts up to 6 KE bullets into paragraph sentences.

    Citation tags ([PMCxxxx:Section]) in the bullets are preserved; URLs are
    stripped so that _hyperlink_inline_pmc can re-link them uniformly.
    """
    summ_m = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    ke_m   = re.search(r"(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if not summ_m or not ke_m:
        return md

    summ_body = summ_m.group(1).strip()
    ke_body   = ke_m.group(1).strip()

    # Split Summary into sentences; take first (context) and last (conclusion)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summ_body) if s.strip()]
    context_sent    = sents[0]  if sents      else ""
    conclusion_sent = sents[-1] if len(sents) > 1 else ""

    # Parse Key Evidence bullets.
    # NOTE: '\u2022' in a raw string is NOT the '•' char; use a line-split approach
    # with the literal bullet characters to avoid silent regex mismatches.
    bullet_parts = re.split(r"\n(?=[•\-\*]\s)", "\n" + ke_body)
    bullets = [re.sub(r"^[•\-\*]\s+", "", p.strip()) for p in bullet_parts if p.strip()]
    # Remove the ke_body itself if the split didn't find sub-bullets
    bullets = [b for b in bullets if b and not b == ke_body.strip()]
    if not bullets:
        return md

    # Regexes for cleaning raw-chunk text before stitching into the Summary
    _url_re        = re.compile(r"(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\(https://[^)]+\)")
    _num_cite_re   = re.compile(r"\[\d+(?:[,\-–]\s*\d+)*\]")           # [13-15], [1,2,3]
    # Any parenthetical containing 'et al.' + a 4-digit year — matches both the
    # single form '(Li et al., 2007)' and multi-author lists
    # '(Smith et al., 2020; Jones et al., 2021)'. [^)] keeps it within one paren.
    _author_cite_re= re.compile(r"\([^)]*\bet\s+al\.[^)]*\d{4}[^)]*\)")
    _frag_re       = re.compile(r"^([a-z][,\s]|\d|Fig|Table|Suppl|Extended|S\d)", re.I)
    _firstperson_re= re.compile(r"^\s*(We|we|Our|our|Here\s+we|I\s+)\b")

    bullet_sents = []
    for b in bullets[:8]:
        b = b.strip().replace("\n", " ")
        b = _url_re.sub(r"\1", b)           # [PMCxxx:Sec](url) → [PMCxxx:Sec]
        b = _num_cite_re.sub("", b)         # strip [13-15] etc.
        b = _author_cite_re.sub("", b)      # strip (Smith et al., 2020; ...) etc.

        # Extract the trailing PMC tag before splitting into sentences
        tag_m = re.search(r"(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\s*$", b)
        pmc_tag = tag_m.group(1) if tag_m else ""
        body_text = b[: tag_m.start()].strip() if tag_m else b

        # Split into individual sentences; pick the first substantive one
        _boilerplate_re = re.compile(
            r'^(RESEARCH IN CONTEXT|Systematic review|Literature was reviewed|'
            r'Interpretation|Added value|[A-Z]{3,}\s+[A-Z]{3,}|'
            r'Background:|Methods:|Results:|Conclusion:)',
            re.IGNORECASE
        )
        sents = re.split(r"(?<=[.!?])\s+", body_text)
        chosen = None
        for s in sents:
            s = s.strip()
            # Skip: too short, panel labels, figure refs, first-person, table refs
            if len(s.split()) < 10:
                continue
            if _frag_re.match(s):
                continue
            if _firstperson_re.match(s):
                continue
            if re.search(r"\bTable\s+\d\b|\bFig\b|\bSuppl\b", s, re.I):
                continue
            if _boilerplate_re.match(s):
                continue
            # Reject raw verbatim methods / sample-inventory narration anywhere in
            # the sentence (not just at the start) — e.g. 'Using ... we completed ...',
            # '... mice were selected because ...', '(163 male and 141 female)'.
            if _RECON_RAW_SENT_RE.search(s):
                continue
            chosen = s
            break

        if not chosen:
            continue

        # Re-attach the PMC tag
        if chosen[-1] not in ".!?":
            chosen += "."
        if pmc_tag:
            chosen = chosen + " " + pmc_tag
        bullet_sents.append(chosen)

    # ── Enumeration path: for list questions, build one compact enumeration ──
    # sentence naming ALL conditions/genes/items rather than 4-6 full sentences.
    _is_enum_q = bool(_LIST_Q_PAT.search(question)) if question else False
    if _is_enum_q:
        # Polarity helpers used by all three strategies for padding.
        # We avoid adding sentences of the OPPOSITE polarity to the question
        # so that _enforce_polarity_focus (which runs later) won't strip them
        # and pull the wc back below 90, triggering _build_polarity_summary_from_ke.
        _pad_lower_re = re.compile(
            r'\b(lower risk|reduced risk|less (?:common|prevalent|frequent)|lower rates?|'
            r'less likely|decreased risk|lower in (?:adults|individuals|people)|protective)\b',
            re.IGNORECASE,
        )
        _pad_higher_re = re.compile(
            r'\b(higher risk|elevated risk|increased risk|at (?:high|higher|elevated) risk|'
            r'more (?:common|prevalent|frequent)|greater risk)\b',
            re.IGNORECASE,
        )
        _pad_q_higher = bool(re.search(r'\bhigher risk\b', question, re.IGNORECASE))
        _pad_q_lower  = bool(re.search(r'\blower risk\b',  question, re.IGNORECASE))

        def _wc_parts(parts: list) -> int:
            return sum(
                len([w for w in re.sub(r'\[PMC[^\]]+\]', '', ss).split() if w])
                for ss in parts
            )

        def _pad_enum_parts(parts: list, seen_set: set, bullet_sents: list) -> None:
            """Append polarity-safe bullet sentences until combined word count >= 90."""
            for _bsp in bullet_sents:
                if _wc_parts(parts) >= 90:
                    break
                # For higher-risk questions, skip pure lower-risk sentences (they'd be stripped)
                if _pad_q_higher and _pad_lower_re.search(_bsp) and not _pad_higher_re.search(_bsp):
                    continue
                # For lower-risk questions, skip pure higher-risk sentences
                if _pad_q_lower and _pad_higher_re.search(_bsp) and not _pad_lower_re.search(_bsp):
                    continue
                _kp = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', _bsp)).strip().lower()
                if _kp and _kp not in seen_set:
                    seen_set.add(_kp)
                    parts.append(_bsp)
            # Second pass: any sentence if still short
            for _bsp in bullet_sents:
                if _wc_parts(parts) >= 90:
                    break
                _kp = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', _bsp)).strip().lower()
                if _kp and _kp not in seen_set:
                    seen_set.add(_kp)
                    parts.append(_bsp)

        # Strategy 0: extract the main condition/entity from EACH KE bullet
        # individually, then build one sentence that names ALL of them.
        # This outperforms Strategies 1/2 when each bullet covers a distinct
        # condition (no single "such as A, B, C" phrase spans all bullets).
        _cond_items: list = []   # [(condition_str, cite_token)]
        # Question polarity: determines which bullets to include and the sentence template.
        _q_is_lower = bool(re.search(
            r'\b(lower risk|less likely|protective|not at (?:higher )?risk|reduced risk|'
            r'lower(?:ed)? (?:rates?|prevalence)|less (?:common|frequent|prevalent)|'
            r'decreased risk|do not develop|rarely develop)\b', question, re.IGNORECASE))
        _q_is_higher = bool(re.search(
            r'\b(higher risk|elevated risk|increased risk|more likely|greater risk|'
            r'more (?:common|frequent|prevalent)|over-represented)\b', question, re.IGNORECASE))
        _s0_lower_re = re.compile(
            r'\b(?:are lower|is lower|lower rates?|lower in adults?|lower prevalence|'
            r'less common|not elevated|reduced in|less prevalent|lower than|'
            r'lower (?:in|compared)\b)\b',
            re.IGNORECASE,
        )
        _s0_high_re = re.compile(
            r'\b(?:at high risk|elevated risk|increased risk|higher risk|'
            r'at higher risk|being at high risk|high risk of)\b',
            re.IGNORECASE,
        )
        for _b0 in bullets[:8]:
            _b0_raw = _b0.strip().replace("\n", " ")
            # For lower-risk questions: include lower-risk bullets, skip pure higher-risk ones.
            # For higher-risk questions (or neutral): skip pure lower-risk bullets.
            if _q_is_lower and not _q_is_higher:
                # Include lower-risk bullets; skip pure higher-risk bullets
                if _s0_high_re.search(_b0_raw) and not _s0_lower_re.search(_b0_raw):
                    continue
            else:
                # Default: skip pure lower-risk bullets
                if _s0_lower_re.search(_b0_raw) and not _s0_high_re.search(_b0_raw):
                    continue
            _pmcm0 = re.search(r'\[PMC(\d+):([^\]]+)\]', _b0_raw)
            _cite0 = f"[PMC{_pmcm0.group(1)}:{_pmcm0.group(2)}]" if _pmcm0 else ""
            _cond0 = _extract_condition_from_ke_bullet(_b0_raw)
            if _cond0 and not _is_junk_enum_item(_cond0):
                _cond_items.append((_cond0, _cite0))

        if len(_cond_items) >= 2:
            _parts0: list = []
            for _c0, _ct0 in _cond_items:
                _tok0 = _c0
                if _ct0:
                    _tok0 = _tok0.rstrip('.,!?') + ' ' + _ct0
                _parts0.append(_tok0)
            # Use polarity-correct sentence template
            if _q_is_lower and not _q_is_higher:
                _enum_phrase = "have lower rates of"
            else:
                _enum_phrase = "are at elevated risk for"
            if len(_parts0) > 1:
                _enum_s0 = (
                    f"Individuals with Down syndrome {_enum_phrase} "
                    + ", ".join(_parts0[:-1])
                    + f", and {_parts0[-1]}."
                )
            else:
                _enum_s0 = (
                    f"Individuals with Down syndrome {_enum_phrase} {_parts0[0]}."
                )
            _seen_s0: set = set()
            def _add_s0(s: str, lst: list) -> None:
                k = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', s)).strip().lower()
                if k not in _seen_s0 and k:
                    _seen_s0.add(k)
                    lst.append(s)
            _pl0: list = []
            if context_sent:
                _add_s0(context_sent, _pl0)
                _seen_s0.add(re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', context_sent)).strip().lower())
            _add_s0(_enum_s0, _pl0)
            _pad_enum_parts(_pl0, _seen_s0, bullet_sents)
            _new_sec0 = "# Summary\n" + " ".join(_pl0) + "\n"
            return md[: summ_m.start()] + _new_sec0 + md[summ_m.end():]
        # else fall through to Strategy 1

        # Strategy 1: two-pass scan of the full Key Evidence body for condition lists.
        # Pass A (priority): search for the canonical DS pattern
        # "at (high|higher|elevated) risk of (others,) (such as) A, B, C"
        # — this targets e.g. "while being at high risk of others, such as
        # Alzheimer's disease, CHD, autism, and severe COVID-19".
        # Pass B (fallback): general "such as / including X, Y" scan, but
        # rejects matches where the candidate itself describes lower-risk items.
        #
        # Lookahead allows [PMCxxxxx:Section] tags after period stop (ke_scan
        # retains plain [PMC...] tags since only URL-linked ones are stripped).
        #
        # Clean the whole KE body for scanning (strip markdown URLs + numeric cites)
        ke_scan = re.sub(r'\[PMC[^\]]+\]\([^)]+\)', '', ke_body)
        ke_scan = re.sub(r'\(\d+(?:[,\-\u2013]\s*\d+)*\)', '', ke_scan)
        ke_scan = re.sub(r'\s+', ' ', ke_scan)

        best_list_str = ""
        best_list_tag = ""

        # Pass A: dedicated high-risk-list pattern (no ambiguity about polarity)
        _HIGH_RISK_LIST_PAT = re.compile(
            r'\bat\s+(?:high|higher|elevated)\s+risk\s+of\s+(?:others?[,\s]+)?'
            r'(?:such\s+as\s+)?'
            r'([A-Za-z][^.;(\n]+?)'
            r'(?=\s*[.;]\s*(?:[A-Z\[]|\Z)|\s*\[PMC|\Z)',
            re.IGNORECASE,
        )
        for m_hr in _HIGH_RISK_LIST_PAT.finditer(ke_scan):
            candidate = m_hr.group(1).strip().rstrip('.,()')
            if candidate.count(',') >= 2:   # need at least 3 items
                if len(candidate) > len(best_list_str):
                    best_list_str = candidate
                    _near = ke_body[
                        max(0, m_hr.start() - 20): m_hr.start() + len(candidate) + 150
                    ]
                    _tag_n = re.search(r'(\[PMC\d{4,}:[A-Za-z0-9_-]+\])', _near)
                    best_list_tag = _tag_n.group(1) if _tag_n else ""

        if not best_list_str:
            # Pass B: general "such as / including X, Y" — prefer high-risk context;
            # reject candidates that describe things DS has LOWER risk for UNLESS the
            # question itself is asking about lower-risk conditions.
            _SUCH_AS_PAT = re.compile(
                r'(?:such as|including)\s+([A-Za-z][^.;({[\n]+?)'
                r'(?=\s*[({[]|\s*[.;]\s*(?:[A-Z\[]|\Z)|\Z)',
                re.IGNORECASE,
            )
            _neg_polarity_re = re.compile(
                r'\b(?:are lower|is lower|lower rates?|lower in adults?|less common|'
                r'not elevated|reduced in|protective|less prevalent|are not at)\b',
                re.IGNORECASE,
            )
            best_is_high_risk = False
            for m_sa in _SUCH_AS_PAT.finditer(ke_scan):
                candidate = m_sa.group(1).strip().rstrip('.,')
                if candidate.count(',') < 1:
                    continue
                # Reject lower-risk candidates ONLY when the question is NOT lower-risk
                if not _q_is_lower and _neg_polarity_re.search(candidate):
                    continue
                before = ke_scan[max(0, m_sa.start() - 250): m_sa.start()]
                is_high_risk = bool(re.search(
                    r'(?:high|elevated|increased|greater)\s+risk', before, re.IGNORECASE
                ))
                if is_high_risk and not best_is_high_risk:
                    take = True
                elif is_high_risk == best_is_high_risk and len(candidate) > len(best_list_str):
                    take = True
                else:
                    take = False
                if take:
                    best_list_str = candidate
                    best_is_high_risk = is_high_risk
                    _near = ke_body[
                        max(0, m_sa.start() - 50): m_sa.start() + len(candidate) + 150
                    ]
                    _tag_n = re.search(r'(\[PMC\d{4,}:[A-Za-z0-9_-]+\])', _near)
                    best_list_tag = _tag_n.group(1) if _tag_n else ""

        if best_list_str:
            s = best_list_str
            tag_part = (" " + best_list_tag) if best_list_tag else ""
            # Use polarity-correct or molecular sentence template
            if _is_gene_like_list(s):
                # Gene/molecule list: avoid "elevated risk for" framing
                enum_sent = (
                    f"Genes showing elevated expression in Down syndrome include "
                    f"{s}{tag_part}."
                )
            elif _q_is_lower and not _q_is_higher:
                enum_sent = (
                    f"Individuals with Down syndrome have lower rates of "
                    f"{s[0].lower()}{s[1:]}{tag_part}."
                )
            else:
                enum_sent = (
                    f"Individuals with Down syndrome are at elevated risk for "
                    f"{s[0].lower()}{s[1:]}{tag_part}."
                )
            _seen_ne: set = set()
            def _add_ne(sent, lst):
                k = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', sent)).strip().lower()
                if k not in _seen_ne and k:
                    _seen_ne.add(k)
                    lst.append(sent)
            parts_ne: list = []
            if context_sent:
                _add_ne(context_sent, parts_ne)
                _seen_ne.add(re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', context_sent)).strip().lower())
            _add_ne(enum_sent, parts_ne)
            _pad_enum_parts(parts_ne, _seen_ne, bullet_sents)
            new_section_ne = "# Summary\n" + " ".join(parts_ne) + "\n"
            return md[: summ_m.start()] + new_section_ne + md[summ_m.end():]

        # Strategy 2: extract subjects from individual clean bullets
        _fp_re2 = re.compile(r'^\s*(We|we|Our|our|Here\s+we|I\s|This\s+study)\b')
        _generic_starts2 = (
            'beyond', 'based on', 'the developmental',
            'chronic condition', 'among those', 'the evidence',
        )
        items = []  # (subject_phrase, pmc_tag)
        for b in bullets:
            b_c2 = re.sub(r'\s*\(https://[^)]+\)', '', b).strip()
            b_c2 = _url_re.sub(r'\1', b_c2)
            b_c2 = _num_cite_re.sub('', b_c2)
            b_c2 = _author_cite_re.sub('', b_c2)
            tag_m2 = re.search(r'(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\s*$', b_c2)
            pmc_tag2 = tag_m2.group(1) if tag_m2 else ''
            body2 = b_c2[: tag_m2.start()].strip() if tag_m2 else b_c2
            first_sent = re.split(r'(?<=[.!?])\s+', body2)[0].strip()
            if _fp_re2.match(first_sent) or _boilerplate_re.match(first_sent):
                continue
            if len(first_sent.split()) < 5:
                continue
            subj = _extract_ke_subject(first_sent)
            if not subj or any(subj.lower().startswith(g) for g in _generic_starts2):
                continue
            items.append((subj, pmc_tag2))
        if len(items) >= 2:
            enum_parts = []
            for subj, tag in items:
                part = subj[0].lower() + subj[1:] if subj else ''
                if tag:
                    part = part.rstrip('.!?,') + ' ' + tag
                enum_parts.append(part)
            # Use polarity-correct or molecular sentence template
            _s2_items_str = ' '.join(p for p, _ in [(ep, None) for ep in enum_parts])
            if _is_gene_like_list(_s2_items_str):
                _s2_verb = 'show elevated expression of'
            elif _q_is_lower and not _q_is_higher:
                _s2_verb = 'have lower rates of'
            else:
                _s2_verb = 'face elevated risk for'
            if len(enum_parts) > 1:
                last_ep = enum_parts[-1]
                rest_ep = ', '.join(enum_parts[:-1])
                enum_sent = f'Individuals with Down syndrome {_s2_verb} {rest_ep}, and {last_ep}.'
            else:
                enum_sent = f'Individuals with Down syndrome {_s2_verb} {enum_parts[0]}.'
            _seen_n2: set = set()
            def _add_n2(s, lst):
                k = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', s)).strip().lower()
                if k not in _seen_n2 and k:
                    _seen_n2.add(k)
                    lst.append(s)
            parts_n2: list = []
            if context_sent:
                _add_n2(context_sent, parts_n2)
                _seen_n2.add(re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', context_sent)).strip().lower())
            _add_n2(enum_sent, parts_n2)
            _pad_enum_parts(parts_n2, _seen_n2, bullet_sents)
            new_section_n2 = '# Summary\n' + ' '.join(parts_n2) + '\n'
            return md[: summ_m.start()] + new_section_n2 + md[summ_m.end():]

        # All three strategies failed to extract enumerable items (e.g. lower-risk
        # question where KE bullets don't have extractable high-risk conditions).
        # Return the original markdown unchanged rather than returning None.
        return md

    # Non-enumeration question: pad the Summary from KE bullet sentences when
    # it is shorter than 110 words so the output is consistently 110–150 words.
    _summ_body_wc = len(
        [w for w in re.sub(r'\[PMC[^\]]+\](?:\([^)]+\))?', '', summ_body).split() if w]
    )
    if _summ_body_wc >= 110 or not bullet_sents:
        return md  # already long enough or no KE material to draw from

    _seen_gen: set = set()
    parts_gen: list = []

    def _add_gen(s: str) -> None:
        k = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', s)).strip().lower()
        if k and k not in _seen_gen:
            _seen_gen.add(k)
            parts_gen.append(s)

    def _wc_gen() -> int:
        return sum(
            len([w for w in re.sub(r'\[PMC[^\]]+\]', '', p).split() if w])
            for p in parts_gen
        )

    # Seed with the existing summary body (context sentence + middle sentences;
    # hold back the conclusion sentence so we can re-append it at the end).
    _existing_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summ_body) if s.strip()]
    _conclusion_sent = _existing_sents[-1] if len(_existing_sents) > 1 else ""
    _body_sents = _existing_sents[:-1] if _conclusion_sent else _existing_sents
    for _s in _body_sents:
        _add_gen(_s)

    # Append KE bullet sentences (already cleaned by earlier stripping) until
    # the combined word count reaches at least 130 (leaving room for truncation
    # at 150 so the summary reliably lands in the 120–150 window).
    _url_re_gen = re.compile(r'(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\(https://[^)]+\)')
    for _bs in bullet_sents:
        if _wc_gen() >= 130:
            break
        _bs_clean = _url_re_gen.sub(r'\1', _bs).strip()
        _add_gen(_bs_clean)

    # Re-append the conclusion sentence last.
    if _conclusion_sent:
        k = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', _conclusion_sent)).strip().lower()
        if k not in _seen_gen:
            parts_gen.append(_conclusion_sent)

    if not parts_gen:
        return md

    new_section_gen = "# Summary\n" + " ".join(parts_gen) + "\n"
    return md[: summ_m.start()] + new_section_gen + md[summ_m.end():]

_LIST_Q_PAT = re.compile(
    r'\b(which|what)\b.{0,80}\b(conditions?|risks?|diseases?|comorbid|higher risk|lower risk|associated|complications?'
    r'|receptors?|genes?|proteins?|pathways?|subunits?|markers?|biomarkers?|mutations?|variants?|types?|kinds?|forms?'
    r'|drugs?|medications?|treatments?|therapies?|interventions?|factors?|mechanisms?|signs?|symptoms?)\b'
    r'|\b(what are the|list the|name the|enumerate)\b',
    re.IGNORECASE,
)


def _fix_inline_section_headers(md: str) -> str:
    """Ensure # Summary / # Key Evidence / # Reasoning headers always start on their own line.

    The LLM occasionally appends a section header inline at the end of the
    preceding sentence without a line break, e.g.:
        "...compared with controls [PMC123:Discussion]. # Key Evidence"
    This breaks all downstream section-span parsers.  Insert \\n\\n before any
    such inline header so it lands at the start of a fresh line.

    Also handles the case where only a single newline precedes the header
    (\\n# Header) and upgrades it to a blank-line separator (\\n\\n# Header).
    """
    # Pass 1: inline headers — no newline at all before #
    md = re.sub(
        r'(?<!\n)(#\s+(?:Summary|Key Evidence|Reasoning)\b)',
        r'\n\n\1',
        md,
    )
    # Pass 2: single-newline headers — exactly one \n before #, not a blank line
    md = re.sub(
        r'(?<!\n)\n(#\s+(?:Summary|Key Evidence|Reasoning)\b)',
        r'\n\n\1',
        md,
    )
    return md


# Transition phrases that are wrong as the very first sentence of a Summary
# because they imply prior context that does not exist yet.
_TRANSITION_OPENER_RE = re.compile(
    r'^(?:'
    r'As a (?:consequence|result)[,\s]+'
    r'|Consequently[,\s]+'
    r'|Therefore[,\s]+'
    r'|Thus[,\s]+'
    r'|Taken together[,\s]+'
    r'|Together[,\s]+these\s+\w+'
    r'|These (?:findings|data|results) (?:therefore |collectively |together )?'
        r'(?:indicate|suggest|demonstrate|confirm|show)\b'
    r'|In conclusion[,\s]+'
    r'|In summary[,\s]+'
    r'|Notably[,\s]+'
    r'|Importantly[,\s]+'
    r'|Strikingly[,\s]+'
    r'|Remarkably[,\s]+'
    r'|Interestingly[,\s]+'
    r')',
    re.IGNORECASE,
)


def _strip_summary_transition_opener(md: str) -> str:
    """Strip a leading transition/conclusion phrase from the FIRST sentence of
    the Summary when it makes no sense as an opener.

    Patterns caught: "As a consequence, ...", "Taken together, ...",
    "These findings suggest ...", "In conclusion, ...", etc.
    The transition prefix is removed and the remaining text is capitalised.
    Non-matching openers are left untouched.
    """
    m = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
    if not m:
        return md
    body = m.group(1)
    # Split at the first sentence boundary
    sent_m = re.match(r'^(.*?[.!?])(\s+.*)', body, re.S)
    if not sent_m:
        return md
    first_sent, rest = sent_m.group(1), sent_m.group(2)
    stripped = _TRANSITION_OPENER_RE.sub('', first_sent.lstrip())
    if stripped == first_sent.lstrip():
        return md  # no transition opener found — unchanged
    # If stripping the transition prefix exposes a dangling demonstrative reference
    # ("this X" or "these X" with no introduced antecedent at position 0), drop the
    # ENTIRE first sentence — it's a forward-reference conclusion misplaced as opener.
    # Example: "Notably, this protective effect extends..." → drop whole sentence.
    if re.match(r'^[Tt]his\s+|^[Tt]hese\s+', stripped):
        new_body = rest.lstrip()
        if new_body:
            new_body = new_body[0].upper() + new_body[1:]
        new_section = "# Summary\n" + new_body
        return md[: m.start()] + new_section + md[m.end():]
    # Regular case: capitalise and substitute
    stripped = stripped[0].upper() + stripped[1:] if stripped else stripped
    new_body = stripped + rest
    new_section = "# Summary\n" + new_body
    return md[: m.start()] + new_section + md[m.end():]


def _flatten_summary_bullets(md: str) -> str:
    """Convert any bullet-marker lines inside the Summary section into prose.

    The Summary must always be a single flowing paragraph. If the LLM or any
    reconstruction step produced lines starting with -, •, or *, strip the
    marker and join everything into one paragraph.
    """
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    lines = body.splitlines()
    prose_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers at start of line
        line = re.sub(r"^[-•\*]\s+", "", line)
        if line:
            prose_parts.append(line)

    # Join into a single paragraph, ensuring sentences end with a period
    sentences = []
    for part in prose_parts:
        # Each part may itself be a run-on; split on existing sentence boundaries
        for s in re.split(r"(?<=[.!?])\s+", part):
            s = s.strip()
            if s:
                sentences.append(s)

    # Deduplicate consecutive identical sentences
    deduped = []
    seen_norms: set = set()
    for s in sentences:
        norm = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', s)).strip().lower()
        if norm and norm not in seen_norms:
            seen_norms.add(norm)
            deduped.append(s)

    new_body = " ".join(deduped)
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]


def _build_polarity_summary_from_ke(md: str, question: str) -> str:
    """When the polarity filter shrinks the Summary too much, rebuild it by extracting
    only the on-topic sentences directly from Key Evidence bullet text.

    For a lower-risk question: gather sentences that contain lower-risk language.
    For a higher-risk question: gather sentences that contain higher-risk language.
    Assembles a clean prose paragraph with citations.

    NOTE: Skipped for enumeration/list questions — they use _reconstruct_list_summary
    which already produces a comprehensive, citation-backed enumeration.
    """
    # Enumeration questions should not be further filtered by polarity
    if _LIST_Q_PAT.search(question):
        return md
    _lower_q = re.compile(
        r'\b(lower risk|less likely|protective|reduced risk|lower(ed)? (rates?|prevalence)|'
        r'less (common|frequent|prevalent)|decreased risk)\b', re.IGNORECASE)
    _higher_q = re.compile(
        r'\b(higher risk|elevated risk|increased risk|more likely|greater risk|'
        r'more (common|frequent|prevalent))\b', re.IGNORECASE)

    q_lower  = bool(_lower_q.search(question))
    q_higher = bool(_higher_q.search(question))
    if not (q_lower ^ q_higher):
        return md  # ambiguous — leave as-is

    # Target polarity: sentences to KEEP
    if q_lower:
        _target = re.compile(
            r'\b(lower risk|lower rates?|less prevalent|less common|less frequent|'
            r'reduced risk|decreased risk|lower(ed)? prevalence|at lower risk|'
            r'less likely|protective|lower in (adults|individuals|people)|'
            r'lower than|less than)\b', re.IGNORECASE)
    else:
        _target = re.compile(
            r'\b(higher risk|elevated risk|increased risk|at (high|higher|elevated) risk|'
            r'more prevalent|more common|more frequent|greater risk)\b', re.IGNORECASE)

    ke_m = re.search(r"(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if not ke_m:
        return md

    ke_body = ke_m.group(1).strip()
    bullet_parts = re.split(r"\n(?=[-•\*]\s)", "\n" + ke_body)
    bullets = [re.sub(r"^[-•\*]\s+", "", p.strip()) for p in bullet_parts if p.strip()]

    _url_re       = re.compile(r"(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\(https://[^)]+\)")
    _num_cite_re  = re.compile(r"\[\d+(?:[,\-–]\s*\d+)*\]")
    _frag_re      = re.compile(r"^([a-z][,\s]|\d|Fig|Table|Suppl|Extended|S\d)", re.I)
    _firstperson  = re.compile(r"^\s*(We|we|Our|our|Here\s+we|I\s+)\b")
    _supp_re      = re.compile(
        r"\s*\((?:Results?|Suppl(?:emental)?|Extended Data|Fig|Table)[\s\.][^)]{0,40}\)",
        re.IGNORECASE)

    on_topic_sents = []
    seen_norms: set = set()

    for b in bullets:
        b = b.strip().replace("\n", " ")
        b = _url_re.sub(r"\1", b)
        b = _num_cite_re.sub("", b)
        b = _supp_re.sub("", b)

        # Collect ALL [PMCxxxxxx:Section] tags in the bullet (may be mid-text or at end)
        all_tags = re.findall(r"\[PMC\d{4,}:[A-Za-z0-9_-]+\]", b)
        pmc_tag  = " ".join(all_tags) if all_tags else ""
        # Remove tags from body so sentence splitting isn't confused
        body_text = re.sub(r"\[PMC\d{4,}:[A-Za-z0-9_-]+\]", "", b).strip()

        sents = re.split(r"(?<=[.!?])\s+", body_text)
        for s in sents:
            s = s.strip()
            if len(s.split()) < 8:
                continue
            if _frag_re.match(s):
                continue
            if _firstperson.match(s):
                continue
            if not _target.search(s):
                continue
            # Truncate at contrast clause where polarity flips
            # e.g. "hypertension is lower; however some are higher" → keep only first part
            flip = re.search(
                r'[;,]?\s*\b(however[,\s]|;\s*(while|but|yet)\s+(some|others?|certain)'
                r'|,?\s*(while|but|yet)\s+(some|others?|certain)'
                r'|while being at (high|higher)|such as Alzheimer'
                r'|at (high|higher|elevated) risk of others'
                r'|including hyperlipidemia|including .*(high|higher) rates)\b.*',
                s, re.IGNORECASE | re.DOTALL)
            if flip:
                s = s[:flip.start()].strip()
                if len(s.split()) < 6:
                    continue
            if s[-1] not in ".!?":
                s += "."
            norm = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', s)).strip().lower()
            if norm in seen_norms:
                continue
            seen_norms.add(norm)
            on_topic_sents.append(s + (" " + pmc_tag if pmc_tag else ""))

    if not on_topic_sents:
        return md  # Nothing found — leave as-is

    # Build a clean paragraph: opener + findings + closer
    if q_lower:
        opener = "Available evidence indicates that individuals with Down syndrome show a reduced prevalence of several medical conditions compared to the general population and other intellectual developmental disabilities."
    else:
        opener = "Available evidence indicates that individuals with Down syndrome show an elevated prevalence of several medical conditions compared to the general population."

    closer_norm = re.sub(r'\s+', ' ', re.sub(r'\[PMC[^\]]+\]', '', opener)).strip().lower()
    seen_norms.add(closer_norm)

    parts = [opener] + on_topic_sents[:5]
    new_body = " ".join(parts)

    summ_m = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if not summ_m:
        return md
    new_section = "# Summary\n" + new_body + "\n"
    return md[: summ_m.start()] + new_section + md[summ_m.end():]


def _strip_uncited_summary_sentences(md: str) -> str:
    """Remove finding sentences (not the first or last sentence) in the Summary
    body that carry no [PMC:Section] citation.

    The first sentence (context) and the last sentence (conclusion) are
    intentionally citation-free per the C-C-C schema, so they are kept
    unconditionally.  All middle sentences without a citation are dropped so
    that `_pad_summary_from_ke_final` can replace them with cited KE prose.
    """
    summ_m = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if not summ_m:
        return md
    body = summ_m.group(1).strip()
    # Split on sentence boundaries; keep markdown bullets intact as single units
    sentences = re.split(r'(?<=[.!?])\s+', body)
    if len(sentences) <= 2:
        return md  # Only context + conclusion — nothing to strip
    _cite_pat = re.compile(r'\[PMC\d+:[^\]]+\]')
    kept = [sentences[0]]  # always keep context (first)
    stripped_count = 0
    for s in sentences[1:-1]:
        if _cite_pat.search(s) or s.startswith('-'):
            kept.append(s)
        else:
            stripped_count += 1
    kept.append(sentences[-1])  # always keep conclusion (last)
    # Guard: if stripping would leave fewer than 60 words, skip — _pad_summary_from_ke_final
    # will rebuild from Key Evidence instead, which is better than stripping a useful sentence.
    if stripped_count > 0:
        _prospective_wc = len([w for w in re.sub(r'\[PMC[^\]]+\](?:\([^)]+\))?', '', ' '.join(kept)).split() if w])
        if _prospective_wc < 60:
            return md
    if stripped_count == 0:
        return md
    new_body = ' '.join(kept)
    new_section = '# Summary\n' + new_body + '\n'
    print(f"  [FINALIZE] _strip_uncited_summary_sentences: removed {stripped_count} uncited sentence(s)")
    return md[: summ_m.start()] + new_section + md[summ_m.end():]


def _pad_summary_from_ke_final(md: str, min_wc: int = 100, target_wc: int = 130) -> str:
    """Last-resort floor enforcement: append KE bullet prose to the Summary when
    the word count after all other post-processing is still below `min_wc`.

    Called at the very end of finalize so that deduplication and stripping steps
    cannot reduce the summary below the minimum.  The conclusion sentence (last
    sentence of the existing summary) is moved to the end so the flow is preserved.
    Pads up to `target_wc` words then lets `_truncate_summary_to_150_words` cap.
    """
    summ_m = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    ke_m = re.search(r"(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if not summ_m or not ke_m:
        return md

    summ_body = summ_m.group(1).strip()
    _cite_re = re.compile(r'\[PMC[^\]]+\](?:\([^)]*\))?')

    def _wc(text: str) -> int:
        return len([w for w in _cite_re.sub('', text).split() if w])

    if _wc(summ_body) >= min_wc:
        return md  # already long enough

    ke_body = ke_m.group(1).strip()

    # Parse KE bullets → first valid sentence of each
    _url_re = re.compile(r'(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\(https://[^)]+\)')
    _num_re = re.compile(r'\[\d+(?:[,\-\u2013]\s*\d+)*\]')
    _frag_re = re.compile(r'^([a-z][,\s]|\d|Fig|Table|Suppl|Extended|S\d)', re.I)
    _fp_re = re.compile(r'^\s*(We|we|Our|our|Here\s+we|I\s+)\b')

    ke_sents: list[str] = []
    for raw_bullet in re.split(r'\n(?=[•\-\*]\s)', '\n' + ke_body):
        raw_bullet = raw_bullet.strip()
        if not raw_bullet:
            continue
        b = re.sub(r'^[•\-\*]\s+', '', raw_bullet).replace('\n', ' ')
        b = _url_re.sub(r'\1', b)
        b = _num_re.sub('', b)
        # Extract trailing tag
        tag_m = re.search(r'(\[PMC\d{4,}:[A-Za-z0-9_-]+\])\s*$', b)
        pmc_tag = tag_m.group(1) if tag_m else ''
        body_text = b[: tag_m.start()].strip() if tag_m else b
        for s in re.split(r'(?<=[.!?])\s+', body_text):
            s = s.strip()
            if len(s.split()) < 10 or _frag_re.match(s) or _fp_re.match(s):
                continue
            if s[-1] not in '.!?':
                s += '.'
            if pmc_tag:
                s = s + ' ' + pmc_tag
            ke_sents.append(s)
            break  # only first valid sentence per bullet

    if not ke_sents:
        return md

    # Build padded body (preserve conclusion at end)
    existing = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summ_body) if s.strip()]
    conclusion = existing[-1] if len(existing) > 1 else ''
    body_parts = existing[:-1] if conclusion else existing

    # Near-duplicate detection: compare content words (gene symbols with digits and
    # DS/T21 boilerplate excluded) so "...in individuals with Down syndrome (DS)..." and
    # "...in individuals with DS..." are recognised as the SAME sentence and not both kept.
    _pad_stop = {"individuals", "patients", "people", "syndrome", "down", "ds", "t21",
                 "trisomy", "with", "the", "and", "has", "been", "associated", "in", "of"}
    def _pad_cwords(text: str) -> set:
        text = _cite_re.sub('', text)
        return {w for w in re.findall(r'\b[a-z]{4,}\b', text.lower()) if w not in _pad_stop}
    def _pad_dup(cw: set, others: list) -> bool:
        if not cw:
            return False
        for ow in others:
            if ow and len(cw & ow) / len(cw | ow) > 0.6:
                return True
        return False

    seen_cw: list[set] = [_pad_cwords(s) for s in body_parts]

    padded = list(body_parts)
    for ks in ke_sents:
        cur_wc = _wc(' '.join(padded))
        if cur_wc >= target_wc:
            break
        # Skip sentences that would overflow target_wc — truncation would cut
        # them mid-phrase producing dangling fragments like "...of only 88 of."
        if _wc(' '.join(padded + [ks])) > target_wc:
            continue
        cw = _pad_cwords(ks)
        if cw and not _pad_dup(cw, seen_cw):
            seen_cw.append(cw)
            padded.append(ks)

    if conclusion:
        if not _pad_dup(_pad_cwords(conclusion), seen_cw):
            padded.append(conclusion)

    if not padded:
        return md

    new_body = ' '.join(padded)
    new_section = '# Summary\n' + new_body + '\n'
    return md[: summ_m.start()] + new_section + md[summ_m.end():]


# ------------------------------------------------------------
# Evidence-chunk pre-processor: strips journal call-out header boilerplate
# (e.g. "RESEARCH IN CONTEXT\nSystematic review: ...") from evidence text
# BEFORE it is passed to the LLM, so the LLM never copies these labels verbatim.
# ------------------------------------------------------------
_BOILERPLATE_LABEL_RE = re.compile(
    r"""
    # "RESEARCH IN CONTEXT" call-out header (with or without trailing newline/colon)
    RESEARCH\s+IN\s+CONTEXT\s*[:\n\r]*\s*
    # Journal call-out section labels used as prefixes (keep content after colon)
    |(?:Systematic\s+review
       |Expert\s+opinion
       |Literature\s+review
       |Added\s+value\s+of\s+this\s+study
       |Evidence\s+before\s+this\s+study
       |Implications\s+of\s+all\s+the\s+available\s+evidence
       |What\s+is\s+(?:new|known|already\s+known)
    ):\s+
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Sentences that are pure database/search-methods narration or first-person verbatim
# from paper body text — no useful science for synthesis.
_METHODS_SENTENCE_RE = re.compile(
    r"""^\s*(?:Literature|[Ss]tudies?)\s+was\s+reviewed\s+using\s
       |^\s*[Dd]atabases?\s+(?:searched|used|consulted)\s+included
       |^\s*[Aa]\s+(?:systematic|structured|narrative)\s+(?:literature\s+)?search\s+was\s+(?:conducted|performed)\b
       |\bour\s+(?:results|data|findings|analyses)\s+suggest\b
       |\btaken\s+together,?\s+our\b
       |\bcould\s+be\s+especially\s+valuable\b
       |\btherapeutic\s+intervention\s+is\s+most\s+likely\s+to\s+be\s+viable\b
       # OR administrative/claims-data methods narration
       |\bChronic\s+Condition\s+Warehouse\b
       |\balgorithms?\s+for\s+Medicare\s+data\b
       |\bidentified\s+from\s+(?:the\s+)?(?:ICD|Medicare|Medicaid|claims)\b
       # OR first-person observation/results narration
       |\bwe\s+observed\b
       |\bwe\s+(?:found|show|report|demonstrate|note|identify|identified)\b
       |\bdespite\s+substantial\s+research\s+efforts\b
       |\bthe\s+situation\s+is\s+even\s+more\s+complex\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _clean_evidence_chunk(text: str) -> str:
    """Strip journal boilerplate labels/headers from an evidence chunk text
    before passing it to the synthesis LLM, so the LLM cannot copy them
    verbatim into the output.

    Removes labels (keeping content after the colon) for:
      - "RESEARCH IN CONTEXT Systematic review: <content>" → "<content>"
      - "Expert opinion: <content>"                        → "<content>"
    Removes entire sentences that are pure database-search narration when
    they appear as standalone sentences.
    """
    # Strip boilerplate labels (keep the content that follows)
    text = _BOILERPLATE_LABEL_RE.sub("", text)
    # Drop standalone pure-methods sentences
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Drop the line if it IS a pure-methods sentence (matched at start)
        if _METHODS_SENTENCE_RE.search(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Collapse runs of whitespace introduced by removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Patterns that mark a sentence as raw experimental/methods data that should
# never appear in a synthesized Summary.
_RAW_DATA_RE = re.compile(
    r"""
    # 3 or more sample-size annotations in one sentence → methods table
    (?:(?:n\s*=\s*\d+.*?){3})
    # OR sex-breakdown notation
    |\b\d+\s+male\s+and\s+\d+\s+female\b
    # OR explicit group counts header
    |Number\s+of\s+(?:animals|participants|subjects|patients|mice|rats|samples)
    # OR wildtype/strain group size descriptions
    |(?:unstimulated|stimulated)\s+WT\s*\(n\s*=
    # OR specific genotype labels (mouse model names like Dp162xIfnrs)
    |Dp\d{2,}x[A-Za-z]
    # OR sentence is a pure experimental arm roster
    |(?:\(n\s*=\s*\d+,\s*\d+\s+male){2,}
    # OR first-person methods/intro narration (verbatim from paper Introduction/Methods)
    |\bwe\s+(?:sought|aimed|wanted|set\s+out|designed|conducted|performed|next\s+sought)\s+to\b
    |\b(?:in\s+this\s+study[,.]|here[,.]?\s+we)\b
    |\bwe\s+(?:used|applied|employed)\s+\w+(?:\s+\w+)?\s+to\s+(?:investigate|examine|test|assess|determine|evaluate|identify)\b
    |\bwe\s+employed\b
    # OR self-referential study description
    |\bthe\s+(?:current|present|this)\s+study\s+(?:aimed|sought|was\s+designed)\b
    # OR verbatim statistical reporting notation copied from paper Results
    |\bRR\s*=\s*[\d.]+
    |\bOR\s*=\s*[\d.]+
    |\bHR\s*=\s*[\d.]+
    |\b95\s*%\s*(?:CI|confidence\s+interval)\b
    |\bp\s*[<>=≤≥]\s*0\.\d{2,4}\b
    # OR journal call-out box headers copied verbatim
    |RESEARCH\s+IN\s+CONTEXT
    |\bLiterature\s+was\s+reviewed\s+using\b
    |\bDatabases?\s+(?:searched|used|consulted)\s+included\b
    # OR first-person "our results / our data / our findings" verbatim from paper text
    |\bour\s+(?:results|data|findings|analyses|study|work|approach)\s+suggest\b
    |\bour\s+(?:results|data|findings|analyses)\s+(?:demonstrate|indicate|show|reveal|support)\b
    |\btaken\s+together,?\s+our\b
    # OR "could be especially valuable" / "is most likely to be viable" promotional filler
    |\bcould\s+be\s+especially\s+valuable\b
    |\bis\s+most\s+likely\s+to\s+be\s+viable\b
    |\btherapeutic\s+intervention\s+is\s+most\s+likely\b
    # OR administrative/claims-data methods narration
    |\bChronic\s+Condition\s+Warehouse\b
    |\balgorithms?\s+for\s+Medicare\s+data\b
    |\bidentified\s+from\s+(?:the\s+)?(?:ICD|Medicare|Medicaid|claims)\b
    # OR first-person observation/results narration copied from paper text
    |\bwe\s+observed\b
    |\bwe\s+(?:found|show|report|demonstrate|note|identify|identified)\b
    # OR first-person methods/results verbs not yet covered above
    |\bwe\s+(?:examined|investigated|analyzed|analysed|characterized|characterised|assessed|evaluated|measured|tested|quantified|profiled|defined|compared|stratified|enrolled|recruited|collected|generated|derived|developed|constructed|utilized|utilised|employed|performed|completed|conducted|undertook|carried\s+out)\b
    |\bdespite\s+substantial\s+research\s+efforts\b
    |\bthe\s+situation\s+is\s+even\s+more\s+complex\b
    # OR verbatim paper-internal cross-references (figure/table/section labels)
    |\bExtended\s+Data\s+(?:Fig|Figs?\.?|Table)\b
    |\(Methods(?:,|\s)
    # OR model/subject selection justification (verbatim from Methods)
    |\bwere\s+selected\s+because\b
    # OR verbatim ordered-prevalence/frequency roster (e.g. "in order of prevalence were: X, Y, Z")
    |\bin\s+order\s+of\s+(?:prevalence|frequency|decreasing|increasing)\b
    # OR truncated comparison — sentence ends with "vs." (second value cut off)
    |\bvs\.\s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


# Subset of _RAW_DATA_RE for Key Evidence bullet cleaning.
# Excludes statistical notation (OR, HR, RR, CI, p-value) which IS legitimate in KE
# but includes all verbatim-methods markers (animal counts, rosters, first-person).
_RAW_DATA_KE_RE = re.compile(
    r"""
    # 3 or more sample-size annotations in one sentence → methods table
    (?:(?:n\s*=\s*\d+.*?){3})
    # OR sex-breakdown notation
    |\b\d+\s+male\s+and\s+\d+\s+female\b
    # OR explicit group counts header
    |Number\s+of\s+(?:animals|participants|subjects|patients|mice|rats|samples)
    # OR wildtype/strain group size descriptions
    |(?:unstimulated|stimulated)\s+WT\s*\(n\s*=
    # OR specific genotype labels (mouse model names like Dp162xIfnrs)
    |Dp\d{2,}x[A-Za-z]
    # OR sentence is a pure experimental arm roster
    |(?:\(n\s*=\s*\d+,\s*\d+\s+male){2,}
    # OR first-person methods/intro narration
    |\bwe\s+(?:sought|aimed|wanted|set\s+out|designed|conducted|performed|next\s+sought)\s+to\b
    |\b(?:in\s+this\s+study[,.]|here[,.]?\s+we)\b
    |\bwe\s+(?:used|applied|employed)\s+\w+(?:\s+\w+)?\s+to\s+(?:investigate|examine|test|assess|determine|evaluate|identify)\b
    |\bwe\s+employed\b
    # OR self-referential study description
    |\bthe\s+(?:current|present|this)\s+study\s+(?:aimed|sought|was\s+designed)\b
    # OR journal call-out box headers copied verbatim
    |RESEARCH\s+IN\s+CONTEXT
    |\bLiterature\s+was\s+reviewed\s+using\b
    |\bDatabases?\s+(?:searched|used|consulted)\s+included\b
    # OR first-person "our results / our data" verbatim from paper text
    |\bour\s+(?:results|data|findings|analyses|study|work|approach)\s+suggest\b
    |\bour\s+(?:results|data|findings|analyses)\s+(?:demonstrate|indicate|show|reveal|support)\b
    |\btaken\s+together,?\s+our\b
    # OR promotional filler copied from paper Discussion
    |\bcould\s+be\s+especially\s+valuable\b
    |\bis\s+most\s+likely\s+to\s+be\s+viable\b
    |\btherapeutic\s+intervention\s+is\s+most\s+likely\b
    # OR administrative/claims-data methods narration
    |\bChronic\s+Condition\s+Warehouse\b
    |\balgorithms?\s+for\s+Medicare\s+data\b
    |\bidentified\s+from\s+(?:the\s+)?(?:ICD|Medicare|Medicaid|claims)\b
    # OR first-person observation/results narration copied from paper text
    |\bwe\s+observed\b
    |\bwe\s+(?:found|show|report|demonstrate|note|identify|identified)\b
    # OR first-person methods/results verbs not yet covered above
    |\bwe\s+(?:examined|investigated|analyzed|analysed|characterized|characterised|assessed|evaluated|measured|tested|quantified|profiled|defined|compared|stratified|enrolled|recruited|collected|generated|derived|developed|constructed|utilized|utilised|employed|performed|completed|conducted|undertook|carried\s+out)\b
    |\bdespite\s+substantial\s+research\s+efforts\b
    |\bthe\s+situation\s+is\s+even\s+more\s+complex\b
    # OR verbatim paper-internal cross-references (figure/table/section labels)
    |\bExtended\s+Data\s+(?:Fig|Figs?\.?|Table)\b
    |\(Methods(?:,|\s)
    # OR model/subject selection justification (verbatim from Methods)
    |\bwere\s+selected\s+because\b
    # OR verbatim ordered-prevalence/frequency roster (e.g. "in order of prevalence were: X, Y, Z")
    |\bin\s+order\s+of\s+(?:prevalence|frequency|decreasing|increasing)\b
    # OR truncated comparison — sentence ends with "vs." (second value cut off)
    |\bvs\.\s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


_LOWER_RISK_Q_RE = re.compile(
    r'\b(lower risk|less likely|protective|not at(?: higher)? risk|reduced risk|'
    r'lower(?:ed)? (?:rates?|prevalence)|less (?:common|frequent|prevalent)|'
    r'decreased risk|do not develop|rarely develop)\b',
    re.IGNORECASE,
)
_HIGHER_RISK_Q_RE = re.compile(
    r'\b(higher risk|elevated risk|increased risk|more likely|greater risk|'
    r'more (?:common|frequent|prevalent)|over-represented)\b',
    re.IGNORECASE,
)

# Patterns that identify a sentence as asserting higher risk
_HIGHER_RISK_SENT_RE = re.compile(
    r'\b(?:at|associated with|face(?:s)?|have|carry)\s+(?:a\s+)?(?:higher|elevated|increased|greater)\s+risk\b'
    r'|\bhigher risk of\b'
    r'|\belevated risk for\b'
    r'|\bincreased incidence of\b'
    r'|\bmore (?:commonly|frequently|often) (?:experienced|observed|affected|present|reported)\b'
    r'|\bover-represented\b'
    r'|\bdisproportionately high(?:er)?\b'
    # "at high risk of" / "being at high risk" — "high" (not just "higher") risk
    r'|\bat high risk (?:for|of)\b'
    r'|\bbeing at (?:high|elevated|increased) risk\b'
    # "present at similar or higher rates" — uses "rates" not "risk"
    r'|\bat (?:similar or )?higher rates\b'
    r'|\bsome (?:\w+\s+){0,5}(?:are|were) (?:present|found|observed) at (?:similar or )?higher rates\b',
    re.IGNORECASE,
)

# Patterns that identify a sentence as asserting lower risk
_LOWER_RISK_SENT_RE = re.compile(
    r'\b(?:lower|reduced|decreased|protective)\s+(?:rates?|risk|prevalence|incidence)\b'
    r'|\blower risk of\b'
    r'|\breduced risk for\b'
    r'|\bless (?:commonly|frequently|often) (?:experienced|observed|affected|present|reported)\b'
    r'|\brarely develop\b'
    r'|\bnot at (?:higher\s+)?risk\b',
    re.IGNORECASE,
)

# ICU/ventilation sentence detector — used to strip acute-care severity metrics
# from Summaries that answer a comorbidity/conditions question.
_ICU_VENTILATION_SENT_RE = re.compile(
    r'\b(ICU admission|mechanical ventilation|ventilat(?:ion|ed)|intubat(?:ion|ed)|'
    r'hospital(?:is|iz)ation rates?|hospital stay|length of stay)\b',
    re.IGNORECASE,
)

# Matches questions that ask about conditions / comorbidities
_COMORBIDITY_Q_RE = re.compile(
    r'\b(conditions?|comorbid(?:ities|ity)?|diseases?|higher risk|elevated risk|'
    r'at risk for|risk factors?|complications?)\b',
    re.IGNORECASE,
)

# Comma-list "at risk for A, B, C" pattern — always wrong in a synthesized Summary
# because each named condition should have its own inline citation immediately
# after it (not a bare comma).  Matches when ≥2 consecutive uncited items follow
# "at elevated/higher/increased risk for".
_COMMALIST_SUMMARY_RE = re.compile(
    r'at (?:elevated|higher|increased) risk for\s+'
    r'(?:[A-Z][^.!?\[\n]*,\s*){2,}',
    re.IGNORECASE,
)


def _strip_opposite_polarity_sentences(md: str, question: str) -> str:
    """Remove Summary middle sentences that assert the OPPOSITE polarity to the question.

    - Lower-risk question → drop sentences asserting elevated/higher risk for specific conditions.
    - Higher-risk question → drop sentences asserting reduced/lower risk for specific conditions.
    First and last sentences (context and conclusion) are always preserved.
    Only applies when the question has an unambiguous polarity.
    """
    is_lower_q = bool(_LOWER_RISK_Q_RE.search(question)) and not bool(_HIGHER_RISK_Q_RE.search(question))
    is_higher_q = bool(_HIGHER_RISK_Q_RE.search(question)) and not bool(_LOWER_RISK_Q_RE.search(question))
    if not is_lower_q and not is_higher_q:
        return md  # ambiguous or neutral question — no stripping

    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    sentences = re.split(r'(?<=[\.\!\?])\s+', body)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return md  # only context + conclusion — nothing to strip

    kept = [sentences[0]]  # always keep context sentence
    for s in sentences[1:-1]:
        if is_lower_q and _HIGHER_RISK_SENT_RE.search(s):
            continue  # drop — talks about higher risk, but question asks about lower
        if is_higher_q and _LOWER_RISK_SENT_RE.search(s):
            continue  # drop — talks about lower risk, but question asks about higher
        kept.append(s)

    # Last sentence (conclusion): apply polarity check too.
    # If the conclusion itself asserts the OPPOSITE polarity AND there are already
    # at least 2 kept sentences (context + ≥1 middle finding), drop it so that the
    # last retained finding serves as the de-facto conclusion.
    # If stripping would leave only 1 sentence (the context), keep the conclusion
    # regardless so the Summary has at least 2 sentences.
    last_sent = sentences[-1]
    if is_lower_q and _HIGHER_RISK_SENT_RE.search(last_sent) and len(kept) >= 2:
        pass  # drop opposite-polarity conclusion — last kept sentence becomes conclusion
    elif is_higher_q and _LOWER_RISK_SENT_RE.search(last_sent) and len(kept) >= 2:
        pass  # drop opposite-polarity conclusion
    else:
        kept.append(last_sent)

    if len(kept) == len(sentences):
        return md  # nothing changed

    new_body = " ".join(kept).strip()
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]


def _strip_verbatim_bracket_sentences(md: str) -> str:
    """Strip verbatim evidence sentences the LLM wraps in square brackets.

    The LLM sometimes duplicates a synthesized finding as a verbatim quote wrapped
    in brackets immediately after the real sentence:
      "...is overexpressed [PMC:Discussion]. [In mouse models, Stat protein
       overexpression was observed...] [PMC:Discussion]."

    Pattern: [non-citation text ≥15 chars ending in .!?] optionally followed by
    one or more [PMC:Section] citations.
    Only applied inside the Summary section.
    """
    # Match: [long non-PMC/non-URL text ending in sentence punctuation]
    # optionally followed by [PMC:Section] citation(s)
    _VERBATIM_BRACKET_RE = re.compile(
        r'\s*\[(?!PMC\d|https?://|#|EVIDENCE_GAP|INCLUDE)[^[\]]{15,}[.!?]\]'
        r'(?:\s*\[PMC\d+:[^\]]+\]\.?)*',
        re.IGNORECASE,
    )
    match = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1)
    new_body = _VERBATIM_BRACKET_RE.sub('', body)
    if new_body != body:
        print('  [FINALIZE] _strip_verbatim_bracket_sentences: removed verbatim bracket quote(s)')
        md = md[: match.start()] + '# Summary\n' + new_body + md[match.end():]
    return md


def _strip_icu_for_comorbidity_q(md: str, question: str) -> str:
    """Strip ICU/ventilation sentences from Summary for comorbidity/conditions questions.

    ICU admission rates and mechanical ventilation rates are acute-care severity metrics,
    not baseline comorbid conditions.  When the question asks which conditions people
    with DS are at higher/elevated risk for, these sentences are off-topic and are stripped
    here so the Summary focuses on named chronic diagnoses.

    The first (context) sentence is always preserved.  Stripping is skipped entirely
    if it would leave fewer than two sentences.
    """
    if not _COMORBIDITY_Q_RE.search(question):
        return md  # not a comorbidity question — no action
    match = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', body)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return md  # nothing to strip safely
    kept = [sentences[0]]  # always keep context sentence
    stripped_count = 0
    for s in sentences[1:]:
        if _ICU_VENTILATION_SENT_RE.search(s):
            stripped_count += 1
        else:
            kept.append(s)
    if stripped_count == 0 or len(kept) < 2:
        return md  # nothing stripped or would leave too few sentences
    print(f'  [FINALIZE] _strip_icu_for_comorbidity_q: removed {stripped_count} ICU/ventilation sentence(s)')
    new_body = ' '.join(kept).strip()
    md = md[: match.start()] + '# Summary\n' + new_body + '\n' + md[match.end():]
    return md


def _strip_comma_list_sentences(md: str) -> str:
    """Strip Summary sentences containing an uncited comma-list 'at risk for A, B, C' pattern.

    The LLM sometimes generates: 'Individuals with DS are at elevated risk for
    Condition A, Condition B, Condition C and Condition D remains a major burden...'
    — a garbled merge of a comma-list opener and a second clause with no per-item
    citations.  This function removes any sentence (not the first/context sentence)
    matching the pattern, regardless of total sentence count.

    Works as a last-resort after _strip_uncited_summary_sentences, _reconstruct_list_summary,
    and _pad_summary_from_ke_final to catch any residual garbled sentences.
    """
    match = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', body)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return md
    kept = [sentences[0]]  # always keep first (context) sentence
    stripped_count = 0
    for s in sentences[1:]:
        if _COMMALIST_SUMMARY_RE.search(s):
            stripped_count += 1
        else:
            kept.append(s)
    if stripped_count == 0 or len(kept) < 2:
        return md  # nothing stripped or would leave only 1 sentence
    print(f'  [FINALIZE] _strip_comma_list_sentences: removed {stripped_count} comma-list sentence(s)')
    new_body = ' '.join(kept).strip()
    md = md[: match.start()] + '# Summary\n' + new_body + '\n' + md[match.end():]
    return md


# Matches ANY risk-intro phrase.  Sentences that contain this AND have no
# [PMC:Section] citation are stripped as garbled/uncited risk summaries.
# Handles all failure modes:
#   "are at elevated risk for Among individuals..."  (fragment merger)
#   "face elevated risk for in infants with DS..."  (fragment merger)
#   "face elevated risk for respiratory and Alzheimer's disease..."  (uncited list)
#   "are at elevated risk for Immune disorders and ASD."  (uncited list, "and" not commas)
_BROKEN_RISK_INTRO_RE = re.compile(
    r'\b(?:are at (?:elevated|higher|increased|greater) risk for|'
    r'at (?:elevated|higher|increased) risk for|'
    r'face[sd]? (?:elevated|higher|increased) risk for|'
    r'experience[sd]? (?:elevated|higher|increased) risk for|'
    r'show(?:s|ed|ing)? (?:elevated|higher|increased|greater) risk for)',
    re.IGNORECASE,
)


def _strip_broken_risk_intro_sentences(md: str) -> str:
    """Strip uncited 'at/face elevated risk for ...' sentences from the Summary.

    Rule: any non-first Summary sentence that contains a risk-intro phrase AND
    has no [PMCxxxxx:Section] citation is stripped.  Sentences with citations are
    kept regardless — they contain cited evidence and are valid.

    Catches all failure modes:
    - Fragment merger: "are at elevated risk for Among individuals with DS..."
    - Uncited comma list: "face elevated risk for respiratory and Alzheimer's..."
    - Uncited "and" list: "are at elevated risk for Immune disorders and ASD."
    """
    match = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', body)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return md
    kept = [sentences[0]]
    stripped_count = 0
    for s in sentences[1:]:
        if _BROKEN_RISK_INTRO_RE.search(s) and not re.search(r'\[PMC\d{4,}:', s):
            stripped_count += 1
        else:
            kept.append(s)
    if stripped_count == 0 or len(kept) < 2:
        return md
    print(f'  [FINALIZE] _strip_broken_risk_intro_sentences: removed {stripped_count} uncited risk-intro sentence(s)')
    new_body = ' '.join(kept).strip()
    md = md[: match.start()] + '# Summary\n' + new_body + '\n' + md[match.end():]
    return md


_FURTHER_RESEARCH_RE = re.compile(
    r'\b(?:further|more|additional)\s+(?:research|studies?|investigation|work|data|evidence|models?)'
    r'(?:\s+(?:is|are|will\s+be))?\s+(?:needed|required|necessary|warranted)\b'
    r'|\bhighlighting\s+the\s+need\s+for\s+(?:further|additional|more|new)\b'
    r'|\bneed\s+for\s+(?:further|additional|more|new)\s+(?:models?|studies?|research|data|evidence|tools?|work)\b'
    r'|\bfuture\s+(?:research|studies?|work|investigation)\s+(?:should|will|must|may|are\s+needed)\b'
    r'|\bremains?\s+to\s+be\s+(?:determined|investigated|established|explored|elucidated)\b',
    re.IGNORECASE,
)


def _strip_further_research_sentences(md: str) -> str:
    """Remove Summary sentences containing banned 'further research needed' hedging.

    Banned: 'further research is needed', 'highlighting the need for additional models',
    'remains to be determined', etc.  Preserves the sentence if stripping would leave
    fewer than 2 sentences in the Summary.
    """
    match = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    sentences = re.split(r'(?<=[.!?])\s+', body)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return md
    kept = []
    stripped_count = 0
    for s in sentences:
        if _FURTHER_RESEARCH_RE.search(s):
            stripped_count += 1
        else:
            kept.append(s)
    if stripped_count == 0 or len(kept) < 2:
        return md
    print(f'  [FINALIZE] _strip_further_research_sentences: removed {stripped_count} hedging sentence(s)')
    new_body = ' '.join(kept).strip()
    md = md[: match.start()] + '# Summary\n' + new_body + '\n' + md[match.end():]
    return md


def _strip_raw_data_sentences(md: str) -> str:
    """Remove raw experimental data from the Summary and Key Evidence.

    Summary: strips individual sentences containing raw data.
    Key Evidence: drops entire bullet lines whose text body contains raw data
    (animal-count tables, group rosters, first-person narration).

    Catches verbatim methods sentences the LLM copies from papers:
    - Sample-size tables: 'WT (n = 25, 14 male and 11 female), Dp (n = 15 ...)'
    - Sex-breakdown lines: '16 male and 9 female'
    - Explicit group-count headers: 'Number of animals—unstimulated WT'
    - Mouse genotype rosters: 'Dp162xIfnrs (n = 8, 8 male)'
    - First-person Methods narration ('we sought to delineate...', 'here, we...')

    If stripping removes all Summary sentences, the Summary is left unchanged
    (fallback handled by downstream reconstruction).
    """
    # --- Summary: strip individual matching sentences ---
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if match:
        body = match.group(1).strip()
        # Strip inline boilerplate labels (e.g. "RESEARCH IN CONTEXT Systematic review:")
        # that the LLM sometimes copies from paper PDF call-out boxes directly into
        # the Summary prose.  _BOILERPLATE_LABEL_RE removes the label prefix and
        # keeps the scientific content that follows the colon.
        body = _BOILERPLATE_LABEL_RE.sub("", body).strip()
        # Strip author-year inline citations carried over from paper text.
        # Broader pattern: any parenthetical containing 'et al.' + 4-digit year.
        # This reliably matches multi-author semicolon lists such as
        # (Sullivan et al., 2016; Waugh et al., 2019; ...; Waugh et al., 2023)
        # regardless of whether the last entry ends in ')' or '; year)'.
        body = re.sub(r"\([^)]*\bet\s+al\.[^)]*\d{4}[^)]*\)", "", body)
        # Strip figure/table inline refs copied verbatim from paper text:
        # (Figure 7F; Table S4H), (Fig. 3A), (Supplementary Table 1), etc.
        body = re.sub(
            r"\((?:Figure|Fig\.?|Table|Suppl\.?|Extended\s+Data|Supplementary)\s*[A-Za-z0-9][^)]*\)",
            "", body,
        )
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
        clean = [s for s in sentences if not _RAW_DATA_RE.search(s)]
        if clean and len(clean) < len(sentences):
            new_body = " ".join(clean).strip()
            new_section = "# Summary\n" + new_body + "\n"
            md = md[: match.start()] + new_section + md[match.end():]
        elif body != match.group(1).strip():
            # Boilerplate was stripped but no sentences were removed — still save
            new_section = "# Summary\n" + body + "\n"
            md = md[: match.start()] + new_section + md[match.end():]

    # --- Key Evidence: drop bullets whose body matches raw-data patterns ---
    ke_match = re.search(r"(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if ke_match:
        ke_body = ke_match.group(1)
        new_ke_lines = []
        for line in ke_body.splitlines(keepends=True):
            stripped = line.strip()
            is_bullet = stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*')
            if not is_bullet:
                new_ke_lines.append(line)
                continue
            # Strip trailing PMC citation tag(s) to get pure text body
            bullet_text = re.sub(r'(\[PMC[^\]]+\](?:\([^)]+\))?)+\s*$', '', stripped).strip()
            bullet_text = re.sub(r'^[-•*]\s+', '', bullet_text).strip()
            if _RAW_DATA_KE_RE.search(bullet_text):
                # Verbatim raw data bullet — drop it
                continue
            new_ke_lines.append(line)
        new_ke_body = ''.join(new_ke_lines)
        md = md[: ke_match.start()] + "# Key Evidence\n" + new_ke_body + md[ke_match.end():]

    return md


def _prose_summary(md: str) -> str:
    """Convert any bullet-formatted content inside the Summary into flowing prose.

    Handles two LLM output patterns:
    1. Line-start bullets: lines beginning with '- ', '• ', '* '
    2. Inline bullets: '...citation] - Next sentence' on the same line
       (LLM embeds ' - ' as separator after a citation or sentence end)
    """
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    lines = body.splitlines()
    prose_lines = []
    for line in lines:
        # Strip leading bullet markers: "- ", "• ", "* "
        cleaned = re.sub(r"^\s*[-•*]\s+", "", line)
        prose_lines.append(cleaned)
    # Join all lines into a single paragraph; collapse internal blank lines
    paragraph = " ".join(l for l in prose_lines if l.strip())
    # Strip inline bullet separators: "] - NextSentence" or ". - NextSentence"
    # (LLM outputs these when it ignores the no-bullet instruction but still
    #  produces output on a single line with inline "- " separators)
    paragraph = re.sub(r"(\]|(?<=[.!?])) - (?=[A-Z])", r"\1 ", paragraph)
    # Clean up double spaces
    paragraph = re.sub(r"  +", " ", paragraph).strip()
    new_section = "# Summary\n" + paragraph + "\n"
    return md[: match.start()] + new_section + md[match.end():]


# Small fallback table for abbreviations the LLM rarely defines anywhere in its output.
# Only truly opaque terms that would confuse a non-specialist reader.
# Ambiguous abbreviations (OR, HR, etc.) are intentionally omitted.
_ACRONYM_FALLBACK: dict[str, str] = {
    "DS":      "Down syndrome",
    "T21":     "trisomy 21",
    "AD":      "Alzheimer's disease",
    "APP":     "amyloid precursor protein",
    "MCI":     "mild cognitive impairment",
    "CHD":     "congenital heart disease",
    "ASD":     "autism spectrum disorder",
    "AML":     "acute myeloid leukemia",
    "ALL":     "acute lymphoblastic leukemia",
    "OSA":     "obstructive sleep apnea",
    "CNS":     "central nervous system",
    "CSF":     "cerebrospinal fluid",
    "PET":     "positron emission tomography",
}

# Regex to detect any "full term (ABBREV)" definition in text.
# Matches 1–6 lowercase/mixed words followed by a parenthetical all-caps abbreviation.
_DEFINITION_PAT = re.compile(
    r'\b([A-Za-z][a-z]+(?:[\s\-][A-Za-z][a-z]+){0,5})\s+\(([A-Z][A-Z0-9\u03b1-\u03c9\u2013\-]{1,15})\)',
)


def _extract_definitions_from_markdown(md: str, skip_section: str = "Summary") -> dict[str, str]:
    """Scan all sections *except* skip_section for 'full term (ABBREV)' patterns.

    Returns a dict mapping abbreviation → full term, built from whatever the LLM
    itself wrote in Key Evidence, Reasoning etc.
    """
    # Strip the Summary from the text we scan so we only look at other sections
    search_text = re.sub(
        r'(?im)^#\s+' + re.escape(skip_section) + r'\s*\n.*?(?=^#\s+|\Z)',
        '', md, flags=re.S | re.M
    )
    found: dict[str, str] = {}
    for m in _DEFINITION_PAT.finditer(search_text):
        full_term = m.group(1).strip()
        abbrev    = m.group(2).strip()
        if abbrev not in found and len(full_term.split()) >= 2:
            found[abbrev] = full_term
    return found


def _expand_acronyms_in_summary(md: str) -> str:
    """Expand undefined acronyms on first use inside the Summary section.

    Strategy (no hardcoded map required):
    1. Auto-detect definitions the LLM already wrote elsewhere in the response
       (Key Evidence, Reasoning) via the pattern 'full term (ABBREV)'.
    2. Merge with a small fallback table for terms the LLM rarely defines.
    3. For each abbreviation found in the Summary without an expansion:
       a. If the full term is already present → append ' (ABBREV)' after it.
       b. If only the bare abbreviation is present → replace first use with
          'full term (ABBREV)'.
    4. Abbreviations already properly expanded or not present are untouched.
    """
    summ_match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not summ_match:
        return md
    body = summ_match.group(1)

    # Build expansion map: dynamic (from LLM output) takes priority over fallback
    expansion_map = dict(_ACRONYM_FALLBACK)  # start with fallback
    expansion_map.update(_extract_definitions_from_markdown(md))  # LLM definitions override

    for abbrev, full in expansion_map.items():
        if not full:
            continue
        # Skip if already expanded in Summary: "full term (ABBREV)"
        already = re.search(
            re.escape(full) + r'\s*\(' + re.escape(abbrev) + r'\)',
            body, re.IGNORECASE
        )
        if already:
            continue

        full_pat   = re.compile(r'\b(' + re.escape(full) + r')\b', re.IGNORECASE)
        abbrev_pat = re.compile(r'(?<!\()(?<!\w)\b' + re.escape(abbrev) + r'\b(?!\))')

        full_m   = full_pat.search(body)
        abbrev_m = abbrev_pat.search(body)

        if full_m:
            # Full term present — annotate it with the abbreviation
            insert_pos = full_m.end()
            following  = body[insert_pos: insert_pos + len(abbrev) + 3]
            if not following.startswith(f" ({abbrev}"):
                body = body[:insert_pos] + f" ({abbrev})" + body[insert_pos:]
        elif abbrev_m:
            # Only bare abbreviation present — expand it
            body = abbrev_pat.sub(f"{full} ({abbrev})", body, count=1)

    new_section = "# Summary\n" + body
    return md[: summ_match.start()] + new_section + md[summ_match.end():]


def _strip_last_sentence_citation(md: str) -> str:
    """Remove trailing PMC citations from the final sentence of the Summary.

    The concluding sentence should be a synthesis/interpretation statement —
    not a cited data point. Stripping its citation avoids the awkward pattern
    of ending a paragraph with a bracket reference.
    Only removes the citation from the very last sentence; all other sentences
    keep their citations untouched.
    """
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    # Split into sentences
    parts = re.split(r'(?<=[\.\!\?])\s+', body)
    if len(parts) < 2:
        return md  # single-sentence summary — leave it alone

    last = parts[-1].strip()
    # Strip one or more trailing PMC citation tokens (linked or bare)
    cleaned = re.sub(r'\s*\[PMC\d{4,}:[A-Za-z0-9_-]+\](?:\([^)]*\))?', '', last).rstrip()
    if not cleaned:
        return md  # entire last sentence was a citation — leave untouched
    if cleaned[-1] not in '.!?':
        cleaned += '.'
    parts[-1] = cleaned
    new_body = ' '.join(parts)
    new_section = '# Summary\n' + new_body + '\n'
    return md[: match.start()] + new_section + md[match.end():]


def _strip_summary_warnings(md: str) -> str:
    """Remove warning/banner lines inside the Summary section."""
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1)
    lines = body.splitlines()
    cleaned = []
    for line in lines:
        if re.search(r"\bwarning\b", line, re.IGNORECASE):
            continue
        if re.search(r"research context only", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    new_body = "\n".join(cleaned).strip()
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]


def _strip_numeric_refs_from_text(body: str) -> str:
    """Apply all numeric-citation-stripping patterns to a plain text body string.

    Shared by _strip_summary_numeric_refs (Summary) and the Key Evidence pass below.
    PMC-format citations ([PMCxxxxxxx:Section]) and real measurements (IL-6, DYRK1A,
    n=25, COVID-19) are never touched.
    """
    # 1. Strip parenthetical numeric references: (5), (1, 2), (1-5), (9, 10, 12, 13)
    #    Lookahead removed — digit-only interior prevents (Figure 1)/(n=5) false positives;
    #    mid-sentence refs like "(9, 10, 12, 13, 15, 17) were associated" require no lookahead.
    body = re.sub(r"\s*\(\d+(?:[,;\s\u2013\-]+\d+)*\)", "", body)

    # 2. Strip parenthetical supplementary/results references: (Results S1), (Suppl. Fig 2)
    body = re.sub(r"\s*\((?:Results?|Suppl(?:emental)?|Extended Data|Fig|Table)[\s\.][^)]{0,40}\)", "", body, flags=re.I)

    # 3. Strip word-attached superscript runs: "death4,54" → "death"
    #    Lookahead includes \[ so that "death4,54[PMC...]" (no space) is also caught.
    body = re.sub(r"(?<=[a-zA-Z])(\d+(?:,\s*\d+)+)(?=[.,;:\s\)\[]|$)", "", body)

    # 4. Strip lone superscript digit glued to end of a word before punctuation/space/bracket
    #    e.g. "infections2." → "infections." / "risk3[PMC..." → "risk[PMC..."
    #    but NOT "COVID-19", "DYRK1A", "IL-6"
    body = re.sub(r"(?<=[a-z])(\d+)(?=[.,;:\s\[])", "", body)

    # 5. Strip space-padded inline numeric reference runs copied from paper text:
    #    e.g. ", 40 , 41 resulting" or "diseases, 42 , 43 further"
    body = re.sub(
        r'(?<![a-zA-Z=\d])(?:[,\s]+\d{1,3}(?:\s*[,;]\s*\d{1,3}){1,9})(?=\s+[a-z]|\s*[,;.\)])',
        '',
        body,
    )

    # 6. Strip standalone superscript-style runs at end of clause: "patients 40,41,44 and"
    body = re.sub(r'\b(\d{1,3}(?:[,\s]+\d{1,3}){1,8})\b(?=\s+(?:and|or|but|resulting|further|which|that|in|of|for|with|to|by|from)\b)', '', body)

    # Collapse any double-spaces left by removals
    body = re.sub(r"  +", " ", body)
    return body


def _strip_summary_numeric_refs(md: str) -> str:
    """Remove inline numeric citation markers carried over from raw paper text.

    Handles:
      • Parenthetical numbers: (5)  (1, 2)  (1–5)  (9, 10, 12)
      • Superscript-style numbers glued to a word: "death4,54" → "death"
      • Standalone trailing numerals before punctuation: "infections2." → "infections."

    Applied to both the Summary prose and each Key Evidence bullet body so that
    verbatim paper text like "leading cause of death4,54 [PMCxx:Discussion]" is
    cleaned regardless of which section it ends up in.

    PMC-format citations ([PMCxxxxxxx:Section]) and real measurements (e.g. IL-6,
    COVID-19, n=25) are not touched.
    """
    # --- Summary section ---
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if match:
        body = _strip_numeric_refs_from_text(match.group(1))
        new_section = "# Summary\n" + body
        md = md[: match.start()] + new_section + md[match.end():]

    # --- Key Evidence bullets ---
    ke_match = re.search(r"(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)", md, re.S)
    if ke_match:
        ke_body = ke_match.group(1)
        new_ke_lines = []
        for line in ke_body.splitlines(keepends=True):
            stripped = line.strip()
            is_bullet = stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*')
            if not is_bullet:
                new_ke_lines.append(line)
                continue
            # Preserve leading bullet marker; strip numeric refs from the rest
            m_bullet = re.match(r'^(\s*[-•*]\s+)(.*)', line, re.DOTALL)
            if m_bullet:
                cleaned = _strip_numeric_refs_from_text(m_bullet.group(2))
                new_ke_lines.append(m_bullet.group(1) + cleaned)
            else:
                new_ke_lines.append(line)
        new_ke_body = ''.join(new_ke_lines)
        md = md[: ke_match.start()] + "# Key Evidence\n" + new_ke_body + md[ke_match.end():]

    return md

def _enforce_polarity_focus(md: str, question: str) -> str:
    """No-op: polarity filtering removed — the LLM answers based on the question naturally."""
    return md


def _strip_summary_synthesis(md: str) -> str:
    """Remove synthesis/interpretive sentences from Summary (evidence-only mode)."""
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    sentences = re.split(r'(?<=[\.\!\?])\s+', body)
    drop_markers = [
        "taken together",
        "overall",
        "in summary",
        "these findings indicate",
        "these findings suggest",
        "this suggests",
        "this indicates",
        "together,",
        "collectively,",
        "in conclusion",
        "therefore",
    ]
    kept = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        s_low = s_strip.lower()
        if any(m in s_low for m in drop_markers):
            continue
        kept.append(s_strip)

    new_body = " ".join(kept).strip()
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]

def _filter_summary_by_question_terms(md: str, question: str) -> str:
    """Keep only sentences that mention key question terms (evidence-only mode)."""
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md

    terms = _extract_key_terms(question)
    if not terms:
        return md

    variants = []
    for t in terms:
        variants.extend(_term_variants(t))
    pat = r"\b(" + "|".join(re.escape(v) for v in variants) + r")\w*\b"
    pattern = re.compile(pat, re.IGNORECASE)

    sentences = re.split(r'(?<=[\.\!\?])\s+', body)
    kept = [s.strip() for s in sentences if s.strip() and pattern.search(s)]
    if not kept:
        return md
    new_body = " ".join(kept).strip()
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]

def _keep_first_summary_sentence(md: str) -> str:
    """Keep only the first sentence in the Summary section."""
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).strip()
    if not body:
        return md
    parts = re.split(r'(?<=[\.\!\?])\s+', body)
    first = parts[0].strip()
    if not first:
        return md
    new_section = "# Summary\n" + first + "\n"
    return md[: match.start()] + new_section + md[match.end():]


def _summary_has_citation(md: str) -> bool:
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return False
    body = match.group(1)
    return bool(re.search(r"\[PMC\d{4,}:[A-Za-z0-9_-]+\]", body))


def _append_summary_citation(md: str, citation: str) -> str:
    """Append a citation tag to the Summary section if missing."""
    match = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", md, flags=re.S | re.M)
    if not match:
        return md
    body = match.group(1).rstrip()
    tag = citation.strip()
    if not tag.startswith("["):
        tag = f"[{tag}]"
    new_body = body + " " + tag
    new_section = "# Summary\n" + new_body + "\n"
    return md[: match.start()] + new_section + md[match.end():]


_DANGLING_STAT_PAREN_RE = re.compile(
    r'\s*\(\s*(?:[pP]|a?OR|a?HR|RR|IRR|95%\s*CI|CI)\b'
    r'[^()\[\]]{0,15}(?=\s*\[PMC)'
)


def _strip_dangling_stat_parens(md: str) -> str:
    """Remove truncated statistical parentheticals glued to a citation tag.

    The improve model sometimes cuts a stat mid-parenthesis, producing e.g.
    "versus 3.5% in controls (p[PMC9796221:Results]." — an opened "(p" (or
    "(OR 2.", "(95% CI") with no closing paren before the citation.  Complete
    parentheticals like "(p=0.013) [PMC...]" are untouched: the pattern only
    matches when NO closing paren occurs between "(" and the "[PMC" tag.
    """
    return _DANGLING_STAT_PAREN_RE.sub(' ', md)


def _repair_mutated_pmc_ids(md: str, valid_pmcids: set) -> str:
    """Repair PMC citation tags whose numeric ID is a digit-level mutation of a
    real corpus ID (e.g. [PMC979796221:Results] — the LLM duplicated "97" while
    typing PMC9796221).  A bad ID is rewritten when exactly one valid corpus ID
    is a substring of it (inserted/duplicated digits) or it is a substring of
    exactly one valid ID (dropped digits).  Ambiguous or unmatched IDs are left
    unchanged for _strip_hallucinated_citations to remove.
    """
    if not valid_pmcids:
        return md
    _norm: set[str] = set()
    for p in valid_pmcids:
        m = re.match(r"(?i)(?:PMC)?(\d+)", str(p).split(":")[0].strip())
        if m:
            _norm.add(m.group(1))
    if not _norm:
        return md

    def _fix(m):
        bad = m.group(1)
        if bad in _norm:
            return m.group(0)
        cands = {v for v in _norm
                 if (len(v) >= 6 and v in bad) or (len(bad) >= 6 and bad in v)}
        if len(cands) == 1:
            good = cands.pop()
            print(f"[CITE_REPAIR] PMC{bad} -> PMC{good}")
            return m.group(0).replace(f"PMC{bad}", f"PMC{good}", 1)
        return m.group(0)

    return re.sub(r'\[PMC(\d+):[^\]]+\](?:\([^)]+\))?', _fix, md)


def _strip_hallucinated_citations(md: str, valid_pmcids: set) -> str:
    """Remove PMC citation tags whose PMC IDs are not in the evidence corpus.

    Protects against the LLM copying exemplar IDs from the prompt (e.g.
    PMC9876543) or otherwise hallucinating citations.

    In the Summary: strips just the citation tag, leaving prose intact so that
    _append_summary_citation can add a real fallback afterward.
    In Key Evidence: removes the entire bullet line when ALL of its PMC IDs
    are hallucinated (copied-exemplar bullets contain fabricated text too).
    """
    if not valid_pmcids:
        return md

    # Pre-pass: strip prompt-exemplar placeholder citations where the PMC "ID"
    # is entirely letters (e.g. [PMCxxxxxxx:Results] copied from the prompt
    # instructions). These never match \d+ so the main loop below would miss them.
    md = re.sub(r'\s*\[PMC[a-zA-Z]{3,}:[^\]]+\](?:\([^)]+\))?', '', md)

    # Build a set of bare numeric IDs (e.g. "11254168") for fast lookup
    _norm: set[str] = set()
    for p in valid_pmcids:
        m = re.match(r"(?i)(?:PMC)?(\d+)", p.split(":")[0].strip())
        if m:
            _norm.add(m.group(1))

    _cite_re = re.compile(r'\[PMC(\d+):[^\]]+\](?:\([^)]+\))?')

    def _is_valid(pmc_num: str) -> bool:
        return pmc_num in _norm

    # --- Summary: strip invalid cite tags, preserve prose -------
    summ_m = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
    if summ_m:
        new_body = _cite_re.sub(
            lambda m: m.group(0) if _is_valid(m.group(1)) else '',
            summ_m.group(1),
        )
        new_body = re.sub(r'  +', ' ', new_body).strip()
        md = md[: summ_m.start()] + f'# Summary\n{new_body}\n' + md[summ_m.end():]

    # --- Key Evidence: drop entire bullet when ALL its PMC IDs are invalid --
    ke_m = re.search(r'(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
    if ke_m:
        ke_body = ke_m.group(1)
        new_ke_lines = []
        for line in ke_body.splitlines(keepends=True):
            stripped = line.strip()
            if not (stripped.startswith('-') or stripped.startswith('•')):
                new_ke_lines.append(line)
                continue
            line_pmcs = re.findall(r'\[PMC(\d+):[^\]]+\]', line)
            if not line_pmcs:
                new_ke_lines.append(line)
                continue
            if any(_is_valid(p) for p in line_pmcs):
                # At least one valid citation — keep bullet, strip invalid tags only
                line = _cite_re.sub(
                    lambda m: m.group(0) if _is_valid(m.group(1)) else '',
                    line,
                )
                new_ke_lines.append(line)
            # else: all citations hallucinated → drop the bullet
        new_ke_body = ''.join(new_ke_lines)
        md = md[: ke_m.start()] + f'# Key Evidence\n{new_ke_body}' + md[ke_m.end():]

    return md


def _llm_review_summary(md: str, question: str, llm) -> str:
    """LLM self-review pass: remove logically inconsistent sentences from the Summary.

    Sends the question + Summary body to the LLM with a focused instruction:
    identify and remove any sentence that contradicts the question's intent,
    asserts the opposite polarity, or is factually unrelated.  The LLM returns
    only the corrected Summary body (plain prose, no headers).

    This is intentionally narrow — it touches ONLY the Summary body, leaves
    Key Evidence and Reasoning untouched, and runs AFTER the initial regex
    polarity pass so the LLM only needs to catch what regex missed.  The pass
    is skipped if the LLM call fails (silent fallback to the unchanged md).
    """
    summ_m = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
    if not summ_m:
        return md
    body = summ_m.group(1).strip()
    if not body or '[EVIDENCE_GAP]' in body:
        return md

    review_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a biomedical editor performing a targeted consistency check.\n"
         "Your ONLY task: read the QUESTION and the SUMMARY paragraph, then return "
         "a corrected version of the Summary with any logically inconsistent sentences "
         "removed.\n\n"
         "RULES:\n"
         "1. Remove any sentence that contradicts the question's intent.\n"
         "   Example: if the question asks about LOWER-risk conditions, remove any "
         "   sentence that says DS individuals are at HIGHER risk of something.\n"
         "   Example: if the question asks about a specific gene, remove sentences "
         "   about unrelated conditions.\n"
         "2. Remove any sentence that introduces a topic the question did NOT ask about.\n"
         "3. Keep the first sentence (context) and last sentence (conclusion) unless "
         "   they directly contradict the question.\n"
         "4. Preserve ALL inline citations exactly as written: [PMCxxxxxxx:Section].\n"
         "5. Do NOT add new sentences. Do NOT rephrase existing sentences. "
         "   Only REMOVE sentences that are logically wrong for this question.\n"
         "6. Return ONLY the corrected paragraph text — no header, no explanation, "
         "   no commentary.\n"
         "7. If every sentence is consistent with the question, return the original "
         "   paragraph unchanged.\n"
        ),
        ("human",
         "QUESTION: {question}\n\n"
         "SUMMARY:\n{summary}\n\n"
         "Return the corrected Summary paragraph (plain prose only, no # headers):"
        ),
    ])
    try:
        result = (review_prompt | llm | StrOutputParser()).invoke(
            {"question": question, "summary": body}
        )
        corrected = result.strip()
        # Sanity guards — reject the LLM result if it looks malformed
        # Strip any section-header text the LLM may have embedded inline in its
        # corrected paragraph (e.g. "...Discussion]. # Key Evidence\n- bullet" or
        # "...Discussion].# Key Evidence").  Use an unconditional search (no ^|\s
        # prefix) so the guard fires regardless of what character precedes '#'.
        _hdr_in_review = re.search(r'(#{1,2}\s+(?:Key Evidence|Reasoning)\b)', corrected)
        if _hdr_in_review:
            corrected = corrected[: _hdr_in_review.start(1)].rstrip()
            print("  [REVIEW] Stripped embedded section header from LLM review output")
        if (
            not corrected
            or len(corrected) < 40
            or re.search(r'(?im)^#\s+', corrected)  # returned headers = wrong format
            or len(corrected) > len(body) * 2         # ballooned = hallucinated new content
        ):
            return md
        # Replace just the Summary body, leave Key Evidence + Reasoning intact
        new_section = "# Summary\n" + corrected + "\n"
        return md[: summ_m.start()] + new_section + md[summ_m.end():]
    except Exception as e:
        print(f"  [REVIEW_WARN] LLM summary review failed: {e}")
        return md


def create_final_node(cfg: RAGConfig, llm=None):
    """Create final processing node."""

    def finalize(state: RAGState) -> RAGState:
        t0 = _now_ms()
        md = state.get("markdown", "")
        original_len = len(md)

        # Normalize section headers — ensure # Key Evidence / # Reasoning always
        # start on their own line.  The LLM occasionally appends a header inline
        # at the end of a sentence without a line break, which breaks all
        # downstream section-span parsers.
        md = _fix_inline_section_headers(md)

        # Ensure Summary header for UI consistency — search anywhere in the doc,
        # not just at position 0, so that a safety-guard banner prepended before
        # "# Summary" does not trigger a second spurious "# Summary\n" insertion
        # (which would produce a duplicate heading and a 0-word parsed summary).
        if md and not re.search(r"(?im)^#\s+Summary", md):
            md = "# Summary\n" + md.lstrip()

        # Fix citations (split semicolon lists, fix unbracketed).
        # NOTE: _fix_citations_after_period is intentionally deferred to just
        # before _hyperlink_inline_pmc so that all intermediate reconstruction
        # steps (_flatten_summary_bullets, _reconstruct_list_summary,
        # _pad_summary_from_ke_final) have already run before the period-fix
        # pass — those steps can themselves introduce post-period citations.
        md = _normalize_pmc_citation_format(md)   # fix [text (PMCxxx):Sec] → [PMCxxx:Sec]
        md = _split_semicolon_citations(md)
        md = _fix_misplaced_bracket_citations(md)
        md = _fix_unbracketed_citations(md)

        # Strip warning banners inside Summary
        md = _strip_summary_warnings(md)

        # Flatten any bullet-marker lines inside Summary into prose paragraph
        md = _flatten_summary_bullets(md)

        # Strip leading transition/conclusion phrases from the first Summary
        # sentence (e.g. "As a consequence, ...", "Taken together, ...",
        # "These findings indicate ...").  These make no sense as openers.
        md = _strip_summary_transition_opener(md)

        # Evidence-only mode: remove synthesis/interpretive sentences
        if cfg.evidence_only_summary:
            md = _strip_summary_synthesis(md)
            md = _filter_summary_by_question_terms(md, state.get("question", ""))
            md = _keep_first_summary_sentence(md)
        elif cfg.focus_question_terms:
            md = _filter_summary_by_question_terms(md, state.get("question", ""))

        # Strip sentences that are verbatim raw experimental data from papers
        md = _strip_raw_data_sentences(md)

        # Strip Summary sentences that assert the OPPOSITE polarity to the question.
        # Lower-risk questions should not contain higher-risk sentences (and vice versa).
        md = _strip_opposite_polarity_sentences(md, state.get("question", ""))

        # Strip verbatim-bracket sentences: LLM sometimes wraps evidence text in
        # [brackets] as a verbatim quote after the real synthesized sentence.
        md = _strip_verbatim_bracket_sentences(md)

        # Strip ICU/ventilation sentences for comorbidity/conditions questions.
        # Acute-care severity metrics (ICU rate, ventilation rate) are not conditions.
        md = _strip_icu_for_comorbidity_q(md, state.get("question", ""))

        # Strip uncited comma-list sentences: "at elevated risk for A, B, C..."
        # These are garbled sentences where the LLM dumps condition names without
        # per-item citations.  First pass runs here before LLM review.
        md = _strip_comma_list_sentences(md)

        # Strip broken "at elevated risk for Among/While/[uncited list]" fragments.
        # The LLM sometimes concatenates a risk-intro phrase with a new sentence
        # starter (e.g. "are at elevated risk for Among individuals...") producing
        # a grammatically broken sentence that _strip_comma_list_sentences misses
        # because the tail uses "and" not commas.
        md = _strip_broken_risk_intro_sentences(md)

        # LLM self-review: catch logical inconsistencies regex cannot handle —
        # wrong-polarity sentences, off-topic claims, hallucinated conditions.
        # Runs after the regex pass so the LLM only sees residual issues.
        # Skipped for EVIDENCE_GAP responses and when no LLM is available.
        if llm is not None and "[EVIDENCE_GAP]" not in md:
            md = _llm_review_summary(md, state.get("question", ""), llm)

        # Re-run raw-data stripping after LLM review — the LLM can re-introduce
        # author-year citations (Smith et al., 2016; ...) and first-person verbs
        # that were stripped before it ran.
        md = _strip_raw_data_sentences(md)

        # Remove duplicate consecutive items within a sentence
        # e.g. 'Ifnar, Ifnar, and Ifngr' → 'Ifnar and Ifngr'
        md = _dedup_inline_list_items(md)

        # Convert any bullet-formatted Summary lines into flowing prose
        md = _prose_summary(md)

        # Hard-enforce 150-word cap on Summary section
        md = _truncate_summary_to_150_words(md)
        truncated = len(md) < original_len

        # Normalize Key Evidence bullets (one per line)
        md = _normalize_key_evidence_bullets(md)

        # Remove Summary sentences that are verbatim copies of Key Evidence bullets.
        # The Nemotron structured output path can write the KE bullets as prose
        # in the Summary, then repeat them as proper bullet points — this strips
        # those duplicates from the Summary so the sections contain distinct content.
        md = _dedup_summary_vs_ke(md)

        # If the LLM omitted the Key Evidence section entirely, synthesize it from
        # the raw retrieved chunks so that _reconstruct_list_summary has material
        # to work with.
        if not re.search(r"(?im)^#\s+Key Evidence", md):
            raw_ev = state.get("raw_evidence") or []
            if raw_ev:
                ke_lines = []
                for item in raw_ev[:8]:
                    tag  = item.get("citation_tag", "")
                    text = (item.get("text") or "").strip()
                    if text and tag:
                        sents   = re.split(r"(?<=[.!?])\s+", text)
                        # Skip fragments: figure/table refs, panel labels (lowercase
                        # letter followed by comma/space like "b, gMFI..."), digit-
                        # prefixed items, and short stubs.
                        good_sents = [
                            s for s in sents
                            if len(s.split()) >= 10
                            and not re.match(
                                r"^[\.\s]*([a-z][,\s]|\d|Fig|Table|Extended|Suppl|S\d)", s, re.I
                            )
                        ]
                        if not good_sents:
                            continue
                        snippet = " ".join(good_sents[:2]).strip()
                        if snippet[-1] not in ".!?":
                            snippet += "."
                        ke_lines.append(f"- {snippet} [{tag}]" if not tag.startswith("[") else f"- {snippet} {tag}")
                if ke_lines:
                    md = md.rstrip("\n") + "\n\n# Key Evidence\n" + "\n".join(ke_lines) + "\n"

        # For any question with a short Summary: the LLM sometimes defers all
        # specifics to Key Evidence. If the Summary word count is still < 90 after
        # all LLM-level retries, rebuild it programmatically from the KE bullets.
        # Also trigger for list questions whose Summary has no inline citations at
        # all (LLM enumerated conditions without per-item [PMC:Section] tags).
        _q_for_recon = state.get("question", "")
        # Gate reconstruction conservatively: only rebuild when the LLM summary is
        # genuinely broken — very short (<70 words; 70-129 is handled by
        # _pad_summary_from_ke_final) or a list answer with no citations at all.
        # think=False now yields reliably good summaries, so we trust the LLM and
        # do NOT force reconstruction merely because no single sentence packs ≥3
        # citations (that over-eager trigger was the main entry into the garbled
        # reconstruction path on an unlucky generation).
        _needs_recon = (
            _summary_word_count(md) < 70
            or (
                not _summary_has_citation(md)
                and bool(_LIST_Q_PAT.search(_q_for_recon))
            )
        )
        # Additional trigger: force reconstruction when ≥2 Summary sentences are
        # verbatim KE bullets even if the word count is ≥110.  _llm_review_summary
        # (which runs before this check) can expand a verbatim-bullet Summary above
        # the 110-word threshold by adding transitions, permanently bypassing this
        # block.  Detecting the copies here ensures reconstruction always runs.
        if not _needs_recon and "[EVIDENCE_GAP]" not in md:
            _ke_dup_m = re.search(r'(?im)^#\s+Key Evidence\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
            _sm_dup_m = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S)
            if _ke_dup_m and _sm_dup_m:
                _cite_d = re.compile(r'\[PMC[^\]]+\](?:\([^)]+\))?')
                _ke_nd = {
                    re.sub(r'\s+', ' ', _cite_d.sub('', b).strip().rstrip('.').lower())
                    for b in re.findall(r'^\s*[-*]\s+(.+)', _ke_dup_m.group(1), re.MULTILINE)
                    if len(_cite_d.sub('', b).strip()) > 30
                }
                if _ke_nd:
                    _sm_sents_d = re.split(r'(?<=[.!?])\s+', _sm_dup_m.group(1).strip())
                    _dup_nd = sum(
                        1 for s in _sm_sents_d
                        if re.sub(r'\s+', ' ', _cite_d.sub('', s).strip().rstrip('.').lower()) in _ke_nd
                    )
                    if _dup_nd >= 2:
                        _needs_recon = True
                        print(f"  [FINAL] {_dup_nd} Summary sentences are verbatim KE bullets — forcing reconstruction")
        if _needs_recon and "[EVIDENCE_GAP]" not in md:
            _md_before_recon = md
            md = _reconstruct_list_summary(md, question=state.get("question", ""))
            # Safety net: if reconstruction still emitted a junk enumeration
            # ("...risk for <discourse marker>..."), revert to the pre-recon summary
            # rather than show garbled text. Generic — checks for a transition
            # marker right after the enumeration verb, not any topic term.
            _recon_sm = re.search(r'(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)', md, re.S | re.M)
            if _recon_sm and re.search(
                r'(?:risk for|include[sd]?|are)\s+(?:on the other hand|by contrast|'
                r'in contrast|thus|however|moreover|therefore|whereas|as (?:previously )?'
                r'(?:described|noted|shown))\b',
                _recon_sm.group(1), re.IGNORECASE,
            ):
                print("  [FINAL] reconstruction produced junk enumeration — reverting to LLM summary")
                md = _md_before_recon
            md = _flatten_summary_bullets(md)
            # Re-enforce 150-word cap after reconstruction (truncation ran before)
            md = _truncate_summary_to_150_words(md)
            # _reconstruct_list_summary draws from Key Evidence bullets, which may
            # themselves contain verbatim methods text (e.g. animal-count tables).
            # Re-run the raw-artifact stripper so nothing re-introduced here survives.
            md = _strip_raw_data_sentences(md)
            # Re-run polarity stripper — reconstruction pulls from KE bullets which
            # may contain higher-risk sentences not relevant to a lower-risk question.
            md = _strip_opposite_polarity_sentences(md, state.get("question", ""))
            # Strip KE bullet sentences re-introduced as padding by _pad_enum_parts
            # (reconstruction sometimes adds them to reach the 90-word minimum).
            md = _dedup_summary_vs_ke(md, floor=40)

        # Strip numeric inline citation markers carried over from raw paper text
        # (e.g. "(5)", "death4,54") — applies whether text came from LLM or reconstruction.
        md = _strip_summary_numeric_refs(md)

        # Strip numeric inline citation markers one more time after reconstruction paths
        md = _strip_summary_numeric_refs(md)

        # Final dedup + 150-word cap — runs AFTER all reconstruction paths
        md = _dedupe_summary_sentences(md)
        md = _truncate_summary_to_150_words(md)

        # Strip citation from the last sentence of the Summary — conclusions
        # should be interpretive statements, not end on a cited data point.
        md = _strip_last_sentence_citation(md)

        # Strip hallucinated PMC citations (IDs not in the actual evidence corpus).
        # Must run BEFORE _hyperlink_inline_pmc so bare [PMCxxxx:Sec] tags are still
        # detectable.  Catches LLM copying of exemplar IDs from the prompt.
        # Remove truncated stat parentheticals glued to citations, e.g. "(p[PMC...]"
        md = _strip_dangling_stat_parens(md)

        _valid_pmcids = set(state.get("citations") or [])
        if _valid_pmcids:
            md = _repair_mutated_pmc_ids(md, _valid_pmcids)
            md = _strip_hallucinated_citations(md, _valid_pmcids)

        # Re-add a fallback citation to the Summary if all citations were stripped
        # (either by _strip_last_sentence_citation or by the hallucination stripper).
        if not _summary_has_citation(md):
            _fallback_cites = state.get("citations") or []
            if _fallback_cites:
                md = _append_summary_citation(md, _fallback_cites[0])

        # Strip finding sentences in the Summary that carry no citation.
        # The LLM sometimes produces factual sentences without [PMC:Section] tags
        # despite prompt instructions. Removing them here lets _pad_summary_from_ke_final
        # replace them with cited KE prose, preserving the context and conclusion.
        if "[EVIDENCE_GAP]" not in md:
            md = _strip_uncited_summary_sentences(md)

        # Final floor: if the Summary is still below 130 words after all stripping,
        # dedup, and citation fixup, pad it from Key Evidence bullet sentences.
        # This is a last-resort safety net that runs AFTER dedup so it cannot undo
        # the sentence-deduplication step but IS undone by truncation at 150 words.
        if _summary_word_count(md) < 60 and "[EVIDENCE_GAP]" not in md:
            md = _pad_summary_from_ke_final(md, min_wc=60, target_wc=120)
            # Re-run dedup after padding — pad draws from KE bullets which may
            # re-introduce a near-duplicate of a sentence the LLM also generated.
            md = _dedupe_summary_sentences(md)
            md = _truncate_summary_to_150_words(md)
            # Re-run polarity stripper after padding — pad draws from KE bullets
            # which may contain opposite-polarity sentences.
            md = _strip_opposite_polarity_sentences(md, state.get("question", ""))
            # Re-run raw-data stripper after padding — pad can introduce verbatim
            # sentences from KE bullets (e.g. "35.6% vs.", "in order of prevalence").
            md = _strip_raw_data_sentences(md)

        # Expand biomedical acronyms on first use in the Summary (e.g. AD → Alzheimer's disease (AD))
        md = _expand_acronyms_in_summary(md)

        # Ensure Reasoning section is present — the LLM occasionally omits it
        # despite being instructed.  A concise fallback is better than a blank UI tab.
        # Also covers the case where the LLM produces '# Reasoning\n' with an empty body.
        _reason_m = re.search(r"(?im)^#\s+Reasoning\s*\n(.*?)(?=^#\s+|\Z)", md, re.S | re.M)
        _reason_has_content = bool(_reason_m and _reason_m.group(1).strip())
        if not _reason_has_content:
            # Strip the empty header if it exists, then append a proper fallback
            md = re.sub(r"(?im)^#\s+Reasoning[ \t]*\n?", "", md)
            md = md.rstrip("\n") + (
                "\n\n# Reasoning\n"
                "- **Evidence selection:** Findings were selected based on direct relevance "
                "to the question, study quality, sample size, and section type "
                "(Results-section quantitative data prioritised over Discussion).\n"
                "- **Reasoning chain:** The evidence collectively addresses the question "
                "through consistent findings across independent Down syndrome cohorts.\n"
                "- **Confidence & gaps:** Confidence is highest where results replicate across "
                "multiple studies; single-study findings should be interpreted with caution.\n"
                "- **Reader caveats:** All findings derive from observational research; consult "
                "a qualified clinician before applying these findings to any clinical decision.\n"
            )

        # Final defence: re-normalise any section header that was put back inline
        # by an intermediate step (e.g. _llm_review_summary returning corrected text
        # that still contains '# Key Evidence' embedded in prose).  Running this a
        # second time is cheap and guarantees that # Key Evidence / # Reasoning
        # always begin on their own line before _hyperlink_inline_pmc fires.
        md = _fix_inline_section_headers(md)

        # Move any citations still placed after a period to before the period.
        # Runs immediately before hyperlinking so it catches citations introduced
        # by _flatten_summary_bullets, _reconstruct_list_summary, _pad_summary_from_ke_final.
        md = _fix_citations_after_period(md)
        # Add missing terminal period when LLM ends a sentence with [PMC:Section]
        # but omits the closing period entirely.
        md = _add_missing_citation_periods(md)

        # Re-split any semicolon-joined citations that were re-introduced by
        # _pad_summary_from_ke_final or _reconstruct_list_summary (which run after
        # the first _split_semicolon_citations call at the top of finalize).
        # Also re-normalise any misplaced-bracket tags added by those steps,
        # and normalize any non-standard citation formats re-introduced by LLM review.
        md = _normalize_pmc_citation_format(md)   # second pass: catch LLM-review variants
        md = _split_semicolon_citations(md)
        md = _fix_misplaced_bracket_citations(md)

        # Inject ', INCLUDE' suffix for INCLUDE-funded papers — runs AFTER all
        # normalisation (so tags are in canonical [PMCxx:Section] form) and
        # BEFORE _hyperlink_inline_pmc (which handles both formats).
        _all_hits = state.get("reranked_hits") or state.get("hits") or []
        md = _inject_include_tags(md, _all_hits)

        # Final dedup of inline list items — runs LAST so duplicates introduced
        # by _reconstruct_list_summary, _pad_summary_from_ke_final, or the LLM
        # are always caught before hyperlinking.
        md = _dedup_inline_list_items(md)

        # Final pass: strip any residual comma-list sentences that survived
        # reconstruction, padding, and all earlier stripping steps.
        md = _strip_comma_list_sentences(md)
        md = _strip_broken_risk_intro_sentences(md)

        # Hyperlink all PMC citations — run LAST so every citation added or
        # preserved by the post-processing steps above gets properly linked.
        md = _hyperlink_inline_pmc(md)

        # Strip any outer bracket wrapping a hyperlinked citation that the improve
        # model or LLM re-introduced: [[PMC:Section](url)] → [PMC:Section](url)
        md = re.sub(
            r'\[(\[PMC\d{4,}:[^\]]+\]\(https?://[^)]+\))\]',
            r'\1',
            md,
        )

        # Keep structured summary in sync after post-processing
        if cfg.evidence_only_summary:
            _structured = state.get("structured")
            if _structured is not None:
                _structured["summary"] = _extract_summary_text(md)

        return {
            **state,
            "markdown": md,
            "trace": [{"agent": "finalize", "timing_ms": _now_ms() - t0,
                       "note": f"citations linked; summary {'truncated to 150w' if truncated else 'within 150w'}"}],
        }

    return finalize


# =============================================================================
# Nemotron Cross-Encoder Reranker Node
# =============================================================================
def create_nemotron_reranker_node(reranker, cfg: RAGConfig):
    """Reranker node — supports DeepInfraReranker (external API) or NemotronCrossEncoderReranker (local).

    Scores are blended with vector scores using cfg.rerank_alpha.
    """
    _reranker_name = (
        f"DeepInfra API ({reranker._url})"
        if isinstance(reranker, DeepInfraReranker)
        else "Nemotron cross-encoder (local)"
    )

    def rerank(state: RAGState) -> RAGState:
        if not cfg.enable_rerank:
            return {
                **state,
                "reranked_hits": state["hits"],
                "trace": [{"agent": "reranker", "timing_ms": 0, "note": "Reranking disabled — using retrieval order"}],
            }

        t0 = _now_ms()
        hits = state["hits"]
        question = state["question"]

        if not hits:
            return {**state, "reranked_hits": [], "trace": [{"agent": "reranker", "timing_ms": 0, "note": "No hits to rerank"}]}

        documents = [(h.get("text") or "").strip()[:512] for h in hits]

        try:
            ce_scores = reranker.score(question, documents)
        except Exception as exc:
            print(f"  [RERANKER_WARN] {_reranker_name} failed: {exc} — using vector scores only")
            return {
                **state,
                "reranked_hits": hits,
                "trace": [{"agent": "reranker", "timing_ms": _now_ms() - t0,
                           "note": f"fallback to vector order: {exc}"}],
            }

        # Normalise vector scores to [0, 100]
        vec_scores = [float(h.get("score", 0.0)) for h in hits]
        vmin, vmax = min(vec_scores) if vec_scores else 0.0, max(vec_scores) if vec_scores else 1.0
        if vmax == vmin:
            norm_vec = lambda v: 50.0
        else:
            norm_vec = lambda v: 100.0 * (v - vmin) / (vmax - vmin)

        ranked = []
        # Use per-request alpha override (for parameter sweeps) if provided
        _alpha = state.get("rerank_alpha_override")
        if _alpha is None:
            _alpha = cfg.rerank_alpha

        # When sections_override is provided (e.g. with_abstract mode), every
        # explicitly requested section competes on equal footing (priority=1.0).
        # Production standard mode (no override): SECTION_PRIORITY applies —
        # Abstract/Introduction are not retrieved at all, so their 0.0 entry is
        # never consulted.
        _sections_override_set: Optional[set] = None
        _so = state.get("sections_override")
        if _so:
            _sections_override_set = {_normalize_section(s) for s in _so}

        for i, h in enumerate(hits):
            ce_s = ce_scores[i] * 100.0          # scale to 0-100
            vs = norm_vec(float(h.get("score", 0.0)))
            combined = _alpha * vs + (1.0 - _alpha) * ce_s

            # Section priority — direct multiplier, no separate β scale
            # final = combined * priority
            # Standard mode: active sections = 1.0, Abstract/Intro = 0.0
            # with_abstract mode (sections_override set): all requested sections
            #   treated equally at 1.0 so the comparison is fair.
            section = _normalize_section(h.get("section") or "NA")
            if _sections_override_set is not None:
                # Equal priority for all sections in the override list;
                # any stray section not in the list keeps SECTION_PRIORITY.
                priority = 1.0 if section in _sections_override_set else SECTION_PRIORITY.get(section, 0.0)
            else:
                priority = SECTION_PRIORITY.get(section, 0.0)
            final = combined * priority

            h2 = dict(h)
            h2["_rerank_llm"]      = round(ce_s, 2)    # cross-encoder score scaled to 0-100
            h2["_rerank_vec"]      = round(vs, 2)
            h2["_rerank_final"]    = round(final, 2)
            h2["_section_priority"] = priority
            ranked.append((final, h2))

        ranked.sort(key=lambda x: x[0], reverse=True)
        reranked = [h for _, h in ranked]

        # Drop chunks below the minimum relevance threshold.
        # _rerank_final is on a [0, ~115] scale (combined score × section bonus).
        # Set RERANK_MIN_SCORE=20 to discard clearly irrelevant chunks before synthesis.
        # Default 0 = no filtering (backward compatible).
        _min_score = float(os.environ.get("RERANK_MIN_SCORE", "0"))
        if _min_score > 0:
            before = len(reranked)
            reranked = [h for h in reranked if h.get("_rerank_final", 0) >= _min_score]
            print(f"  [RERANKER] score filter ≥{_min_score}: {before} → {len(reranked)} chunks")

        # Paper diversity: cap chunks per PMCID in reranked order so that a single
        # dominant paper (e.g. one large cohort study) cannot crowd out evidence from
        # other papers.  Runs after reranking so the cap respects reranker scores —
        # the highest-scoring chunks per paper are kept, the rest are dropped.
        _max_per_paper = int(os.environ.get("MAX_CHUNKS_PER_PAPER", "2"))
        before_div = len(reranked)
        reranked = apply_paper_diversity(
            reranked,
            target_k=len(reranked),   # keep all after diversity cap — synthesis node further limits via context chars
            max_chunks_per_paper=_max_per_paper,
        )
        print(f"  [RERANKER] paper diversity (max {_max_per_paper}/paper): {before_div} → {len(reranked)} chunks")

        t1 = _now_ms()
        return {
            **state,
            "reranked_hits": reranked,
            "trace": [{"agent": "reranker", "timing_ms": t1 - t0,
                       "note": f"{_reranker_name} re-ranked {len(reranked)} hits"}],
        }

    return rerank


# =============================================================================
# LLM-as-Judge + Improve Node
# =============================================================================
_PIPELINE_JUDGE_SYSTEM = (
    "You are an expert biomedical reviewer specialising in Down syndrome (Trisomy 21) research.\n"
    "Evaluate the following summary based on three criteria and return a JSON object ONLY.\n"
    "Criteria (score 1-5):\n"
    "  accuracy:        Is the information factually correct and well-supported by evidence?\n"
    "  informativeness: Does it provide specific, meaningful information (not vague)?\n"
    "  completeness:    Does it cover the main aspects of the question without major omissions?\n"
    "Scoring guide (use the FULL 1-5 range; do NOT use a 1-10 scale): 5=Excellent  4=Good  3=Fair  2=Poor  1=Very Poor\n"
    "Output discipline: reply with the JSON object ONLY. No reasoning, no preamble, no markdown, no code fences.\n"
    'Return ONLY valid JSON: {"accuracy":<1-5>,"informativeness":<1-5>,"completeness":<1-5>,"justification":"<one concise sentence>"}'
)

_PIPELINE_IMPROVE_SYSTEM = (
    "You are an expert biomedical writer specialising in Down syndrome research.\n"
    "Rewrite the provided summary guided by the judge's feedback. Rules:\n"
    "  - 130-150 words (citations excluded from the count).\n"
    "  - Preserve ALL inline citations exactly ([PMCxxxxxxx:Section]); do NOT add or remove any.\n"
    "  - Address every point in JUDGE FEEDBACK; do not fabricate new facts.\n"
    "  - No em dashes; no repeated enumeration; no duplicate tokens in a row.\n"
    "  - Every finding sentence: citation BEFORE the period ('...upregulated [PMC:Results].'\n"
    "    NOT '...upregulated. [PMC:Results].' or '...upregulated.[PMC:Results].').\n"
    "  - CRITICAL: Reply with ONLY the rewritten prose. No headers, no preamble, no\n"
    "    'Here is...', 'Let me...', no code fences, no quotation marks."
)


def create_judge_improve_node(cfg: RAGConfig):
    """Pipeline node: auto-score the finalized summary, then rewrite it guided by judge notes."""

    _use_deepinfra = (cfg.provider_chat == "deepinfra")

    _ollama_base  = os.environ.get("OLLAMA_REMOTE_BASE_URL", "https://ollama.com")
    _ollama_key   = os.environ.get("OLLAMA_API_KEY", "")
    # Judge / improve models — DeepInfra when provider_chat=deepinfra, else Ollama remote.
    _judge_model  = cfg.deepinfra_judge_model if _use_deepinfra else "gpt-oss:120b-cloud"
    _improve_model = cfg.deepinfra_chat_model if _use_deepinfra else os.environ.get("NEMOTRON_CHAT_MODEL", "nemotron-3-super")
    # Whichever provider is active must have a credential set for the node to run.
    _judge_enabled_key = cfg.deepinfra_api_key if _use_deepinfra else _ollama_key

    def _get_client():
        from ollama import Client  # type: ignore[import-untyped]
        # Non-streaming calls: a scalar httpx timeout bounds the TOTAL call.
        return Client(host=_ollama_base, headers={"Authorization": f"Bearer {_ollama_key}"},
                      timeout=_ollama_call_settings()["timeout"])

    def _di_chat(model: str, system: str, user: str, seed: Optional[int] = None,
                 reasoning_effort: Optional[str] = None) -> str:
        """One-shot chat via DeepInfra's OpenAI-compatible endpoint. Returns the message content."""
        import requests as _requests
        url = cfg.deepinfra_base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if seed is not None:
            payload["seed"] = seed
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        resp = _requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {cfg.deepinfra_api_key}", "Content-Type": "application/json"},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _score(question: str, summary_plain: str) -> Optional[Dict[str, Any]]:
        try:
            if _use_deepinfra:
                raw = _di_chat(
                    _judge_model,
                    _PIPELINE_JUDGE_SYSTEM,
                    f"QUESTION:\n{question}\n\nSUMMARY:\n{summary_plain}",
                    reasoning_effort=cfg.deepinfra_reasoning_effort,  # gpt-oss reasons too — suppress for speed + clean JSON
                )
            else:
                # Bounded Ollama judge with retry-once, then DeepInfra fallback so a
                # slow judge can never hang the request.
                raw = None
                for _attempt in (1, 2):
                    _t = time.time()
                    try:
                        resp = _get_client().chat(
                            model=_judge_model,
                            messages=[
                                {"role": "system", "content": _PIPELINE_JUDGE_SYSTEM},
                                {"role": "user",   "content": f"QUESTION:\n{question}\n\nSUMMARY:\n{summary_plain}"},
                            ],
                            think=False,   # gpt-oss judge reasons too — disable for speed + clean JSON
                            options={"temperature": 0.0, "seed": 42, "top_k": 1, "top_p": 1.0},
                        )
                        raw = resp["message"]["content"].strip()
                        print(f"  [OLLAMA_TIMING] judge _score try{_attempt}: {time.time() - _t:.1f}s")
                        break
                    except Exception as e:
                        print(f"  [OLLAMA_WARN] _score try{_attempt} failed: {type(e).__name__}: {e}")
                if raw is None and cfg.deepinfra_api_key:
                    raw = _di_chat(
                        cfg.deepinfra_judge_model, _PIPELINE_JUDGE_SYSTEM,
                        f"QUESTION:\n{question}\n\nSUMMARY:\n{summary_plain}",
                        reasoning_effort=cfg.deepinfra_reasoning_effort,
                    )
                if raw is None:
                    return None
            m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            if not {"accuracy", "informativeness", "completeness", "justification"}.issubset(data):
                return None
            for k in ("accuracy", "informativeness", "completeness"):
                data[k] = max(1, min(5, int(data[k])))
            return data
        except Exception as exc:
            print(f"  [JUDGE_NODE_WARN] scoring failed: {exc}")
            return None

    def _improve(question: str, summary_with_cites: str, scores: Dict[str, Any]) -> Optional[str]:
        try:
            _labels = {1: "Very Poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
            weak = [
                f"{d.title()} ({scores[d]}/5 \u2014 {_labels.get(scores[d], '')})"
                for d in ("accuracy", "informativeness", "completeness")
                if scores.get(d, 5) < 4
            ]
            weak_str = ", ".join(weak) if weak else "general improvement"
            parts = [
                f"QUESTION:\n{question}",
                f"ORIGINAL SUMMARY:\n{summary_with_cites}",
                f"WEAK CRITERIA: {weak_str}",
            ]
            if scores.get("justification"):
                parts.append(f"JUDGE FEEDBACK:\n{scores['justification']}")
            parts.append("Please rewrite the summary to address the judge's feedback.")
            if _use_deepinfra:
                text = _di_chat(
                    _improve_model, _PIPELINE_IMPROVE_SYSTEM, "\n\n".join(parts), seed=42,
                    reasoning_effort=cfg.deepinfra_reasoning_effort,  # suppress Nemotron CoT in the rewrite
                )
            else:
                # Bounded Ollama rewrite with retry-once, then DeepInfra fallback.
                text = None
                for _attempt in (1, 2):
                    _t = time.time()
                    try:
                        resp = _get_client().chat(
                            model=_improve_model,
                            messages=[
                                {"role": "system", "content": _PIPELINE_IMPROVE_SYSTEM},
                                {"role": "user",   "content": "\n\n".join(parts)},
                            ],
                            think=False,   # nemotron rewrite — disable reasoning trace for speed
                            options={"temperature": 0.0, "seed": 42, "top_k": 1, "top_p": 1.0},
                        )
                        text = resp["message"]["content"].strip()
                        print(f"  [OLLAMA_TIMING] improve _rewrite try{_attempt}: {time.time() - _t:.1f}s")
                        break
                    except Exception as e:
                        print(f"  [OLLAMA_WARN] _improve try{_attempt} failed: {type(e).__name__}: {e}")
                        text = None
                if not text and cfg.deepinfra_api_key:
                    _cs = _ollama_call_settings()
                    text = _di_chat(
                        _cs["fallback_model"], _PIPELINE_IMPROVE_SYSTEM, "\n\n".join(parts), seed=42,
                        reasoning_effort=cfg.deepinfra_reasoning_effort,
                    )
            if not text:
                return None
            # Safety net: reject rewrites that leaked reasoning/preamble instead of a clean summary.
            # Falling back to None keeps the original (already clean) summary rather than splicing
            # a chain-of-thought dump into the user-visible answer.
            _head = text.lstrip("#> \n\t").lower()[:60]
            _leak_markers = ("we need to", "we must", "let me", "here is", "here's",
                             "the user", "first, i", "okay,", "i will", "i need to",
                             "to answer", "let's", "<think")
            if _head.startswith(_leak_markers):
                print("  [IMPROVE_NODE_WARN] rewrite looked like a reasoning dump — discarding, keeping original summary")
                return None
            return text
        except Exception as exc:
            print(f"  [IMPROVE_NODE_WARN] rewrite failed: {exc}")
            return None

    def judge_improve(state: RAGState) -> RAGState:
        t0 = _now_ms()
        _skip = {"judge_scores": None, "improved_summary": None}

        if not cfg.enable_judge:
            return {**state, **_skip, "trace": [{"agent": "judge_improve", "timing_ms": 0, "note": "disabled"}]}
        if not _judge_enabled_key:
            _miss = "DEEPINFRA_API_KEY not set" if _use_deepinfra else "OLLAMA_API_KEY not set"
            return {**state, **_skip, "trace": [{"agent": "judge_improve", "timing_ms": 0, "note": _miss}]}

        md = state.get("markdown", "")
        if "Insufficient Evidence" in md or "[EVIDENCE_GAP]" in md:
            return {**state, **_skip, "trace": [{"agent": "judge_improve", "timing_ms": 0, "note": "skipped: evidence gap"}]}

        # Extract summary section
        _m = re.search(r"#{1,2}\s*Summary(?:\s*\(.*?\))?\s*\n(.*?)(?=\n#{1,2}\s|\Z)", md, re.DOTALL | re.IGNORECASE)
        _sum_raw = (_m.group(1) if _m else md).strip()

        # Plain text (no links, no tags) for judge scoring
        _sum_plain = re.sub(r"\]\(https?://[^)]+\)", "]", _sum_raw)
        _sum_plain = re.sub(r"\[PMC[^\]]+\]", "", _sum_plain).strip()

        # Citation tags without URL part (for improve model to preserve)
        # Also strip ', INCLUDE' — the model mangles it into [[PMC...] INCLUDE] in its output
        _sum_cites = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"[\1]", _sum_raw).strip()
        _sum_cites = re.sub(r",\s*INCLUDE", "", _sum_cites)

        question = state.get("question", "")

        scores = _score(question, _sum_plain)
        if scores is None:
            return {**state, **_skip, "trace": [{"agent": "judge_improve", "timing_ms": _now_ms() - t0, "note": "scoring failed"}]}

        # Only rewrite when at least one criterion scores below 4.
        # If all criteria are ≥ 4 the summary is already good — skip the improve call.
        # Set SKIP_JUDGE_IMPROVE=true to always skip the rewrite (keeps scoring, saves ~80s).
        _skip_improve = os.environ.get("SKIP_JUDGE_IMPROVE", "").strip().lower() in ("1", "true", "yes", "y", "on")
        # Rewrite only when a criterion is genuinely poor (< 3), not merely "Good" (3).
        # The rewrite is a second hosted-120B call (~8 s); gating it at < 3 instead of
        # < 4 lets most answers skip it (typical latency ~11 s), paying the extra cost
        # only for genuinely weak answers. When it does fire, `weak` (built at < 4 above)
        # still guides it to fix every sub-4 criterion.
        _needs_improve = (not _skip_improve) and any(scores.get(k, 5) < 3 for k in ("accuracy", "informativeness", "completeness"))
        improved_raw = _improve(question, _sum_cites, scores) if _needs_improve else None
        if improved_raw:
            _all_hits = state.get("reranked_hits") or state.get("hits") or []
            improved_linked = _hyperlink_inline_pmc(
                _inject_include_tags(improved_raw, _all_hits)
            )
        else:
            improved_linked = None

        avg = sum(scores.get(k, 0) for k in ("accuracy", "informativeness", "completeness")) / 3
        _note = (
            f"scored avg={avg:.1f}/5; skipped: SKIP_JUDGE_IMPROVE=true"
            if _skip_improve
            else f"scored avg={avg:.1f}/5; all ≥3 — no rewrite needed"
            if not _needs_improve
            else f"scored avg={avg:.1f}/5; improved={'yes' if improved_linked else 'no'}"
        )
        return {
            **state,
            "judge_scores": scores,
            "improved_summary": improved_linked,
            "trace": [{"agent": "judge_improve", "timing_ms": _now_ms() - t0, "note": _note}],
        }

    return judge_improve


# =============================================================================
# LangGraph Pipeline Builder
# =============================================================================
def build_rag_graph(  # type: ignore[return]
    cfg: RAGConfig,
    retriever: LangChainMilvusRetriever,
    llm,
    nemotron_reranker,
):
    """Build the LangGraph RAG pipeline."""

    # Create the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("question_judge", create_question_judge_node(llm, cfg))
    workflow.add_node("end_invalid", create_invalid_end_node())
    workflow.add_node("retrieve", create_retrieval_node(retriever, cfg))
    workflow.add_node("rerank", create_nemotron_reranker_node(nemotron_reranker, cfg))
    workflow.add_node("synthesize", create_synthesis_node(llm, cfg))
    workflow.add_node("raw_evidence", create_raw_evidence_node(cfg))
    workflow.add_node("safety_guard", create_safety_guard_node(cfg))
    workflow.add_node("finalize", create_final_node(cfg, llm))

    # Define edges — judge first, then conditional branch
    workflow.set_entry_point("question_judge")
    workflow.add_conditional_edges("question_judge", _judge_route, {
        "retrieve": "retrieve",
        "end_invalid": "end_invalid",
    })
    workflow.add_edge("end_invalid", END)
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "synthesize")
    workflow.add_edge("synthesize", "raw_evidence")
    workflow.add_edge("raw_evidence", "safety_guard")
    workflow.add_edge("safety_guard", "finalize")
    workflow.add_node("judge_improve", create_judge_improve_node(cfg))
    workflow.add_edge("finalize", "judge_improve")
    workflow.add_edge("judge_improve", END)

    return workflow.compile()


# =============================================================================
# Main RAG Class
# =============================================================================
class LangChainMultiAgentRAG:
    """LangChain-based Multi-Agent RAG orchestrator."""

    def __init__(self, cfg: RAGConfig):
        cfg.require()
        self.cfg = cfg

        # Initialize LangChain components
        self.embeddings = make_langchain_embeddings(cfg)
        self.retriever = LangChainMilvusRetriever(cfg, self.embeddings)

        # Create LLM for synthesis
        self.llm = make_langchain_llm(cfg, role="chat")

        # Load Nemotron cross-encoder reranker
        device = cfg.nemotron_device if cfg.nemotron_device else None
        if cfg.provider_rerank == "deepinfra":
            if not cfg.deepinfra_api_key:
                raise RuntimeError("DEEPINFRA_API_KEY not set while PROVIDER_RERANK=deepinfra")
            self.nemotron_reranker = DeepInfraReranker(
                api_key=cfg.deepinfra_api_key,
                url=cfg.deepinfra_rerank_url,
            )
            print(f"[+] Reranker: DeepInfra API ({cfg.deepinfra_rerank_url})")
        else:
            self.nemotron_reranker = NemotronCrossEncoderReranker(
                model_name=cfg.nemotron_rerank_model,
                batch_size=cfg.nemotron_rerank_batch_size,
                device=device,
            )

        # Build the graph
        self.graph = build_rag_graph(cfg, self.retriever, self.llm, self.nemotron_reranker)

        print(f"[+] LangChain RAG initialized: chat={cfg.provider_chat}, embed={cfg.provider_embed}")

    def _no_answer(self, reason: str, trace=None):
        _no_answer_reasoning = (
            "\n\n# Reasoning\n"
            "- **Evidence selection:** No relevant passages were retrieved from the indexed corpus.\n"
            "- **Reasoning chain:** Without retrievable evidence, a citation-backed synthesis cannot be constructed.\n"
            "- **Confidence & gaps:** This reflects a corpus coverage limitation, not necessarily the absence of published research.\n"
            "- **Reader caveats:** Try rephrasing the question or consulting PubMed directly for broader coverage.\n"
        )
        return {
            "markdown": f"# Summary\n{self.cfg.no_answer_text}\n" + _no_answer_reasoning,
            "structured": None,
            "hits": [],
            "re_ranked_hits": [],
            "citations": [],
            "safety": {"flagged": False, "patterns": []},
            "raw_evidence": [],
            "trace": trace or [],
        }

    def _advice_seeking_response(self, question: str, trace=None):
        """Decline response for personal treatment-advice-seeking questions.

        No retrieval or synthesis is run; the response explains that the tool
        reports published research only and suggests a research-framed rephrase.
        """
        suggestion = _research_rephrase_suggestion(question)
        msg = (
            "# Summary\n"
            "This tool cannot recommend or prescribe any drug or treatment. Your question asks "
            "for personal treatment advice, which is outside the scope of this research "
            "assistant. It summarizes findings from the peer-reviewed Down syndrome (Trisomy 21) "
            "literature; it does not provide medical advice.\n\n"
            f"To see what the published literature reports on this topic, rephrase it as a "
            f"research question, for example: \"{suggestion}\"\n\n"
            "For any treatment decision, please consult a qualified clinician.\n"
            "\n# Reasoning\n"
            "- **Evidence selection:** No evidence was retrieved; the question was declined "
            "before retrieval because it requests personal treatment advice.\n"
            "- **Reasoning chain:** This tool reports what the published Down syndrome literature "
            "says; it cannot recommend, prescribe, or advise on individual treatment.\n"
            "- **Confidence & gaps:** The underlying research topic may be well covered in the "
            "corpus; rephrasing as a research question will surface the published evidence.\n"
            "- **Reader caveats:** Consult a qualified clinician for any diagnosis or treatment "
            "decision.\n"
        )
        return {
            "markdown": msg,
            "structured": None,
            "hits": [],
            "re_ranked_hits": [],
            "citations": [],
            "safety": {"flagged": True, "patterns": ["advice_seeking_question"]},
            "raw_evidence": [],
            "trace": trace or [],
        }

    def _evidence_gap_response(self, trace=None, missing_terms: Optional[List[str]] = None, question: Optional[str] = None, judge_reason: Optional[str] = None):
        # If the question judge already supplied a reason, use it directly — no extra LLM call.
        if judge_reason:
            justification = judge_reason
        else:
            # Generate a short question-specific justification via a lightweight LLM call.
            justification = ""
            if question and self.llm is not None:
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage
                    _sys = SystemMessage(content=(
                        "You are a scientific assistant. In 1–2 sentences, explain concisely and factually "
                        "WHY the following research question cannot be fully answered using the Down syndrome "
                        "(Trisomy 21) peer-reviewed literature corpus. "
                        "Distinguish between two cases: "
                        "(1) If the question's premise is biologically incorrect or the topic is genuinely "
                        "outside the scope of DS/Trisomy 21 research, state that biological reason directly. "
                        "(2) If the question is scientifically valid and plausible within DS research but the "
                        "specific aspect is simply not well-covered in this particular indexed corpus, say so "
                        "honestly — e.g. 'this specific aspect of DS research is not well-represented in the "
                        "indexed corpus of publications.' Do not state a specific paper count — it is not "
                        "necessary for the justification and would drift out of date as the corpus grows. "
                        "Do NOT fabricate biological facts or claim a phenomenon does not exist purely "
                        "because it is absent from this corpus. "
                        "Be direct and informative. Output only the justification sentence(s), no preamble."
                    ))
                    _human = HumanMessage(content=f"Question: {question}")
                    _resp = self.llm.invoke([_sys, _human])
                    justification = (_resp.content if hasattr(_resp, "content") else str(_resp)).strip()
                except Exception:
                    justification = ""

        _reasoning_section = (
            "\n\n# Reasoning\n"
            "- **Evidence selection:** No sufficiently relevant passages were retrieved from "
            "the indexed corpus for this question.\n"
            "- **Reasoning chain:** Without retrievable evidence, a citation-backed synthesis "
            "cannot be constructed.\n"
            "- **Confidence & gaps:** This response reflects a corpus coverage limitation, "
            "not necessarily the absence of published research on the topic.\n"
            "- **Reader caveats:** Try rephrasing the question or consulting PubMed directly "
            "for broader literature coverage.\n"
        )
        if justification:
            msg = (
                "# Summary\n"
                "The INCLUDE Publication Corpus does not contain sufficient evidence to answer "
                "this question.\n\n"
                f"**Why:** {justification}\n\n"
                "Please try rephrasing your question or asking about a different aspect of Down syndrome research.\n"
                + _reasoning_section
            )
        else:
            msg = (
                "# Summary\n"
                "The INCLUDE Publication Corpus does not contain sufficient evidence to answer "
                "this question. This tool is designed for research questions about Down syndrome "
                "(Trisomy 21) supported by the indexed peer-reviewed literature. Please try "
                "rephrasing your question or asking about a different aspect of Down syndrome research.\n"
                + _reasoning_section
            )
        return {
            "markdown": msg,
            "structured": None,
            "hits": [],
            "re_ranked_hits": [],
            "citations": [],
            "safety": {"flagged": False, "patterns": []},
            "raw_evidence": [],
            "trace": trace or [],
        }

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        rerank_alpha_override: Optional[float] = None,
        sections_override: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the RAG pipeline."""

        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "top_k": top_k or self.cfg.top_k_default,  # dynamic boost applied inside retrieval node
            "hits": [],
            "reranked_hits": [],
            "deduped_hits": [],
            "markdown": "",
            "structured": None,
            "citations": [],
            "raw_evidence": [],
            "safety_flagged": False,
            "safety_patterns": [],
            "missing_terms": [],
            "unmatched_entities": [],
            "question_verdict": "VALID",
            "trace": [],
            "error": None,
            "judge_scores": None,
            "improved_summary": None,
            "rerank_alpha_override": rerank_alpha_override,
            "sections_override": sections_override,
        }

        # Run the graph
        try:
            result = self.graph.invoke(initial_state)
        except Exception as e:
            import traceback
            _tb = traceback.format_exc()
            try:
                with open("/tmp/rag_pipeline_error.txt", "a") as _f:
                    _f.write(f"\n=== PIPELINE ERROR q={question!r} top_k={top_k} ===\n{_tb}\n")
            except Exception:
                pass
            return self._no_answer(f"Pipeline error: {str(e)}")

        # Question judge: if the LLM pre-flight classified this as INVALID, use the
        # judge's reason directly as the evidence-gap justification (skip all other checks).
        _verdict = result.get("question_verdict", "VALID")
        if _verdict.upper().startswith("INVALID_ADVICE"):
            # Personal treatment-advice-seeking question: dedicated decline +
            # rephrase suggestion (NOT the evidence-gap framing — the topic may
            # be well covered; the phrasing, not the corpus, is the problem).
            return self._advice_seeking_response(question, trace=result.get("trace", []))
        if _verdict.upper().startswith("INVALID"):
            # Extract the reason after "INVALID: "
            _reason = _verdict[len("INVALID"):].lstrip(": ").strip() or None
            return self._evidence_gap_response(
                trace=result.get("trace", []),
                question=question,
                judge_reason=_reason,
            )

        # In strict mode, enforce minimum hits and similarity thresholds
        if self.cfg.strict_mode:
            if not result.get("hits") or len(result["hits"]) < self.cfg.min_hits:
                return self._no_answer("no_hits", trace=result.get("trace", []))

            hits = result.get("hits", [])
            if hits:
                # Use top-hit score instead of average: broad meta-questions retrieve
                # many only-somewhat-relevant papers, pulling the average below threshold
                # even when the best hit clearly addresses the question.
                top_sim = float(hits[0].get("score", 0))
                if top_sim < self.cfg.min_avg_sim:
                    return self._no_answer(f"low_similarity<{self.cfg.min_avg_sim}", trace=result.get("trace", []))

        # Evidence-first rule: if key question terms are absent from evidence, report missing evidence.
        # Threshold = 1: a single missing content term (e.g. "chloroplast", "BRCA1") is sufficient
        # to flag an evidence gap.  Meta-research words ("dysfunction", "affect", "mechanism" etc.)
        # are already excluded by the stoplist in _extract_key_terms, so false positives from
        # question framing words are not a concern.
        missing_terms = result.get("missing_terms") or []
        if missing_terms:
            return self._evidence_gap_response(trace=result.get("trace", []), missing_terms=missing_terms, question=question)

        # Check for Evidence Gap: if the LLM flagged insufficient evidence, do not
        # return the partial summary — ask the user to rephrase instead.
        _md = result.get("markdown", "")
        if "[EVIDENCE_GAP]" in _md:
            return self._evidence_gap_response(trace=result.get("trace", []), question=question)

        # Self-negating summary detector: catch cases where the LLM explicitly
        # admits the key concept is absent or inapplicable (e.g. out-of-scope
        # questions like "chloroplast dysfunction in DS") but generates a bridging
        # response anyway instead of flagging [EVIDENCE_GAP].
        # Pattern: Summary contains phrases that directly say the topic is missing/inapplicable.
        _summ_m = re.search(r"(?im)^#\s+Summary\s*\n(.*?)(?=^#\s+|\Z)", _md, re.S)
        if _summ_m:
            _summ_body = _summ_m.group(1).lower()
            _self_negate_pats = [
                # Topic absent from evidence corpus
                r'\b(was not|were not|is not|are not) (directly )?(addressed|mentioned|discussed|covered|found|present) in\b',
                r'\b(no explicit|no direct) (connection|link|relationship|mention|reference|evidence)\b',
                r'\bremains unsupported by\b',
                r'\bany inference.{0,60}(remains|is) (unsupported|speculative|unsubstantiated)\b',
                r'\b(not|never) (discussed|mentioned|addressed|covered) in the (provided|available|retrieved|given)\b',
                r'\b(cannot|can\'t) (be )?(directly )?(answered|addressed) (from|by|with|using) the (provided|available|retrieved)\b',
                r'\bno (evidence|information|data) (in the|from the|within the) (provided|available|retrieved|given)\b',
                # Topic is biologically inapplicable to humans / DS
                r'\bnot applicable to human\b',
                r'\bfound exclusively in (plant|algal|bacterial|prokaryo|fungi)\b',
                r'\babsent (in|from) human (cell|biology|tissue|organ)\b',
                r'\bno scientific evidence linking\b',
                r'\borganelles? (found|present|occur) (only |exclusively )?(in|within) plant\b',
                r'\b(humans?|human cells?|mammal) (do|does|don\'t|doesn\'t) (not )?(have|contain|possess) (chloroplast|plastid|thylakoid)\b',
                r'\bthis (concept|organelle|process|pathway) (does not|is not|has no) (exist|apply|occur|relevance) in human\b',
                r'\bc(annot|an\'t) (directly )?affect.{0,60}(down syndrome|human (condition|biology|cell|physiology))\b',
            ]
            if any(re.search(p, _summ_body) for p in _self_negate_pats):
                return self._evidence_gap_response(trace=result.get("trace", []), question=question)

        # Format output
        return {
            "markdown": result.get("markdown", ""),
            "structured": result.get("structured"),
            "hits": result.get("hits", []),
            "re_ranked_hits": result.get("reranked_hits", []),
            "citations": result.get("citations", []),
            "safety": {
                "flagged": result.get("safety_flagged", False),
                "patterns": result.get("safety_patterns", []),
            },
            "raw_evidence": result.get("raw_evidence", []),
            "trace": result.get("trace", []),
            # Expose the judge-improved summary text so main.py can splice it in.
            # judge_improve stores this in state["improved_summary"]; if the judge
            # decided no rewrite was needed it will be None.
            "improved_summary": result.get("improved_summary"),
        }


# =============================================================================
# Backward Compatibility Alias
# =============================================================================
MultiAgentRAG = LangChainMultiAgentRAG


# =============================================================================
# CLI
# =============================================================================
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="LangChain Multi-Agent RAG for DS-Chat")
    ap.add_argument("--question", "-q", type=str, required=True, help="Question to answer")
    ap.add_argument("--top-k", type=int, default=None, dest="top_k", help="Number of results")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    args = ap.parse_args()

    cfg = RAGConfig().override_with_env()
    rag = LangChainMultiAgentRAG(cfg)
    out = rag.answer(args.question, top_k=args.top_k)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out["markdown"])


if __name__ == "__main__":
    _cli()
