"""
PaperQA2 adapter for the T21 benchmark.

Runs PaperQA2 (https://github.com/future-house/paper-qa) over the SAME 1,602 papers
the T21 Research Assistant indexes, with the SAME generator model, so that retrieval
and answering machinery is the only thing that differs between the arms.

PaperQA2 is a closer comparison than MedRAG: it is built for full-text scientific
papers, does its own chunking, retrieval, per-chunk evidence summarisation and cited
synthesis, and is agentic (it can issue several searches per question). What it does
NOT do is prioritise Results/Methods over background text, which is the T21 claim
under test.

Wiring notes:

- PaperQA2 has THREE LLM roles (llm, summary_llm, agent.agent_llm). All three are
  pointed at nemotron-3-super so no part of the pipeline silently falls back to GPT-4o,
  which is the shipped default and would confound the comparison with model quality.
- The generator reaches Ollama Cloud through LiteLLM's openai-compatible provider with
  a per-model api_base, rather than by setting OPENAI_API_BASE globally. A global
  override would also redirect the embedding calls to ollama.com, which does not serve
  OpenAI embedding models.
- Embeddings stay on PaperQA2's shipped default (text-embedding-3-small) so this arm is
  "PaperQA2 as published". That is deliberate: the arm answers "how does an off-the-shelf
  scientific RAG system do on this corpus", not "does section-awareness help in
  isolation".
- parsing.use_doc_details is disabled. It triggers Crossref/Semantic Scholar metadata
  lookups per document, which are slow, rate-limited, and unnecessary here because the
  citation header is already embedded at the top of every text file.

Usage:
    from paperqa_adapter import make_settings, ask_one
    s = make_settings()
    resp = ask_one("What is the prevalence of ...?", s)
"""
import os
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SRV = BENCH.parent.parent / "srv"
CORPUS = BENCH / "paperqa_corpus"
INDEX_DIR = BENCH / "paperqa_index"

OLLAMA_BASE = "https://ollama.com/v1"


def _env():
    from dotenv import dotenv_values
    return dotenv_values(SRV / ".env")


def _llm_config(model: str, api_base: str, api_key: str) -> dict:
    """LiteLLM router config pinning api_base/api_key to this model only."""
    return {
        "model_list": [{
            "model_name": model,
            "litellm_params": {
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "temperature": 0.0,
            },
        }],
    }


def quiet_litellm(input_cost_per_token: float = 0.0, output_cost_per_token: float = 0.0,
                  max_tokens: int = 32768):
    """Stop LiteLLM's cost-tracking noise for models it does not know.

    LiteLLM prices every call against a built-in table. nemotron-3-super is served
    through an OpenAI-compatible endpoint but is not in that table, so each call logs
    "Failed to calculate cost ... This model isn't mapped yet" plus a provider list.
    Harmless, but during a 1,602-paper index it prints thousands of times and floods the
    notebook. Registering the model with explicit costs removes the cause rather than
    hiding it; set real per-token prices here if you want cost accounting to work.
    """
    import logging
    import litellm

    env = _env()
    model = (env.get("NEMOTRON_CHAT_MODEL") or "nemotron-3-super")
    entry = {
        "max_tokens": max_tokens,
        "max_input_tokens": max_tokens,
        "max_output_tokens": 8192,
        "input_cost_per_token": input_cost_per_token,
        "output_cost_per_token": output_cost_per_token,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    try:
        litellm.register_model({model: entry, f"openai/{model}": entry})
    except Exception as exc:
        print(f"[warn] could not register {model} with litellm: {str(exc)[:80]}")
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    for name in ("LiteLLM", "litellm", "litellm.utils", "litellm.cost_calculator"):
        logging.getLogger(name).setLevel(logging.ERROR)


def make_settings(paper_directory: Path = CORPUS,
                  index_directory: Path = INDEX_DIR,
                  evidence_k: int = 12,
                  answer_max_sources: int = 8,
                  provider: str = "ollama_remote",
                  embed_provider: str = "openai"):
    """Settings for PaperQA2 over the T21 corpus with the shared generator.

    embed_provider:
      "openai" (default) keeps PaperQA2's shipped text-embedding-3-small, so the arm is
        "PaperQA2 as published" and answers how an off-the-shelf scientific RAG system
        performs on this corpus.
      "t21" swaps in the exact embedder the T21 pipeline uses
        (nvidia/llama-nemotron-embed-vl-1b-v2, 2048-dim, via DeepInfra's
        OpenAI-compatible endpoint). Combined with the shared generator this makes the
        arm a tight ablation: same corpus, same embedder, same LLM, so the only
        remaining differences are PaperQA2's chunking, its per-chunk evidence
        summarisation, and the absence of section prioritisation and reranking.

    Note the two routes that do NOT work, verified 2026-09-02: LiteLLM's native
    deepinfra provider rejects this embedding model as unmapped, and Ollama Cloud
    serves no /v1/embeddings endpoint at all (404). The OpenAI-compatible path below
    is the one that works, and it is the same path embed_and_insert.py uses.
    """
    from paperqa import Settings
    from paperqa.settings import AgentSettings, AnswerSettings, IndexSettings, ParsingSettings

    env = _env()
    if provider == "ollama_remote":
        key = env.get("OLLAMA_API_KEY")
        if not key:
            raise RuntimeError("OLLAMA_API_KEY missing from srv/.env")
        model = "openai/" + (env.get("NEMOTRON_CHAT_MODEL") or "nemotron-3-super")
        base = OLLAMA_BASE
    elif provider == "deepinfra":
        key = env.get("DEEPINFRA_API_KEY")
        if not key:
            raise RuntimeError("DEEPINFRA_API_KEY missing from srv/.env")
        model = "openai/openai/gpt-oss-120b"
        base = "https://api.deepinfra.com/v1/openai"
    else:
        raise ValueError(provider)

    if embed_provider == "openai":
        # PaperQA2's shipped default, so the arm is "PaperQA2 as published"
        embedding, embedding_config = "text-embedding-3-small", None
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing from srv/.env")
        os.environ.setdefault("OPENAI_API_KEY", env["OPENAI_API_KEY"])
    elif embed_provider == "t21":
        di_key = env.get("DEEPINFRA_API_KEY")
        if not di_key:
            raise RuntimeError("DEEPINFRA_API_KEY missing from srv/.env")
        embed_model = env.get("DEEPINFRA_EMBED_MODEL",
                              "nvidia/llama-nemotron-embed-vl-1b-v2")
        di_base = env.get("DEEPINFRA_BASE_URL",
                          "https://api.deepinfra.com/v1/openai").rstrip("/")
        # "openai/<model>" plus api_base is the OpenAI-compatible route; the kwargs are
        # forwarded straight to litellm.aembedding by LiteLLMEmbeddingModel.
        embedding = f"openai/{embed_model}"
        embedding_config = {"kwargs": {"api_base": di_base, "api_key": di_key}}
    else:
        raise ValueError(f"unknown embed_provider: {embed_provider!r}")

    quiet_litellm()          # suppress the unmapped-model cost warnings, see above

    cfg = _llm_config(model, base, key)
    return Settings(
        embedding=embedding, embedding_config=embedding_config,
        llm=model, llm_config=cfg,
        summary_llm=model, summary_llm_config=cfg,
        agent=AgentSettings(agent_llm=model, agent_llm_config=cfg,
                            index=IndexSettings(paper_directory=str(paper_directory),
                                                index_directory=str(index_directory))),
        answer=AnswerSettings(evidence_k=evidence_k,
                              answer_max_sources=answer_max_sources),
        parsing=ParsingSettings(use_doc_details=False),
        temperature=0.0,
    )


def ask_one(question: str, settings=None):
    """Agentic path. Currently unusable: paper-qa 2026.8.12 calls
    LiteLLMModel.get_router(), which fhlmi 1.0.7 removed, so the ToolSelector agent
    raises AttributeError. Kept for when the dependency pin is fixed upstream.
    """
    from paperqa import ask
    return ask(question, settings=settings or make_settings())


def _run(coro):
    """Run a coroutine from either a script or a live notebook.

    Jupyter already owns a running event loop, so asyncio.run() raises
    "cannot be called from a running event loop". nest_asyncio makes the existing
    loop re-entrant, which is the standard fix and keeps one code path for both
    contexts rather than forcing the notebook to use top-level await.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)          # plain script: no loop yet
    try:
        import nest_asyncio
    except ImportError as exc:
        raise RuntimeError(
            "running inside an event loop (Jupyter); pip install nest_asyncio"
        ) from exc
    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


def build_docs(settings, paper_directory: Path = CORPUS, limit: int | None = None,
               progress_every: int = 100, concurrency: int = 8):
    """Index the corpus with PaperQA2's core (non-agentic) Docs API.

    This is the deterministic path: chunk, embed, retrieve, summarise evidence per
    chunk, then synthesise a cited answer. It sidesteps the agent tool-selector, which
    is broken by the fhlmi version mismatch, and it is the better choice for a
    benchmark regardless: the agent issues a variable number of searches per question,
    which would add run-to-run variance to an arm that is supposed to be a fixed
    comparator.

    Build once and reuse across all benchmark questions.
    """
    from paperqa import Docs

    files = sorted(Path(paper_directory).glob("*.txt"))
    if limit:
        files = files[:limit]

    docs = Docs()

    async def _load():
        """Index with bounded concurrency.

        Sequential indexing measured 8.6 s per paper, nearly all of it network
        latency (LLM-free: just parsing plus embedding calls), which extrapolates
        to about 4 hours for 1,765 papers. Papers are independent, so a semaphore
        of `concurrency` papers in flight cuts this to roughly the sequential time
        divided by the concurrency, while staying inside one event loop (asyncio
        interleaves only at await points, and each paper gets a unique docname, so
        Docs state updates do not conflict).
        """
        import asyncio as _aio
        sem = _aio.Semaphore(concurrency)
        done = {"ok": 0, "failed": 0}

        async def _one(f):
            async with sem:
                try:
                    await docs.aadd(f, settings=settings)
                    done["ok"] += 1
                except Exception as exc:
                    done["failed"] += 1
                    if done["failed"] <= 5:
                        print(f"  [warn] {f.name}: {str(exc)[:90]}", flush=True)
                n = done["ok"] + done["failed"]
                if n % progress_every == 0:
                    print(f"  indexed {n}/{len(files)} (ok {done['ok']}, "
                          f"failed {done['failed']})", flush=True)

        await _aio.gather(*(_one(f) for f in files))
        return done["ok"], done["failed"]

    added, failed = _run(_load())
    print(f"PaperQA2 corpus indexed: {added} papers ({failed} failed)")
    return docs


def query_docs(docs, question: str, settings):
    """Ask one open-ended question against a prebuilt Docs index."""
    return _run(docs.aquery(question, settings=settings))


def answer_text(resp) -> str:
    """Extract the answer string across PaperQA2 return shapes."""
    session = getattr(resp, "session", resp)
    return (getattr(session, "answer", "") or "").strip()
