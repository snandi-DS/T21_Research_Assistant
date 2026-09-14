"""
MedRAG adapter for the T21 benchmark.

Runs the MedRAG toolkit (https://github.com/Teddy-XiongGZ/MedRAG) as a controlled
general-biomedical RAG baseline, answering the same open-ended questions the T21
Research Assistant is evaluated on, with the same generator model, so that retrieval
is the only variable between the two arms.

Three adaptations are required and each is deliberate:

1. Free-text output. MedRAG ships multiple-choice templates by default
   ("answer a multi-choice medical question ... output json with answer_choice"),
   which would be meaningless against open-ended expert reference answers. The
   toolkit already provides simple_medrag_system / simple_medrag_prompt for
   non-MCQ use; we swap those in and pass options=None.

2. Shared generator. MedRAG's OpenAI path honours openai.api_base, so it is pointed
   at the same model the T21 pipeline uses (nemotron-3-super via the Ollama cloud
   OpenAI-compatible endpoint), or at DeepInfra. Holding the generator constant is
   what makes this a controlled comparison rather than a model comparison.

3. Context budget. MedRAG only sets max_length / context_length for model names
   containing "gpt-3.5" or "gpt-4"; any other name leaves both attributes unset and
   raises AttributeError during retrieval. We set them explicitly.

Usage:
    from medrag_adapter import build_medrag, answer_freetext
    mr = build_medrag(corpus="Textbooks", retriever="MedCPT")
    text, snippets = answer_freetext(mr, "What is the prevalence of ...?", k=32)
"""
import os
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
MEDRAG_DIR = BENCH_DIR / "MedRAG"
SRV_DIR = BENCH_DIR.parent.parent / "srv"


def _load_env():
    """Read API credentials from the project's .env (never hardcode keys)."""
    from dotenv import dotenv_values
    return dotenv_values(SRV_DIR / ".env")


def configure_generator(provider: str = "ollama_remote"):
    """Point MedRAG's OpenAI client at the same LLM the T21 pipeline uses.

    provider="ollama_remote" -> nemotron-3-super via https://ollama.com/v1
                                (identical to the T21 synthesis model)
    provider="deepinfra"     -> gpt-oss-120b via DeepInfra, supports
                                reasoning_effort=none for faster answers
    Returns the model name to pass as llm_name suffix.
    """
    env = _load_env()
    if provider == "ollama_remote":
        key = env.get("OLLAMA_API_KEY")
        if not key:
            raise RuntimeError("OLLAMA_API_KEY missing from srv/.env")
        os.environ["OPENAI_API_BASE"] = "https://ollama.com/v1"
        os.environ["OPENAI_API_KEY"] = key
        model = env.get("NEMOTRON_CHAT_MODEL") or "nemotron-3-super"
    elif provider == "deepinfra":
        key = env.get("DEEPINFRA_API_KEY")
        if not key:
            raise RuntimeError("DEEPINFRA_API_KEY missing from srv/.env")
        os.environ["OPENAI_API_BASE"] = "https://api.deepinfra.com/v1/openai"
        os.environ["OPENAI_API_KEY"] = key
        model = "openai/gpt-oss-120b"
    else:
        raise ValueError(f"unknown provider: {provider}")
    os.environ["OPENAI_API_TYPE"] = "openai"
    return model


def build_medrag(corpus: str = "Textbooks", retriever: str = "MedCPT",
                 provider: str = "ollama_remote", rag: bool = True,
                 context_length: int = 30000):
    """Construct a MedRAG instance wired for open-ended answering."""
    model = configure_generator(provider)
    sys.path.insert(0, str(MEDRAG_DIR))
    os.chdir(MEDRAG_DIR)                      # MedRAG resolves ./corpus relatively
    from src.medrag import MedRAG
    from src.template import simple_medrag_system, simple_medrag_prompt

    # Auto-register corpus names not in MedRAG's static table (e.g. dated rebuilds
    # like "T21_2026_09_02" living in corpus/t21_2026_09_02/chunk; the directory is the
    # lowercased corpus name). MedRAG loads its utils module TWICE under two names:
    # medrag.py does "from utils import ..." (top-level, via its own sys.path insert of
    # src/), while external code sees it as the package module "src.utils". These are
    # distinct module objects with distinct corpus_names dicts, and RetrievalSystem's
    # assert reads the top-level one, so the name must be registered in BOTH or the
    # registration silently misses the copy that matters.
    import sys as _sys
    for _name in ("utils", "src.utils"):
        _mod = _sys.modules.get(_name)
        if _mod is not None and hasattr(_mod, "corpus_names"):
            _mod.corpus_names.setdefault(corpus, [corpus.lower()])

    mr = MedRAG(llm_name=f"OpenAI/{model}", rag=rag,
                retriever_name=retriever, corpus_name=corpus,
                db_dir=str(MEDRAG_DIR / "corpus"))

    # adaptation 3: MedRAG leaves these unset for non-GPT model names
    mr.max_length = context_length + 2000
    mr.context_length = context_length

    # adaptation 1: replace the multiple-choice templates with free-text ones
    mr.templates["medrag_system"] = simple_medrag_system
    mr.templates["medrag_prompt"] = simple_medrag_prompt
    return mr


def answer_freetext(mr, question: str, k: int = 32, **kwargs):
    """Ask an open-ended question. Returns (answer_text, retrieved_snippets)."""
    result = mr.answer(question=question, options=None, k=k, **kwargs)
    answer, snippets = result[0], result[1]
    text = answer[0] if isinstance(answer, list) else answer
    return re.sub(r"\s+", " ", str(text)).strip(), snippets
