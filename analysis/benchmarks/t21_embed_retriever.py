"""
Give MedRAG the SAME embedding model the T21 Research Assistant uses.

Motivation. With MedRAG's default MedCPT retriever, the two arms differ in several
ways at once (embedding model, chunking, section prioritization, reranking), so a
gap in answer quality cannot be attributed to any one of them. Swapping in T21's
embedder removes the biggest confound and leaves the comparison much closer to the
controlled ablation the reviewer asked for:

    shared : corpus (same 1,602 papers), embedding model, generator (nemotron-3-super)
    differs: flat ~1000-char chunks vs section-aware 1200/200 chunks,
             no section prioritization, no cross-encoder rerank, no multi-agent pipeline

Implementation notes.

- Embeddings come from DeepInfra's nvidia/llama-nemotron-embed-vl-1b-v2 (2048-dim),
  read from srv/.env, i.e. the exact model and endpoint embed_and_insert.py uses.
- Encoding is SYMMETRIC: corpus chunks and queries both go through the same call with
  no instruction prefix, matching T21's DeepInfraEmbeddings. Asymmetric encoding here
  would silently degrade the baseline and make the comparison unfair.
- Only `content` is embedded, not "title + content". MedRAG's other retrievers prepend
  the title, but our titles are bare PMCIDs, which would inject identifier noise that
  T21's embedder never sees.
- MedRAG's retrieval logic is left untouched: the .npy shards and faiss.index are
  written in exactly the layout construct_index() expects (IndexFlatIP, inner product,
  same metric family T21 uses in Milvus), so search, RRF and snippet extraction all
  behave normally.

Usage:
    conda run -n medrag python t21_embed_retriever.py          # build the index
    # then, in the adapter: build_medrag(retriever="T21Embed", corpus="T21")
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

BENCH = Path(__file__).resolve().parent
MEDRAG_DIR = BENCH / "MedRAG"
SRV = BENCH.parent.parent / "srv"

RETRIEVER_ID = "t21-deepinfra-embed"     # index dir name; not a HuggingFace repo
CHUNK_DIR = MEDRAG_DIR / "corpus" / "t21" / "chunk"
INDEX_DIR = MEDRAG_DIR / "corpus" / "t21" / "index" / RETRIEVER_ID
EMBED_DIM = 2048


def _env():
    from dotenv import dotenv_values
    return dotenv_values(SRV / ".env")


class DeepInfraEncoder:
    """SentenceTransformer-compatible encoder backed by T21's embedding endpoint.

    Exposes .encode(texts) -> np.ndarray so it can be dropped into MedRAG's
    Retriever.embedding_function without touching retrieval code.
    """

    def __init__(self, batch_size=None, max_retries=5):
        env = _env()
        self.model = env.get("DEEPINFRA_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
        base = env.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
        self.url = base + "/embeddings"
        key = env.get("DEEPINFRA_API_KEY")
        if not key:
            raise RuntimeError("DEEPINFRA_API_KEY missing from srv/.env")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.batch_size = int(batch_size or env.get("DEEPINFRA_EMBED_BATCH_SIZE", 32))
        self.max_retries = max_retries

    def _post(self, batch):
        payload = {"input": batch, "model": self.model, "encoding_format": "float"}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, json=payload, headers=self.headers, timeout=180)
                if r.status_code in (429, 500, 502, 503):
                    last = f"HTTP {r.status_code}"
                    time.sleep(min(60, 3 * 2 ** attempt))
                    continue
                r.raise_for_status()
                rows = sorted(r.json()["data"], key=lambda d: d["index"])
                vecs = [row["embedding"] for row in rows]
                # guard against a short response silently dropping chunks
                if len(vecs) != len(batch):
                    raise RuntimeError(f"expected {len(batch)} embeddings, got {len(vecs)}")
                return vecs
            except requests.RequestException as exc:
                last = str(exc)[:150]
                time.sleep(min(60, 3 * 2 ** attempt))
        raise RuntimeError(f"embedding request failed after {self.max_retries} tries: {last}")

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else " ".join(t) for t in texts]   # MedCPT passes [title, content]
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(texts[i:i + self.batch_size]))
        return np.asarray(out, dtype=np.float32)

    def eval(self):
        return self          # SentenceTransformer API compatibility


def build_index():
    import faiss
    enc = DeepInfraEncoder()
    save_dir = INDEX_DIR / "embedding"
    save_dir.mkdir(parents=True, exist_ok=True)

    fnames = sorted(f for f in os.listdir(CHUNK_DIR) if f.endswith(".jsonl"))
    print(f"chunk files: {len(fnames)} | model: {enc.model} | batch: {enc.batch_size}")

    total = 0
    for fname in fnames:
        out_path = save_dir / fname.replace(".jsonl", ".npy")
        if out_path.exists():
            total += len(np.load(out_path))
            print(f"  {fname}: already embedded, skipping")
            continue
        items = [json.loads(l) for l in (CHUNK_DIR / fname).read_text().strip().split("\n") if l.strip()]
        texts = [it["content"] for it in items]          # content only, see module docstring
        t0 = time.time()
        vecs = enc.encode(texts)
        assert vecs.shape == (len(texts), EMBED_DIM), f"unexpected shape {vecs.shape}"
        np.save(out_path, vecs)
        total += len(texts)
        print(f"  {fname}: {len(texts)} chunks in {time.time()-t0:.0f}s -> {out_path.name}", flush=True)

    print(f"\nembedded {total:,} chunks; building faiss index (IndexFlatIP, dim {EMBED_DIM})")
    sys.path.insert(0, str(MEDRAG_DIR))
    from src.utils import construct_index
    construct_index(index_dir=str(INDEX_DIR), model_name=RETRIEVER_ID, h_dim=EMBED_DIM)
    print(f"[done] index at {INDEX_DIR}")


def patch_medrag():
    """Register the retriever and make MedRAG use DeepInfraEncoder for queries.

    Call BEFORE constructing MedRAG. Leaves all MedRAG retrieval logic intact.
    """
    sys.path.insert(0, str(MEDRAG_DIR))
    from src import utils as mru
    mru.retriever_names.setdefault("T21Embed", [RETRIEVER_ID])
    _orig = mru.CustomizeSentenceTransformer

    def _factory(model_name, *a, **kw):
        if model_name == RETRIEVER_ID:
            return DeepInfraEncoder()
        return _orig(model_name, *a, **kw)

    mru.CustomizeSentenceTransformer = _factory
    return mru


if __name__ == "__main__":
    build_index()
