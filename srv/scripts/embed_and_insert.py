#!/usr/bin/env python3
"""
Embed and Insert Pipeline for T21 Research Assistant v7
========================================================

This script:
1. Reads JSON publication files from a directory
2. Computes embeddings for specific sections using Nemotron or OpenAI
3. Stores embeddings in pickle files (intermediate cache)
4. Inserts embeddings into Milvus vector database

Usage:
    # Full pipeline (compute + insert):
    python embed_and_insert.py --input ../output/DS_PMC_05_2026 --compute --insert

    # Compute embeddings only (save to pickle):
    python embed_and_insert.py --input ../output/DS_PMC_05_2026 --compute

    # Insert from existing pickles:
    python embed_and_insert.py --insert

    # Custom pickle directory:
    python embed_and_insert.py --input ../output/my_pubs --pickle-dir ../output/my_embeddings --compute

Configuration:
    - Set environment variables in .env file
    - Or export them before running:
        export PROVIDER_EMBED=nemotron
        export NEMOTRON_DEVICE=cpu
        export MILVUS_URI=http://localhost:19530
"""

import os
import sys
import json
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import Counter
from dotenv import load_dotenv

# ===========================================================================
# Configuration
# ===========================================================================

# Load .env from project root
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print(f"[OK] Loaded .env from: {ENV_PATH}")
else:
    print(f"[WARNING] .env not found at {ENV_PATH}, using environment variables")

# Sections to embed
TARGET_SECTIONS = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embedding provider
PROVIDER_EMBED = os.getenv("PROVIDER_EMBED", "nemotron")

# Milvus settings
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", None)
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", os.getenv("MILVUS_COLLECTION_NAME", "DS_PMC_v7_DEEPINFRA_2048"))
EMBED_DIM = int(os.getenv("MILVUS_DIM", "4096"))
INDEX_TYPE = os.getenv("MILVUS_INDEX", "IVF_FLAT")
INDEX_METRIC = os.getenv("MILVUS_METRIC", "IP")

# Insert batch size
INSERT_BATCH = 500

print("\n" + "="*70)
print("T21 Research Assistant v7 - Embedding Pipeline")
print("="*70)
print(f"Provider       : {PROVIDER_EMBED}")
print(f"Embed Dim      : {EMBED_DIM}")
print(f"Chunk Size     : {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
print(f"Milvus URI     : {MILVUS_URI}")
print(f"Collection     : {COLLECTION_NAME}")
print(f"Index          : {INDEX_TYPE} / {INDEX_METRIC}")
print("="*70 + "\n")


# ===========================================================================
# Embedding Functions
# ===========================================================================

class EmbeddingProvider:
    """Abstract base for embedding providers."""
    
    def __init__(self):
        self.model_name = "unknown"
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class NemotronEmbeddings(EmbeddingProvider):
    """Nemotron local embeddings (CPU or GPU)."""
    
    def __init__(self):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
        
        self.model_name = os.getenv("NEMOTRON_EMBED_MODEL", "nvidia/llama-embed-nemotron-8b")
        self.max_length = int(os.getenv("NEMOTRON_EMBED_MAX_LENGTH", "512"))
        self.batch_size = int(os.getenv("NEMOTRON_EMBED_BATCH_SIZE", "2"))
        
        device_env = os.getenv("NEMOTRON_DEVICE", "").strip().lower()
        if device_env:
            self.device = device_env
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()
        
        self.torch = torch
        self.F = F
        print(f"[OK] Model loaded on {self.device}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            
            with self.torch.no_grad():
                outputs = self.model(**encoded)
            
            attn = encoded["attention_mask"]
            tok_emb = outputs.last_hidden_state
            mask_exp = attn.unsqueeze(-1).expand(tok_emb.size()).float()
            
            emb = self.torch.sum(tok_emb * mask_exp, dim=1)
            emb = emb / self.torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            
            # L2 normalize
            emb = self.F.normalize(emb, p=2, dim=1)
            
            out.extend(emb.cpu().tolist())
        
        return out


class DeepInfraEmbeddings(EmbeddingProvider):
    """DeepInfra cloud embeddings via the OpenAI-compatible endpoint.

    Model: nvidia/llama-nemotron-embed-vl-1b-v2 (output dim: 2048).
    Encodes documents symmetrically (no instruction prefix) to match the query-side
    encoding used by the API at retrieval time. Requires MILVUS_DIM=2048.
    """

    def __init__(self):
        import requests

        self.requests = requests
        self.model_name = os.getenv("DEEPINFRA_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
        self.base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
        self.url = self.base_url + "/embeddings"
        self.batch_size = int(os.getenv("DEEPINFRA_EMBED_BATCH_SIZE", "32"))

        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not set in environment")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"[OK] DeepInfra embeddings initialized (model: {self.model_name}, batch: {self.batch_size})")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import time as _time

        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {"input": batch, "model": self.model_name, "encoding_format": "float"}
            # Retry transient API failures (429/5xx/network) with backoff. Without this,
            # one hiccup aborts the whole paper via the caller's catch-all except, which
            # then counts it as skipped_error and moves on silently.
            last_err = None
            for attempt in range(5):
                try:
                    resp = self.requests.post(self.url, json=payload,
                                              headers=self.headers, timeout=180)
                    if resp.status_code in (429, 500, 502, 503):
                        last_err = f"HTTP {resp.status_code}"
                        _time.sleep(min(60, 2 * 2 ** attempt))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except self.requests.exceptions.RequestException as exc:
                    last_err = str(exc)[:150]
                    _time.sleep(min(60, 2 * 2 ** attempt))
            else:
                raise RuntimeError(f"DeepInfra embeddings failed after 5 tries: {last_err}")
            rows = sorted(data["data"], key=lambda d: d["index"])
            vecs = [r["embedding"] for r in rows]
            # A short response here would silently drop chunks downstream, because the
            # caller zips texts with vectors and zip stops at the shorter sequence.
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"embedding API returned {len(vecs)} vectors for {len(batch)} texts")
            out.extend(vecs)
        if len(out) != len(texts):
            raise RuntimeError(f"embedded {len(out)} of {len(texts)} texts")
        return out


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI cloud embeddings."""
    
    def __init__(self):
        from openai import OpenAI
        
        self.model_name = os.getenv("MODEL_EMBED", "text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", None)
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[OK] OpenAI client initialized (model: {self.model_name})")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API supports batch embedding
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        
        # Sort by index to maintain order
        data_sorted = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in data_sorted]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to create embedding provider."""
    
    if PROVIDER_EMBED == "nemotron":
        return NemotronEmbeddings()
    elif PROVIDER_EMBED == "deepinfra":
        return DeepInfraEmbeddings()
    elif PROVIDER_EMBED == "openai":
        return OpenAIEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {PROVIDER_EMBED}")


# ===========================================================================
# Helper Functions
# ===========================================================================

def make_splitter():
    """Create text splitter for chunking."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def stable_pk(pmid: str, section: str, chunk_id: int, model_name: str) -> int:
    """Generate deterministic 63-bit primary key."""
    raw = f"{pmid}|{section}|{chunk_id}|{model_name}"
    h = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


def is_empty(value) -> bool:
    """Check if section value is absent or placeholder."""
    if value is None:
        return True
    s = str(value).strip()
    return s in ("", "N/A", "NA", "n/a", "na")


def section_present(value) -> bool:
    """Check if section has content."""
    return not is_empty(value)


# ===========================================================================
# Compute Embeddings
# ===========================================================================

def compute_embeddings(input_dir: Path, pickle_dir: Path, force: bool = False) -> Dict[str, int]:
    """
    Compute embeddings for all JSON files and save to pickle.
    
    Args:
        input_dir: Directory containing JSON publication files
        pickle_dir: Directory to save pickle files
        force: If True, recompute even if pickle exists
    
    Returns:
        Statistics dictionary
    """
    
    print("\n" + "-"*70)
    print("STEP 1: COMPUTE EMBEDDINGS")
    print("-"*70)
    
    # Ensure pickle directory exists
    pickle_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {input_dir}")
        return {}
    
    # Filter to papers with target sections
    def has_target_section(path: Path) -> bool:
        try:
            with open(path) as fh:
                data = json.load(fh)
            sections = data.get("sections") or {}
            return any(section_present(sections.get(s)) for s in TARGET_SECTIONS)
        except Exception:
            return False
    
    eligible_files = [f for f in json_files if has_target_section(f)]
    
    print(f"Total JSON files    : {len(json_files)}")
    print(f"Eligible papers     : {len(eligible_files)} (have ≥1 target section)")
    print(f"Pickle directory    : {pickle_dir}")
    print()
    
    # Initialize embedding provider
    embedder = get_embedding_provider()
    splitter = make_splitter()
    
    # Statistics
    stats = {
        "processed": 0,
        "skipped_empty": 0,
        "skipped_error": 0,
        "already_done": 0,
        "total_chunks": 0
    }
    
    # Process each file
    for json_file in tqdm(eligible_files, desc="Computing embeddings"):
        
        pkl_path = pickle_dir / (json_file.stem + ".pkl")
        
        # Skip if already done (unless force)
        if pkl_path.exists() and not force:
            stats["already_done"] += 1
            continue
        
        try:
            # Load JSON
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            sections = data.get("sections") or {}
            pmid = str(data.get("pmid") or "").strip()
            
            # Prepare metadata
            def _str(v, default=""):
                return "" if (v is None or str(v).strip() in ("N/A", "NA")) else str(v).strip()
            
            def _int(v, default=0):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return default
            
            def _json_list(v):
                if not v:
                    return ""
                return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            
            meta_common = {
                "pmid": pmid,
                "pmcid": _str(data.get("pmcid")),
                "doi": _str(data.get("doi")),
                "openalex_id": _str(data.get("openalex_id")),
                "title": _str(data.get("title")),
                "year": _int(data.get("year")),
                "pub_type": _str(data.get("type")),
                "source": _str(data.get("source")),
                "language": _str(data.get("language")),
                "topic": _str(data.get("topic")),
                "subfield": _str(data.get("subfield")),
                "pub_field": _str(data.get("field")),
                "domain": _str(data.get("domain")),
                "sdg": _str(data.get("sdg")),
                "citations": _int(data.get("citations")),
                "keywords": _json_list(data.get("keywords")),
                "mesh_terms": _json_list(data.get("mesh_terms")),
                "authors": _json_list(data.get("authors")),
                "institutions": _json_list(data.get("institutions")),
                "funders_openalex": _json_list(data.get("funders_openalex")),
                "grants": _json_list(data.get("grants")),
                # INCLUDE provenance (present in JSONs from fetch_pmc_publications.py)
                "is_include":    int(bool(data.get("is_include") or False)),
                "condition":     _str(data.get("condition")),
                "grant_number":  _str(data.get("grant_number")),
                "rfa_pa_number": _str(data.get("rfa_pa_number")),
                "fiscal_year":   _str(data.get("fiscal_year")),
            }
            
            # Collect chunks
            texts: List[str] = []
            chunk_metas: List[Dict] = []
            
            for sec in TARGET_SECTIONS:
                content = sections.get(sec)
                if is_empty(content):
                    continue
                
                for cid, chunk in enumerate(splitter.split_text(str(content))):
                    c = chunk.strip()
                    if not c:
                        continue
                    texts.append(c)
                    chunk_metas.append({"section": sec, "chunk_id": cid})
            
            if not texts:
                stats["skipped_empty"] += 1
                continue
            
            # Compute embeddings
            vectors = embedder.embed_texts(texts)
            
            # Build rows
            rows: List[Dict[str, Any]] = []
            for text, cmeta, vec in zip(texts, chunk_metas, vectors):
                row = {
                    **meta_common,
                    "section": cmeta["section"],
                    "chunk_id": cmeta["chunk_id"],
                    "text": text,
                    "vector": vec,
                    "pk": stable_pk(pmid, cmeta["section"], cmeta["chunk_id"], embedder.model_name),
                }
                rows.append(row)
            
            # Save pickle
            with open(pkl_path, "wb") as fh:
                pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
            
            stats["processed"] += 1
            stats["total_chunks"] += len(rows)
        
        except Exception as exc:
            print(f"\n[ERROR] Error processing {json_file.name}: {exc}")
            stats["skipped_error"] += 1
            continue
    
    # Print summary
    print("\n" + "-"*70)
    print("EMBEDDING COMPUTATION SUMMARY")
    print("-"*70)
    for k, v in stats.items():
        print(f"{k:20s}: {v:,}")
    print(f"Pickle directory  : {pickle_dir}")
    print("-"*70 + "\n")
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Insert to Milvus
# ═══════════════════════════════════════════════════════════════════════
def build_milvus_fields() -> list:
    """Build Milvus collection schema fields."""
    from pymilvus import FieldSchema, DataType
    
    return [
        # Primary key
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        
        # Core publication identifiers
        FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="pmcid", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="openalex_id", dtype=DataType.VARCHAR, max_length=128),
        
        # Bibliographic metadata
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="pub_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
        
        # OpenAlex classification
        FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="subfield", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="pub_field", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="sdg", dtype=DataType.VARCHAR, max_length=256),
        
        # Citation count
        FieldSchema(name="citations", dtype=DataType.INT64),
        
        # List fields stored as JSON strings
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="mesh_terms", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="institutions", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="funders_openalex", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="grants", dtype=DataType.VARCHAR, max_length=16384),
        
        # Chunk fields
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),

        # INCLUDE provenance
        FieldSchema(name="is_include", dtype=DataType.INT8),          # 1=INCLUDE, 0=non-INCLUDE
        FieldSchema(name="condition", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="grant_number", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="rfa_pa_number", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="fiscal_year", dtype=DataType.VARCHAR, max_length=64),

        # Embedding vector
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]


def connect_milvus():
    """Connect to Milvus server."""
    from pymilvus import connections
    
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
    )
    print(f"[OK] Connected to Milvus at {MILVUS_URI}")


def ensure_collection(name: str):
    """Ensure Milvus collection exists with correct schema."""
    from pymilvus import Collection, CollectionSchema, utility
    
    fields = build_milvus_fields()
    expected_count = len(fields)
    
    if utility.has_collection(name):
        col = Collection(name)
        existing_count = len(col.schema.fields)
        
        if existing_count == expected_count:
            print(f"[OK] Collection '{name}' exists with correct schema ({existing_count} fields)")
            col.load()
            return col
        else:
            print(f"[WARNING] Collection '{name}' has {existing_count} fields but expected {expected_count}")
            print(f"  Dropping and recreating...")
            utility.drop_collection(name)
    
    # Create new collection
    schema = CollectionSchema(
        fields=fields,
        description="INCLUDE-pub section embeddings (Abstract/Introduction/Methods/Results/Discussion/Conclusion)",
    )
    col = Collection(name=name, schema=schema)
    
    # Create index
    index_params = {
        "index_type": INDEX_TYPE,
        "metric_type": INDEX_METRIC,
        "params": {"nlist": 128},
    }
    col.create_index(field_name="vector", index_params=index_params)
    col.load()
    
    print(f"[OK] Collection '{name}' created with {expected_count} fields, {INDEX_TYPE}/{INDEX_METRIC} index")
    return col


def _truncate(value, max_len: int) -> str:
    """Safely convert to str and truncate to Milvus VARCHAR limit."""
    return str(value or "")[:max_len]


def upsert_batch(col, rows: List[Dict[str, Any]], mark_include: bool = False):
    """Delete existing PKs then insert rows (upsert pattern).

    Parameters
    ----------
    mark_include : if True, override is_include=1 for every row in this batch.
        Use when re-inserting the original 363 INCLUDE pickles whose row dicts
        do not yet contain an is_include field.
    """
    if not rows:
        return
    
    pk_list = [r["pk"] for r in rows]
    expr = f"pk in [{', '.join(str(p) for p in pk_list)}]"
    
    try:
        col.delete(expr)
    except Exception:
        pass  # Empty collection on first run
    
    # Prepare data in column-wise format
    data = [
        [r["pk"] for r in rows],
        [_truncate(r.get("pmid"), 32) for r in rows],
        [_truncate(r.get("pmcid"), 32) for r in rows],
        [_truncate(r.get("doi"), 256) for r in rows],
        [_truncate(r.get("openalex_id"), 128) for r in rows],
        [_truncate(r.get("title"), 1024) for r in rows],
        [int(r.get("year") or 0) for r in rows],
        [_truncate(r.get("pub_type"), 64) for r in rows],
        [_truncate(r.get("source"), 512) for r in rows],
        [_truncate(r.get("language"), 16) for r in rows],
        [_truncate(r.get("topic"), 256) for r in rows],
        [_truncate(r.get("subfield"), 256) for r in rows],
        [_truncate(r.get("pub_field"), 256) for r in rows],
        [_truncate(r.get("domain"), 128) for r in rows],
        [_truncate(r.get("sdg"), 256) for r in rows],
        [int(r.get("citations") or 0) for r in rows],
        [_truncate(r.get("keywords"), 2048) for r in rows],
        [_truncate(r.get("mesh_terms"), 4096) for r in rows],
        [_truncate(r.get("authors"), 4096) for r in rows],
        [_truncate(r.get("institutions"), 2048) for r in rows],
        [_truncate(r.get("funders_openalex"), 2048) for r in rows],
        [_truncate(r.get("grants"), 16384) for r in rows],
        [_truncate(r.get("section"), 64) for r in rows],
        [int(r.get("chunk_id") or 0) for r in rows],
        [_truncate(r.get("text"), 4096) for r in rows],
        [1 if mark_include else int(bool(r.get("is_include") or False)) for r in rows],
        [_truncate(r.get("condition"), 512) for r in rows],
        [_truncate(r.get("grant_number"), 256) for r in rows],
        [_truncate(r.get("rfa_pa_number"), 256) for r in rows],
        [_truncate(r.get("fiscal_year"), 64) for r in rows],
        [r["vector"] for r in rows],
    ]
    
    col.insert(data)


def insert_embeddings(pickle_dir: Path, mark_include: bool = False) -> Dict[str, int]:
    """
    Insert embeddings from pickle files into Milvus.

    Parameters
    ----------
    pickle_dir   : Directory containing pickle files
    mark_include : If True, override is_include=1 for all rows (use when
                   re-inserting the original 363 INCLUDE pickles that pre-date
                   the is_include schema field)

    Returns
    -------
    Statistics dictionary
    """
    
    print("\n" + "-"*70)
    print("STEP 2: INSERT TO MILVUS")
    print("-"*70)
    
    # Get pickle files
    pkl_files = sorted(pickle_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"[ERROR] No pickle files found in {pickle_dir}")
        return {}
    
    print(f"Pickle files found  : {len(pkl_files)}")
    print(f"Pickle directory    : {pickle_dir}")
    if mark_include:
        print(f"mark_include        : True  (all rows forced is_include=1)")
    print()
    
    # Connect to Milvus
    connect_milvus()
    col = ensure_collection(COLLECTION_NAME)
    
    print(f"Entity count before : {col.num_entities:,}")
    print()
    
    # Statistics
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_chunks": 0
    }
    
    # Process each pickle file
    for pkl_path in tqdm(pkl_files, desc="Inserting to Milvus"):
        
        try:
            # Load rows from pickle
            with open(pkl_path, "rb") as fh:
                rows: List[Dict[str, Any]] = pickle.load(fh)
            
            if not rows:
                stats["files_skipped"] += 1
                continue
            
            # Insert in sub-batches
            for start in range(0, len(rows), INSERT_BATCH):
                upsert_batch(col, rows[start : start + INSERT_BATCH], mark_include=mark_include)
            
            stats["files_processed"] += 1
            stats["total_chunks"] += len(rows)
        
        except Exception as exc:
            print(f"\n[ERROR] Error inserting {pkl_path.name}: {exc}")
            stats["files_skipped"] += 1
            continue
    
    # Milvus auto-flushes in the background; no explicit flush needed here.

    # Print summary
    print("\n" + "-"*70)
    print("MILVUS INSERTION SUMMARY")
    print("-"*70)
    for k, v in stats.items():
        print(f"{k:20s}: {v:,}")
    print(f"Entity count after  : {col.num_entities:,}")
    print("-"*70 + "\n")
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Compute embeddings and insert into Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline:
  python embed_and_insert.py --input ../output/DS_PMC_05_2026 --compute --insert
  
  # Compute only:
  python embed_and_insert.py --input ../output/DS_PMC_05_2026 --compute
  
  # Insert only (from existing pickles):
  python embed_and_insert.py --insert
  
  # Custom pickle directory:
  python embed_and_insert.py --input ../output/my_pubs --pickle-dir ../output/my_pkl --compute --insert
  
  # Force recompute embeddings:
  python embed_and_insert.py --input ../output/DS_PMC_05_2026 --compute --force
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory containing JSON publication files"
    )
    
    parser.add_argument(
        "--pickle-dir",
        type=Path,
        help="Directory for pickle files (default: ../output/embeddings_pickle/<input_dir_name>)"
    )
    
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Compute embeddings and save to pickle"
    )
    
    parser.add_argument(
        "--insert",
        action="store_true",
        help="Insert embeddings from pickle into Milvus"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute embeddings even if pickle exists"
    )

    parser.add_argument(
        "--mark-include",
        action="store_true",
        help="Override is_include=1 for ALL inserted rows (use when re-inserting existing INCLUDE pickles)"
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not args.compute and not args.insert:
        parser.error("Must specify at least one of: --compute, --insert")
    
    if args.compute and not args.input:
        parser.error("--input required when using --compute")
    
    # Determine pickle directory
    if args.pickle_dir:
        pickle_dir = args.pickle_dir.resolve()
    elif args.input:
        output_base = Path(__file__).parent.parent / "output" / "embeddings_pickle"
        pickle_dir = output_base / f"{args.input.name}_{PROVIDER_EMBED}_{EMBED_DIM}"
    else:
        # --insert with neither --pickle-dir nor --input given: there is no correct directory
        # to guess. A previous fallback silently defaulted to a fixed, unrelated corpus name
        # here (INCLUDE_PUB_by_2025_11_14_...) -- fail loudly instead of inserting from the
        # wrong pickle directory.
        parser.error("--pickle-dir required for --insert when --input is not also given")
    
    # Execute pipeline
    if args.compute:
        input_dir = args.input.resolve()
        if not input_dir.exists():
            print(f"[ERROR] Input directory not found: {input_dir}")
            sys.exit(1)
        
        compute_embeddings(input_dir, pickle_dir, force=args.force)
    
    if args.insert:
        if not pickle_dir.exists():
            print(f"[ERROR] Pickle directory not found: {pickle_dir}")
            sys.exit(1)

        mark_include = getattr(args, "mark_include", False)
        insert_embeddings(pickle_dir, mark_include=mark_include)
    
    print("\n[SUCCESS] Pipeline complete!\n")


if __name__ == "__main__":
    main()