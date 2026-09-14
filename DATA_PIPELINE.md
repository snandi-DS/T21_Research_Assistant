# T21 Research Assistant: Complete Data Pipeline
## PMID → XML → JSON → Embeddings → Milvus

---

## Overview

The pipeline processes Down Syndrome research papers from MEDLINE/PMC through 4 major stages:

```
STAGE 1: FETCH       STAGE 2: EXTRACT     STAGE 3: EMBED        STAGE 4: INDEX
 PMID/PMCID    →    XML → JSON Metadata  →  Chunk & Embed  →   Milvus Vector DB
(input files)        (metadata + sections)    (2048-dim)         (68,509 vectors)
```

---

## STAGE 1: FETCH PUBLICATION METADATA

Script: `srv/scripts/fetch_pmc_publications.py`

### Input Sources
- MEDLINE Export: `PMC_DownSynd_05_18_2026.txt`, 1,936 papers; journal articles indexed in MEDLINE
- INCLUDE Registry: `INCLUDE_Publications_by_2025-11-14.xlsx`, 428 papers; NIH INCLUDE-funded publications
- EMODS Registry: `EMODS_publications_extract_20250717_additional_details (1).xlsx`, ~388 papers; additional open-access publications

### Inclusion Criteria (Applied)

- Publication Stage: Published journal article (NOT preprint)
- Publication Type: Article (NOT case reports, news, letters, editorials)
- Collection: PMC Open Access Subset only (full text must be available)
- Embargo Status: NOT embargoed articles
- Down Syndrome Focus: Papers indexed under "Down Syndrome" or "Trisomy 21"

### Exclusion Criteria (Applied)

- Preprints: exclude early/unreviewed versions
- Case Reports: too narrow/specialized
- News & Views: editorial/opinion content
- Letters/Editorials: correspondence only
- Embargoed: restricted access articles
- Non-English: language filter
- Duplicate PMIDs: keep only unique papers

### Output
- Format: One JSON file per PMID
- Location: `srv/output/DS_PMC_05_2026/` (example)
- Fields: Metadata from OpenAlex, NCBI, CrossRef, PMC

Example JSON structure:
```json
{
  "pmid": "12345678",
  "pmcid": "PMC12345678",
  "title": "Transcriptomic study of Down Syndrome...",
  "authors": ["Smith J", "Johnson A"],
  "abstract": "Background: Down syndrome...",
  "pub_type": "article",
  "pub_date": "2024-05-15",
  "journal": "Nature Genetics",
  "sections": {
    "Abstract": "Background: ...",
    "Introduction": "Down syndrome (DS) is...",
    "Methods": "We analyzed transcriptomic data...",
    "Results": "We identified X genes...",
    "Discussion": "Our findings suggest...",
    "Conclusion": "In conclusion..."
  },
  "is_include": true,
  "condition": "Down Syndrome; Developmental Delay",
  "grant_number": "R01 HD109765",
  "rfa_pa_number": "PA-16-275",
  "fiscal_year": "2024"
}
```

---

## STAGE 2: EXTRACT SECTIONS FROM XML (Automated in Fetch Step)

During the fetch step, the pipeline retrieves and parses PMC XML to extract structured sections.

### Section Extraction Logic

The pipeline extracts 6 paper sections from PMC XML:

- Abstract: 100-300 words; summary of study
- Introduction: 500-1500 words; background & motivation
- Methods: 500-2000 words; study design & procedures
- Results: 500-2000 words; key findings
- Discussion: 1000-3000 words; interpretation & implications
- Conclusion: 100-500 words; main takeaways

### Section Availability in Corpus

(measured against the 1,602 pub_type=article documents in the live collection, 2026-09-02)

- Abstract: 1,602 documents (100.0%)
- Methods: 1,601 documents (99.9%)
- Results: 1,593 documents (99.4%)
- Discussion: 1,584 documents (98.9%)
- Introduction: 1,541 documents (96.2%)
- Conclusion: 867 documents (54.1%; lower, common in shorter papers)

### Why Some Sections Missing?
- No Conclusion: Short reports, editorials, or unusual paper structures
- No Introduction: Brief case reports or summary papers
- Empty Results: Review/perspective pieces (rare among filtered articles)

---

## STAGE 3: EMBEDDING & CHUNKING

Script: `srv/scripts/embed_and_insert.py`

### Step 3a: Filter Eligible Papers

```python
# Only embed papers with ≥1 target section
TARGET_SECTIONS = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]

# Filter:  has ≥1 of these sections AND has metadata
eligible = papers that have at least one section with content
```

Result (corpus selection): 1,655 papers stored: 1,501 legacy papers carried over from the corrected previous corpus (only those with 5+ sections; `srv/output/active_pmcids_1501_5plus_sections.txt`) + 389 newly ingested PMIDs (`scripts/ingest_pmids.py`, Aug 2026) minus 235 later purged (`srv/output/v7_new_388_purge_list.json`). All 1,655 have ≥1 target section. Of these, 1,602 are `pub_type=article` (plus 51 reviews and 2 preprints, stored but excluded from retrieval; see Stage 4).

### Step 3b: Text Chunking

Goal: Break long sections into manageable pieces for embedding

Settings:
- Chunk Size: 1,200 characters (~250-300 tokens; default, configurable via `CHUNK_SIZE`)
- Chunk Overlap: 200 characters (default, configurable via `CHUNK_OVERLAP`)
- Splitter: `RecursiveCharacterTextSplitter` from LangChain with `length_function=len` (character-based), separators `["\n\n", "\n", ". ", " ", ""]`
  - Splits on paragraph, then sentence boundaries (`. `)
  - Falls back to word boundaries
  - Preserves semantic coherence

Example:
```
Section: "Results" (12,000 characters)
Chunk 1: chars [0-1200]      (1,200 chars, 0 overlap)
Chunk 2: chars [1000-2200]   (1,200 chars, 200 overlap with Chunk 1)
... and so on to the end of the section
```

Result: 68,509 total chunks across 1,655 papers
- Results: 20,652 chunks (30.1%)
- Discussion: 17,107 chunks (25.0%)
- Methods: 17,187 chunks (25.1%)
- Introduction: 8,554 chunks (12.5%)
- Abstract: 3,614 chunks (5.3%)
- Conclusion: 1,395 chunks (2.0%)

### Step 3c: Embedding Computation

Current Provider: DeepInfra API (via OpenAI-compatible endpoint)

Model: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Dimensionality: 2048 dimensions
- Type: Dense, symmetric embeddings (no instruction prefix)
- Batch Size: 32 chunks per API call
- Latency: ~5-10 seconds per 32 chunks

Alternative Providers:
- `nemotron`: Local Nemotron 8B model (CPU/GPU, ~16GB VRAM)
- `openai`: OpenAI `text-embedding-3-small` (1,536-dim)
- `google`: Google Gemini embeddings

### Output: Pickle Files

Location: `srv/output/embeddings_pickle/DS_PMC_05_2026_deepinfra_2048/`

Format: One pickle file per JSON (serialized Python objects)

Contains per chunk:
```python
{
    "pmid": "12345678",
    "pmcid": "PMC12345678",
    "section": "Results",
    "chunk_id": 0,
    "text": "We identified X genes showing elevated...",
    "vector": [0.123, -0.456, 0.789, ...],  # 2048 floats
    "pk": <deterministic hash of pmid|section|chunk_id|model>,
    # ... plus all shared paper metadata fields (title, year, mesh_terms, is_include, ...)
}
```

---

## STAGE 4: INDEX INTO MILVUS VECTOR DATABASE

Script: `srv/scripts/embed_and_insert.py` (with `--insert` flag)

### Step 4a: Create Collection Schema

Collection Name: `DS_INCLUDE_2026_09_02_DEEPINFRA_2048`

Schema (31 fields):

- `pk` (INT64, primary key): unique ID, deterministic hash of pmid+section+chunk_id+model
- `pmid` (VARCHAR): PubMed ID
- `pmcid` (VARCHAR): PMC ID (used for citations)
- `section` (VARCHAR): paper section name
- `chunk_id` (INT64): sequence within section
- `text` (VARCHAR): actual chunk text (not searched, metadata only)
- `pub_type` (VARCHAR): publication type (article, review, preprint); retrieval filters on this
- `year` (INT64): publication year
- `title` (VARCHAR): paper title
- `authors` (VARCHAR): author list
- `source` (VARCHAR): journal / source name
- `mesh_terms` (VARCHAR): MeSH terms
- `is_include` (INT8): 1 if NIH INCLUDE-funded, 0 otherwise
- `condition` (VARCHAR): INCLUDE Condition (if applicable)
- `grant_number` (VARCHAR): INCLUDE Grant Number
- `vector` (FLOAT_VECTOR, dim 2048): dense vector; the only indexed field (IVF_FLAT)
- plus 15 more metadata fields: `doi`, `openalex_id`, `language`, `topic`, `subfield`, `pub_field`, `domain`, `sdg`, `citations`, `keywords`, `institutions`, `funders_openalex`, `grants`, `rfa_pa_number`, `fiscal_year`

### Step 4b: Insert Embeddings

Process:
1. Load pickles from embedding stage
2. Batch insert by 500 chunks per transaction
3. Milvus auto-flushes via Write Ahead Log (WAL)
4. Build IVF_FLAT index (`nlist=128`); searches use `nprobe=64`

Final State (live collection, verified 2026-09-02):
- Total Vectors: 68,509
- Total Documents (unique PMCIDs): 1,655 stored (1,602 articles + 51 reviews + 2 preprints)
- Active in retrieval: 1,602 articles (66,195 vectors); reviews/preprints are stored but excluded by the retrieval-time `pub_type == "article"` filter
- INCLUDE-funded Docs: 285 articles marked `is_include=1` (plus 2 INCLUDE-funded reviews, stored but not retrieved)
- Index Type: IVF_FLAT (Inverted File Flat), `nlist=128`
- Index Metric: Inner Product (IP) for cosine similarity
- Search Params: `nprobe=64` (higher precision, ~64 buckets searched)

---

## DATA QUALITY ASSURANCE

### Document Completeness

```
  1,574 documents (98.3%)  Have all 3 core sections (Methods + Results + Discussion)
    18 documents (1.1%)      Missing Discussion section
     9 documents (0.6%)      Missing Results section
     1 document  (0.1%)      Missing Methods section
────────────────────────────────────────────
  1,602 articles (100%)       Have at least 2 of 3 core sections
```

### Publication Type Filter

Retrieval filters at query time to `pub_type == "article"`:
- INCLUDED: Article (1,602 documents, active in retrieval)
- EXCLUDED: Review (51 documents, stored in Milvus but excluded from retrieval)
- EXCLUDED: Preprint (2 documents, stored in Milvus but excluded from retrieval)

Note: The 28 incomplete documents above are all classified as "article" (pub_type='article'), but 18 lack Discussion sections and 9 lack Results. This is normal for brief/short reports.

### INCLUDE Provenance

- 285 articles (17.8% of the 1,602 retrieval-active articles) tagged with `is_include=1` (NIH INCLUDE-funded)
- 2 additional INCLUDE-funded reviews are stored but excluded from retrieval by the pub_type filter

---

## PIPELINE FLOW SUMMARY

```
1. FETCH
   Input:  MEDLINE TXT + INCLUDE Excel + EMODS Excel
   │
   ├─ Parse PMIDs from MEDLINE
   ├─ Fetch metadata (OpenAlex, NCBI, CrossRef, PMC)
   ├─ Cross-reference INCLUDE Excel for funding tags
   ├─ Apply inclusion/exclusion filters
   └─ Save: 1 JSON per PMID (2,178 files cached across all fetch runs)

2. EXTRACT
   Input:  cached JSON files (with XML section text)
   │
   ├─ Parse 6 paper sections
   ├─ Validate section content (not empty/N/A)
   └─ Output: Sections embedded in JSON

3. EMBED  (selection: 1,501 legacy 5+-section papers + 389 new - 235 purged = 1,655)
   Input:  1,655 selected JSON files
   │
   ├─ Filter: papers with ≥1 target section
   ├─ Chunk: split sections into 1,200-character pieces (200 overlap)
   ├─ Embed: 68,509 chunks → 2048-dim vectors (DeepInfra API)
   └─ Save: 1,655 pickle files with embeddings

4. INDEX
   Input:  1,655 pickle files with embeddings
   │
   ├─ Create Milvus collection schema (31 fields)
   ├─ Insert: 68,509 vectors + metadata
   ├─ Build IVF_FLAT index (nlist=128; searched with nprobe=64)
   └─ Live: Ready for RAG retrieval queries (article-type docs only)

```

---

## Configuration (`.env`)

```bash
# STAGE 1: Fetch
NCBI_EMAIL=your.email@example.com              # Required for NCBI API
NCBI_API_KEY=xxxxx                             # Optional NCBI key
CROSSREF_EMAIL=your.email@example.com          # Required for CrossRef API

# STAGE 3: Embed
PROVIDER_EMBED=deepinfra                       # deepinfra | nemotron | openai
DEEPINFRA_API_KEY=xxxxx                        # DeepInfra API key
CHUNK_SIZE=1200                                # Chunk size in tokens
CHUNK_OVERLAP=200                              # Overlap between chunks

# STAGE 4: Index
MILVUS_URI=http://localhost:19530              # Milvus server
MILVUS_COLLECTION=DS_INCLUDE_2026_09_02_DEEPINFRA_2048   # Collection name (current live)
MILVUS_DIM=2048                                # Embedding dimension
MILVUS_INDEX=IVF_FLAT                          # Index type
MILVUS_METRIC=IP                               # Metric (IP = cosine)
```

---

## Running the Pipeline

> On the deployment server, run every stage through Docker via the `data-processor` service; the full step-by-step Docker operations guide is in [DEPLOYMENT.md](DEPLOYMENT.md).

### Full Pipeline (4 Stages)
```bash
cd srv

# STAGE 1: Fetch metadata (2-3 hours)
python scripts/fetch_pmc_publications.py \
  --medline input/PMC_DownSynd_05_18_2026.txt \
  --include-excel input/INCLUDE_Publications_by_2025-11-14.xlsx \
  --out-dir output/DS_PMC_05_2026

# STAGE 2: Extract sections (automatic in Stage 1)

# STAGE 3: Compute embeddings (1-2 hours)
python scripts/embed_and_insert.py \
  --input output/DS_PMC_05_2026 \
  --compute

# STAGE 4: Insert into Milvus (30 minutes)
# On the deployment server, via the data-processor Docker service (no GPU, no conda env)
sudo docker compose --profile data run --rm data-processor \
  python scripts/embed_and_insert.py \
  --insert
```

### Quick Re-Insert (Stages 3-4, pickles already exist)
```bash
# On the deployment server: insert the current production pickles via Docker (5-15 minutes, CPU-only)
# Milvus containers must be running: sudo docker ps --filter name=milvus_shared
sudo docker compose --profile data run --rm data-processor \
  python scripts/embed_and_insert.py \
  --pickle-dir output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048 \
  --insert

# Then restart the API and clear the cache
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear
```

---

## Troubleshooting

- PMID not found: network error or deleted paper; retry, check NCBI API status
- Missing section: PDF parsing failed or no section present; acceptable, the paper is still indexed with its other sections
- Embedding timeout: DeepInfra API overloaded; increase `DEEPINFRA_TIMEOUT` and retry
- Milvus insert error: collection schema mismatch; drop and recreate the collection
- Low vector count: filtering too aggressive; verify `TARGET_SECTIONS` and `pub_type`

---

## Summary Statistics

(verified against the live collection, 2026-09-02)

- Stored Papers: 1,655 (1,501 legacy 5+-section + 389 newly ingested - 235 purged)
- Active in retrieval (article only): 1,602
- Complete (Methods+Results+Discussion): 1,574 (98.3% of articles)
- Total Sections Indexed: 6 (Abstract, Intro, Methods, Results, Discussion, Conclusion)
- Total Chunks Created: 68,509
- Embedding Dimension: 2048
- Index Type: IVF_FLAT with IP metric
- INCLUDE-funded: 285 articles (17.8%)
- Search Coverage: Results, Discussion, Methods, Conclusion only (54,400 chunks active)

