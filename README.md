# T21 Research Assistant v7

A multi-agent Retrieval-Augmented Generation (RAG) system for evidence-based question answering over the **INCLUDE Publication Corpus** — a curated collection of Down Syndrome (Trisomy 21) research papers.

---

## Overview

The system retrieves section-level text from peer-reviewed publications stored in a Milvus vector database, reranks the results using the NVIDIA Nemotron cross-encoder **via the DeepInfra REST API** (no local GPU required), and synthesizes a structured response with inline citations using **Nemotron-3-Super via the Ollama remote cloud API**. The deployment runs a **hybrid stack** — embeddings (2048-dim) + reranking on DeepInfra, chat/synthesis + LLM-as-judge on Ollama remote — so it needs no GPU and loads no local model weights. (Ollama remote is the reliable default for chat because DeepInfra's `NVIDIA-Nemotron-3-Super-120B` returned `engine_overloaded`/429 under load.) The backend is a FastAPI REST API; the frontend is an interactive Streamlit chat interface.

> **Performance & determinism (Ollama-remote path):** `nemotron-3-super` and `gpt-oss:120b-cloud` are reasoning models. Every Ollama chat call passes **`think=False`** to suppress the hidden chain-of-thought — without it a single synthesis call took **~153 s** (the model generates a large unseen reasoning trace); with it, synthesis runs in seconds. All chat calls also use **greedy decoding** (`temperature=0.0, seed=42, top_k=1, top_p=1.0`) so repeated questions return closely consistent answers. All prompts were also **heavily condensed**: `SYNTH_PROMPT` ~6,700 → ~2,100 tokens (~69%); then a further ~1,627 tokens removed across all prompts (`_JUDGE_SYSTEM`, `_PIPELINE_IMPROVE_SYSTEM`, `lenient_system`, `system_content`, human-message boilerplate) — cutting per-call input processing latency while preserving every behavioral rule.

**Current corpus:** **1,594 open-access original research articles**, published **1970 – September 2026** (earliest indexed paper: PMC6460451, 1970; the "1962" coverage figure quoted in earlier drafts belonged to the superseded May-2026 v7 fetch and no 1962–1969 paper survives the current criteria) — the collection stores only papers that satisfy every criterion, so the stored count, the searchable count, and the `/health`/GUI count are the same number. Derivation from the 2026-09-02 PMC snapshot (`srv/nbs/1_download_ds_papers.ipynb` + `2_embed_and_index_ds_papers.ipynb`): 2,075 snapshot records → 1,891 eligible (publication-type gate, PMCID present, not retracted) → 1,837 with usable PMC full text → 1,669 with ≥4 of 6 sections → **1,594 with `pub_type == "article"`** (drops 64 reviews, 8 preprints, 2 errata, 1 other). Embedding vectors are computed only for these 1,594 papers, after all criteria pass. | **158 NIH INCLUDE-funded articles** tagged with provenance badge in citations
**Sections indexed (stored):** Results, Discussion, Methods, Conclusion, Abstract, Introduction (**65,677 total vectors**) | **Active retrieval:** Results, Discussion, Methods, Conclusion only (**54,246 chunks**); Abstract/Introduction are stored but excluded from retrieval by the Milvus search filter. The `pub_type == "article"` query filter remains in place as a no-op safety net (everything stored is already an article)

**Funding:** NIH INCLUDE Project — grant R01 HD109765  
**Developed by:** Costello Lab, University of Colorado Anschutz

> **Operating the deployment?** See [DEPLOYMENT.md](DEPLOYMENT.md) — a step-by-step guide for running every operation (start/stop, code and `.env` changes, corpus insert/re-index, rollback, troubleshooting) entirely through Docker on the deployment server.

---

## Architecture

```
User Question
     |
     v
[Streamlit UI]  <-->  [FastAPI REST API]
                              |
                              v
                   [LangGraph Multi-Agent Pipeline]
                       |
                       +-- 0. question_judge  (LLM pre-flight classifier — VALID / INVALID, fail-open)
                       +-- 1. retrieve        (Milvus IVF_FLAT vector search, 5x oversampling)
                       +-- 2. rerank          (Nemotron cross-encoder reranking via DeepInfra REST API — no local GPU)
                       +-- 3. synthesize      (C-C-C structured response, max 150 words)
                       +-- 4. raw_evidence    (top-K chunk extraction)
                       +-- 5. safety_guard    (prescription/diagnostic language filter)
                       +-- 6. finalize        (citation hyperlinking + 150-word cap + INCLUDE tag injection)
                       +-- 7. judge_improve   (LLM-as-Judge scoring + guided rewrite)
```

Pipeline entry: `question_judge` → (VALID) → `retrieve → rerank → synthesize → raw_evidence → safety_guard → finalize → judge_improve → END`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└→ (INVALID) → `end_invalid → END`

### Section Priority Weights

Only the following sections are retrieved from Milvus and receive a reranking bonus:

- Results: 1.0
- Discussion: 1.0
- Methods: 1.0
- Conclusion: 1.0
- Abstract: 0.0 (excluded)
- Introduction: 0.0 (excluded)

### Output Sections

Each response contains:

- **Summary** — Publication-quality paragraph (max 150 words), following the Context–Content–Conclusion (C-C-C) scheme, with inline PMC citations. Expanded by default. When `ENABLE_JUDGE=true`, the pipeline scores the Summary (Accuracy / Informativeness / Completeness, 1–5) and, **only if any criterion scores below 4**, rewrites it guided by the judge notes. The rewritten version is spliced in at the API level (`main.py`) before the response is returned, so what the user sees IS the judge-improved Summary. If all scores are ≥ 4 the original finalized Summary is kept unchanged.
- **Key Evidence** — 4–6 high-impact findings with effect sizes, sample sizes, and p-values where available. Collapsed by default.
- **Reasoning** — 4 bullet points (Evidence selection, Reasoning chain, Confidence & gaps, Reader caveats) explaining why these findings were prioritised, how the evidence chain supports the Summary conclusion, where gaps or caveats exist, and what the reader should bear in mind. Collapsed by default.
- **Deep Dive** — Collapsible on-demand section below each response. Clicking the **"Deep Dive"** button runs a two-step atomic cycle: (1) scores the original summary on Accuracy / Informativeness / Completeness (1–5 stars) via an independent LLM judge (`gpt-oss:120b-cloud`); (2) generates a comprehensive multi-paragraph extended synthesis (`nemotron-3-super`), guided by the judge scores and drawing from an extended citation pool of up to 30 papers from the full retrieved evidence. Judge scores are shown in a nested collapsed expander **"Original Summary — LLM-as-Judge scores"**. **Re-score Deep Dive** and **Re-generate Deep Dive** buttons allow further on-demand refinement. Deep Dive text is included in all Export & Copy formats when present.
- **Highlighted Topic, MeSH Terms and Keywords from Top Retrieved Papers** — HTML card display per cited paper showing Topic (with PMCID hyperlink), MeSH Terms (teal pills), and Keywords (orange pills); terms appearing in the Summary are highlighted. Collapsed by default.

---

## Streamlit UI Pages

The Streamlit interface (port 8878) includes the following navigation pages accessible from the top nav bar:

| Page | URL | Description |
|------|-----|-------------|
| **Chat** (default) | `/` or `?` | Main Q&A interface — multi-conversation chat with sidebar |
| **About** | `?page=about` | Project overview, pipeline summary, corpus stats |
| **Team** | `?page=team` | PI, Collaborators (Matthew Galbraith, Kelly D. Sullivan, Joaquin Espinosa, Milton Pividori), and Research Team member cards |
| **FAQ** | `?page=faq` | 11 expandable frequently asked questions (includes Deep Dive, pipeline, corpus, citations) |
| **GitHub** | [github.com/snandi-DS/T21_Research_Assistant](https://github.com/snandi-DS/T21_Research_Assistant.git) | Source code repository (opens in new tab) |
| **Contact** | `?page=contact` | Contact form — sends directly to James Costello, PhD |

**Navigation:** Click the **logo** in the sidebar or press **+ New Chat** from any sub-page to return to the main chat.

---

## Project Structure

The codebase is split into two top-level directories:

- **`srv/`** — server-side deployment (API, UI, Docker, data pipeline notebooks)
- **`analysis/`** — analysis and evaluation notebooks with their own input/output

```
T21_research_assistant_v7/
├── README.md, ...                 # Project documentation
│
├── srv/                                      # Server-side deployment
│   ├── .env                                  # Environment config (API keys, settings)
│   ├── api/
│   │   ├── main.py                           # FastAPI application (port 8698)
│   │   └── synthesizer_langchain.py          # LangChain/LangGraph RAG engine
│   ├── scripts/
│   │   ├── fetch_pmc_publications.py         # Fetch + enrich metadata for expanded DS PMC corpus
│   │   └── embed_and_insert.py               # Compute Nemotron embeddings → Milvus (31-field schema)
│   ├── ui/
│   │   └── ui_streamlit_ds_chat.py           # Streamlit chat frontend (port 8878)
│   ├── input/                                # Server input data
│   │   ├── INCLUDE_Publications_by_2025-11-14.xlsx   # 428 INCLUDE-funded papers (provenance reference)
│   │   ├── PMC_DownSynd_05_18_2026.txt               # MEDLINE export — 1,934 DS journal articles
│   │   └── AI_research_assistant_Sample_questions_revised_JIM_Negative_Ctrl_dt_03_12_2026.xlsx
│   ├── output/                               # Server outputs
│   │   ├── embeddings_pickle/                # Per-PMID embedding pickle files
│   │   ├── DS_PMC_05_2026/                   # Per-PMID metadata JSON files (2,184 papers)
│   │   ├── batch_eval/                       # Batch evaluation spreadsheets
│   │   └── survey/                           # Qualtrics survey CSVs
│   ├── nbs/                                  # Data pipeline notebooks (1–4)
│   │   ├── 1_fetch_publication_metadata.ipynb
│   │   ├── 2_publication_metadata_analysis.ipynb
│   │   ├── 3_compute_embeddings_to_pickle.ipynb
│   │   └── 4_insert_embeddings_to_milvus.ipynb
│   ├── local_hugging_face_model/             # Nemotron embed + rerank model weights
│   │   ├── embed/                            # nvidia/llama-embed-nemotron-8b (~14 GB)
│   │   └── rerank/                           # nvidia/llama-nemotron-rerank-1b-v2 (~2 GB)
│   ├── docker-compose.yml                    # T21 app: API + UI (requires Milvus)
│   ├── docker-compose.milvus.yml             # Shared Milvus vector database
│   └── requirements.txt                      # Python dependencies
│
└── analysis/                                 # Analysis & evaluation
    ├── .env                                  # Symlink to ../srv/.env
    ├── input/                                # Copies of benchmark Excel files
    │   └── AI_research_assistant_Sample_questions_revised_JIM_Negative_Ctrl_dt_03_12_2026.xlsx
    ├── output/                               # Analysis outputs
    │   ├── batch_eval/                       # Benchmark results, model responses
    │   ├── umap/                             # UMAP figures (PDF + JPEG)
    │   └── survey/                           # Survey CSV outputs
    └── nbs/                                  # Analysis & evaluation notebooks (0, 5–8)
        ├── 0_ollma_model_test.ipynb
        ├── 5_benchmark_pipeline_vs_ground_truth.ipynb
        ├── 6_umap_questions.ipynb
        ├── 7_expert_survey_qualtrics_design.ipynb
        └── 8_qualtrics_survey_dataset_builder.ipynb
```

---

## Requirements

- Python 3.11+
- Conda environment: `dschat` (for local/manual deployment)
- Milvus (running at `localhost:19530`)
- DeepInfra API key (`DEEPINFRA_API_KEY`) — the default stack runs chat, embeddings, reranking, and the judge through DeepInfra; Anthropic, Google, OpenAI, and Ollama remote are also supported alternatives

**Docker Deployment (Recommended):** See the [Docker Deployment](#docker-deployment-recommended-for-production) section below for containerized deployment.

### Python dependencies (key packages)

```
fastapi
uvicorn
streamlit
langchain-core
langchain-openai
langchain-anthropic
langchain-google-genai
langchain-community
langchain-milvus
langgraph
pymilvus
python-dotenv
pydantic
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd /path/to/project/T21_research_assistant_v7/srv
```

### 2. Configure environment variables

```bash
# Edit .env and fill in your API keys
nano .env
```

Key variables to set:

- `OPENAI_API_KEY`: OpenAI API key (required for OpenAI)
- `ANTHROPIC_API_KEY`: Anthropic API key (required for Anthropic)
- `PROVIDER_CHAT`: LLM provider - options: `ollama_remote` (default — `nemotron-3-super` via Ollama cloud for chat + judge, reliable), `deepinfra` (DeepInfra Nemotron, can be `engine_overloaded`/429), `openai`, `anthropic`, `google`. The `judge_improve` node uses Ollama remote (`gpt-oss:120b-cloud`) whenever `PROVIDER_CHAT != deepinfra`.
- `PROVIDER_EMBED`: Embedding provider - options: `deepinfra` (default, 2048-dim), `openai`, `google`, `nemotron`. **Cannot switch without re-embedding corpus + matching `MILVUS_DIM`/collection.**
- `PROVIDER_RERANK`: Reranking provider - **current production default: `deepinfra`** (uses DeepInfra REST API, no local GPU needed). Set to `nemotron` to use the local `NemotronCrossEncoderReranker` (requires GPU).
- `DEEPINFRA_API_KEY`: Bearer key for all DeepInfra calls (chat, embed, rerank, judge) — **read from `.env`, never hardcoded**.
- `DEEPINFRA_BASE_URL`: OpenAI-compatible base for DeepInfra chat + embeddings (default: `https://api.deepinfra.com/v1/openai`)
- `DEEPINFRA_CHAT_MODEL`: DeepInfra synthesis/improve model (default: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`)
- `DEEPINFRA_EMBED_MODEL`: DeepInfra embedding model, dim=2048 (default: `nvidia/llama-nemotron-embed-vl-1b-v2`)
- `DEEPINFRA_EMBED_BATCH_SIZE`: Batch size for DeepInfra embedding API calls (default: `32`)
- `DEEPINFRA_JUDGE_MODEL`: LLM-as-judge scoring model (default: `openai/gpt-oss-20b`)
- `MILVUS_URI`: Milvus connection URI, optional, overrides host/port (default: unset)
- `MILVUS_COLLECTION`: Vector collection name (default: `DS_INCLUDE_2026_09_02_DEEPINFRA_2048`; previous `DS_PMC_v7_DEEPINFRA_2048` and legacy 4096-dim `DS_PMC_05_2026_NEMOTRON_4096` kept for rollback)
- `MILVUS_DIM`: Embedding vector dimension (default: `2048`; was `4096` under local Nemotron)
- `OPENAI_CHAT_MODEL`: Chat model name (default: `gpt-5.4`)
- `OPENAI_COMPARE_MODEL`: Model used for GPT responses in survey builder (default: `gpt-5.4`)
- `ANTHROPIC_COMPARE_MODEL`: Model used for Claude responses in survey builder (default: `claude-sonnet-4-6`)
- `OLLAMA_BASE_URL`: Local Ollama server URL, used when `PROVIDER_CHAT=ollama` (default: `http://localhost:11434`). **NOT RECOMMENDED on this server — local nemotron-3-super (86 GB) is unusably slow on CPU. Use `ollama_remote` instead.**
- `OLLAMA_REMOTE_BASE_URL`: Remote Ollama API endpoint for notebooks 5 and 6, Nemotron (default: unset)
- `OLLAMA_API_KEY`: Bearer token for the remote Ollama API endpoint (default: unset)
- `OLLAMA_NEMOTRON_MODEL`: Nemotron model name used in benchmark/UMAP notebooks (default: `nemotron-3-super`)
- `NEMOTRON_CHAT_MODEL`: Nemotron chat model, used when `PROVIDER_CHAT=ollama_remote` (default: `nemotron-3-super`)
- `NEMOTRON_EMBED_MODEL`: Nemotron embedding model path (default: `./local_hugging_face_model/embed`). Model weights stored in the project directory. Docker overrides to `/app/local_hugging_face_model/embed`. **HuggingFace is blocked on the deployment server — download on dev machine, then rsync into `local_hugging_face_model/embed/`. See [HuggingFace Model Transfer](#step-2b-transfer-huggingface-models-to-deployment-server) below.**
- `NEMOTRON_RERANK_MODEL`: Nemotron cross-encoder reranker path (default: `./local_hugging_face_model/rerank`). **Only loaded when `PROVIDER_RERANK=nemotron` (GPU required). Not used in current production deployment** (`PROVIDER_RERANK=deepinfra`).
- `NEMOTRON_DEVICE`: Device for the local Nemotron **embedding** model. Left blank in `docker-compose.yml` for auto-detection (`cpu` on this server, `cuda` when GPU is present). Do **not** hard-code `cuda` on a CPU-only server.
- `NEMOTRON_EMBED_BATCH_SIZE`: Batch size for Nemotron embeddings (default: `8`)
- `NEMOTRON_RERANK_BATCH_SIZE`: Batch size for Nemotron cross-encoder (default: `4`)
- `TOP_K_DEFAULT`: Default retrieval count (default: `40`)
- `RERANK_ALPHA`: Vector vs LLM rerank blend, range 0-1 (default: `0.4`)
- `BACKEND_URL`: API URL for Streamlit (default: `http://127.0.0.1:8698`)
- `T21_API_URL`: T21 API URL used by survey builder notebook (default: `http://localhost:8698/answer`)
- `QUALTRICS_BASE_URL`: Qualtrics datacenter base URL (required for Qualtrics integration)
- `QUALTRICS_API_TOKEN`: Qualtrics API token (required for Qualtrics integration)

### 3. Ensure Milvus is running

```bash
# Check Milvus health
curl http://localhost:19530/healthz
```

---

## Running the Services

> **Prerequisites**: Milvus must be running at `localhost:19530` and the conda env `dschat` must be active.

### Quick Start (both services at once)

```bash
cd /path/to/project/T21_research_assistant_v7/srv

# Start API + UI
sudo docker compose up -d
```

- **API** available at: `http://localhost:8698`
- **UI** available at: `http://<server-ip>:8878`

---

### Restart API

```bash
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo "Cache cleared" && \
curl -s http://localhost:8698/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Status:', d.get('status'))"
```

### Restart Streamlit UI

```bash
sudo docker compose restart ui && sleep 10 && sudo docker compose ps ui
```

### Apply .env changes (collection cutover, provider switch)

A plain `restart` does NOT pick up `.env` changes: the api container's environment is
baked in at container creation via `env_file:`, and `/app/.env` is not mounted into it,
so the container must be RECREATED with `up -d`. Use this after editing
`MILVUS_COLLECTION`, provider settings, or any other `.env` value (and keep
`PICKLE_DIR` in `docker-compose.yml` pointing at the pickles of the ACTIVE collection,
or an empty-collection recovery would restore the wrong corpus):

```bash
cd /path/to/project/T21_research_assistant_v7/srv
sudo docker compose up -d api ui && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo " cache cleared" && \
curl -s http://localhost:8698/health
```

Verify the health output shows the intended `milvus_collection`, `total_vectors`,
`estimated_documents` and `include_count`. Rollback: restore the previous
`MILVUS_COLLECTION` line in `.env` (kept as a comment next to the active one) and run
the same `up -d` command again.

---

## Docker Deployment — `<deployment-server>`

> **Target server:** `<deployment-server>` (CU Anschutz HPC/bioinformatics server)
> **No GPU available** — all LLM inference routes through external cloud APIs.
> All `docker compose` commands run from the `srv/` directory.

### Docker Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build (targets: `api`, `ui`, `data-processor`) |
| `docker-compose.yml` | **Production** — API + UI + data-processor; connects to an existing Milvus instance on the host |
| `docker-compose.milvus.yml` | Shared Milvus vector DB — run **once per server**, never restarted |
| `docker-compose.gpu.yml` | GPU overlay (future use — not active on this server) |
| `start.sh` | Auto-detecting launcher; merges GPU overlay only when `nvidia-smi` is found |

### System Requirements

| Requirement | Value | Notes |
|-------------|-------|-------|
| OS | Linux Ubuntu 20.04+ | |
| Docker | 24.0+ | |
| Docker Compose | 2.20+ | |
| RAM | 8 GB sufficient | No local model loaded under the DeepInfra stack (legacy local Nemotron embed needed ~14 GB) |
| Storage | 60 GB free | Project + Milvus volumes (local model weights no longer required under DeepInfra) |
| GPU | **Not required** | All four components (synthesis, embeddings, reranking, judge) → DeepInfra REST API |
| Internet | Required (outbound HTTPS) | DeepInfra — needs outbound port 443 |

### Current Deployment: Zero-GPU Hybrid (Ollama remote for chat/judge, DeepInfra for embed/rerank)

| Component | Mode | Config key |
|-----------|------|-----------|
| **LLM synthesis / judge-improve / question-judge** | Ollama remote cloud (`nemotron-3-super`) — reliable; DeepInfra Nemotron was `engine_overloaded`/429 | `PROVIDER_CHAT=ollama_remote` |
| **LLM-as-judge scoring** | Ollama remote (`gpt-oss:120b-cloud`) — used when `PROVIDER_CHAT != deepinfra` | `OLLAMA_API_KEY` |
| **Embeddings** | DeepInfra (`nvidia/llama-nemotron-embed-vl-1b-v2`, **2048-dim**, no local model) | `PROVIDER_EMBED=deepinfra` |
| **Reranking** | DeepInfra REST API — no local model, no GPU | `PROVIDER_RERANK=deepinfra` |
| **Vector DB** | Milvus on host (`host.docker.internal:19530`), collection `DS_INCLUDE_2026_09_02_DEEPINFRA_2048` | `MILVUS_HOST=host.docker.internal`, `MILVUS_DIM=2048` |

> All four model components authenticate with a single `DEEPINFRA_API_KEY` read from `.env`. Under the DeepInfra stack no HuggingFace weights are loaded; the `TRANSFORMERS_OFFLINE=1` / local-weights guidance applies only to the legacy `PROVIDER_EMBED=nemotron` path.

---

### Step 0 — Transfer Project to Deployment Server

Run from your **dev machine**:

```bash
DEV_PROJECT=/path/to/T21_research_assistant_v7
REMOTE=<username>@<deployment-server>
REMOTE_PATH=/path/on/server/T21_research_assistant_v7

# Transfer everything except git history, Milvus volumes, and Python caches
rsync -avz --progress \
  --exclude='.git' \
  --exclude='srv/volumes/' \
  --exclude='**/__pycache__' \
  --exclude='**/*.pyc' \
  --exclude='srv/local_hugging_face_model/' \
  $DEV_PROJECT/ $REMOTE:$REMOTE_PATH/
```

> Transfer `local_hugging_face_model/embed/` separately in Step 2 (it is ~14 GB).

---

### Step 1 — Configure Environment Variables

SSH into the server, then:

```bash
cd /path/on/server/T21_research_assistant_v7/srv
cp .env.example .env   # if starting fresh
nano .env
```

**Required keys for the current no-GPU deployment (hybrid stack):**

```bash
# ── LLM: nemotron-3-super via Ollama remote cloud ──────────────────────────
PROVIDER_CHAT=ollama_remote
NEMOTRON_CHAT_MODEL=nemotron-3-super
OLLAMA_REMOTE_BASE_URL=https://ollama.com
OLLAMA_API_KEY=<your-ollama-bearer-token>

# ── Embeddings: DeepInfra REST API — 2048-dim, no GPU, no local model ──────
PROVIDER_EMBED=deepinfra
DEEPINFRA_API_KEY=<your-deepinfra-api-key>       # https://deepinfra.com/
DEEPINFRA_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
DEEPINFRA_EMBED_BATCH_SIZE=32

# ── Reranker: DeepInfra REST API — no GPU, no local model ──────────────────
PROVIDER_RERANK=deepinfra
DEEPINFRA_RERANK_URL=https://api.deepinfra.com/v1/inference/nvidia/llama-nemotron-rerank-vl-1b-v2

# ── Milvus ─────────────────────────────────────────────────────────────────
MILVUS_HOST=host.docker.internal      # set in docker-compose.yml environment block
MILVUS_PORT=19530
MILVUS_URI=http://host.docker.internal:19530
MILVUS_COLLECTION=DS_INCLUDE_2026_09_02_DEEPINFRA_2048
MILVUS_DIM=2048

# ── Feature flags ──────────────────────────────────────────────────────────
ENABLE_RERANK=true
ENABLE_JUDGE=true
ENABLE_SAFETY_GUARD=true

# ── API / UI ───────────────────────────────────────────────────────────────
DEFAULT_TOPK=12
BACKEND_URL=http://127.0.0.1:8698

# ── Other API keys (for benchmark notebooks) ───────────────────────────────
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>
```

> **Do not set** `NEMOTRON_DEVICE` in `.env` — it is left **blank** in `docker-compose.yml` so the code auto-detects CPU on this server.

---

### Step 2 — Transfer HuggingFace Embedding Model

> `huggingface.co` is **blocked** on the deployment server. Download on your dev machine and rsync across.

**On your dev machine:**

```bash
# Download nvidia/llama-embed-nemotron-8b (~14 GB) into project-local dir
python3 -c "
from huggingface_hub import snapshot_download
snap = snapshot_download('nvidia/llama-embed-nemotron-8b')
print('Downloaded to:', snap)
"

# Copy snapshot into project-local embed dir (replace <HASH> with output above)
cp -rL ~/.cache/huggingface/hub/models--nvidia--llama-embed-nemotron-8b/snapshots/<HASH>/. \
  /path/to/T21_research_assistant_v7/srv/local_hugging_face_model/embed/
```

**Sync to deployment server (~14 GB, use tmux/screen):**

```bash
REMOTE=<username>@<deployment-server>
REMOTE_SRV=$REMOTE:/path/on/server/T21_research_assistant_v7/srv

rsync -avz --progress \
  /path/to/T21_research_assistant_v7/srv/local_hugging_face_model/embed/ \
  $REMOTE_SRV/local_hugging_face_model/embed/

# Also sync UI images (logos)
rsync -avz --progress \
  /path/to/T21_research_assistant_v7/srv/ui/images/ \
  $REMOTE_SRV/ui/images/
```

> `local_hugging_face_model/rerank/` weights are **not needed** — reranking uses DeepInfra REST API, no local model is loaded.

**Verify on the server:**

```bash
ls srv/local_hugging_face_model/embed/
# Expected: config.json  model-00001-of-00004.safetensors  tokenizer.json  tokenizer_config.json ...
```

---

### Step 3 — Start Shared Milvus (run once per server)

If Milvus is already running on this server, **skip this step** — do not restart or recreate the containers.

```bash
# Check if Milvus is already running
sudo docker ps --filter name=milvus_shared

# If all three containers are listed, skip to Step 4.
# If none appear, start Milvus:
cd /path/on/server/T21_research_assistant_v7/srv
sudo docker compose -f docker-compose.milvus.yml up -d

# Wait ~60 s then verify
curl http://localhost:9091/healthz    # → {"status":"ok"}
curl http://localhost:19530/healthz   # → {"status":"ok"}
```

**Container names:** `milvus_shared`, `milvus_shared_etcd`, `milvus_shared_minio`

> **Never** stop/remove Milvus containers to refresh vector data. To reload vectors, drop only the *collection* inside Milvus and re-insert from pickles — see [Corpus Re-indexing](#corpus-re-indexing-embedding-pipeline).

---

### Step 4 — Build and Deploy T21 App (API + UI)

```bash
cd /path/on/server/T21_research_assistant_v7/srv

# First deploy — build images and start
sudo docker compose up -d --build

# Subsequent starts (no rebuild needed unless Dockerfile or requirements.txt changed)
sudo docker compose up -d

# Check status
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

**Container names:** `t21_res_assi_api` (port 8698), `t21_res_assi_ui` (port 8878)

**Access:**

| Endpoint | URL |
|----------|-----|
| Streamlit UI (direct) | `http://localhost:8878/t21-res-assi` |
| Streamlit UI (via Apache proxy) | `http://<deployment-server>/t21-res-assi` |
| API health | `http://localhost:8698/health` |
| API docs | `http://localhost:8698/docs` |

> The Streamlit UI runs with `--server.baseUrlPath /t21-res-assi` (set in `docker-compose.yml`) for Apache reverse proxy compatibility.

---

### Step 5 — Verify Deployment

```bash
# 1. Check all containers are healthy
sudo docker ps --format 'table {{.Names}}\t{{.Status}}'

# 2. Confirm API is up and Milvus is connected
curl -s http://localhost:8698/health | python3 -c \
  "import sys,json; d=json.load(sys.stdin); \
   print('Status:', d.get('status')); \
   print('Vectors:', d.get('database',{}).get('total_vectors')); \
   print('Model:', d.get('config',{}).get('chat_model'))"

# 3. Test a question end-to-end
curl -s -X POST http://localhost:8698/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"What cardiac conditions are common in Down syndrome?","top_k":8}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['structured']['summary'][:300])"

# 4. Check UI is responding
curl -sf http://localhost:8878/t21-res-assi/_stcore/health && echo "UI OK"
```

---

### Ongoing Management

#### Restart API (after editing `api/` files)

```bash
cd /path/on/server/T21_research_assistant_v7/srv
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo "Cache cleared" && \
curl -s http://localhost:8698/health | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('Status:', d.get('status'))"
```

#### Restart UI (after editing `ui/` files)

```bash
sudo docker compose restart ui && sleep 10 && sudo docker compose ps ui
```

#### View Logs

```bash
sudo docker compose logs -f api          # API live log
sudo docker compose logs --tail=100 api  # API last 100 lines
sudo docker compose logs -f ui           # UI live log
```

#### Clear Response Cache

```bash
curl -X POST http://localhost:8698/cache/clear
```

#### Stop All Services (leaves Milvus running)

```bash
sudo docker compose down
```

#### Full Rebuild (after `requirements.txt` or `Dockerfile` changes)

```bash
sudo docker compose down
sudo docker compose up -d --build
```

---

## API Reference

Base URL: `http://localhost:8698`

### `GET /`

Returns service name, version, status, and available endpoints.

### `GET /health`

Returns configuration details and Milvus database statistics (total vectors, unique documents).

**Response example:**
```json
{
  "status": "healthy",
  "config": {
    "architecture": "LangChain/LangGraph",
    "provider_chat": "ollama_remote",
    "chat_model": "nemotron-3-super",
    "milvus_collection": "DS_INCLUDE_2026_09_02_DEEPINFRA_2048",
    "top_k_default": 15,
    "rerank_alpha": 0.8
  },
  "database": {
    "total_vectors": "65,677",
    "estimated_documents": "1,594",
    "include_count": "158"
  }
}
```

> The collection stores only papers satisfying every corpus criterion (including `pub_type == "article"`), so `estimated_documents` equals the stored document count exactly. How 1,594 comes about: 2,075 records in the 2026-09-02 PMC snapshot → 1,891 eligible after the metadata gate → 1,837 with usable full text → 1,669 with ≥4 of 6 sections → 1,594 original research articles (see `srv/nbs/2_embed_and_index_ds_papers.ipynb`, "How the document count comes about").

### `POST /answer`

Main RAG endpoint. Returns a synthesized answer with citations.

**Request:**
```json
{
  "question": "What is the prevalence of autism in Down syndrome?",
  "top_k": 12,
  "include_hits": false
}
```

**Response fields:**

- `markdown`: Full synthesized response in markdown
- `structured`: Parsed section dict
- `citations`: List of cited PMC IDs
- `trace`: Pipeline execution trace with timing
- `raw_evidence`: Top-K evidence chunks used for synthesis
- `hits`: Raw vector search hits (if `include_hits=true`)
- `re_ranked_hits`: Reranked hits (if `include_hits=true`)

Responses are cached by MD5(question + top_k). Call `POST /cache/clear` to invalidate.

### `POST /search`

Debug endpoint — vector search only, no synthesis.

**Request:**
```json
{
  "query": "cardiac defects trisomy 21",
  "top_k": 10
}
```

### `POST /cache/clear`

Clears the in-memory query response cache.

### `POST /embed`

Embeds a single text using the already-loaded `nvidia/llama-embed-nemotron-8b` model (4096-dim).

**Request:**
```json
{ "text": "What cardiac conditions are common in Down syndrome?", "mode": "query" }
```

| `mode` | Instruction prefix | Use for |
|--------|-------------------|---------|
| `query` | Added | Retrieval queries |
| `document` | None | Passages, model responses, ground truth |

**Response:** `{ "embedding": [...], "dim": 4096 }`

### `POST /embed_batch`

Batch-embeds multiple texts (order preserved). Used by Notebook 5 (BERTScore) and Notebook 6 (UMAP).

**Request:**
```json
{ "texts": ["text one", "text two"], "mode": "document" }
```

**Response:** `{ "embeddings": [[...], [...]], "dim": 4096, "count": 2 }`

---

## RAG Pipeline Details

### 1. Retrieval

- Milvus IVF_FLAT index (`nlist=128`, `nprobe=64`), Inner Product metric, embedding dim 4096 (Nemotron)
- Retrieves `top_k * 5` candidates (5x oversampling for diversity)
- **Milvus filter:** `pub_type != "review" AND section IN ["Results", "Discussion", "Methods", "Conclusion"]`
- **Dynamic top_k boost:** complex questions (containing keywords like `compare`, `versus`, `mechanisms`, `therapies`, etc.) or questions longer than 100 chars get `top_k` boosted by 3–5; broad comorbidity/list questions (`which/what conditions|diseases|comorbid|complications|risks` or `higher/lower risk`) get an additional +8 boost — capped at 20
- Paper diversity: max 2 chunks per PMCID (configurable via `MAX_CHUNKS_PER_PAPER`)
- Section diversity: enforces minimum chunks from Results (`DIVERSITY_MIN_RESULTS=6`), Discussion (`DIVERSITY_MIN_DISCUSSION=4`), Methods (`DIVERSITY_MIN_METHODS=3`)

### 0. Question Judge (pre-flight)

Runs as the graph entry point before any retrieval.

- Single fast LLM call (`_JUDGE_SYSTEM` prompt) classifies the question as `VALID` or `INVALID: <one sentence reason>`
- **Requires DS relevance** — VALID only when the question is explicitly about Down syndrome / Trisomy 21, OR about a condition/gene/mechanism/comorbidity known to co-occur with or be studied in the context of DS (e.g. Alzheimer's, leukemia, hypothyroidism, DYRK1A, APP, Ts65Dn/Dp16 models). General biomedical questions with no DS connection are INVALID even if scientifically valid in another context.
- **INVALID cases:** (1) completely off-topic domains (cryptocurrency, astrophysics, etc.); (2) questions that depend on a clearly non-existent or non-DS biological entity; (3) personal/biographical identity questions about a named individual (e.g. "Who is [person]?") that are not biomedical research questions; (4) general biomedical questions with no DS connection — e.g. "What causes type 2 diabetes?" is INVALID, but "What causes type 2 diabetes in Down syndrome?" is VALID
- **Fails open** — any LLM error defaults verdict to `VALID` so no valid question is silently blocked
- `_JUDGE_CACHE` (process-lifetime dict) deduplicates repeated identical questions — skips the LLM call on repeats
- `question_verdict: str` stored in `RAGState`; conditional edge `_judge_route` routes `INVALID…` → `end_invalid → END`, everything else → `retrieve`
- When INVALID, the judge's reason is passed directly to `_evidence_gap_response` as `judge_reason` — skips the second LLM justification call (no double-billing)

### 2. Reranking

- **Current production mode:** `PROVIDER_RERANK=deepinfra` → uses `DeepInfraReranker`, which calls `POST https://api.deepinfra.com/v1/inference/nvidia/llama-nemotron-rerank-vl-1b-v2` with Bearer auth. No local model is loaded; no GPU required.
- **Alternative (GPU servers):** `PROVIDER_RERANK=nemotron` → uses `NemotronCrossEncoderReranker` (`nvidia/llama-nemotron-rerank-1b-v2`, local HuggingFace, requires CUDA).
- Scoring formula (two steps):
  1. `combined = alpha * vector_score_norm + (1 - alpha) * ce_score * 100`
  2. `final = combined * section_priority`
- `alpha = 0.4` (configurable via `RERANK_ALPHA`; higher = more vector weight)
- Section priorities (standard mode): Results/Discussion/Methods/Conclusion = **1.00**, Abstract/Introduction/FullText/NA = **0.00** — direct multiplier, no β
- In **`with_abstract` mode** (`sections_override` set): all explicitly requested sections receive `priority = 1.0` so Abstract/Introduction compete on equal footing with active sections
- `SECTION_BONUS_SCALE` removed — section weighting is now a direct priority multiplier with no extra β parameter
- Toggle with `ENABLE_RERANK=true/false`

### 3. Synthesis

The LLM generates three output sections (all required):

**Summary** — max 150 words, C-C-C scheme:
1. **Context** — ONE orienting sentence on clinical/biological significance. **Never opens with a generic DS description** ("Down syndrome is a chromosomal condition…") — the reader is a researcher; assume knowledge of DS.
2. **Content** — 3–5 finding sentences. **Inverted pyramid: most important quantitative finding first**, subsequent sentences add context/nuance. Each sentence must include a `[PMCxxxxxxx:Section]` citation and quantitative data where available.
3. **Conclusion** — ONE synthesis sentence that directly answers the question, naming the specific entity (gene/pathway/condition — never "these factors" or "this pathway"). Must be a synthesis statement; raw data or "further research is needed" are banned.

**Writing-style rules baked into the synthesis prompt:**
- No generic DS boilerplate openers — start immediately with a finding or clinical fact
- No herald phrases: `Notably,` `Importantly,` `Interestingly,` `It is worth noting that…` `This underscores…` `These results highlight…`
- No hedging stacks: `may potentially suggest` → pick ONE level of certainty
- No meta-commentary: `Based on the evidence provided…` `The available evidence suggests that…`
- No redundant attribution: `research has shown` `studies indicate` — cite directly
- Person-first language: "individuals with DS", not "DS patients"

**Key Evidence** — 4–6 bullets with precise statistics and section-aware citations (`[PMCxxxxxxx:Results]`).

**Reasoning** — Exactly 4 bullet points:
- **Evidence selection** — why these specific findings were prioritised (quality, section type, sample size, replication)
- **Reasoning chain** — how the evidence logically supports the Summary conclusion
- **Confidence & gaps** — where evidence is strongest and where limitations exist
- **Reader caveats** — what the reader should bear in mind when interpreting

No citations required in Reasoning; bare `PMCxxxxxxx` IDs are auto-hyperlinked in the UI. A fallback Reasoning section is guaranteed at three layers: (1) the **finalize node** checks for `# Reasoning` AND a non-empty body — if the section is absent or the LLM produced only the header with no bullet content, the empty header is stripped and a 4-bullet fallback is injected; (2) the **API cache guard** (`main.py`) refuses to cache any response where the Reasoning body is empty; (3) the **UI safety net** (`ui_streamlit_ds_chat.py`) patches stored session data at render time using the same empty-body check. All three locations use `re.search(r"(?im)^#\s+Reasoning\s*\n(.*?)(?=^#\s+|\Z)", md, re.S|re.M)` and verify `.group(1).strip()` contains content before considering the section present.

- Temperature: `0.0`, seed: `42` (deterministic output)
- Deduplication: highest-scoring chunk per (PMCID, section) pair kept
- **Evidence gap handling:** when evidence is insufficient, returns a single `# Summary` section with an LLM-generated 1–2 sentence justification explaining *why* the question cannot be answered from the DS literature. The justification distinguishes truly off-topic questions (biological reason stated) from valid DS questions not well-covered in this 1,594-article corpus (corpus scope acknowledged — does not fabricate biology). When the `question_judge` node provides a `judge_reason`, that reason is used directly — no second LLM call. False-gap prevention operates at four layers: (1) **Pre-flight judge** — `question_judge` classifies borderline questions before any retrieval (VALID only if explicitly about DS, Trisomy 21, or a known DS comorbidity/mechanism — general biomedical questions with no DS connection are INVALID); (2) **Two-tier off-topic filter** — `_missing_terms_in_hits` hard-blocks only questions containing a `_HARD_OFFTOPIC_TERMS` term (plant biology, non-human microbiology, astrophysics, etc.) absent from all retrieved evidence; (3) **Stoplist** — `_extract_key_terms` strips question-framing, meta-research, generic anatomical, and plain-language risk/frequency framing words (`prone`, `blood`, `pressure`, `increased`, `decreased`, `symptom`, `disease`, etc.); (4) **Misspelling normalization** — `_SPELL_NORMALIZE` maps common medical misspellings to canonical forms before the stoplist (`disfunction` → `dysfunction`, `leukaemia` → `leukemia`, `cognative` → `cognitive`) so user typos are absorbed rather than extracted as unmatched terms. `_term_variants` expands clinical synonyms (`hypertension` ↔ `blood pressure`, `seizure` ↔ `epilepsy`, `obesity` ↔ `obese`, `dementia` ↔ `alzheimer`, `leukemia` ↔ `aml`, `diaphragmatic`/`diaphragm` ↔ `cdh`, `hernia` ↔ `cdh`) so framing variations across papers do not trigger a false gap.
- **Named entity check (`_unmatched_named_entities`):** gene-symbol-like tokens (e.g. `FX84A9`) absent from all retrieved evidence are stored in `state["unmatched_entities"]`; the synthesis prompt is injected with a `[NAMED ENTITY CHECK]` block forcing the LLM to note the absent entity rather than silently producing a generic response. `_NAMED_ENTITY_EXCLUDE` prevents common acronyms (`DS`, `IQ`, `MRI`, etc.) from being flagged.
- **Structured output quality gate (Nemotron path):** if `_synthesize_structured` returns output, the `# Summary` body is checked for ≥ 100 words, ≥ 1 inline `[PMCxxxxxxx:Section]` citation, AND no raw-data artifact. If any check fails, the result is discarded and the standard markdown chain runs instead.
- **Raw-data artifact stripping (`_strip_raw_data_sentences`):** applied to both Summary and Key Evidence after synthesis. The Summary body is pre-cleaned before sentence analysis: journal call-out box labels (e.g. `RESEARCH IN CONTEXT Systematic review:`) are stripped by `_BOILERPLATE_LABEL_RE`, and author-year inline citations (e.g. `(Li et al., 2007)`) are removed by regex. Summary sentences containing verbatim sample inventories (e.g. `n = 25, 16 male and 9 female`), genotype labels (`unstimulated WT`, `Dp162x…`), first-person methods narration (including standalone `we employed` without requiring a `to investigate` tail), or journal-section headers are then dropped via `_RAW_DATA_RE`. Additional `_RAW_DATA_RE` patterns: `in order of (prevalence|frequency|decreasing|increasing)` (catches verbatim ordered-prevalence rosters) and `\bvs\.\s*$` (catches truncated comparison sentences ending with "vs." where the second value was cut off). Both patterns are also in `_RAW_DATA_KE_RE`. Key Evidence bullets whose prose body matches `_RAW_DATA_KE_RE` (same patterns minus legitimate statistical notation OR/HR/CI/p-values) are dropped entirely. `_strip_raw_data_sentences` is also called after `_pad_summary_from_ke_final` in the finalize pipeline, because the padding step draws from KE bullets which can introduce raw-data artifacts not previously caught. Parenthetical numeric citation markers (e.g. `(45)`, `(9, 10, 12, 13, 15, 17)`), word-glued superscripts (e.g. `death4,54`), and space-padded inline numeric runs are removed from both the **Summary** and each **Key Evidence** bullet by `_strip_summary_numeric_refs` via a shared helper `_strip_numeric_refs_from_text`. The word-glued superscript lookahead was extended to include `[` so that `death4,54[PMC...]` (no space before the bracket) is cleaned correctly.
- **Hallucination filter (`_strip_hallucinated_citations`):** PMC citation tags whose IDs are absent from the actual retrieved evidence corpus are stripped. A **pre-pass** strips prompt-exemplar placeholder citations where the PMC "ID" is entirely letters (e.g. `[PMCxxxxxxx:Results]` copied verbatim from the synthesis prompt instructions) via `re.sub(r'\s*\[PMC[a-zA-Z]{3,}:[^\]]+\](?:\([^)]+\))?', '', md)` — these never match the `\d+` pattern in the main loop. After the pre-pass, the main loop strips tags whose numeric IDs are absent from the corpus. In the Summary, only the tag is removed so the fallback citation can be re-added. In Key Evidence, entire bullets are dropped when all their cited PMC IDs are hallucinated. Prompt exemplar IDs were changed to `PMC00000001`/`PMC00000002` so any LLM copy of the exemplar is caught by both the pre-pass (if letters) and the main loop (if formatted as digits). The strip is preceded by `_repair_mutated_pmc_ids`: a citation whose numeric ID is a digit-level mutation of exactly one real corpus ID (duplicated, inserted, or dropped digits — e.g. `PMC979796221` → `PMC9796221`) is rewritten to the real ID instead of being stripped, keeping the sentence cited (logged `[CITE_REPAIR]`); ambiguous or unmatched IDs fall through to the strip. `_strip_dangling_stat_parens` additionally removes truncated statistical parentheticals glued to a citation (e.g. `(p[PMC...]`, `(OR 3.4, [PMC...]`) while never touching complete ones like `(p=0.013) [PMC...]`. All three run in the finalize node AND are re-applied in `main.py`'s judge-improve splice path, because the improve model bypasses finalize.
- **Summary length floor (`_pad_summary_from_ke_final`):** last-resort minimum word count enforcement called after all stripping, dedup, and citation steps. If `_summary_word_count < 100`, appends clean prose sentences extracted from Key Evidence bullets until the count reaches 130, then re-applies the 150-word cap. Ensures consistent 100–150-word output regardless of how much earlier passes stripped. After padding: `_dedupe_summary_sentences` re-runs (pad can reintroduce near-duplicates), then `_strip_opposite_polarity_sentences` re-runs (pad can introduce KE bullets with opposite polarity), then `_strip_raw_data_sentences` re-runs (pad can introduce verbatim KE artifact sentences such as truncated comparisons ending with "vs." or ordered-prevalence rosters).
- **Verbatim bracket sentence stripping (`_strip_verbatim_bracket_sentences`):** detects and removes Summary sentences where the LLM echoed a raw evidence chunk verbatim inside brackets, e.g. `[In mouse models, Stat protein overexpression was observed...] [PMC:Discussion]`. Pattern: `\[(?!PMC\d|https?://|#|EVIDENCE_GAP|INCLUDE)[^[\]]{15,}[.!?]\]`. Applied in the finalize pipeline before `_strip_icu_for_comorbidity_q`.
- **ICU/ventilation sentence filtering (`_strip_icu_for_comorbidity_q`):** applied when the question matches `_COMORBIDITY_Q_RE` (conditions/comorbidities/diseases/higher risk). Strips Summary sentences containing ICU admission rates, mechanical ventilation rates, intubation rates, or hospital length-of-stay — acute-care severity metrics that contaminate comorbidity-enumeration answers. The first context sentence is always preserved; stripping is skipped if it would leave fewer than 2 sentences.
- **Comma-list sentence stripping (`_strip_comma_list_sentences`):** two-pass filter (early in finalize; again just before `_hyperlink_inline_pmc`). Removes uncited sentences matching `at (elevated|higher|increased) risk for X, Y, and Z` with ≥ 2 comma-separated items (`_COMMALIST_SUMMARY_RE`) — a degenerate list-question output pattern from the retry path. First sentence always preserved; skipped if stripping would leave fewer than 2 sentences.

### 4. Citation Format

Inline citations use the format: `[PMCxxxxxxx:Section]`
These are converted to hyperlinks: `[PMCxxxxxxx:Results](https://pmc.ncbi.nlm.nih.gov/articles/PMCxxxxxxx/)`

**UI deep-linking (Text Fragments):** the Streamlit UI independently rebuilds every citation URL client-side (`_process_pmc_citations` in `ui_streamlit_ds_chat.py`). When the actual retrieved chunk text for that citation's PMCID + section is available, the rebuilt link appends a browser [Text Fragment](https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Fragment/Text_fragments) — `#:~:text=<snippet>` — so Chrome/Edge scroll to and highlight the exact cited passage on the live PMC page instead of just opening the article. PMC has no stable, predictable per-section anchor id, so this uses the browser's native page-text search rather than a guessed anchor; unsupported browsers (e.g. Safari) just ignore the fragment and land on the article as before. Applies to on-screen citation links and evidence-card PMC links only — the Export & Copy downloads still use the backend's plain (non-fragment) URLs.

### 5. Safety Guard

Scans output for prescription-directive language directed at patients (e.g. "take X tablets", "you should stop"). Drug/treatment findings reported from the literature (e.g. dosages, adverse events in trials) are allowed. If flagged, prepends a research disclaimer banner. Toggle with `ENABLE_SAFETY_GUARD=true/false`.

### 6. Finalize

Converts inline citations `[PMCxxxxxxx:Section]` to clickable hyperlinks and enforces the 150-word summary cap.

### 7. judge_improve (LLM-as-Judge)

Enabled via `ENABLE_JUDGE=true`. Skipped for evidence-gap responses or when `OLLAMA_API_KEY` is unset.

| Step | Model | What happens |
|------|-------|--------------|
| **Score** | `gpt-oss:120b-cloud` | Single LLM call — scores Summary on **Accuracy**, **Informativeness**, **Completeness** (1–5); returns structured JSON `{"accuracy":…,"informativeness":…,"completeness":…,"justification":"…"}`. `temperature=0.0` for deterministic scoring. |
| **Rewrite** *(conditional)* | `nemotron-3-super` | Called **only if any criterion < 4**. Rewrites the Summary guided by the weak dimensions and judge notes. `temperature=0.0`, `seed=42` for deterministic output. If all scores ≥ 4 the rewrite is skipped entirely. |
| **Splice** | `main.py` | If a rewrite was produced, it first passes a **quality gate** that discards the improved text if: (1) `_diversity_regressed` — fewer unique papers than original (when original had ≥2); (2) `_citation_stripped` — fewer inline `[PMCxx:Section]` tags than original (when original had ≥3, allows drop of 1); (3) `_semicolon_runon` — ≥3 `[cite];` patterns (degenerate output). Discarded rewrites log `[IMPROVE_QG]`. If the gate passes, the improved text replaces the Summary body in the final `markdown`. The post-splice pipeline runs: `_strip_raw_data_sentences` → `_fix_citations_after_period` → `_strip_dangling_stat_parens` → `_repair_mutated_pmc_ids` (digit-mutated IDs rewritten to the real corpus ID, `[CITE_REPAIR]` log) → `_strip_hallucinated_citations` (re-applied here — the improve model bypasses finalize) → **post-strip quality gate** (re-counts the surviving citations after fabricated IDs are removed; if they vanished or regressed the Summary reverts to the original finalize output — `[IMPROVE_QG] reverted improved_summary post-strip`) → `_hyperlink_inline_pmc` → outer-bracket strip → `_dedup_inline_list_items` → `_dedupe_summary_sentences` → `_strip_comma_list_sentences` → `_strip_broken_risk_intro_sentences` → `_strip_further_research_sentences` → `_strip_uncited_summary_sentences` → `_truncate_summary_to_150_words`. A 60-word floor (`_pad_summary_from_ke_final(min_wc=60, target_wc=120)`) is enforced after the splice step regardless of whether a rewrite was accepted, followed by a final citation guarantee: a Summary left with zero citations while valid evidence citations exist gets the top evidence citation appended and hyperlinked (`[CITE_GUARANTEE]` log). |

Trace field reports: `"scored avg=X.X/5; all ≥4 — no rewrite needed"` or `"scored avg=X.X/5; improved=yes/no"`.
- **`_strip_further_research_sentences`:** called in the post-splice pipeline and also in the finalize node (before `_strip_last_sentence_citation`). Removes Summary sentences containing "further research needed" hedging — caught by `_FURTHER_RESEARCH_RE`: `further/more/additional research/investigation needed/required`, `highlighting the need for further/additional`, `remains to be determined/investigated`, `future research should/will/must/may/are needed`. Preserves the sentence if stripping would leave fewer than 2 Summary sentences.
 **Citation normalization (applied at two points—before and after all reconstruction passes):** `_fix_misplaced_bracket_citations` corrects `[PMCxx]:Section]` (closing bracket misplaced before section label) → `[PMCxx:Section]`; `_split_semicolon_citations` expands `[PMC1:Section; PMC2:Section]` into separate `[PMC1:Section] [PMC2:Section]` tokens. Citation processing order: `_strip_last_sentence_citation` → `_strip_dangling_stat_parens` (truncated stat parentheticals glued to a citation) → `_repair_mutated_pmc_ids` (digit-mutated IDs rewritten to the real corpus ID) → `_strip_hallucinated_citations` (drops non-corpus PMC tags from Summary; drops entire Key Evidence bullets when all their IDs are hallucinated) → `_append_summary_citation` (re-adds a real fallback cite if Summary still has none). Handles `[EVIDENCE_GAP]` — if the LLM determines evidence is insufficient, returns a standardised "Insufficient Evidence" response with an LLM-generated 1–2 sentence justification. Also guarantees the `# Reasoning` section has substantive content: if the LLM omitted it **or** produced the header with an empty body, the empty header is stripped and a concise 4-bullet fallback is injected before hyperlinking. The API cache guard likewise refuses to cache responses where the Reasoning body is empty.

---

## Corpus Re-indexing (Embedding Pipeline)

Use this section whenever you need to add new papers to the database, expand the corpus, or recover from a dropped Milvus collection. The pipeline runs entirely via the **`data-processor` Docker service** (profile: `data`) and consists of two scripts.

### Prerequisites — add to `srv/.env`

```bash
NCBI_EMAIL=your@email.edu        # Required by fetch_pmc_publications.py
CROSSREF_EMAIL=your@email.edu    # Required by fetch_pmc_publications.py
# NCBI_API_KEY=...               # Optional — raises NCBI rate limit from 3 to 10 req/s
```

### Script Overview

| Script | What it does | Requires GPU? |
|--------|-------------|---------------|
| `scripts/fetch_pmc_publications.py` | Queries OpenAlex, NCBI Entrez, CrossRef, and PMC OAI to build one enriched JSON per PMID; cross-references INCLUDE Excel to set `is_include` provenance | No |
| `scripts/embed_and_insert.py` | Loads JSON metadata, computes 2048-dim DeepInfra embeddings via REST API (`--compute`), saves pickles, and/or upserts into Milvus (`--insert`). No GPU needed — `--compute` calls `nvidia/llama-nemotron-embed-vl-1b-v2` via `DEEPINFRA_API_KEY`. | No (DeepInfra API) |

> The Milvus collection uses a **31-field schema**. `ensure_collection()` auto-drops and recreates the collection when the field count changes — after any schema change, all data must be re-inserted in full.

### Source Data — MEDLINE Export (`PMC_DownSynd_05_18_2026.txt`)

Down syndrome-related publication metadata were retrieved from PubMed (<https://pubmed.ncbi.nlm.nih.gov>) using the MeSH term search query `"Down syndrome"[MeSH Terms] AND "open access"[filter]`, with results filtered to journal articles available as open-access full text in PubMed Central (PMC). Search results were sorted by most recent publication date to prioritize current literature. Following the search, all matching records were selected using the bulk selection option, and metadata were exported in PubMed MEDLINE format via the **Send to → Citation Manager → PubMed format → Create file** workflow. The downloaded file, saved in plain-text MEDLINE format, contains structured bibliographic fields including PubMed ID (PMID), PMC accession number, title, abstract, authors, MeSH terms, journal details, and publication dates — providing a comprehensive metadata snapshot of the open-access Down syndrome literature available as of the download date.

**Multi-API enrichment (`fetch_pmc_publications.py`):** The MEDLINE export serves as the initial PMID/PMCID list. For each paper, `fetch_pmc_publications.py` then queries four external APIs in sequence to build a fully enriched JSON record:

| Step | API | Data fetched |
|------|-----|-------------|
| 1 | **OpenAlex** (`api.openalex.org`) | OpenAlex ID, DOI, abstract (inverted-index), authors, institutions, funder names, Sustainable Development Goal tags |
| 2 | **NCBI Entrez** (`eutils.ncbi.nlm.nih.gov`) | MeSH terms, keywords, publication type, journal, volume/issue/pages, ISSN |
| 3 | **CrossRef** (`api.crossref.org`) | DOI confirmation, license, referenced-by count |
| 4 | **PMC OAI** (`www.ncbi.nlm.nih.gov/pmc/oai`) | Full-text section-level content (Results, Discussion, Methods, Conclusion) |

The resulting per-PMID JSON files (written to `output/DS_PMC_05_2026/`) are the input to `embed_and_insert.py` for embedding and Milvus insertion.

### Step-by-Step Workflow

All commands run from `srv/`:

**Step 1 — Fetch metadata** (~1.5–3 h, network I/O)

```bash
nohup sudo docker compose --profile data run --rm data-processor \
    python scripts/fetch_pmc_publications.py \
    --medline input/PMC_DownSynd_05_18_2026.txt \
    --include-excel input/INCLUDE_Publications_by_2025-11-14.xlsx \
    --out-dir output/DS_PMC_05_2026 \
    > /tmp/t21v4_fetch.log 2>&1 &
tail -f /tmp/t21v4_fetch.log   # monitor
```

- Resume-safe: skips JSONs that already have an `is_include` field
- Test with `--limit 10` before the full run

> **No GPU required — current stack uses DeepInfra API.** `--compute` calls `nvidia/llama-nemotron-embed-vl-1b-v2` (2048-dim) via the DeepInfra REST API; no local model is loaded. Pickles are computed and inserted directly on the deployment server — no rsync from a GPU dev machine is needed.

**Step 2 — Compute embeddings** (DeepInfra API, no GPU, ~1–2 h for full corpus)

Run via `conda` on the deployment server:

```bash
# On the deployment server (conda env dschat), from srv/
conda run -n dschat --no-capture-output python scripts/embed_and_insert.py \
    --input output/DS_PMC_05_2026 \
    --pickle-dir output/embeddings_pickle/DS_PMC_05_2026_deepinfra_2048 \
    --compute 2>&1 | tee /tmp/t21v7_embed.log
```

- Resume-safe: already-pickled papers are skipped — safe to rerun after interruption
- Reads `DEEPINFRA_API_KEY` from `.env`; batches at `DEEPINFRA_EMBED_BATCH_SIZE=32`

**Step 3 — Insert pickles into Milvus** (no GPU needed, ~5–15 min)

> **Milvus containers must already be running** (check with `sudo docker ps --filter name=milvus_shared`). Never stop or recreate the containers to refresh data — only drop and recreate the *collection* inside Milvus.

Drop the old collection if it exists (optional — `ensure_collection()` drops automatically on schema mismatch):

```bash
# From srv/ — runs inside the data-processor container (no conda env needed)
sudo docker compose --profile data run --rm data-processor python3 -c "
from pymilvus import connections, utility
connections.connect(host='host.docker.internal', port=19530)
if utility.has_collection('DS_INCLUDE_2026_09_02_DEEPINFRA_2048'):
    utility.drop_collection('DS_INCLUDE_2026_09_02_DEEPINFRA_2048')
    print('Collection dropped.')
else:
    print('Collection does not exist — nothing to drop.')
"
```

Then insert from pickles via the `data-processor` Docker service (preferred on the deployment server — reads pickles and writes to Milvus only, no model loaded, CPU-only):

```bash
# On the deployment server, from srv/
sudo docker compose --profile data run --rm data-processor \
    python scripts/embed_and_insert.py \
    --pickle-dir output/embeddings_pickle/DS_PMC_05_2026_deepinfra_2048 \
    --insert 2>&1 | tee /tmp/t21v7_insert.log
```

Alternative (conda, outside Docker):

```bash
conda run -n dschat --no-capture-output python scripts/embed_and_insert.py \
    --pickle-dir output/embeddings_pickle/DS_PMC_05_2026_deepinfra_2048 \
    --insert 2>&1 | tee /tmp/t21v7_insert.log
```

> To re-insert the **current production corpus** (the 2026-09-02 build) rather than a new corpus build, use the pickle dir given under *Quick Re-Insert* below.


Reads `MILVUS_COLLECTION=DS_INCLUDE_2026_09_02_DEEPINFRA_2048` and `MILVUS_DIM=2048` from `.env`; auto-creates the collection at the correct dimension. (The current production collection is built by `srv/nbs/2_embed_and_index_ds_papers.ipynb`, which targets it explicitly regardless of `.env`.)

> **Implementation notes:**
> - **No `col.flush()` call** — `embed_and_insert.py` does NOT call `col.flush()` after inserting. Explicit flush caused `MilvusException: channel not found` on freshly-created collections (DataNode channel registration race condition). Milvus auto-flushes via WAL; data is durably inserted without it.
> - **No GPU needed** — the entire DeepInfra stack (`--compute` + `--insert`) runs without any GPU.
> - **`/health` total_vectors** — uses `collection.query(expr="pk >= 0", output_fields=["count(*)"])` rather than `collection.num_entities`, which only counts sealed (fully flushed) segments and returns `0` for recently inserted data. The `count(*)` query counts both sealed and growing segments.

**Step 4 — Restart API and clear cache**

```bash
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo "Cache cleared" && \
curl -s http://localhost:8698/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Database:', d.get('database',{}))"
```

### Quick Re-Insert (pickles already exist, no embedding needed)

> Milvus containers must be running. This only drops and recreates the collection — containers are never touched.
> **Current corpus:** 1,594 embedded pickles at `output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048/` (65,677 section-level vectors, articles only), built from `output/ds_include_pipeline_2026-09-02/json_cache/` (the canonical build dir; the identical 09-05 rerun was hardlink-merged into it and archived). The pre-purge 1,669-paper set (incl. 75 non-articles) and the other rollback pickle sets are archived under `output/_archive_superseded_2026-09-06/embeddings_pickle/`.

```bash
# 1. Check Milvus containers are running
sudo docker ps --filter name=milvus_shared

# 2. Drop the existing collection (optional — auto-dropped on schema mismatch)
sudo docker compose --profile data run --rm data-processor python3 -c "
from pymilvus import connections, utility
connections.connect(host='host.docker.internal', port=19530)
if utility.has_collection('DS_INCLUDE_2026_09_02_DEEPINFRA_2048'):
    utility.drop_collection('DS_INCLUDE_2026_09_02_DEEPINFRA_2048')
    print('Collection dropped.')
else:
    print('No existing collection.')
"

# 3. Re-insert from pickles (~5–15 min, no GPU, no API calls)
sudo docker compose --profile data run --rm data-processor \
    python scripts/embed_and_insert.py \
    --pickle-dir output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048 \
    --insert 2>&1 | tee /tmp/t21v7_insert.log

# 4. Restart API, clear cache, verify vector count
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo "Cache cleared" && \
curl -s http://localhost:8698/health | python3 -c \
    "import sys,json; d=json.load(sys.stdin); db=d.get('database',{}); print('Vectors:', db.get('total_vectors'))"
```

---

## Data Pipeline (Notebooks)

The `nbs/` folder contains the full ingestion pipeline, evaluation tools, and survey builder:

- `0_ollama_model_test.ipynb`: Test Ollama/Nemotron model connectivity and generation
- `1_fetch_publication_metadata.ipynb`: Fetch metadata (title, abstract, MeSH terms, keywords, sections) for each PMID from PubMed/PMC
- `2_publication_metadata_analysis.ipynb`: Exploratory analysis of the corpus (topic distribution, wordclouds)
- `3_compute_embeddings_to_pickle.ipynb`: Compute section-level embeddings and save as pickle files
- `4_insert_embeddings_to_milvus.ipynb`: Insert embeddings into the Milvus collection
- `5_benchmark_pipeline_vs_ground_truth.ipynb`: Multi-model benchmark — T21 RAG vs GPT-5.4 vs Claude Sonnet 4.6 vs Nemotron-3-Super; BERTScore P/R/F + semantic similarity; size-matched extraction; navy-blue heatmaps
- `6_umap_questions.ipynb`: Figure 4-style 3-panel UMAP per positive-control question — Panel A: corpus manifold by section; Panel B: retrieved chunks (navy→white by rerank score) with A./B./C. topic annotations; Panel C: model response positions vs Ground Truth. Shortcut path: load embeddings from `responses_latest.csv`, embed via Nemotron `/embed` API, fit UMAP on full background corpus, save per-question PDF + JPEG (300 dpi) to `output/umap/`
- `7_expert_survey_qualtrics_design.ipynb`: Survey design and Qualtrics question layout prototyping
- `8_qualtrics_survey_dataset_builder.ipynb`: Generate model responses, blind-randomize, and publish expert ranking survey to Qualtrics

---

## Benchmark & Batch Evaluation

The file `input/AI_research_assistant_Sample_questions_revised_JIM_Negative_Ctrl_dt_03_12_2026.xlsx` contains **24 curated benchmark questions** with expected answers and citations, covering:

- Clinical conditions: Higher/lower-risk conditions, blood pressure, tongue size
- Immune system: Interferon receptors, STAT1 signaling, B cell lymphopenia
- Genetics: Three types of Down syndrome, APP / Alzheimer's, JAM2
- Mouse models: Ts65Dn vs Dp16, behavioral phenotypes, hearing loss (DYRK1A)
- Comorbidities: Thyroid dysfunction, neurodegeneration, cancer risk, liver function

### UI Integration

The Streamlit welcome screen loads all questions from the Excel file automatically. The first 6 are shown as prominent clickable cards in a 2-column grid; the remaining questions appear under **"More sample questions"** (collapsible).

**Streamlit UI features:**

- **Navigation bar:** About, Team, FAQ, GitHub, **Publication** (links to the bioRxiv preprint, opens in a new tab — renamed from "Paper"), **Useful Links** (hover dropdown — NIH INCLUDE Project, INCLUDE DCC, Human Trisome Project, Linda Crnic Institute), Contact — displayed above the title row. Fully **mobile-responsive** — on screens ≤768 px the nav bar spans full width (sidebar hidden), buttons shrink, and all layouts reflow to single column.
- **Running indicator:** while a query is processing, a single **circular elapsed-time ring** is shown — a rotating "comet" arc with the live elapsed time in its center (e.g. `8s` → `1:23`) and a "T21 RA is running…" label beside it (rendered by `_render_t21_ring`; a background worker runs the request while the main thread polls every 0.2 s).
- **Multi-conversation:** create, switch, delete conversations (up to 10 shown in sidebar)
- **Search Settings sidebar:** conversations list, Clear Response Cache button, System Info expander (API status, models, Milvus collection, vector count); **Costello Lab logo** displayed at the bottom of the sidebar
- **Response sections:** Summary (expanded), Key Evidence (collapsed), Reasoning (collapsed), Highlighted Topic / MeSH Terms / Keywords (collapsed)
- **Evidence cards:** each cited paper shown with Topic label + PMCID hyperlink (deep-linked via a Text Fragment when the retrieved chunk text yields a usable snippet — see *Citation Format* above), MeSH Terms (teal pills), Keywords (orange pills); terms appearing in the Summary are highlighted bright, others faded
- **One-click copy buttons:** every Summary / Key Evidence / Reasoning box, and the whole-response "Copy All" fallback for plain, section-less answers, has a real one-click clipboard-copy button (`render_copy_button`) — not just the manual Ctrl+A/Ctrl+C text area described below. Implemented via `st.components.v1.html` + inline JS, since Streamlit widgets can't call the browser clipboard directly: tries `navigator.clipboard.writeText()` first, falling back to the legacy `document.execCommand('copy')` when the Clipboard API is unavailable (e.g. served over plain HTTP), so it works regardless of deployment TLS. Copies clean plain text (HTML from citation linking is stripped) and the button briefly shows "✓ Copied" on success.
- **Deep Dive** — collapsible `st.expander("Deep Dive", expanded=False)` shown below each response. Clicking the single **"Deep Dive"** button runs a two-step atomic scoring + synthesis cycle that iteratively improves the original summary using LLM-as-Judge feedback:

  | Step | Model | What happens |
  |------|-------|-------------|
  | **1 — Score** | `gpt-oss:120b-cloud` | Scores the original API summary on three criteria — **Accuracy**, **Informativeness**, **Completeness** (1–5 stars each) — and produces structured `{"accuracy":…,"informativeness":…,"completeness":…,"notes":"…"}` JSON with judge commentary identifying specific weaknesses |
  | **2 — Generate** | `nemotron-3-super` | Reads the judge scores and notes; if any criterion is below 4, rewrites the Summary as a comprehensive multi-paragraph synthesis *guided by the specific low-score dimensions and judge notes*; also draws from an extended `AVAILABLE CITATIONS` catalogue (up to 30 unique `[PMCxx:Section]` tags from the full retrieved evidence pool) so the Deep Dive can cite beyond the original summary's papers |

  - Judge scores (Accuracy / Informativeness / Completeness, 1–5 stars + notes) are shown in a nested collapsed expander **"Original Summary — LLM-as-Judge scores"** — hidden by default; open to inspect which dimensions triggered the rewrite.
  - The improved synthesis is rendered below with a `st.caption("Detailed summary")` label and full citation rendering (INCLUDE badges, hyperlinks).
  - **Re-score Deep Dive** and **Re-generate Deep Dive** buttons allow further iterative refinement on demand — each re-run re-scores the current output and rewrites again guided by updated judge notes, narrowing the gap on any remaining weak dimensions.
  - The Deep Dive text is fully included in all Export & Copy formats when present (`.txt`, `.md`, `.json`, `.pdf`).

- **Export & Copy per response:** collapsible " Export & Copy" expander below each response. Contains:
  - **Copy text area** — full plain-text response (markdown stripped) in a scrollable box; press Ctrl+A then Ctrl+C to copy. When a Deep Dive has been generated, the Deep Dive text is appended below a `--- Deep Dive ---` separator so the entire extended response is captured in one copy action.
  - **Download buttons** (4 columns):

    | Button | File | Content |
    |--------|------|---------|
    | Text | `response_N.txt` | Plain text; Deep Dive appended (if present) under `--- Deep Dive ---` separator |
    | Markdown | `response_N.md` | Full markdown; Deep Dive appended (if present) as `## Deep Dive` section |
    | JSON | `response_N.json` | Full API response object; Deep Dive appended (if present) as `"deep_dive"` field |
    | PDF | `response_N.pdf` | PDF via `fpdf`; Deep Dive included (if present) in same document |

  When a Deep Dive has been generated the text area shows a **"Includes Deep Dive results"** caption so the user knows the extended content is included.
- **System info sidebar:** API status, model, provider, embed model, Milvus collection, vector count (cached 60 s)
- **Fixed footer bar:** compact bar pinned below the chat input — © 2026 Costello Lab | James Costello, PhD contact | NIH grant R01 HD109765 | research-use disclaimer

### Batch Evaluation Script

Run all questions through the API and save results to an Excel file:

```bash
# Basic run
python scripts/batch_eval.py

# Custom top-k and fresh (no cache)
python scripts/batch_eval.py --top-k 12 --clear-cache

# Remote backend
python scripts/batch_eval.py --backend http://<server-ip>:8698
```

**Arguments:**

- `--top-k`: Chunks retrieved per question (default: `12`)
- `--backend`: API base URL (default: `http://localhost:8698`)
- `--clear-cache`: Clears response cache before run (default: off)

**Output** — saved to `output/batch_eval/eval_results_<timestamp>.xlsx` with columns:

- `question`: Original question
- `expected_answer`: Ground-truth answer from benchmark
- `expected_citation`: Expected PMID citations
- `status`: `OK`, `HTTP_xxx`, or `EXCEPTION`
- `elapsed_sec`: Response time
- `generated_summary`: AI-generated summary (citations stripped)
- `generated_key_evidence`: AI-generated key evidence bullets
- `cited_pmcs`: PMC IDs cited in the response

---

## Expert Ranking Survey (Qualtrics)

Notebook `8_qualtrics_survey_dataset_builder.ipynb` builds a blind expert evaluation survey:

### Workflow

1. **Generate responses** — Calls T21 API, GPT, and Claude for each benchmark question. All responses are capped at 150 words and end at a complete sentence. T21 responses are taken verbatim from the API summary; GPT and Claude are prompted to answer in 2–3 sentences. If T21 cannot find sufficient evidence, the response is replaced with: *"Could not find sufficient evidence to answer your query."*
2. **Blind randomisation** — Responses are shuffled into `Response 1 / 2 / 3` slots per question using a fixed seed. A hidden answer key records the model-to-slot mapping.
3. **Qualtrics upload** — Creates a Matrix/Likert survey via the Qualtrics REST API (one question per benchmark item; rows = responses, columns = rank 1–3). A JS lock prevents duplicate rank selections.
4. **Activate & share** — Survey is activated via API and the respondent URL is printed.

### Survey Output Files

- `output/survey/model_responses_for_survey.csv`: Raw model responses before randomisation
- `output/survey/qualtrics_ranking_upload.csv`: Blind randomised responses sent to Qualtrics
- `output/survey/qualtrics_ranking_hidden_key.csv`: Model identity per slot (kept private during survey)
- `output/survey/qualtrics_quality_check.csv`: Word count QC report per question

### Required `.env` Variables for Survey Builder

- `T21_API_URL`: T21 API endpoint (default: `http://localhost:8698/answer`)
- `OPENAI_API_KEY`: OpenAI API key for GPT responses
- `OPENAI_COMPARE_MODEL`: GPT model used for survey (default: `gpt-5.4`)
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude responses
- `ANTHROPIC_COMPARE_MODEL`: Claude model used for survey (default: `claude-sonnet-4-6`)
- `QUALTRICS_BASE_URL`: Qualtrics datacenter base URL
- `QUALTRICS_API_TOKEN`: Qualtrics API token

---

## LLM Provider Support

**Provider capabilities:**

- **deepinfra** (current default): Chat (Yes), Embeddings (Yes, dim=2048), Reranking (Yes, `DeepInfraReranker`) — Default chat: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`, Default embed: `nvidia/llama-nemotron-embed-vl-1b-v2`; judge: `openai/gpt-oss-20b`
- **OpenAI**: Chat (Yes), Embeddings (Yes), Reranking (No¹) — Default chat: `gpt-5.4`, Default embed: `text-embedding-3-small` (dim=1536)
- **Anthropic**: Chat (Yes), Embeddings (No), Reranking (No¹) — Default chat: `claude-sonnet-4-6`
- **Google**: Chat (Yes), Embeddings (Yes), Reranking (No¹) — Default chat: `gemini-1.5-pro`, Default embed: `models/embedding-001`
- **Ollama**: Chat (Yes), Embeddings (Yes), Reranking (No¹) — Default chat: `llama3.1`, Default embed: `nomic-embed-text`
- **ollama_remote**: Chat (Yes), Embeddings (No), Reranking (No¹) — Default chat: `nemotron-3-super`
- **nemotron**: Chat (No), Embeddings (Yes), Reranking (Yes, local cross-encoder) — Default embed: `nvidia/llama-embed-nemotron-8b` (dim=4096)

¹ LLM-prompt-based reranking was removed. Reranking routes through either `DeepInfraReranker` (`PROVIDER_RERANK=deepinfra`) or the local `NemotronCrossEncoderReranker` (`PROVIDER_RERANK=nemotron`). The `*_RERANK_MODEL` env vars (`OPENAI_RERANK_MODEL`, `ANTHROPIC_RERANK_MODEL`, `GOOGLE_RERANK_MODEL`, `OLLAMA_RERANK_MODEL`) are not used.

Switch providers by setting `PROVIDER_CHAT` and `PROVIDER_EMBED`. **The current default is a hybrid stack:** `PROVIDER_CHAT=ollama_remote` (chat + judge via Ollama remote), `PROVIDER_EMBED=deepinfra` (2048-dim), `PROVIDER_RERANK=deepinfra` — no GPU, no local model weights. Set `PROVIDER_CHAT=deepinfra` for an all-DeepInfra path when its Nemotron deployment is not overloaded.

---

## Configuration Reference

All settings are read from `.env` (project root) via `python-dotenv`.

**Provider Settings:**
- `PROVIDER_CHAT`: LLM provider - options: `ollama_remote` (default — chat + judge), `deepinfra`, `openai`, `anthropic`, `google`, `ollama`
- `PROVIDER_EMBED`: Embedding provider - options: `deepinfra` (default, 2048-dim), `openai`, `google`, `nemotron`
- `PROVIDER_RERANK`: Reranking provider - `deepinfra` (default) or `nemotron`

**DeepInfra Models (current default stack):**
- `DEEPINFRA_API_KEY`: Bearer key for all DeepInfra calls — read from `.env`, never hardcoded
- `DEEPINFRA_BASE_URL`: OpenAI-compatible base (default: `https://api.deepinfra.com/v1/openai`)
- `DEEPINFRA_CHAT_MODEL`: Synthesis/improve model (default: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`)
- `DEEPINFRA_EMBED_MODEL`: Embedding model, dim=2048 (default: `nvidia/llama-nemotron-embed-vl-1b-v2`)
- `DEEPINFRA_EMBED_BATCH_SIZE`: Embedding batch size (default: `32`)
- `DEEPINFRA_JUDGE_MODEL`: LLM-as-judge model (default: `openai/gpt-oss-20b`)
- `DEEPINFRA_RERANK_URL`: Rerank inference endpoint (default: `https://api.deepinfra.com/v1/inference/nvidia/llama-nemotron-rerank-vl-1b-v2`)

**Nemotron Models (legacy local stack):**
- `NEMOTRON_EMBED_MODEL`: Nemotron embedding model (default: `nvidia/llama-embed-nemotron-8b`, dim=4096)
- `NEMOTRON_RERANK_MODEL`: Nemotron cross-encoder reranker (default: `nvidia/llama-nemotron-rerank-1b-v2`)
- `NEMOTRON_CHAT_MODEL`: Nemotron chat model for `PROVIDER_CHAT=ollama_remote` (default: `nemotron-3-super`)
- `NEMOTRON_DEVICE`: Device for local Nemotron models (default: `cuda`)
- `NEMOTRON_EMBED_BATCH_SIZE`: Batch size for Nemotron embeddings (default: `8`)
- `NEMOTRON_RERANK_BATCH_SIZE`: Batch size for Nemotron cross-encoder (default: `4`)

**OpenAI Models:**
- `OPENAI_CHAT_MODEL`: OpenAI chat model (default: `gpt-5.4`)
- `OPENAI_RERANK_MODEL`: OpenAI rerank model - UNUSED; reranking always uses Nemotron cross-encoder (default: `gpt-5.4`)
- `OPENAI_EMBED_MODEL`: OpenAI embedding model, dim=1536 (default: `text-embedding-3-small`)

**Anthropic Models:**
- `ANTHROPIC_CHAT_MODEL`: Anthropic chat model (default: `claude-sonnet-4-6`)
- `ANTHROPIC_RERANK_MODEL`: Anthropic rerank model - UNUSED; reranking always uses Nemotron cross-encoder (default: `claude-sonnet-4-6`)

**Google Models:**
- `GOOGLE_CHAT_MODEL`: Google chat model (default: `gemini-1.5-pro`)
- `GOOGLE_EMBED_MODEL`: Google embedding model (default: `models/embedding-001`)
- `GOOGLE_RERANK_MODEL`: Google rerank model - UNUSED; reranking always uses Nemotron cross-encoder (default: `gemini-1.5-pro`)

**Ollama Settings:**
- `OLLAMA_BASE_URL`: Local Ollama server URL, used by the RAG API for `PROVIDER_CHAT=ollama` (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL`: Ollama chat model (default: `llama3.1`)
- `OLLAMA_EMBED_MODEL`: Ollama embedding model (default: `nomic-embed-text`)
- `OLLAMA_RERANK_MODEL`: Ollama rerank model - UNUSED; reranking always uses Nemotron cross-encoder (default: `llama3.1`)
- `OLLAMA_REMOTE_BASE_URL`: Remote Ollama API endpoint, used by benchmark/UMAP notebooks for Nemotron via `ollama.Client` with Bearer auth (default: unset)
- `OLLAMA_API_KEY`: Bearer token for the remote Ollama API endpoint, notebooks only (default: unset)
- `OLLAMA_NEMOTRON_MODEL`: Nemotron model name used in benchmark/UMAP notebooks (default: `nemotron-3-super`)

**Milvus Configuration:**
- `MILVUS_HOST`: Milvus host for local connection (default: `127.0.0.1`)
- `MILVUS_PORT`: Milvus port for local connection (default: `19530`)
- `MILVUS_URI`: If set, use URI-based connection for Zilliz or managed Milvus (default: unset)
- `MILVUS_TOKEN`: Auth token for managed Milvus (default: unset)
- `MILVUS_COLLECTION`: Vector collection name (default: `DS_INCLUDE_2026_09_02_DEEPINFRA_2048`; previous `DS_PMC_v7_DEEPINFRA_2048` and legacy `DS_PMC_05_2026_NEMOTRON_4096` kept for rollback)
- `MILVUS_DIM`: Embedding vector dimension (default: `2048`; was `4096` under local Nemotron)
- `MILVUS_SEARCH_PARAMS`: Milvus IVF_FLAT search params as JSON (default: `{"metric_type":"IP","params":{"nprobe":64}}`)

**RAG Pipeline Parameters:**
- `TOP_K_DEFAULT`: Default retrieval count (default: `40`)
- `DEFAULT_TOPK`: Default top-k sent to the API; no user-visible selectbox — system auto-adapts via dynamic boost (default: `12`)
- `RERANK_ALPHA`: Vector/LLM blend, higher = more vector weight (default: `0.4`)
- `MAX_CONTEXT_CHARS`: Max chars fed to synthesis LLM (default: `12000`)
- `ENABLE_RERANK`: Toggle LLM reranking (default: `true`)
- `ENABLE_SAFETY_GUARD`: Toggle prescription language filter (default: `true`)
- `DIVERSITY_MIN_RESULTS`: Minimum Results-section chunks after retrieval (default: `6`)
- `DIVERSITY_MIN_DISCUSSION`: Minimum Discussion-section chunks after retrieval (default: `4`)
- `DIVERSITY_MIN_METHODS`: Minimum Methods-section chunks after retrieval (default: `3`)
- `MAX_CHUNKS_PER_PAPER`: Max chunks allowed from any single PMCID (default: `2`)

**API & UI Configuration:**
- `USE_LANGCHAIN_OPENAI_EMBEDDINGS`: Set to `1` to use LangChain wrapper instead of direct OpenAI client (default: unset)
- `BACKEND_URL`: API URL used by Streamlit UI (default: `http://127.0.0.1:8698`)
- `T21_API_URL`: T21 API endpoint for survey builder (default: `http://localhost:8698/answer`)
- `OPENAI_COMPARE_MODEL`: GPT model for survey builder (default: `gpt-5.4`)
- `ANTHROPIC_COMPARE_MODEL`: Claude model for survey builder (default: `claude-sonnet-4-6`)

**Qualtrics Integration:**
- `QUALTRICS_BASE_URL`: Qualtrics datacenter base URL (required for Qualtrics)
- `QUALTRICS_API_TOKEN`: Qualtrics API token (required for Qualtrics)

---

## CPU-Only Deployment (No GPU)

The project runs on a CPU-only Linux server with no code changes. Only the `.env` provider settings need updating.

### What uses a GPU

**GPU usage by component:**

- `NemotronEmbeddings` (`nvidia/llama-embed-nemotron-8b`, 8 B params, 4096-dim): High GPU usage — very slow on CPU
- `NemotronCrossEncoderReranker` (`nvidia/llama-nemotron-rerank-1b-v2`, 1 B params): **Requires GPU — will fail on CPU. Must set `ENABLE_RERANK=false` on CPU-only servers.**
- `NemotronOllamaLLM` chat (`nemotron-3-super` via remote Ollama): No GPU usage — remote API call, no local model

The embed model (`nvidia/llama-embed-nemotron-8b`, 8 B params) falls back to CPU automatically (`torch.cuda.is_available()` → `"cpu"`) but runs slowly (~10–30 s/query). The rerank model does **not** fall back gracefully to CPU — always disable reranking (`ENABLE_RERANK=false`) when no GPU is present.

### Quickest CPU-only config (keeps existing 4096-dim Milvus collection)

```bash
# .env overrides — no collection rebuild required
PROVIDER_CHAT=openai            # or anthropic — remote API, no local model
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o

PROVIDER_EMBED=nemotron         # keep dimension match with existing 4096-dim collection
NEMOTRON_DEVICE=cpu             # explicit fallback (default is auto-detect)
NEMOTRON_EMBED_BATCH_SIZE=1     # reduce for CPU memory

ENABLE_RERANK=false             # required on CPU — rerank model needs GPU
```

### Full CPU-only config (rebuild collection with OpenAI 1536-dim embeddings)

If you rebuild the Milvus collection using notebooks `3_compute_embeddings_to_pickle.ipynb` and `4_insert_embeddings_to_milvus.ipynb` with `PROVIDER_EMBED=openai`, **all inference becomes remote API calls** — no local model is loaded at all:

```bash
PROVIDER_CHAT=openai
PROVIDER_EMBED=openai
OPENAI_EMBED_MODEL=text-embedding-3-small
USE_LANGCHAIN_OPENAI_EMBEDDINGS=1
MILVUS_DIM=1536
MILVUS_COLLECTION=<your_new_1536_collection_name>
ENABLE_RERANK=false
```

> **Note:** `MILVUS_DIM` and `MILVUS_COLLECTION` must match the embedding dimension used when the collection was built. Mixing 4096-dim vectors with a 1536-dim query (or vice versa) will cause Milvus search to fail.

---

## Troubleshooting

### "Server Offline" / API unreachable, but containers show "healthy"

This has two known distinct root causes on this deployment. Check both `docker compose ps` output and `curl http://localhost:8698/health` — a `Connection refused` or `Expecting value: line 1 column 1 (char 0)` response means nothing is actually listening on port 8698, regardless of what the container status says.

**Cause A — Milvus standalone crash-loop from an orphaned MinIO bind mount (deployment server, disk-full aftermath).**
If the host disk fills up (`df -h /`), Milvus's standalone container can enter `stage=Abnormal` and, in the worst case, something on a shared/managed server can end up deleting `milvus_shared_minio`'s bind-mounted host data directory *while the container is still running*. Docker keeps the container alive on the now-orphaned mount (invisible to `ls`/`df`/`du` from the host), but every MinIO write starts failing with `"Storage backend has reached its minimum free drive threshold"` — a misleading message; it's not really about free space once this happens. That cascades into Milvus's internal coordinators repeatedly failing session registration (`CompareAndSwap error`, `expectedNodeID=X, actualNodeID=Y: node not match`) and the whole `milvus_shared` container crash-looping (`RestartCount` climbing, `ExitCode: 0` — a clean self-exit, not a real crash).

Diagnose:
```bash
# Confirm free disk space isn't the (whole) problem
df -h /

# Check Milvus standalone health + restart count
sudo docker ps -a --filter name=milvus --format '{{.Names}}\t{{.Status}}'
sudo docker inspect milvus_shared --format='RestartCount: {{.RestartCount}}'

# Look for the orphaned-mount signature specifically
MINIO_PID=$(sudo docker inspect milvus_shared_minio --format='{{.State.Pid}}')
sudo cat /proc/$MINIO_PID/mountinfo | grep minio_data
# A path ending in "//deleted" confirms the bind-mount source is gone from the host
```

Fix (safe — the orphaned mount is typically empty by the time this is caught, so there's nothing to lose):
```bash
# 1. Recreate the missing host directory (match ownership of its sibling etcd/ dir)
sudo mkdir -p /srv/shiny-server/T21_research_assistant_v3/volumes/milvus_shared/minio
sudo chown --reference=/srv/shiny-server/T21_research_assistant_v3/volumes/milvus_shared/etcd \
    /srv/shiny-server/T21_research_assistant_v3/volumes/milvus_shared/minio

# 2. Restart minio so it re-mounts against the now-valid directory
sudo docker restart milvus_shared_minio
sleep 30

# 3. Clean stop -> clear stale etcd session keys -> start on the standalone node
#    (a plain `docker restart` is too fast to let stale etcd session leases clear —
#    it must be a full stop, then explicit session-key delete, then start)
sudo docker stop milvus_shared
sudo docker exec milvus_shared_etcd etcdctl del "by-dev/meta/session/" --prefix
sudo docker start milvus_shared
sleep 70

# 4. Verify
sudo docker inspect milvus_shared --format='RestartCount: {{.RestartCount}}'   # should stay 0
curl -s http://localhost:8698/health | python3 -m json.tool                    # total_vectors should be a real number
```

If `df -h /` is genuinely low on space first, reclaim it before touching Milvus — `sudo docker image prune -f` (removes only dangling/untagged images, never tagged images belonging to other projects on a shared server) and `sudo docker builder prune -a -f` (pure build cache, zero risk) are both safe on a multi-tenant server and can reclaim large amounts of space.

**Cause B — Docker Compose project-name collision with a sibling project (local/dev machines).**
Compose derives its default project name from the current directory's *basename* when no explicit `name:` is set in the compose file. If a sibling project (e.g. `T21_Experiment_Model_Research_Assistant`) also keeps its compose file in a directory literally named `srv/`, both projects default to the **same** project name — and `docker compose ps`/`up`/`restart`/`logs`, run from *either* directory, will operate on whichever project's containers were most recently built, silently displacing the other. Symptom: `docker compose ps` shows a container for the `api` service that's actually running completely different application code (check the startup logs — if you see `[Experimental Models]`/`MultiAgentAnnotator` instead of the RAG pipeline's `[OLLAMA_TIMING]`/LangGraph node logs, this is it).

This project's `docker-compose.yml` now pins an explicit `name: t21_research_assistant_v7` at the top (above `services:`) specifically to prevent this. If it ever happens again — or you're setting up a new sibling project on the same machine — give every project's compose file its own explicit unique `name:`, and if a container name is already taken by a stale container from the collision, remove it (`sudo docker rm -f <name>`, safe for the stateless `ui` container) before re-running `docker compose up -d`.

---

## Disclaimer

This tool is intended for **research and educational use only**. It does not provide medical advice. Always consult qualified clinicians for treatment decisions. Citations link to PubMed Central (PMC) for source verification.
