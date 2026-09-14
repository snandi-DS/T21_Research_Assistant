# DEPLOYMENT.md - T21 Research Assistant v7 (Docker, step-by-step)

Complete step-by-step guide for operating this project entirely through Docker on the
deployment server. Every operation here runs in a container; no conda environment and
no GPU are required (the current hybrid stack calls DeepInfra and Ollama remote APIs,
so no model weights are loaded locally).

All commands run from the compose directory unless stated otherwise:

```bash
cd /path/to/project/T21_research_assistant_v7/srv
```

Related documents:

- README.md: full project reference, troubleshooting details
- DATA_PIPELINE.md: corpus build pipeline internals

---

## 0. Standard deployment sequence (checklist)

The deployment server runs the identical project configuration as the dev copy
(same `.env`, same `docker-compose.yml`, same directory layout), so every step
below works unchanged on both.

**First-time deployment (empty server):**

1. Transfer the project from the dev machine — section 3, step 0.
2. Verify prerequisites: Docker, Milvus up, `.env` keys — section 2.
3. Build images and start API + UI — section 3.
4. Load the corpus into Milvus (auto-insert on first start, or manually via
   drop + re-insert) — section 3 step 5 / section 5.1.
5. Verify `/health`, the UI, and one smoke-test query — section 3.

**Routine update (code changed, corpus unchanged — the common case):**

1. rsync the changed source files from dev — section 4, "Deploy code changes".
2. `sudo docker compose restart api` (+ `ui` if UI code changed).
3. Clear the response cache, verify `/health`, smoke-test one query.

**Corpus update (pickles changed):**

1. rsync the pickle directory — section 4, "Deploy code changes", step 4.
2. Drop the existing collection IF it exists, then re-insert from pickles via
   the data-processor container — section 5.1.
3. Restart API, clear cache, verify vector/document counts.

**Config update (`.env` or `docker-compose.yml` changed):**

1. Edit the file(s) on the server (or rsync them deliberately — the code-sync
   commands never touch them).
2. `sudo docker compose up -d api ui` — a plain `restart` is NOT enough;
   the container must be recreated — section 4, "Apply .ENV changes".
3. Clear cache, verify `/health` reports the expected collection and counts.

---

## 1. What gets deployed

The compose file (`docker-compose.yml`, project name `t21_research_assistant_v7`) defines
three services on the `t21_res_assi_network` bridge network:

- `api` (container `t21_res_assi_api`): FastAPI backend on port 8698. Runs the LangGraph
  RAG pipeline. Connects to the host's existing Milvus via `host.docker.internal:19530`.
  On startup it waits for Milvus, checks the collection entity count, and if the
  collection is empty auto-inserts from `PICKLE_DIR` as a background coroutine.
- `ui` (container `t21_res_assi_ui`): Streamlit chat UI on port 8878, served under the
  base path `/t21-res-assi` (for the Apache reverse proxy). Starts only after `api`
  reports healthy.
- `data-processor` (container `t21_res_assi_data_processor`, profile `data`): on-demand
  container for corpus operations (fetch, embed, insert, batch eval). Never runs unless
  invoked with `--profile data run`.

Each service builds into a uniquely named image (`t21_res_assi_api:v7`,
`t21_res_assi_ui:v7`, `t21_res_assi_data_processor:v7`), pinned via `image:` in the
compose file so rebuilds never collide with sibling projects (see Rebuild images,
section 4).

Milvus itself is NOT part of this compose file. It is a shared instance already running
on the host (containers named `milvus_shared*`). Never stop or recreate the Milvus
containers to refresh data; only drop and recreate collections inside Milvus.

---

## 2. Prerequisites

1. Docker Engine with the compose plugin (`sudo docker compose version`).
2. Milvus running on the host:

   ```bash
   sudo docker ps --filter name=milvus_shared
   curl http://localhost:19530/healthz
   ```

3. `srv/.env` configured. Minimum required for the current hybrid stack:

   - `DEEPINFRA_API_KEY` (embeddings + reranking + LLM failover)
   - `OLLAMA_API_KEY` and `OLLAMA_REMOTE_BASE_URL=https://ollama.com` (chat + judge)
   - `PROVIDER_CHAT=ollama_remote`, `PROVIDER_EMBED=deepinfra`, `PROVIDER_RERANK=deepinfra`
   - `MILVUS_COLLECTION=DS_INCLUDE_2026_09_02_DEEPINFRA_2048`, `MILVUS_DIM=2048`

   The `.env` file contains live API keys; never commit it to a public repository.

4. `PICKLE_DIR` in `docker-compose.yml` must point at the ACTIVE collection's pickle
   directory (currently
   `/app/output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048`).
   This is the startup auto-insert recovery path; if it points at the wrong corpus, an
   empty-collection recovery silently restores the wrong data.

---

## 3. First-time deployment

**Step 0 — transfer the project from the dev machine** (skip if the project is
already on the server). The configuration is identical on both machines, so the
whole tree transfers as-is; only volumes, caches, and git history are excluded.
`.env` IS transferred (same config, and it carries the live API keys the server
needs) — never commit it anywhere public.

```bash
# From the dev machine
REMOTE=<username>@<deployment-server>
REMOTE_PATH=/path/on/server/T21_research_assistant_v7

rsync -avz --progress \
  --exclude='.git' \
  --exclude='srv/volumes/' \
  --exclude='**/__pycache__' \
  --exclude='**/*.pyc' \
  --exclude='srv/local_hugging_face_model/' \
  --exclude='srv/hf_cache/' \
  /path/to/project/T21_research_assistant_v7/ \
  $REMOTE:$REMOTE_PATH/
```

The hybrid stack loads no local model, so `local_hugging_face_model/` is NOT
needed on the server (only transfer it if rolling back to the legacy
`PROVIDER_EMBED=nemotron` stack; see README.md, Docker Deployment, Step 2).
The pickle directory `srv/output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048/`
IS included in the transfer above — it is required for the corpus load in step 5.

**Steps 1-4 — on the server, from `$REMOTE_PATH/srv`:**

```bash
# 1. Verify Milvus is up
sudo docker ps --filter name=milvus_shared

# 2. Review configuration (identical to dev; confirm keys survived the transfer)
nano .env

# 3. Build images and start API + UI (detached)
sudo docker compose up -d --build

# 4. Watch startup (API healthcheck allows up to 5 min start period)
sudo docker compose logs -f api
```

Alternatively use the launcher, which auto-detects an NVIDIA GPU and merges
`docker-compose.gpu.yml` only when `nvidia-smi` is present (irrelevant on the CPU-only
server, but safe everywhere):

```bash
./start.sh build     # rebuild images then start
./start.sh           # start without rebuild
```

**Step 5 — corpus load.** On startup the API checks the collection entity count;
if the collection named by `MILVUS_COLLECTION` is empty or missing, it
auto-inserts from `PICKLE_DIR` as a background coroutine — watch the api log
until the insert completes. If the collection exists but holds wrong or partial
data (auto-insert will NOT fix that), do the explicit drop + re-insert of
section 5.1 instead: drop the collection if it exists, then re-insert the
embedding vectors from the pickles with the data-processor container.

**Step 6 — verify:**

```bash
curl -s http://localhost:8698/health | python3 -m json.tool
sudo docker compose ps
```

Expect in the health output: `status: healthy`, `milvus_collection:
DS_INCLUDE_2026_09_02_DEEPINFRA_2048`, `total_vectors: 65677`,
`estimated_documents: 1594`, `include_count: 158`.

Then run one smoke-test query and confirm the Summary carries citations:

```bash
curl -s -X POST http://localhost:8698/answer -H 'Content-Type: application/json' \
  -d '{"question":"What conditions are individuals with Down syndrome at increased risk for?","top_k":12}' \
  | python3 -c "import sys,json,re; md=json.load(sys.stdin)['markdown']; \
print('Summary citations:', len(re.findall(r'\[PMC\d+:', md)))"
```

Access the UI at `http://<server>:8878/t21-res-assi` (or through the Apache reverse
proxy path if configured).

---

## 4. Everyday operations

### Start / stop / status

```bash
sudo docker compose up -d          # start API + UI
sudo docker compose down           # stop API + UI (leaves Milvus running)
sudo docker compose ps             # container status
./start.sh down                    # same stop, via the launcher
```

### Logs

```bash
sudo docker compose logs -f api            # API live
sudo docker compose logs --tail=100 api    # API last 100 lines
sudo docker compose logs -f ui             # UI live
```

Useful API log markers: `[OLLAMA_TIMING]` (per-call latency), `[OLLAMA_WARN]` /
`[OLLAMA_FALLBACK]` (degraded Ollama remote, DeepInfra failover engaged),
`[IMPROVE_QG]` (judge-improve quality gate discarded a rewrite; the variant
`reverted improved_summary post-strip` means the rewrite's surviving citations
vanished or regressed after hallucinated IDs were removed, so the original
finalize Summary was restored), `[CITE_REPAIR]` (a digit-mutated PMC ID was
repaired to the real corpus ID, e.g. `PMC979796221 -> PMC9796221`), and
`[CITE_GUARANTEE]` (the Summary reached the final check with zero citations;
the top evidence citation was appended as a fallback).

### Apply CODE changes (api/, ui/, scripts/ are bind-mounted; restart is enough)

```bash
# API code changed
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo "Cache cleared" && \
curl -s http://localhost:8698/health | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('Status:', d.get('status'))"

# UI code changed
sudo docker compose restart ui && sleep 10 && sudo docker compose ps ui
```

Docker does not auto-reload on file saves; always restart the container after editing
source files.

### Deploy code changes from the dev machine to the production server

`api/`, `ui/`, and `scripts/` are bind-mounted into the containers, so deploying a
code change is: rsync the changed files, restart the affected container, clear the
cache, verify. No image rebuild is needed unless `requirements.txt` or the
`Dockerfile` changed (see "Rebuild images" below).

The deployment server runs the identical project configuration as the dev copy —
same `.env` (`MILVUS_COLLECTION=DS_INCLUDE_2026_09_02_DEEPINFRA_2048`,
`MILVUS_DIM=2048`, hybrid `ollama_remote` + DeepInfra providers), same
`docker-compose.yml` (pinned project `name:`, image names, `PICKLE_DIR`), same
directory layout. So after syncing code, NO `.env` or compose edits are needed on
the server — only the restart. The rsync commands below deliberately touch only
source files and never `.env`, `docker-compose.yml`, or `volumes/`.

**Step 1 — from the dev machine, sync the changed source files:**

```bash
REMOTE=<username>@<deployment-server>
REMOTE_PATH=/path/on/server/T21_research_assistant_v7   # actual project path on the server

# API code only (e.g. the 2026-09-11 citation-integrity fixes touched exactly these two files)
rsync -avz --progress \
  srv/api/main.py srv/api/synthesizer_langchain.py \
  $REMOTE:$REMOTE_PATH/srv/api/

# Or sync all backend + UI code (no data, no volumes, no model weights)
rsync -avz --progress \
  --exclude='**/__pycache__' --exclude='**/*.pyc' \
  srv/api/ $REMOTE:$REMOTE_PATH/srv/api/
rsync -avz --progress \
  --exclude='**/__pycache__' --exclude='**/*.pyc' \
  srv/ui/ $REMOTE:$REMOTE_PATH/srv/ui/
```

**Step 2 — on the server, restart the API, clear the cache, verify health:**

```bash
cd $REMOTE_PATH/srv
sudo docker compose restart api && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo " cache cleared" && \
curl -s http://localhost:8698/health | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('Status:', d.get('status'))"
# If UI code was synced too:
sudo docker compose restart ui && sleep 10 && sudo docker compose ps ui
```

**Step 3 — on the server, smoke-test one query and check the citation-integrity markers:**

```bash
curl -s -X POST http://localhost:8698/answer -H 'Content-Type: application/json' \
  -d '{"question":"What conditions are individuals with Down syndrome at increased risk for?","top_k":12}' \
  | python3 -c "import sys,json,re; md=json.load(sys.stdin)['markdown']; \
print('Summary citations:', len(re.findall(r'\[PMC\d+:', md)))"
# Expect a non-zero citation count. Then confirm no unexpected degradation markers:
sudo docker compose logs --tail=200 api | grep -E 'CITE_REPAIR|CITE_GUARANTEE|IMPROVE_QG|Traceback' || echo "no markers (clean run)"
```

`[CITE_REPAIR]` lines are normal (a mistyped ID was fixed); frequent
`[CITE_GUARANTEE]` or `[IMPROVE_QG] reverted` lines mean the improve model is
fabricating citations often and are worth a look; any `Traceback` is a bug.

**Step 4 — only if the embedding pickles changed (corpus update):** sync the new
pickle directory to the server, then drop the existing Milvus collection and
re-insert the vectors from the pickles using the data-processor container —
the exact drop-if-exists + insert + verify commands are in section 5.1 below.

```bash
# From the dev machine: sync the pickle directory named by PICKLE_DIR in docker-compose.yml
rsync -avz --progress \
  srv/output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048/ \
  $REMOTE:$REMOTE_PATH/srv/output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048/

# On the server: drop the collection if it exists, re-insert, restart API, verify
# -> run section 5.1 steps 1-4 exactly as written
```

### Apply .ENV changes (restart is NOT enough; the container must be recreated)

The api container's environment is baked at creation via `env_file:` and `/app/.env`
is not mounted into it, so a plain `restart` keeps the old values.

```bash
sudo docker compose up -d api ui && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && echo " cache cleared" && \
curl -s http://localhost:8698/health
```

Confirm `milvus_collection`, `total_vectors`, `estimated_documents`, `include_count`
in the health output. When changing `MILVUS_COLLECTION`, also update `PICKLE_DIR` in
`docker-compose.yml` to the matching pickle directory.

### Rebuild images (requirements.txt or Dockerfile changed)

Each service builds into an explicitly named, project-unique image (pinned via
`image:` in `docker-compose.yml`, mirroring the pinned `container_name` and project
`name:` so sibling projects can never collide in the image namespace):

- `t21_res_assi_api:v7`
- `t21_res_assi_ui:v7`
- `t21_res_assi_data_processor:v7`

Standard rebuild (uses the build cache; rebuilds only changed layers, then restarts):

```bash
sudo docker compose up -d --build      # or: ./start.sh build
```

Rebuild a single service:

```bash
sudo docker compose build api && sudo docker compose up -d api
```

Full clean rebuild (ignore the build cache; use when a cached layer is stale, for
example after a base-image or pip index change):

```bash
sudo docker compose build --no-cache && sudo docker compose up -d
```

The on-demand `data-processor` image is not rebuilt by a plain `up -d --build`
(profile `data` is inactive by default); rebuild it explicitly when its
dependencies change:

```bash
sudo docker compose --profile data build data-processor
```

Verify the images and clean up old unnamed layers left behind by rebuilds:

```bash
sudo docker images | grep t21_res_assi
sudo docker image prune -f              # removes dangling (untagged) images only
```

### Clear the response cache

Responses are cached in memory by MD5(question + top_k). Clear after any code or
data change:

```bash
curl -X POST http://localhost:8698/cache/clear
```

---

## 5. Data operations (data-processor container)

All corpus work runs through the on-demand `data-processor` service. It reads
`input/` and `output/` via bind mounts and reaches Milvus at
`host.docker.internal:19530`. The insert step only reads pickles and writes to
Milvus; it never loads a model and runs fine on CPU.

### 5.1 Drop and re-insert the production corpus (Docker)

Use this to (re)load the collection from pickles — recovery from a corrupted
collection, or a refresh after the pickles changed (the current corpus: 1,594
pickles, 65,677 vectors). The flow is always **drop-if-exists, then insert**:
if a collection with the same name and schema already holds vectors, running
`--insert` into it again would duplicate every vector, so the existing
collection must be dropped first. The drop reads `MILVUS_COLLECTION` from
`.env` (the data-processor container loads it via `env_file:`), so it always
targets the same collection the insert step will create. Only the collection
inside Milvus is dropped — never stop or recreate the Milvus containers
themselves.

```bash
# 1. Check Milvus containers are running (they must be; do NOT restart them)
sudo docker ps --filter name=milvus_shared

# 2. Drop the collection IF it exists (name read from .env MILVUS_COLLECTION)
sudo docker compose --profile data run --rm data-processor python3 -c "
import os
from pymilvus import connections, utility
name = os.environ['MILVUS_COLLECTION']
connections.connect(host='host.docker.internal', port=19530)
if utility.has_collection(name):
    utility.drop_collection(name)
    print(f'Dropped existing collection: {name}')
else:
    print(f'No existing collection ({name}) - nothing to drop.')
"

# 3. Re-insert the embedding vectors from pickles (5-15 min, no GPU, no external API calls)
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

Expected health output after re-insert: `total_vectors` 65,677,
`estimated_documents` 1,594, `include_count` 158.

Notes:

- The `--pickle-dir` must match `PICKLE_DIR` in `docker-compose.yml` and the
  collection named by `MILVUS_COLLECTION` in `.env` — all three describe the
  same corpus build.
- `embed_and_insert.py` reads `MILVUS_COLLECTION` and `MILVUS_DIM` from `.env` and
  auto-creates the collection at the correct dimension (31-field schema).
- `ensure_collection()` auto-drops the collection on a schema field-count
  mismatch, but NOT on a same-schema re-insert — that is why step 2 is required,
  not optional.
- There is intentionally no `col.flush()` after insert; Milvus auto-flushes via WAL.
  An explicit flush caused `MilvusException: channel not found` on fresh collections.
- `/health` counts vectors with `count(*)` (not `num_entities`), so the total is
  accurate immediately after insert, before segments are sealed.

### 5.2 Build a NEW corpus (full pipeline, all in Docker)

Step A: fetch metadata from a MEDLINE export (1.5-3 h, network I/O; resume-safe):

```bash
nohup sudo docker compose --profile data run --rm data-processor \
    python scripts/fetch_pmc_publications.py \
    --medline input/<MEDLINE_export>.txt \
    --include-excel input/INCLUDE_Publications_by_2025-11-14.xlsx \
    --out-dir output/<json-dir> \
    > /tmp/t21v7_fetch.log 2>&1 &
tail -f /tmp/t21v7_fetch.log
```

Test first with `--limit 10`. Then top up INCLUDE-only papers (resume-safe):

```bash
sudo docker compose --profile data run --rm data-processor \
    python scripts/fetch_pmc_publications.py \
    --include-only \
    --include-excel input/INCLUDE_Publications_by_2025-11-14.xlsx \
    --out-dir output/<json-dir>
```

Step B: compute embeddings via the DeepInfra API (1-2 h, no GPU; resume-safe) and
insert, in one run:

```bash
sudo docker compose --profile data run --rm data-processor \
    python scripts/embed_and_insert.py \
    --input output/<json-dir> \
    --pickle-dir output/embeddings_pickle/<new-pickle-dir> \
    --compute --insert 2>&1 | tee /tmp/t21v7_reindex.log
```

Step C: cut over. Point `.env` `MILVUS_COLLECTION` (and `MILVUS_DIM` if changed) at
the new collection, update `PICKLE_DIR` in `docker-compose.yml` to the new pickle
directory, then recreate the containers (section 4, .env changes). Keep the previous
collection in Milvus untouched for rollback.

### 5.3 Batch evaluation

```bash
sudo docker compose --profile data run --rm data-processor \
    python scripts/batch_eval.py --top-k 12 --clear-cache
```

---

## 6. Rollback

Collection cutover rollback: restore the previous (commented) `MILVUS_COLLECTION`
value in `.env`, restore the matching `PICKLE_DIR` in `docker-compose.yml` (the
previous values are kept as comments there), then:

```bash
sudo docker compose up -d api ui && sleep 15 && \
curl -s -X POST http://localhost:8698/cache/clear && \
curl -s http://localhost:8698/health
```

Available rollback collections: `DS_PMC_v7_DEEPINFRA_2048` (previous 2048-dim corpus)
and `DS_PMC_05_2026_NEMOTRON_4096` (legacy local-Nemotron 4096-dim; also requires
`PROVIDER_EMBED=nemotron` and `MILVUS_DIM=4096`).

---

## 7. Troubleshooting

Server offline / API unreachable; the two known root causes (full detail in
README.md, Troubleshooting):

1. Milvus crash-loop after a disk-full event (orphaned MinIO bind mount). Symptoms:
   `RestartCount` climbing, `stage=Abnormal` or `node not match` in
   `sudo docker logs milvus_shared`.
2. Compose project-name collision with a sibling project whose compose file also
   lives in a directory named `srv/`. This project pins `name:
   t21_research_assistant_v7` at the top of `docker-compose.yml`; do not remove it.
   Symptom: `docker compose ps` shows an `api` container running visibly different
   application code (for example `[Experimental Models]` log lines instead of
   LangGraph node output).

Other checks:

```bash
# Container states and restart counts
sudo docker ps --format 'table {{.Names}}\t{{.Status}}'

# API health from inside the network (bypasses host port mapping)
sudo docker compose exec api curl -s http://localhost:8698/health

# Milvus reachability from a data container
sudo docker compose --profile data run --rm data-processor python3 -c "
from pymilvus import connections, utility
connections.connect(host='host.docker.internal', port=19530)
print('Collections:', utility.list_collections())
"
```

Slow or hanging answers: check the API log for `[OLLAMA_WARN]` / `[OLLAMA_FALLBACK]`.
Ollama remote calls are bounded by `OLLAMA_TIMEOUT` (default 20 s) and
`OLLAMA_DEADLINE` (default 25 s) and fail over to DeepInfra automatically; a fully
degraded call site is bounded at roughly 110 s instead of hanging.

`total_vectors` shows 0 right after an insert: the API's `/health` uses `count(*)`
and should be accurate; if you queried Milvus directly, `num_entities` only counts
sealed segments and undercounts fresh data.

---

Research and educational use only. Not medical advice.
