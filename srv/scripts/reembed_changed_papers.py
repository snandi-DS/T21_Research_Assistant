"""
reembed_changed_papers.py

Phase B of the section-extraction fix rollout. Consumes the diff log produced
by refresh_section_extraction.py (Phase A, already applied to the local JSON
cache), and for every paper it flagged as CHANGED:
  1. Purges ALL existing Milvus rows for that pmcid (by pmcid, not by the
     newly-computed pk -- required because a paper whose section text changed
     will very likely re-chunk into a different chunk_id sequence, so a
     pk-only delete would leave the old, now-stale chunks orphaned alongside
     the new correct ones).
  2. Re-chunks + re-embeds the paper's now-corrected sections (reusing
     embed_and_insert.py's own compute_embeddings()/insert_embeddings(), not
     a reimplementation, for full consistency with the production pipeline).
  3. Re-inserts the fresh chunks.

This is a live, destructive Milvus write. Requires DEEPINFRA_API_KEY (paid
embedding calls). Run with --dry-run first to see exactly what would change.

Usage (run from srv/):
    conda run -n dschat python scripts/reembed_changed_papers.py --dry-run
    conda run -n dschat python scripts/reembed_changed_papers.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import embed_and_insert as eai

DIFF_LOG_PATH  = Path(__file__).parent.parent / "output" / "section_extraction_refresh_log.json"
JSON_CACHE_DIR = Path(__file__).parent.parent / "output" / "DS_PMC_05_2026"
TEMP_INPUT_DIR = Path(__file__).parent.parent / "output" / "_reembed_temp_input"
TEMP_PICKLE_DIR = Path(__file__).parent.parent / "output" / "_reembed_temp_pickles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Show what would be purged/re-inserted without touching Milvus or calling the embedding API")
    args = ap.parse_args()

    if not DIFF_LOG_PATH.exists():
        print(f"[ERROR] {DIFF_LOG_PATH} not found -- run refresh_section_extraction.py first")
        return

    diff_log = json.loads(DIFF_LOG_PATH.read_text())
    pmcids = [d["pmcid"] for d in diff_log]
    print(f"Diff log: {len(diff_log)} papers flagged as changed by Phase A")
    print(f"Sample: {pmcids[:5]}")

    if args.dry_run:
        print("\n[DRY RUN] Would purge and re-embed these papers. No writes performed.")
        total_old_chars = 0
        total_new_chars = 0
        for d in diff_log:
            old_total = sum(d["old"].values())
            new_total = sum(d["new"].values())
            total_old_chars += old_total
            total_new_chars += new_total
        print(f"Total section characters before: {total_old_chars:,}")
        print(f"Total section characters after : {total_new_chars:,}")
        print(f"Net new characters to embed    : {total_new_chars - total_old_chars:,}")
        return

    # ── Step 1: purge existing Milvus rows for every changed pmcid ──────────
    eai.connect_milvus()
    col = eai.ensure_collection(eai.COLLECTION_NAME)
    print(f"Entity count before purge: {col.num_entities:,}")

    for i in range(0, len(pmcids), 200):
        chunk = pmcids[i : i + 200]
        expr = "pmcid in [" + ",".join(f'"{p}"' for p in chunk) + "]"
        col.delete(expr)
    print(f"Purged existing rows for {len(pmcids)} pmcids")

    # ── Step 2/3: re-chunk, re-embed, re-insert via the production pipeline ─
    if TEMP_INPUT_DIR.exists():
        shutil.rmtree(TEMP_INPUT_DIR)
    if TEMP_PICKLE_DIR.exists():
        shutil.rmtree(TEMP_PICKLE_DIR)
    TEMP_INPUT_DIR.mkdir(parents=True)
    TEMP_PICKLE_DIR.mkdir(parents=True)

    pmid_set = {d["pmid"] for d in diff_log}
    n_linked = 0
    for fp in JSON_CACHE_DIR.glob("publication_metadata_PMID_*.json"):
        pmid = fp.stem.replace("publication_metadata_PMID_", "")
        if pmid in pmid_set:
            (TEMP_INPUT_DIR / fp.name).symlink_to(fp.resolve())
            n_linked += 1
    print(f"Linked {n_linked} updated JSON files into {TEMP_INPUT_DIR}")

    eai.compute_embeddings(TEMP_INPUT_DIR, TEMP_PICKLE_DIR, force=True)
    eai.insert_embeddings(TEMP_PICKLE_DIR, mark_include=False)

    shutil.rmtree(TEMP_INPUT_DIR)
    shutil.rmtree(TEMP_PICKLE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
