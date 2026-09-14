"""
move_non_db_files.py
Moves pickle and JSON files whose PMID is NOT in the Milvus collection
to _others/ subdirectories inside each output folder.

Usage:
    python scripts/move_non_db_files.py          # preview (dry-run)
    python scripts/move_non_db_files.py --move   # actually move files
"""
import os, re, shutil, argparse
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection

ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

COLLECTION_NAME = os.environ["MILVUS_COLLECTION"]  # required — no silent fallback to a stale collection

BASE     = os.path.join(os.path.dirname(__file__), "..", "output")
PKL_DIR  = os.path.join(BASE, "embeddings_pickle", "v7_corpus_deepinfra_2048")
JSON_DIR = os.path.join(BASE, "DS_PMC_05_2026")
PKL_OTHERS  = os.path.join(PKL_DIR,  "_others")
JSON_OTHERS = os.path.join(JSON_DIR, "_others")

_re = re.compile(r"PMID_(\d+)\.")

def pmid_from(fname):
    m = _re.search(fname)
    return m.group(1) if m else None

def main(dry_run=True):
    # 1. Fetch all unique PMIDs from Milvus
    connections.connect(host="127.0.0.1", port=19530)
    col = Collection(COLLECTION_NAME)
    col.load()

    db_pmids = set()
    last_pk = 0
    while True:
        rows = col.query(
            expr=f'pk > {last_pk} && pub_type == "article"',
            output_fields=["pk", "pmid"],
            limit=16000,
        )
        if not rows:
            break
        for r in rows:
            if r.get("pmid"):
                db_pmids.add(str(r["pmid"]))
        last_pk = max(r["pk"] for r in rows)
        if len(rows) < 16000:
            break

    print(f"Unique article PMIDs in Milvus: {len(db_pmids)}")

    # 2. Find files not in DB
    pkl_extras  = [f for f in os.listdir(PKL_DIR)  if f.endswith(".pkl")  and pmid_from(f) not in db_pmids]
    json_extras = [f for f in os.listdir(JSON_DIR) if f.endswith(".json") and pmid_from(f) not in db_pmids]

    print(f"\nPickle files not in DB : {len(pkl_extras)}")
    print(f"JSON files not in DB   : {len(json_extras)}")

    if dry_run:
        print("\n[DRY RUN] Pass --move to actually move files.")
        if pkl_extras:
            print(f"  Would move {len(pkl_extras)} pickles → {PKL_OTHERS}/")
        if json_extras:
            print(f"  Would move {len(json_extras)} JSONs   → {JSON_OTHERS}/")
        return

    # 3. Move files
    for others_dir, src_dir, extras in [
        (PKL_OTHERS,  PKL_DIR,  pkl_extras),
        (JSON_OTHERS, JSON_DIR, json_extras),
    ]:
        if not extras:
            continue
        os.makedirs(others_dir, exist_ok=True)
        for fname in extras:
            src  = os.path.join(src_dir,   fname)
            dest = os.path.join(others_dir, fname)
            shutil.move(src, dest)
        print(f"Moved {len(extras)} files → {others_dir}/")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", action="store_true", help="Actually move files (default: dry-run)")
    args = parser.parse_args()
    main(dry_run=not args.move)
