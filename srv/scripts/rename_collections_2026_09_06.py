"""One-time collection rename, 2026-09-06 (user-approved cutover).

1. Archives the current live collection:
   DS_INCLUDE_2026_09_02_DEEPINFRA_2048  ->  DS_INCLUDE_2026_09_02_DEEPINFRA_2048_ARCHIVED
2. Gives its name to the 4-plus-section subset collection:
   DS_INCLUDE_2026_09_05_DEEPINFRA_2048  ->  DS_INCLUDE_2026_09_02_DEEPINFRA_2048

The current live collection is the articles-only rebuild (1,687 articles, 66,742
vectors). After this swap, the API (whose .env MILVUS_COLLECTION is unchanged) serves
the 4-plus-section subset corpus instead: 1,669 papers, 68,946 vectors. Restart the API and clear the cache afterwards:

    cd srv && sudo docker compose restart api && sleep 15 && \
    curl -s -X POST http://localhost:8698/cache/clear && curl -s http://localhost:8698/health

Rollback: run this script again with ROLLBACK = True.
"""
from pymilvus import connections, utility, Collection

LIVE     = "DS_INCLUDE_2026_09_02_DEEPINFRA_2048"
SUBSET   = "DS_INCLUDE_2026_09_05_DEEPINFRA_2048"
ARCHIVED = "DS_INCLUDE_2026_09_02_DEEPINFRA_2048_ARCHIVED"
ROLLBACK = False

connections.connect(host="127.0.0.1", port=19530)

def count(name):
    c = Collection(name)
    c.load()
    return c.query(expr="pk >= 0", output_fields=["count(*)"])[0]["count(*)"]

if not ROLLBACK:
    assert utility.has_collection(LIVE), f"{LIVE} not found"
    assert utility.has_collection(SUBSET), f"{SUBSET} not found"
    assert not utility.has_collection(ARCHIVED), f"{ARCHIVED} already exists"
    print(f"before: {LIVE} = {count(LIVE):,} vectors (full corpus, to be archived)")
    print(f"        {SUBSET} = {count(SUBSET):,} vectors (4+ section subset)")
    utility.rename_collection(LIVE, ARCHIVED)
    print(f"renamed {LIVE} -> {ARCHIVED}")
    utility.rename_collection(SUBSET, LIVE)
    print(f"renamed {SUBSET} -> {LIVE}")
    print(f"after : {LIVE} = {count(LIVE):,} vectors (expected 68,946)")
else:
    utility.rename_collection(LIVE, SUBSET)
    utility.rename_collection(ARCHIVED, LIVE)
    print(f"rolled back: {LIVE} = {count(LIVE):,} vectors (expected 66,742)")

print("DS_INCLUDE collections now:",
      [x for x in utility.list_collections() if "DS_INCLUDE" in x])
print("Done. Now restart the API and clear the cache (see docstring).")
