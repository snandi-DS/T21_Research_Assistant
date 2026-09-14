#!/usr/bin/env python3
"""Purge non-article papers from the live production collection.

2026-09-06, at user request: DS_INCLUDE_2026_09_02_DEEPINFRA_2048 must store ONLY the
papers that satisfy every corpus criterion, including pub_type == "article", so the
stored document count equals the searchable count reported by /health and the GUI.

How the 1,594 number comes about (the full criteria chain, from
srv/nbs/1_download_ds_papers.ipynb + 2_embed_and_index_ds_papers.ipynb):

    2,075  records in the PMC query snapshot (input/PMC_DownSynd_09_02_2026.txt)
    1,891  eligible after the metadata gate (publication-type exclusions, PMCID
           present, not retracted per MEDLINE tag + Crossref)
    1,837  retained after the full-text criterion (PMC OAI serves a usable body)
    1,669  selected for indexing (n_sections >= 4 of the six target sections,
           and at least one retrieval section)
    1,594  pub_type == "article" (drops 64 reviews, 8 preprints, 2 errata, 1 other)

The 75 non-article papers carry 3,269 chunks; after the purge the collection holds
65,677 vectors across 1,594 papers (158 INCLUDE-funded).

Dry-run by default: reports what would be deleted and exits. Pass --apply to delete.

Rollback: the non-article pickles remain on disk in
output/_archive_superseded_2026-09-06/embeddings_pickle/
ds_include_2026-09-05_ge4sections_deepinfra_2048/ (the pre-purge 1,669-paper set),
so re-inserting them restores the papers exactly.

Run from srv/:  python scripts/purge_non_articles_2026_09_06.py [--apply]
"""
import sys
from collections import Counter

from pymilvus import connections, Collection

COLLECTION = "DS_INCLUDE_2026_09_02_DEEPINFRA_2048"
EXPECT_AFTER = {"vectors": 65_677, "papers": 1_594, "include": 158}


def main() -> None:
    apply = "--apply" in sys.argv[1:]
    connections.connect(host="localhost", port=19530)
    col = Collection(COLLECTION)
    col.load()

    total_before = col.query(expr="pk >= 0", output_fields=["count(*)"])[0]["count(*)"]

    pks, papers, kinds = [], set(), Counter()
    it = col.query_iterator(expr='pub_type != "article"',
                            output_fields=["pk", "pmcid", "pub_type"],
                            batch_size=16384)
    while True:
        rows = it.next()
        if not rows:
            break
        for r in rows:
            pks.append(r["pk"])
            papers.add(r["pmcid"])
            kinds[r["pub_type"]] += 1
    it.close()

    print(f"collection            : {COLLECTION}")
    print(f"vectors before        : {total_before:,}")
    print(f"non-article chunks    : {len(pks):,} across {len(papers)} papers")
    print(f"by pub_type           : {dict(kinds.most_common())}")
    print(f"vectors after purge   : {total_before - len(pks):,}")

    if not apply:
        print("\nDRY RUN ONLY. Re-run with --apply to delete the chunks above.")
        return

    for i in range(0, len(pks), 1000):
        batch = pks[i:i + 1000]
        col.delete(f"pk in [{', '.join(str(p) for p in batch)}]")
    print(f"\ndeleted {len(pks):,} chunks in {(len(pks) + 999) // 1000} batches")

    # verify
    total = col.query(expr="pk >= 0", output_fields=["count(*)"])[0]["count(*)"]
    docs, inc = set(), set()
    it = col.query_iterator(expr="pk >= 0",
                            output_fields=["pmcid", "is_include", "pub_type"],
                            batch_size=16384)
    leftovers = Counter()
    while True:
        rows = it.next()
        if not rows:
            break
        for r in rows:
            docs.add(r["pmcid"])
            if r["is_include"] == 1:
                inc.add(r["pmcid"])
            if r["pub_type"] != "article":
                leftovers[r["pub_type"]] += 1
    it.close()

    print(f"vectors after         : {total:,}   (expected {EXPECT_AFTER['vectors']:,})")
    print(f"papers after          : {len(docs):,}   (expected {EXPECT_AFTER['papers']:,})")
    print(f"INCLUDE papers after  : {len(inc)}   (expected {EXPECT_AFTER['include']})")
    if leftovers:
        print(f"WARNING: non-article chunks still present: {dict(leftovers)}")
    ok = (total == EXPECT_AFTER["vectors"] and len(docs) == EXPECT_AFTER["papers"]
          and len(inc) == EXPECT_AFTER["include"] and not leftovers)
    print("PURGE VERIFIED OK" if ok else
          "COUNTS DIFFER FROM EXPECTED. A slightly short vector count right after a "
          "delete can be growing-segment settling; wait a minute and re-run without "
          "--apply to re-check.")
    print("\nNow clear the API cache: curl -X POST http://localhost:8698/cache/clear")


if __name__ == "__main__":
    main()
