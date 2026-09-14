#!/usr/bin/env python3
"""Export the versioned corpus manifest and screening audit for the GUI download buttons.

Answers the manuscript reviewer's reproducibility request: a complete, versioned list
of the corpus with PMCID, DOI, title, publication year, journal, MeSH terms, keywords,
retrieval date, inclusion/exclusion status, and INCLUDE provenance.

Writes two CSVs into srv/ui/data/ (the only project directory mounted into the UI
container), named for the corpus snapshot:

1. corpus_manifest_DS_INCLUDE_2026_09_02.csv
   One row per INCLUDED paper (1,594 original research articles): identifiers,
   bibliographic metadata, MeSH terms, keywords, INCLUDE provenance, sections present,
   and the number of indexed chunks per paper.

2. screening_audit_DS_INCLUDE_2026_09_02.csv
   One row per SCREENED record (2,075 snapshot records): the final status
   (included, or excluded with the exact criterion that fired) covering both the
   download-notebook screening gate and the indexing selection
   (>= 4 sections, pub_type == "article").

Sources: output/ds_include_pipeline_2026-09-02/ (master_table.csv,
step2_screening_audit.csv, json_cache/) and the article pickle set
output/embeddings_pickle/ds_include_2026-09-02_ge4sections_articles_deepinfra_2048/.

Run from srv/:  python scripts/export_corpus_manifest.py
Re-run after any corpus rebuild so the GUI serves the current list.
"""
import json
import pickle
from datetime import date, datetime
from pathlib import Path

import pandas as pd

SRV = Path(__file__).resolve().parent.parent
SNAPSHOT = "2026-09-02"
BUILD = SRV / "output" / "ds_include_pipeline_2026-09-02"
PICKLES = SRV / "output" / "embeddings_pickle" / "ds_include_2026-09-02_ge4sections_articles_deepinfra_2048"
OUT_DIR = SRV / "ui" / "data"
MIN_SECTIONS = 4
TARGET = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]


def present(v):
    return v is not None and str(v).strip() not in ("", "N/A", "NA", "n/a", "na")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    m = pd.read_csv(BUILD / "master_table.csv", dtype={"pmid": str})
    enough = (m["n_sections"] >= MIN_SECTIONS) & (m["n_retrieval"] > 0)
    is_article = m["type"] == "article"
    keep = m[enough & is_article]

    # ---- 1. corpus manifest: one row per included paper ----
    rows = []
    for pmid in keep["pmid"]:
        f = BUILD / "json_cache" / f"publication_metadata_PMID_{pmid}.json"
        d = json.loads(f.read_text())
        pkl = PICKLES / f"publication_metadata_PMID_{pmid}.pkl"
        n_chunks = len(pickle.loads(pkl.read_bytes())) if pkl.exists() else 0
        secs = d.get("sections") or {}
        doi = str(d.get("doi") or "").replace("https://doi.org/", "")
        rows.append({
            "pmid": str(d.get("pmid") or pmid),
            "pmcid": d.get("pmcid") or "",
            "doi": doi,
            "title": d.get("title") or "",
            "year": d.get("year") or "",
            "journal": d.get("source") or "",
            "publication_type": d.get("type") or "",
            "language": d.get("language") or "",
            "mesh_terms": "; ".join(d.get("mesh_terms") or []),
            "keywords": "; ".join(d.get("keywords") or []),
            "topic": d.get("topic") or "",
            "field": d.get("field") or "",
            "is_include": bool(d.get("is_include")),
            "include_condition": d.get("condition") or "",
            "include_grant_number": d.get("grant_number") or "",
            "include_rfa_pa_number": d.get("rfa_pa_number") or "",
            "include_fiscal_year": d.get("fiscal_year") or "",
            "sections_present": "; ".join(s for s in TARGET if present(secs.get(s))),
            "n_sections": sum(1 for s in TARGET if present(secs.get(s))),
            "n_indexed_chunks": n_chunks,
            "metadata_retrieved": datetime.fromtimestamp(f.stat().st_mtime).date().isoformat(),
            "corpus_snapshot": SNAPSHOT,
            "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{d.get('pmcid')}/",
        })
    manifest = pd.DataFrame(rows).sort_values("pmid").reset_index(drop=True)
    assert len(manifest) == len(keep), "manifest row count != selected papers"
    assert manifest["pmcid"].is_unique and manifest["pmid"].is_unique
    out1 = OUT_DIR / f"corpus_manifest_DS_INCLUDE_{SNAPSHOT.replace('-', '_')}.csv"
    manifest.to_csv(out1, index=False)
    print(f"manifest : {out1.name}  ({len(manifest)} papers, "
          f"{int(manifest['is_include'].sum())} INCLUDE, "
          f"{int(manifest['n_indexed_chunks'].sum()):,} chunks)")

    # ---- 2. screening audit: one row per screened record, with final status ----
    audit = pd.read_csv(BUILD / "step2_screening_audit.csv", dtype={"pmid": str}).fillna("")
    mt = m.set_index("pmid")
    included = set(keep["pmid"])

    def final_status(r):
        if r["verdict"] == "excluded":
            return "excluded (screening)", r["reason"]
        pmid = r["pmid"]
        if pmid in included:
            return "included", ""
        row = mt.loc[pmid] if pmid in mt.index else None
        if row is None:
            return "excluded (indexing)", "not in master table"
        if row["n_sections"] < MIN_SECTIONS or row["n_retrieval"] == 0:
            return "excluded (indexing)", f"fewer than {MIN_SECTIONS} of 6 sections"
        if row["type"] != "article":
            return "excluded (indexing)", f"non-article publication type: {row['type']}"
        return "excluded (indexing)", "unclassified"

    audit[["status", "exclusion_reason"]] = audit.apply(
        lambda r: pd.Series(final_status(r)), axis=1)
    audit.loc[audit["status"] == "excluded (screening)", "exclusion_reason"] = \
        audit.loc[audit["status"] == "excluded (screening)", "reason"]
    audit_out = audit[["pmid", "pmcid", "doi", "status", "exclusion_reason",
                       "pub_types", "n_sections"]].copy()
    audit_out["corpus_snapshot"] = SNAPSHOT
    n_inc = int((audit_out["status"] == "included").sum())
    assert n_inc == len(manifest), f"audit included={n_inc} != manifest {len(manifest)}"
    out2 = OUT_DIR / f"screening_audit_DS_INCLUDE_{SNAPSHOT.replace('-', '_')}.csv"
    audit_out.to_csv(out2, index=False)
    print(f"audit    : {out2.name}  ({len(audit_out)} records screened, {n_inc} included)")
    print("status breakdown:")
    print(audit_out.groupby(["status", "exclusion_reason"]).size().to_string())
    print(f"\ngenerated {date.today().isoformat()} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
