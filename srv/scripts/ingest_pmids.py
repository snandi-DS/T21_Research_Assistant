"""
ingest_pmids.py

Incremental corpus-update pipeline. Takes a list of unique PMIDs and, for each:

  1. Eligibility gate (fetch_pmc_publications.check_pmid_eligibility) --
     Publication Type filter (no case reports, retracted publications,
     conference proceedings, comments, editorials, letters, personal
     narratives, biographies, historical articles), PMC open-access
     availability (must have a PMCID), and an independent CrossRef
     retraction cross-check.
  2. If eligible: full metadata enrichment via
     fetch_pmc_publications.fetch_and_save_metadata() -- the same
     OpenAlex -> NCBI Entrez -> CrossRef -> PMC OAI chain used for the main
     corpus, including the section-extraction fix applied during the
     Scientific Reports revision (fetch_pmc_publications._parse_pmc_xml()).
  3. Chunk + embed (embed_and_insert.compute_embeddings) and insert/update
     into Milvus (embed_and_insert.insert_embeddings) for every newly
     eligible PMID in this run.

Idempotent and resume-safe: a PMID whose JSON cache already exists is
skipped (not re-fetched, not re-embedded) unless --force is given, so this
can be re-run periodically with an updated PMID list to grow the corpus
incrementally without redoing prior work. insert_embeddings()'s own
upsert-by-deterministic-pk logic means re-running with --force on an
already-ingested PMID correctly updates it rather than duplicating rows.

Usage (run from srv/):
    conda run -n dschat python scripts/ingest_pmids.py --pmid-file input/new_pmids.txt
    conda run -n dschat python scripts/ingest_pmids.py --pmids 12345678 23456789
    conda run -n dschat python scripts/ingest_pmids.py --pmid-file input/new_pmids.txt --force
    conda run -n dschat python scripts/ingest_pmids.py --pmid-file input/new_pmids.txt --dry-run
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fetch_pmc_publications as fpp
import embed_and_insert as eai

JSON_CACHE_DIR = Path(__file__).parent.parent / "output" / "DS_PMC_05_2026"
LOG_DIR        = Path(__file__).parent.parent / "output" / "ingest_pmids_logs"
TEMP_INPUT_DIR  = Path(__file__).parent.parent / "output" / "_ingest_temp_input"
TEMP_PICKLE_DIR = Path(__file__).parent.parent / "output" / "_ingest_temp_pickles"
INCLUDE_EXCEL   = Path(__file__).parent.parent / "input" / "INCLUDE_Publications_by_2025-11-14.xlsx"


def _apply_include_tags(pmids: list) -> int:
    """
    Cross-reference freshly-fetched PMIDs against the INCLUDE excel and patch
    is_include/condition/grant_number/rfa_pa_number/fiscal_year in their cached
    JSON. fetch_and_save_metadata() -- unlike the batch fetch functions -- never
    sets these fields itself (confirmed 2026-08-31: 80 of 388 papers ingested via
    this script were is_include=True per the INCLUDE excel but landed in Milvus
    as is_include=0/empty since this call was missing). Always run this after
    fetching, or genuinely INCLUDE-funded papers silently lose that provenance.

    Returns the number of PMIDs actually tagged is_include=True.
    """
    if not INCLUDE_EXCEL.exists():
        print(f"[WARN] INCLUDE excel not found at {INCLUDE_EXCEL} -- skipping is_include tagging")
        return 0
    lookup = fpp.load_include_lookup(INCLUDE_EXCEL)
    n_tagged = 0
    for pmid in pmids:
        path = JSON_CACHE_DIR / f"publication_metadata_PMID_{pmid}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        rec = lookup.get(pmid)
        is_inc = rec is not None
        changed = bool(d.get("is_include")) != is_inc
        d["is_include"] = is_inc
        for field in ("condition", "grant_number", "rfa_pa_number", "fiscal_year"):
            val = (rec or {}).get(field, "") or ""
            if d.get(field, "") != val:
                changed = True
            d[field] = val
        if changed:
            path.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        if is_inc:
            n_tagged += 1
    return n_tagged


def _load_pmid_list(args) -> list:
    pmids = []
    if args.pmid_file:
        pmids.extend(Path(args.pmid_file).read_text().split())
    if args.pmids:
        pmids.extend(args.pmids)
    # Dedupe, preserve order
    seen, out = set(), []
    for p in pmids:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmid-file", type=str, default=None, help="File of PMIDs, one per line")
    ap.add_argument("--pmids", nargs="+", default=None, help="PMIDs given directly on the command line")
    ap.add_argument("--force", action="store_true",
                     help="Re-check eligibility and re-fetch even if a JSON cache entry already exists")
    ap.add_argument("--dry-run", action="store_true",
                     help="Run the eligibility gate only, report the verdicts, do not fetch/chunk/embed/insert")
    args = ap.parse_args()

    pmids = _load_pmid_list(args)
    if not pmids:
        print("[ERROR] No PMIDs given. Use --pmid-file or --pmids.")
        sys.exit(1)

    print(f"PMIDs given: {len(pmids)}")
    JSON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    already_cached, eligible, excluded, errors = [], [], [], []

    for i, pmid in enumerate(pmids, 1):
        existing = JSON_CACHE_DIR / f"publication_metadata_PMID_{pmid}.json"
        if existing.exists() and not args.force:
            already_cached.append(pmid)
            print(f"  [{i}/{len(pmids)}] {pmid}: already cached, skipping")
            continue

        verdict = fpp.check_pmid_eligibility(pmid)
        if not verdict["eligible"]:
            excluded.append(verdict)
            print(f"  [{i}/{len(pmids)}] {pmid}: EXCLUDED -- {verdict['reason']}")
            continue

        eligible.append(verdict)
        print(f"  [{i}/{len(pmids)}] {pmid}: ELIGIBLE -- {verdict['pmcid']} -- {verdict['title'][:60]}")

    # ── Log the eligibility run regardless of --dry-run ──────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": ts,
        "n_input": len(pmids),
        "n_already_cached": len(already_cached),
        "n_eligible": len(eligible),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "eligible_pmids": [v["pmid"] for v in eligible],
    }
    log_path = LOG_DIR / f"ingest_run_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"Already cached : {len(already_cached)}")
    print(f"Eligible (new) : {len(eligible)}")
    print(f"Excluded       : {len(excluded)}")
    if excluded:
        from collections import Counter
        reasons = Counter(v["reason"].split(":")[0] for v in excluded)
        for reason, n in reasons.most_common():
            print(f"  {reason:30s}: {n}")
    print(f"Log saved to   : {log_path}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY RUN] Stopping before fetch/chunk/embed/insert.")
        return

    if not eligible:
        print("Nothing new and eligible to ingest.")
        return

    # ── Enrichment: fetch full metadata + section-extraction-fixed full text ─
    print("\nFetching full metadata for eligible PMIDs...")
    fetched_ok = []
    for i, v in enumerate(eligible, 1):
        pmid = v["pmid"]
        try:
            fpp.fetch_and_save_metadata(
                pmid, out_dir=JSON_CACHE_DIR, fetch_fulltext=True,
                overwrite=args.force, sleep_between=0.5,
            )
            fetched_ok.append(pmid)
            print(f"  [{i}/{len(eligible)}] {pmid}: fetched and saved")
        except Exception as exc:
            errors.append({"pmid": pmid, "stage": "enrichment", "error": str(exc)[:200]})
            print(f"  [{i}/{len(eligible)}] {pmid}: [ERROR] {exc}")

    if fetched_ok:
        n_tagged = _apply_include_tags(fetched_ok)
        print(f"\nINCLUDE tagging: {n_tagged}/{len(fetched_ok)} papers marked is_include=True")

    if not fetched_ok:
        print("No papers were successfully enriched. Stopping before embed/insert.")
        return

    # ── Chunk + embed + insert, scoped to only this run's new papers ─────────
    for d in (TEMP_INPUT_DIR, TEMP_PICKLE_DIR):
        if d.exists():
            shutil.rmtree(d)
    TEMP_INPUT_DIR.mkdir(parents=True)
    TEMP_PICKLE_DIR.mkdir(parents=True)

    n_linked = 0
    for pmid in fetched_ok:
        src = JSON_CACHE_DIR / f"publication_metadata_PMID_{pmid}.json"
        if src.exists():
            (TEMP_INPUT_DIR / src.name).symlink_to(src.resolve())
            n_linked += 1
    print(f"\nLinked {n_linked} newly-fetched papers for embedding")

    eai.compute_embeddings(TEMP_INPUT_DIR, TEMP_PICKLE_DIR, force=True)
    eai.insert_embeddings(TEMP_PICKLE_DIR, mark_include=False)

    shutil.rmtree(TEMP_INPUT_DIR)
    shutil.rmtree(TEMP_PICKLE_DIR)

    print("\n[SUCCESS] Incremental ingestion complete.")
    print(f"  Newly ingested: {len(fetched_ok)}")
    print(f"  Errors        : {len(errors)}")
    if errors:
        err_path = LOG_DIR / f"ingest_errors_{ts}.json"
        with open(err_path, "w", encoding="utf-8") as fh:
            json.dump(errors, fh, indent=2, ensure_ascii=False)
        print(f"  Error log     : {err_path}")


if __name__ == "__main__":
    main()
