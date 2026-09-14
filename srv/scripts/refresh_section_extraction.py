"""
refresh_section_extraction.py

Re-fetches PMC OAI XML for each active-corpus paper and re-parses it with the
fixed _parse_pmc_xml() (fetch_pmc_publications.py), updating each paper's local
JSON cache 'sections' field in place. Local filesystem only -- does not touch
Milvus or call any embedding API. Checkpointed and resume-safe.

Usage (run from srv/):
    conda run -n dschat python scripts/refresh_section_extraction.py [--limit N] [--pmcid-file PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fetch_pmc_publications as fpp

JSON_CACHE_DIR = Path(__file__).parent.parent / "output" / "DS_PMC_05_2026"
DIFF_LOG_PATH  = Path(__file__).parent.parent / "output" / "section_extraction_refresh_log.json"
CHECKPOINT_PATH = Path(__file__).parent.parent / "output" / "section_extraction_refresh_checkpoint.json"
SLEEP_BETWEEN  = 0.4


def _section_signature(sections: dict) -> dict:
    """Char length of each COMMON_SECTIONS entry, MISSING/'N/A' treated as 0."""
    out = {}
    for k in fpp.COMMON_SECTIONS:
        v = sections.get(k, fpp.MISSING) or fpp.MISSING
        out[k] = 0 if str(v).strip() in ("N/A", "") else len(str(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Process at most N papers (for testing)")
    ap.add_argument("--pmcid-file", type=str, default=None,
                     help="Optional file of PMCIDs (one per line) to restrict processing to")
    args = ap.parse_args()

    json_files = sorted(JSON_CACHE_DIR.glob("publication_metadata_PMID_*.json"))
    print(f"Found {len(json_files)} JSON cache files")

    restrict = None
    if args.pmcid_file:
        restrict = set(Path(args.pmcid_file).read_text().split())
        print(f"Restricting to {len(restrict)} PMCIDs from {args.pmcid_file}")

    # Resume support
    done = {}
    if CHECKPOINT_PATH.exists():
        done = json.loads(CHECKPOINT_PATH.read_text())
        print(f"Resuming -- {len(done)} papers already processed in a prior run")

    diff_log = []
    if DIFF_LOG_PATH.exists():
        diff_log = json.loads(DIFF_LOG_PATH.read_text())

    targets = json_files
    if args.limit:
        targets = targets[: args.limit]

    n_processed = n_changed = n_skipped = n_error = 0
    for i, fp in enumerate(targets, 1):
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  [{i}/{len(targets)}] SKIP (bad json) {fp.name}: {e}")
            n_error += 1
            continue

        pmcid = str(data.get("pmcid") or "").strip()
        pmid  = str(data.get("pmid") or "").strip()

        if not pmcid:
            n_skipped += 1
            continue
        if restrict is not None and pmcid not in restrict:
            n_skipped += 1
            continue
        if pmid in done:
            continue

        old_sections = data.get("sections") or {}
        old_sig = _section_signature(old_sections)

        new_sections = fpp._fetch_pmc_sections(pmcid)
        new_sig = _section_signature(new_sections)

        changed = old_sig != new_sig
        if changed:
            data["sections"] = new_sections
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
            n_changed += 1
            diff_entry = {
                "pmid": pmid, "pmcid": pmcid, "title": str(data.get("title") or "")[:80],
                "old": old_sig, "new": new_sig,
            }
            diff_log.append(diff_entry)
            tag = "CHANGED"
        else:
            tag = "same"

        n_processed += 1
        done[pmid] = changed
        print(f"  [{i}/{len(targets)}] {tag:8s} {pmcid}  {pmid}")

        if n_processed % 50 == 0:
            CHECKPOINT_PATH.write_text(json.dumps(done))
            DIFF_LOG_PATH.write_text(json.dumps(diff_log, indent=1, ensure_ascii=False))
            print(f"  [checkpoint] {n_processed} processed, {n_changed} changed so far")

        time.sleep(SLEEP_BETWEEN)

    CHECKPOINT_PATH.write_text(json.dumps(done))
    DIFF_LOG_PATH.write_text(json.dumps(diff_log, indent=1, ensure_ascii=False))

    print()
    print("=" * 60)
    print(f"Processed : {n_processed}")
    print(f"Changed   : {n_changed}")
    print(f"Skipped   : {n_skipped}")
    print(f"Errors    : {n_error}")
    print(f"Diff log  : {DIFF_LOG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
