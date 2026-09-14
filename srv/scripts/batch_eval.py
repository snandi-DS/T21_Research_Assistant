"""
scripts/batch_eval.py
---------------------
Batch evaluation of all sample questions against the T21 Research Assistant API.

Loads questions from:
  input/AI_research_assistant_Sample_questions_revised_JIM_Negative_Ctrl_dt_04_29_2026.xlsx

Runs each question through POST /answer and saves results to:
  output/batch_eval/eval_results_<timestamp>.xlsx

Usage:
    python scripts/batch_eval.py [--top-k 12] [--backend http://localhost:8698] [--clear-cache]
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_XLSX = ROOT / "input" / "AI_research_assistant_Sample_questions_revised_JIM_Negative_Ctrl_dt_04_29_2026.xlsx"
OUTPUT_DIR = ROOT / "output" / "batch_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_section(markdown: str, section: str) -> str:
    """Extract text of a markdown section (# Section or ## Section)."""
    pattern = rf"#{1,2}\s*{re.escape(section)}[^\n]*\n(.*?)(?=\n#{1,2}\s|\Z)"
    m = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def strip_citations(text: str) -> str:
    """Remove inline [PMCxxxxxxx:Section] citations for clean text comparison."""
    return re.sub(r"\[PMC\d{4,}:[A-Za-z0-9_-]+\](?:\([^)]+\))?", "", text).strip()


def run_eval(backend_url: str, top_k: int, clear_cache: bool):
    # ── Load questions ─────────────────────────────────────────────────────
    print(f"\n[INFO] Loading questions from: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX, sheet_name="Questions_for_testing")
    df = df.rename(columns={
        "Original question": "question",
        "Answer":            "expected_answer",
        "Citation":          "expected_citation"
    })
    df["question"] = df["question"].fillna("").str.strip()
    df = df[df["question"] != ""].reset_index(drop=True)
    print(f"[INFO] {len(df)} questions loaded")

    # ── Optionally clear cache ────────────────────────────────────────────
    if clear_cache:
        try:
            r = requests.post(f"{backend_url}/cache/clear", timeout=10)
            print(f"[INFO] Cache cleared: {r.status_code}")
        except Exception as e:
            print(f"[WARN] Could not clear cache: {e}")

    # ── Results storage ───────────────────────────────────────────────────
    results = []

    for idx, row in df.iterrows():
        q = row["question"]
        expected = str(row.get("expected_answer", "") or "")
        citations = str(row.get("expected_citation", "") or "")

        print(f"\n[{idx+1}/{len(df)}] {q[:80]}...")

        t0 = time.time()
        try:
            resp = requests.post(
                f"{backend_url}/answer",
                json={"question": q, "top_k": top_k, "include_hits": False},
                timeout=120
            )
            elapsed = round(time.time() - t0, 2)

            if resp.status_code == 200:
                data = resp.json()
                md   = data.get("markdown", "")
                summary    = strip_citations(extract_section(md, "Summary"))
                key_evid   = extract_section(md, "Key Evidence")
                cited_pmcs = data.get("citations", [])

                status = "OK"
                error  = ""
                print(f"  [OK] {elapsed}s | {len(cited_pmcs)} citations")
            else:
                summary = key_evid = ""
                cited_pmcs = []
                status = f"HTTP_{resp.status_code}"
                error  = resp.text[:200]
                elapsed = round(time.time() - t0, 2)
                print(f"  [ERROR] HTTP {resp.status_code}")

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            summary = key_evid = ""
            cited_pmcs = []
            status = "EXCEPTION"
            error  = str(e)
            print(f"  [EXCEPTION] {e}")

        results.append({
            "index":             idx + 1,
            "question":          q,
            "expected_answer":   expected,
            "expected_citation": citations,
            "status":            status,
            "elapsed_sec":       elapsed,
            "generated_summary": summary,
            "generated_key_evidence": key_evid,
            "cited_pmcs":        ", ".join(cited_pmcs) if cited_pmcs else "",
            "error":             error,
        })

        # Small pause to avoid rate-limit
        time.sleep(0.5)

    # ── Save results ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUT_DIR / f"eval_results_{timestamp}.xlsx"

    results_df = pd.DataFrame(results)
    results_df.to_excel(out_path, index=False)

    # ── Summary stats ─────────────────────────────────────────────────────
    total   = len(results_df)
    ok      = (results_df["status"] == "OK").sum()
    failed  = total - ok
    avg_t   = results_df["elapsed_sec"].mean()

    print(f"\n{'='*60}")
    print(f"BATCH EVAL COMPLETE")
    print(f"  Total questions : {total}")
    print(f"  Successful      : {ok}")
    print(f"  Failed          : {failed}")
    print(f"  Avg response    : {avg_t:.1f}s")
    print(f"  Results saved   : {out_path}")
    print(f"{'='*60}\n")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation of T21 Research Assistant")
    parser.add_argument("--top-k",   type=int, default=12,                       help="Number of chunks to retrieve (default: 12)")
    parser.add_argument("--backend", type=str, default="http://localhost:8698",  help="API base URL (default: http://localhost:8698)")
    parser.add_argument("--clear-cache", action="store_true",                    help="Clear API response cache before running")
    args = parser.parse_args()

    # Check API is reachable
    try:
        r = requests.get(f"{args.backend}/", timeout=5)
        info = r.json()
        print(f"[INFO] API connected: {info.get('name')} v{info.get('version')} [{info.get('status')}]")
    except Exception as e:
        print(f"[ERROR] Cannot reach API at {args.backend}: {e}")
        sys.exit(1)

    run_eval(
        backend_url=args.backend,
        top_k=args.top_k,
        clear_cache=args.clear_cache
    )
