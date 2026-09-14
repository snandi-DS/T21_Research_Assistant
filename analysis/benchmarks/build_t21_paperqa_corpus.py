"""
Build a PaperQA2 corpus from the T21 papers.

PaperQA2 indexes a directory of documents. We give it the SAME 1,602 papers the T21
Research Assistant indexes, written as one plain-text file per paper, so the corpus is
identical across benchmark arms and only the retrieval and answering machinery differs.

Why text files rather than the downloaded PDFs:

- Coverage. Every paper has full text in the JSON cache; only about 80 percent have a
  fetchable PDF, and that download is still running. Benchmarking on PDFs would silently
  drop roughly a fifth of the corpus and make the arms non-comparable.
- Fidelity. The JSON sections are exactly the text T21 embeds. Feeding PaperQA2 PDFs
  would introduce a different text-extraction path (PDF parsing artifacts, headers,
  figure captions, reference lists) and confound the comparison with OCR quality.
- Reproducibility. Text files are deterministic; PDF parsers are not.

Each file starts with a title line and the PMCID, because PaperQA2 uses the document
header for citation keys, so answers remain traceable to a paper the same way the T21
and MedRAG answers are.

Sections are concatenated in reading order with their headings retained. PaperQA2 does
its own chunking and has no notion of section priority, which is the point of the
comparison.

Usage:  conda run -n paperqa python build_t21_paperqa_corpus.py
Output: analysis/benchmarks/paperqa_corpus/<PMCID>.txt  + t21_paperqa_manifest.csv
"""
import csv
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SRV = BENCH.parent.parent / "srv"
JSON_CACHE = SRV / "output" / "DS_PMC_05_2026"
OUT_DIR = BENCH / "paperqa_corpus"

SECTION_ORDER = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]


def present(v):
    return v is not None and str(v).strip() not in ("", "N/A", "NA", "n/a", "na")


def live_pmids():
    """The PMIDs the T21 system actually answers from, so the arms match exactly."""
    try:
        from pymilvus import connections, Collection
        connections.connect(host="localhost", port=19530)
        col = Collection("DS_PMC_v7_DEEPINFRA_2048")
        col.load()
        out = set()
        it = col.query_iterator(expr='pub_type == "article"', output_fields=["pmid"], batch_size=16384)
        while True:
            rows = it.next()
            if not rows:
                break
            out.update(r["pmid"] for r in rows)
        it.close()
        print(f"live Milvus articles: {len(out)}")
        return out
    except Exception as exc:
        print(f"[WARN] Milvus unavailable ({str(exc)[:70]}); using the full JSON cache")
        return None


def main():
    keep = live_pmids()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.txt"):
        old.unlink()

    rows, n = [], 0
    for f in sorted(JSON_CACHE.glob("publication_metadata_PMID_*.json")):
        d = json.loads(f.read_text())
        pmid = str(d.get("pmid") or f.stem.replace("publication_metadata_PMID_", ""))
        if keep is not None and pmid not in keep:
            continue
        pmcid = str(d.get("pmcid") or "").strip() or pmid
        secs = d.get("sections") or {}
        body_parts = [f"## {s}\n{str(secs[s]).strip()}"
                      for s in SECTION_ORDER if present(secs.get(s))]
        if not body_parts:
            continue

        title = str(d.get("title") or "").strip()
        year = d.get("year") or ""
        journal = str(d.get("source") or "").strip()
        header = (f"{title}\n"
                  f"{journal} ({year})\n"
                  f"PMCID: {pmcid}  PMID: {pmid}\n"
                  f"DOI: {d.get('doi') or ''}\n")
        text = header + "\n" + "\n\n".join(body_parts) + "\n"
        (OUT_DIR / f"{pmcid}.txt").write_text(text, encoding="utf-8")
        n += 1
        rows.append({"pmid": pmid, "pmcid": pmcid, "year": year,
                     "chars": len(text), "sections": len(body_parts),
                     "title": title[:120]})

    with open(BENCH / "t21_paperqa_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pmid", "pmcid", "year", "chars", "sections", "title"])
        w.writeheader()
        w.writerows(rows)

    total_chars = sum(r["chars"] for r in rows)
    print(f"\npapers written : {n}")
    print(f"total text     : {total_chars/1e6:.1f} M chars (~{total_chars/4/1e6:.1f} M tokens)")
    print(f"directory      : {OUT_DIR}")
    print(f"manifest       : t21_paperqa_manifest.csv")


if __name__ == "__main__":
    main()
