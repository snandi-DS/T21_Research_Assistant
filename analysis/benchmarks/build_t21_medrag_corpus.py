"""
Build a MedRAG-format corpus from the T21 papers, for the controlled ablation.

The reviewer asks for a controlled general biomedical RAG baseline. Running MedRAG
over the SAME documents the T21 Research Assistant indexes, with the same generator,
makes retrieval the only variable: MedRAG applies generic dense retrieval over
uniformly chunked full text, with no section labels, no section prioritization and
no cross-encoder rerank, which is exactly the ablation being asked for.

Design choices, all deliberate:

- Document set: the PMIDs currently live in the T21 Milvus collection, so both arms
  answer from an identical corpus. Falls back to every cached JSON if Milvus is down.
- Full text, not the retrieval subset: sections are concatenated in natural reading
  order (Abstract, Introduction, Methods, Results, Discussion, Conclusion). MedRAG
  therefore sees Abstract and Introduction too, which T21 stores but excludes from
  retrieval. That asymmetry is the point: it is what a generic RAG would do.
- Chunking: ~1000 characters to match MedRAG's own snippet size (measured mean 810,
  max 999 on their Textbooks corpus), NOT T21's 1200/200, so the baseline follows the
  baseline system's conventions rather than inheriting ours.
- title = PMCID, so retrieved snippets remain traceable to a paper and the generated
  answers can be checked for grounding the same way the T21 answers are.

Output: MedRAG/corpus/t21/chunk/*.jsonl  plus  t21_corpus_manifest.csv
Usage:  conda run -n medrag python build_t21_medrag_corpus.py
"""
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SRV = BENCH.parent.parent / "srv"
JSON_CACHE = SRV / "output" / "DS_PMC_05_2026"
OUT_CHUNK = BENCH / "MedRAG" / "corpus" / "t21" / "chunk"

SECTION_ORDER = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
CHUNK_CHARS = 1000
PAPERS_PER_FILE = 200          # keep jsonl files a manageable size


def present(v):
    return v is not None and str(v).strip() not in ("", "N/A", "NA", "n/a", "na")


def live_pmids():
    """PMIDs currently in the T21 Milvus collection (the set T21 answers from)."""
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
        print(f"[WARN] Milvus unavailable ({str(exc)[:80]}); falling back to the full JSON cache")
        return None


def split_text(text, size=CHUNK_CHARS):
    """Sentence-aware split at ~size chars, mirroring MedRAG's snippet granularity."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if not s.strip():
            continue
        if len(cur) + len(s) + 1 <= size:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            # a single sentence longer than the budget is hard-split
            while len(s) > size:
                chunks.append(s[:size])
                s = s[size:]
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def main():
    keep = live_pmids()
    files = sorted(JSON_CACHE.glob("publication_metadata_PMID_*.json"))
    print(f"cached JSONs: {len(files)}")

    OUT_CHUNK.mkdir(parents=True, exist_ok=True)
    for old in OUT_CHUNK.glob("*.jsonl"):
        old.unlink()

    manifest, buf, part, n_chunks, n_papers = [], [], 0, 0, 0
    for f in files:
        d = json.loads(f.read_text())
        pmid = str(d.get("pmid") or f.stem.replace("publication_metadata_PMID_", ""))
        if keep is not None and pmid not in keep:
            continue
        pmcid = str(d.get("pmcid") or "").strip()
        secs = d.get("sections") or {}
        body = "\n\n".join(str(secs[s]).strip() for s in SECTION_ORDER if present(secs.get(s)))
        if not body:
            continue

        pieces = split_text(body)
        for i, c in enumerate(pieces):
            buf.append({"id": f"{pmcid or pmid}_{i}", "title": pmcid or pmid, "content": c})
        n_chunks += len(pieces)
        n_papers += 1
        manifest.append({"pmid": pmid, "pmcid": pmcid, "chunks": len(pieces),
                         "chars": len(body), "title": str(d.get("title") or "")[:120]})

        if n_papers % PAPERS_PER_FILE == 0:
            (OUT_CHUNK / f"t21_{part:03d}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in buf) + "\n")
            buf, part = [], part + 1

    if buf:
        (OUT_CHUNK / f"t21_{part:03d}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in buf) + "\n")

    import csv
    with open(BENCH / "t21_corpus_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pmid", "pmcid", "chunks", "chars", "title"])
        w.writeheader()
        w.writerows(manifest)

    avg = n_chunks / max(n_papers, 1)
    print(f"\npapers written : {n_papers}")
    print(f"chunks written : {n_chunks:,}  (mean {avg:.1f} per paper)")
    print(f"jsonl files    : {len(list(OUT_CHUNK.glob('*.jsonl')))} -> {OUT_CHUNK}")
    print(f"manifest       : t21_corpus_manifest.csv")


if __name__ == "__main__":
    main()
