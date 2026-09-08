"""
bench_mem.py — benchmark multilingual-e5-small INT8 on NPU vs CPU.

Corpus: 113 chunks from the repo's own docs. Queries: 26 hand-written (20 EN +
6 ID, incl. cross-lingual ID queries over EN chunks), each with gold chunk IDs.

Pipeline per device:
  1. reshape IR to static [1, SEQ_LEN] (NPU rejects dynamic shapes — known pitfall)
  2. embed all chunks once (passage: prefix), time it
  3. embed each query (query: prefix), cosine top-k, recall@1/3/5
  4. report overall + monolingual/cross-lingual/hard splits + latency

E5 convention: queries get "query: ", passages get "passage: ". Without prefixes
the model still runs, but retrieval quality drops — the prefixes are part of the
model's training contract, not decoration.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

CORPUS = Path("corpus")
SEQ_LEN = 256


def load_all():
    chunks = json.loads((CORPUS / "chunks.json").read_text(encoding="utf-8"))
    queries = json.loads((CORPUS / "queries.json").read_text(encoding="utf-8"))
    return chunks, queries


def build_engine(model_dir, device):
    import numpy as np
    import openvino as ov
    from transformers import AutoTokenizer

    core = ov.Core()
    model = core.read_model(str(Path(model_dir) / "openvino_model.xml"))
    model.reshape({"input_ids": [1, SEQ_LEN],
                   "attention_mask": [1, SEQ_LEN],
                   "token_type_ids": [1, SEQ_LEN]})
    compiled = core.compile_model(model, device)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    def embed(texts, prefix):
        vecs = []
        for t in texts:
            enc = tok(prefix + t, return_tensors="np", padding="max_length",
                      truncation=True, max_length=SEQ_LEN)
            feed = {k: enc[k].astype(np.int64)
                    for k in ("input_ids", "attention_mask")}
            # the IR has a token_type_ids input but the tokenizer does not emit
            # it; all zeros is correct (single segment, no sentence pair)
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            out = compiled(feed)
            h = next(iter(out.values()))
            mask = enc["attention_mask"][0]
            v = (h[0] * mask[:, None]).sum(0) / mask.sum()
            n = float((v ** 2).sum() ** 0.5) or 1.0
            vecs.append(v / n)
        return vecs

    return compiled, embed


def recall_at(gold, ranked, k):
    return 1.0 if any(r in gold for r in ranked[:k]) else 0.0


def run_device(model_dir, device, chunks, queries):
    import numpy as np

    print(f"\n=== {device} ===", flush=True)
    t0 = time.time()
    try:
        compiled, embed = build_engine(model_dir, device)
    except Exception as e:
        print(f"COMPILE FAILED: {type(e).__name__}: {e}")
        return {"device": device, "compile_ok": False,
                "error": f"{type(e).__name__}: {e}"}
    try:
        dev = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        dev = "n/a (encoder-only, verify via LUID)"
    compile_s = time.time() - t0
    print(f"compile: {compile_s:.2f}s  EXECUTION_DEVICES={dev}", flush=True)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    t0 = time.time()
    cvecs = embed(texts, "passage: ")
    corpus_ms = (time.time() - t0) * 1000
    print(f"corpus: {len(texts)} chunks in {corpus_ms:.0f}ms "
          f"({corpus_ms/len(texts):.1f}ms/chunk)", flush=True)
    C = np.stack(cvecs)

    rows, lat = [], []
    for q in queries:
        t0 = time.time()
        qv = embed([q["text"]], "query: ")[0]
        ms = (time.time() - t0) * 1000
        lat.append(ms)
        sims = C @ qv
        ranked = [ids[i] for i in np.argsort(-sims)]
        rows.append({"id": q["id"], "lang": q["lang"], "hard": q["hard"],
                     "r1": recall_at(q["gold"], ranked, 1),
                     "r3": recall_at(q["gold"], ranked, 3),
                     "r5": recall_at(q["gold"], ranked, 5),
                     "top1": ranked[0], "gold": q["gold"], "ms": round(ms, 1)})
        mark = "" if rows[-1]["r3"] else "  <-- MISS@3"
        print(f"  {q['id']} [{q['lang']}] r@1={rows[-1]['r1']:.0f} "
              f"r@3={rows[-1]['r3']:.0f} top1={ranked[0]} "
              f"gold={q['gold'][0]} {ms:6.1f}ms{mark}", flush=True)

    def agg(sel):
        s = [r for r in rows if sel(r)] or rows
        return {f"recall@{k}": round(sum(r[f"r{k}"] for r in s) / len(s), 4)
                for k in (1, 3, 5)} | {"n": len(s)}

    out = {"device": device, "compile_ok": True,
           "compile_s": round(compile_s, 2), "exec_devices": str(dev),
           "seq_len": SEQ_LEN, "corpus_chunks": len(texts),
           "corpus_ms": round(corpus_ms, 1),
           "query_p50_ms": round(statistics.median(lat), 1),
           "overall": agg(lambda r: True),
           "monolingual_en": agg(lambda r: r["lang"] == "en"),
           "crosslingual_id": agg(lambda r: r["lang"] == "id"),
           "hard": agg(lambda r: r["hard"]),
           "rows": rows}
    for k, v in (("overall", out["overall"]), ("en", out["monolingual_en"]),
                 ("id", out["crosslingual_id"]), ("hard", out["hard"])):
        print(f"  [{k}] n={v['n']} r@1={v['recall@1']} r@3={v['recall@3']} "
              f"r@5={v['recall@5']}", flush=True)
    print(f"  query p50 {out['query_p50_ms']}ms", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/e5-small-int8-ov")
    ap.add_argument("--devices", default="NPU,CPU")
    ap.add_argument("--out", default="mem_bench.json")
    args = ap.parse_args()

    chunks, queries = load_all()
    print(f"corpus: {len(chunks)} chunks, queries: {len(queries)} "
          f"({sum(1 for q in queries if q['lang']=='id')} ID)")

    results = [run_device(args.model, d.strip(), chunks, queries)
               for d in args.devices.split(",")]
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    ok = [r for r in results if r.get("compile_ok")]
    if len(ok) > 1:
        agree = sum(1 for a, b in zip(ok[0]["rows"], ok[1]["rows"])
                    if a["top1"] == b["top1"])
        print(f"top1 agreement {ok[0]['device']} vs {ok[1]['device']}: "
              f"{agree}/{len(ok[0]['rows'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
