"""
test_rerank.py — does a cross-encoder reranker fix the bi-encoder's misses?

Protocol: bi-encoder top-10 per query (from mem_bench.json top1 agreement run —
recomputed here on the fly), score all 10 pairs with mMiniLM cross-encoder,
re-rank, compare recall@1 before/after. Latency per pair measured too, since a
reranker that costs 10x the query budget for +0.02 recall is a bad trade.

Honest framing: same 26 hand-written queries, so the gain is measured on the same
small-n set as the baseline. A gain here means "worth keeping in the server",
not "state of the art".
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from bench_mem import load_all, build_engine, SEQ_LEN

RERANK_MODEL = "../models/mmMiniLM-reranker-int8-ov"
RERANK_SEQ = 384
TOPK = 10


def build_reranker(device):
    import openvino as ov
    from transformers import AutoTokenizer

    core = ov.Core()
    model = core.read_model(str(Path(RERANK_MODEL) / "openvino_model.xml"))
    model.reshape({"input_ids": [1, RERANK_SEQ],
                   "attention_mask": [1, RERANK_SEQ]})
    compiled = core.compile_model(model, device)
    tok = AutoTokenizer.from_pretrained(RERANK_MODEL, local_files_only=True)

    def score(query, passages):
        out = []
        for p in passages:
            enc = tok(query, p, return_tensors="np", padding="max_length",
                      truncation=True, max_length=RERANK_SEQ)
            feed = {k: enc[k].astype(np.int64)
                    for k in ("input_ids", "attention_mask")}
            logits = next(iter(compiled(feed).values()))
            out.append(float(logits[0][0]))
        return out

    return compiled, score


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "CPU"
    chunks, queries = load_all()
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    _, embed = build_engine("../models/e5-small-int8-ov",
                            "CPU")  # bi-encoder: device-independent recall
    C = np.stack(embed(texts, "passage: "))

    t0 = time.time()
    try:
        compiled, score = build_reranker(device)
    except Exception as e:
        print(f"RERANK COMPILE FAILED on {device}: {type(e).__name__}: {e}")
        return 1
    try:
        dev = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        dev = "n/a"
    print(f"reranker compile on {device}: {time.time()-t0:.2f}s exec={dev}",
          flush=True)

    score(queries[0]["text"], [texts[0]])  # warmup
    lat, before1, after1, rows = [], [], [], []
    for q in queries:
        qv = embed([q["text"]], "query: ")[0]
        ranked = [ids[i] for i in np.argsort(-(C @ qv))]
        cand = ranked[:TOPK]
        cand_texts = [texts[ids.index(c)] for c in cand]
        t0 = time.time()
        s = score(q["text"], cand_texts)
        ms = (time.time() - t0) * 1000
        lat.append(ms / TOPK)
        re_ranked = [c for _, c in sorted(zip(s, cand), reverse=True)]
        b = 1.0 if cand[0] in q["gold"] else 0.0
        a = 1.0 if re_ranked[0] in q["gold"] else 0.0
        before1.append(b)
        after1.append(a)
        rows.append({"id": q["id"], "lang": q["lang"], "bi_top1": cand[0],
                     "re_top1": re_ranked[0], "gold": q["gold"],
                     "fixed": bool(a > b), "broken": bool(a < b)})
        tag = "FIXED" if a > b else ("BROKEN" if a < b else "")
        print(f"  {q['id']} [{q['lang']}] bi={cand[0]} re={re_ranked[0]} "
              f"gold={q['gold'][0]} {ms:6.1f}ms/10 {tag}", flush=True)

    n = len(queries)
    print(f"\ndevice={device} top{TOPK} rerank:",
          f"r@1 {sum(before1)/n:.4f} -> {sum(after1)/n:.4f} "
          f"(+{sum(after1)/n - sum(before1)/n:+.4f})", flush=True)
    print(f"fixed: {sum(r['fixed'] for r in rows)}, "
          f"broken: {sum(r['broken'] for r in rows)}, "
          f"pair p50: {statistics.median(lat):.1f}ms", flush=True)
    for r in rows:
        if r["fixed"] or r["broken"]:
            print(f"  {'FIXED ' if r['fixed'] else 'BROKEN'} {r['id']}: "
                  f"{r['bi_top1']} -> {r['re_top1']} (gold {r['gold']})")
    Path(f"rerank_{device}.json").write_text(json.dumps(
        {"device": device, "topk": TOPK, "r1_before": sum(before1) / n,
         "r1_after": sum(after1) / n, "pair_p50_ms": statistics.median(lat),
         "rows": rows}, indent=2))
    print(f"wrote rerank_{device}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
