"""
bench_hybrid.py — does BM25 + RRF fusion beat pure dense on the 39 queries?

Motivation (from the skill's own open list): pure dense cosine has no exact-term
bias. Queries containing filenames, error codes, or hex IDs should favor the chunk
that literally contains them, but dense ranking can prefer a paraphrase instead.

Method, no new dependencies (stdlib only — same policy as the inline ROUGE in
bench_aux):
  BM25 (k1=1.5, b=0.75) over whitespace/alnum tokens, lowercased. No stemming
  (code tokens must not be stemmed).
  RRF fusion: score = 1/(60+rank_dense) + 1/(60+rank_bm25), re-rank top-20.
Report recall@1/3/5 for dense-only vs hybrid, overall + EN/ID splits. BM25 costs
nothing at query time to speak of; the question is purely whether it helps.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from bench_mem import load_all, build_engine

K1, B = 1.5, 0.75
FUSE_DEPTH = 20
RRF_K = 60


def toks(s):
    return re.findall(r"[a-z0-9_.$/-]+", s.lower())


def build_index(texts):
    docs = [toks(t) for t in texts]
    df = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    avgdl = sum(map(len, docs)) / max(n, 1)
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    return docs, idf, avgdl


def bm25_rank(query, docs, idf, avgdl, ids):
    qt = toks(query)
    scores = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc) or 1
        s = 0.0
        for t in qt:
            if t not in idf:
                continue
            f = tf.get(t, 0)
            s += idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
        scores.append(s)
    return [ids[i] for i in np.argsort([-x for x in scores])]


def rrf_fuse(dense_ranked, bm25_ranked, depth=FUSE_DEPTH):
    pool = list(dict.fromkeys(dense_ranked[:depth] + bm25_ranked[:depth]))
    pos_d = {c: i + 1 for i, c in enumerate(dense_ranked)}
    pos_b = {c: i + 1 for i, c in enumerate(bm25_ranked)}
    scored = [(1 / (RRF_K + pos_d.get(c, 10 ** 9))
               + 1 / (RRF_K + pos_b.get(c, 10 ** 9)), c) for c in pool]
    scored.sort(reverse=True)
    return [c for _, c in scored]


def main():
    chunks, queries = load_all()
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    docs, idf, avgdl = build_index(texts)

    _, embed = build_engine("../models/e5-base-int8-ov", "CPU")
    C = np.stack(embed(texts, "passage: "))

    rows = []
    for q in queries:
        qv = embed([q["text"]], "query: ")[0]
        dense = [ids[i] for i in np.argsort(-(C @ qv))]
        bm = bm25_rank(q["text"], docs, idf, avgdl, ids)
        fused = rrf_fuse(dense, bm)
        rows.append({
            "id": q["id"], "lang": q["lang"], "gold": q["gold"],
            "dense_top1": dense[0], "bm25_top1": bm[0], "fused_top1": fused[0],
            "d_r1": any(g in dense[:1] for g in q["gold"]),
            "d_r3": any(g in dense[:3] for g in q["gold"]),
            "f_r1": any(g in fused[:1] for g in q["gold"]),
            "f_r3": any(g in fused[:3] for g in q["gold"]),
            "f_r5": any(g in fused[:5] for g in q["gold"]),
        })

    def agg(sel, key):
        s = [r for r in rows if sel(r)]
        return sum(r[key] for r in s) / len(s), len(s)

    print(f"{'split':10s}{'n':>4s}{'dense_r1':>10s}{'hyb_r1':>9s}"
          f"{'dense_r3':>10s}{'hyb_r3':>9s}{'hyb_r5':>9s}")
    for name, sel in (("overall", lambda r: True),
                      ("en", lambda r: r["lang"] == "en"),
                      ("id", lambda r: r["lang"] == "id")):
        (dr1, n), (fr1, _) = agg(sel, "d_r1"), agg(sel, "f_r1")
        (dr3, _), (fr3, _) = agg(sel, "d_r3"), agg(sel, "f_r3")
        (fr5, _) = agg(sel, "f_r5")
        print(f"{name:10s}{n:>4d}{dr1:>10.4f}{fr1:>9.4f}"
              f"{dr3:>10.4f}{fr3:>9.4f}{fr5:>9.4f}")

    print("\nper-query moves (hybrid vs dense @1):")
    fixed = broken = 0
    for r in rows:
        if r["f_r1"] and not r["d_r1"]:
            fixed += 1
            print(f"  FIXED  {r['id']}: {r['dense_top1']} -> {r['fused_top1']}")
        elif r["d_r1"] and not r["f_r1"]:
            broken += 1
            print(f"  BROKEN {r['id']}: {r['dense_top1']} -> {r['fused_top1']}")
    print(f"\nfixed={fixed} broken={broken}")
    Path("hybrid_test.json").write_text(json.dumps(
        {"k1": K1, "b": B, "fuse_depth": FUSE_DEPTH, "rrf_k": RRF_K,
         "fixed": fixed, "broken": broken, "rows": rows}, indent=2))
    print("wrote hybrid_test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
