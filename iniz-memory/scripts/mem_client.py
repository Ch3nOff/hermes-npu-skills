"""
mem_client.py — end-to-end smoke test for mem_server.py over real HTTP.

Replays all 39 hand-written queries through the server and recomputes recall@1/3/5
from the HTTP responses. r@3/r@5 are device-stable (33/39 and 36/39 on both NPU
and CPU); r@1 ties flip on device numerics (22 NPU vs 23 CPU), so r@1 is accepted
in {22, 23} rather than exactly. Error handling: empty body, missing query, bad
top_k, unknown endpoint.
"""

import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8012"
WORK = Path(__file__).resolve().parent
# queries.json lives in corpus/ next to scripts/ (repo layout) or in corpus/
# next to this file (dev work/ layout) — same resolution story as the server.
_qc = [WORK / "corpus" / "queries.json", WORK.parent / "corpus" / "queries.json",
        Path(os.environ["INIZ_MEM_QUERIES"]) if os.environ.get("INIZ_MEM_QUERIES") else None]
QUERIES = next((p for p in _qc if p is not None and p.exists()), None)
if QUERIES is None:
    raise FileNotFoundError("queries.json not found — set INIZ_MEM_QUERIES.")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    print("=== /health ===")
    h = get("/health")
    for k, v in h.items():
        print(f"  {k}: {v}")

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    print(f"\n=== POST /search ({len(queries)} queries from {QUERIES}) ===")
    r1 = r3 = r5 = 0
    lat = []
    for q in queries:
        d = post("/search", {"query": q["text"], "top_k": 5})
        got = [x["id"] for x in d["hits"]]
        a = any(g in got[:1] for g in q["gold"])
        b = any(g in got[:3] for g in q["gold"])
        c = any(g in got for g in q["gold"])
        r1 += a
        r3 += b
        r5 += c
        lat.append(d["_ms"])
        mark = "" if b else "  <-- MISS@3"
        print(f"  {q['id']} [{q['lang']}] r@1={a:.0f} r@3={b:.0f} "
              f"top1={got[0]} {d['_ms']:6.1f}ms{mark}")

    n = len(queries)
    print(f"\n  served recall: r@1={r1/n:.4f} r@3={r3/n:.4f} r@5={r5/n:.4f} "
          f"(offline e5-base: r@3=33/39 r@5=36/39 both devices; r@1 22-23)")
    print(f"  server latency p50={statistics.median(lat):.1f}ms")
    match = (r3, r5) == (33, 36) and r1 in (22, 23)
    print(f"  matches offline bench: {match}")

    print("\n=== error handling ===")
    checks = []
    for name, fn in [
        ("empty query", lambda: post("/search", {"query": "  "})),
        ("missing query", lambda: post("/search", {"foo": 1})),
        ("bad top_k", lambda: post("/search", {"query": "x", "top_k": "many"})),
    ]:
        try:
            fn()
            print(f"  {name:14s} -> UNEXPECTED 200")
            checks.append(False)
        except urllib.error.HTTPError as ex:
            print(f"  {name:14s} -> HTTP {ex.code} "
                  f"{json.loads(ex.read()).get('error', '')[:44]}")
            checks.append(True)
    try:
        get("/nope")
        checks.append(False)
    except urllib.error.HTTPError as ex:
        print(f"  {'unknown GET':14s} -> HTTP {ex.code}")
        checks.append(True)

    h2 = get("/health")
    print(f"\n=== /health after load ===")
    print(f"  requests_served={h2['requests_served']} "
          f"latency_p50_ms={h2['latency_p50_ms']}")

    ok = all(checks) and match and h2["requests_served"] == n
    print(f"\nVERDICT: {'PASS' if ok else 'CHECK NEEDED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
