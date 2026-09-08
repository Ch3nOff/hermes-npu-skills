"""
mem_client.py — end-to-end smoke test for mem_server.py over real HTTP.

Replays all 26 hand-written queries through the server and recomputes recall@1/3/5
from the HTTP responses, so the served numbers are checked against the offline
benchmark (r@1=0.615, r@3=0.808, r@5=0.962) rather than taken on faith. Also
exercises error handling: empty body, missing query, bad top_k, unknown endpoint.
"""

import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8012"
WORK = Path(__file__).resolve().parent


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

    queries = json.loads((WORK / "corpus" / "queries.json").read_text(
        encoding="utf-8"))
    print(f"\n=== POST /search ({len(queries)} queries) ===")
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
          f"(offline: 0.6154 / 0.8077 / 0.9615)")
    print(f"  server latency p50={statistics.median(lat):.1f}ms")
    match = (r1, r3, r5) == (16, 21, 25)
    print(f"  matches offline bench exactly: {match}")

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
