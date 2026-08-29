"""
aux_client.py — end-to-end HTTP smoke test for aux_server.py.

Checks all three endpoints against the same ground truth bench_aux.py uses, so the
server's numbers can be compared to the offline benchmark rather than taken on faith.

Also exercises error handling: empty body, missing field, oversized text, missing
question on /extract, unknown endpoint.
"""

import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8011"
DATA = Path("aux_data")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.loads(r.read())


def post(path, obj):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    print("=== /health ===")
    h = get("/health")
    for k, v in h.items():
        print(f"  {k}: {v}")

    print("\n=== POST /summarize (3 CNN articles) ===")
    arts = json.loads((DATA / "summarize.json").read_text(encoding="utf-8"))[:3]
    lat = []
    for a in arts:
        d = post("/summarize", {"text": a["article"]})
        lat.append(d["_ms"])
        print(f"  {a['id']}  {d['_ms']:8.1f}ms")
        print(f"     {d['summary'][:110]}")

    print("\n=== POST /sentiment (12 SST-2, balanced) ===")
    sents = json.loads((DATA / "sentiment.json").read_text(encoding="utf-8"))
    sub = sents[:6] + sents[-6:]
    ok = 0
    slat = []
    for s in sub:
        d = post("/sentiment", {"text": s["text"]})
        ok += d["sentiment"] == s["label"]
        slat.append(d["_ms"])
        mark = "" if d["sentiment"] == s["label"] else "  <-- wrong"
        print(f"  {s['id']}  gold={s['label']:8s} got={d['sentiment']:9s} "
              f"{d['_ms']:7.1f}ms{mark}")
    print(f"  accuracy {ok}/{len(sub)}")

    print("\n=== POST /extract (all 10) ===")
    exts = json.loads((DATA / "extract.json").read_text(encoding="utf-8"))
    exact = grounded_wrong = 0
    elat = []
    for it in exts:
        d = post("/extract", {"text": it["text"], "question": it["question"]})
        elat.append(d["_ms"])
        hit = d["answer"] == it["answer"]
        exact += hit
        if not hit and d["grounded"]:
            grounded_wrong += 1
        tag = "exact" if hit else ("grounded-but-wrong" if d["grounded"]
                                   else "not-grounded")
        print(f"  {it['id']}  {tag:19s} gold={it['answer']!r} got={d['answer']!r}")
    print(f"  exact {exact}/{len(exts)}, grounded-but-wrong {grounded_wrong}")

    print("\n=== error handling ===")
    checks = []
    cases = [
        ("empty body", lambda: post("/summarize", {})),
        ("missing text", lambda: post("/sentiment", {"foo": "bar"})),
        ("oversized text", lambda: post("/summarize", {"text": "x" * 20000})),
        ("extract no question", lambda: post("/extract", {"text": "a file a.py"})),
    ]
    for name, fn in cases:
        try:
            fn()
            print(f"  {name:20s} -> UNEXPECTED 200")
            checks.append(False)
        except urllib.error.HTTPError as ex:
            msg = json.loads(ex.read()).get("error", "")
            print(f"  {name:20s} -> HTTP {ex.code} {msg[:48]}")
            checks.append(True)
    try:
        get("/nope")
        checks.append(False)
    except urllib.error.HTTPError as ex:
        print(f"  {'unknown GET':20s} -> HTTP {ex.code}")
        checks.append(True)

    h2 = get("/health")
    print(f"\n=== /health after load ===")
    print(f"  requests_served={h2['requests_served']}")
    for t, v in h2.get("tasks", {}).items():
        print(f"  {t:10s} n={v['requests']:3d} p50={v['latency_p50_ms']}ms")

    passed = (all(checks) and ok >= len(sub) * 0.75 and exact >= 6
              and h2["requests_served"] > 0)
    print(f"\nVERDICT: {'PASS' if passed else 'CHECK NEEDED'}")
    print(f"  summarize p50 {statistics.median(lat):.1f}ms | "
          f"sentiment p50 {statistics.median(slat):.1f}ms | "
          f"extract p50 {statistics.median(elat):.1f}ms")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
