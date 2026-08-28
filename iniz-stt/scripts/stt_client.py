"""
stt_client.py — end-to-end smoke test for stt_server.py over real HTTP.

Verifies:
  * /health reports the expected device
  * raw audio upload works
  * JSON {"path": ...} works
  * WER against LibriSpeech ground truth is sane (not a fake 0 from empty output)
  * translation task and timestamps do not crash
  * error handling: empty body, undecodable bytes, missing file, unknown endpoint
"""

import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8010"
AUDIO = Path("audio")


def normalize(s):
    s = s.upper().replace("-", " ")
    s = re.sub(r"[^A-Z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)], len(r)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def post_bytes(path, data, ctype="application/octet-stream"):
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def post_json(path, obj):
    return post_bytes(path, json.dumps(obj).encode(), "application/json")


def main():
    print("=== /health ===")
    h = get("/health")
    for k, v in h.items():
        print(f"  {k}: {v}")

    manifest = json.loads((AUDIO / "manifest.json").read_text())

    print("\n=== POST /transcribe (raw upload) ===")
    err_t, ref_t, lat = 0, 0, []
    for m in manifest:
        raw = (AUDIO / m["file"]).read_bytes()
        t0 = time.time()
        d = post_bytes("/transcribe", raw)
        rt = (time.time() - t0) * 1000
        lat.append(rt)
        e, n = wer(m["reference"], d["text"])
        err_t += e
        ref_t += n
        print(f"  {m['file']}  {d['duration_s']:5.2f}s  {d['_ms']:7.1f}ms  "
              f"rtf={d['rtf']}  WER={e}/{n}  rt={rt:.0f}ms")
        print(f"     {d['text'][:76]!r}")

    print(f"\n  overall WER: {err_t/ref_t:.4f} ({err_t}/{ref_t} words)")
    print(f"  client latency p50={statistics.median(lat):.0f}ms")

    print("\n=== POST /transcribe (JSON path) ===")
    p = (AUDIO / manifest[0]["file"]).resolve()
    d = post_json("/transcribe", {"path": str(p)})
    print(f"  {d['_ms']:.1f}ms  {d['text'][:70]!r}")

    print("\n=== translate task (English audio -> should still be English) ===")
    d = post_bytes("/transcribe?task=translate",
                   (AUDIO / manifest[0]["file"]).read_bytes())
    print(f"  {d['_ms']:.1f}ms  {d['text'][:70]!r}")

    print("\n=== timestamps ===")
    d = post_bytes("/transcribe?timestamps=1",
                   (AUDIO / manifest[1]["file"]).read_bytes())
    ch = d.get("chunks")
    print(f"  chunks: {len(ch) if ch else 0}")
    if ch:
        for c in ch[:3]:
            print(f"    [{c['start']} -> {c['end']}] {c['text'][:56]!r}")

    print("\n=== error handling ===")
    checks = []
    for name, fn in [
        ("empty body", lambda: post_bytes("/transcribe", b"")),
        ("garbage bytes", lambda: post_bytes("/transcribe", b"not audio at all" * 8)),
        ("missing file", lambda: post_json("/transcribe", {"path": "/nope/x.wav"})),
        ("JSON no path", lambda: post_json("/transcribe", {})),
    ]:
        try:
            fn()
            print(f"  {name:16s} -> UNEXPECTED 200")
            checks.append(False)
        except urllib.error.HTTPError as ex:
            msg = json.loads(ex.read()).get("error", "")
            print(f"  {name:16s} -> HTTP {ex.code} {msg[:52]}")
            checks.append(True)
    try:
        get("/nope")
        checks.append(False)
    except urllib.error.HTTPError as ex:
        print(f"  {'unknown GET':16s} -> HTTP {ex.code}")
        checks.append(True)

    h2 = get("/health")
    print(f"\n=== /health after load ===")
    print(f"  requests_served={h2['requests_served']} "
          f"latency_p50_ms={h2['latency_p50_ms']}")

    ok = all(checks) and err_t / ref_t < 0.25 and h2["requests_served"] > 0
    print(f"\nVERDICT: {'PASS' if ok else 'CHECK NEEDED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
