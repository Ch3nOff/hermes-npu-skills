"""
stt_client_zhtw.py — HTTP smoke test with Traditional-Chinese audio.

Confirms the server handles zh: language pinning via query string, CER against
Traditional references, and what orthography comes back over the wire.
"""

import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_check import script_verdict, to_traditional

BASE = "http://127.0.0.1:8010"
AUDIO = Path("audio_zhtw")
PUNCT = "，。！？、；：「」『』（）《》〈〉…—～·,.!?;:\"'()[]{}<>-– "


def norm_chars(s):
    return [c for c in re.sub(r"\s+", "", s) if c not in PUNCT]


def cer(ref, hyp):
    r, h = norm_chars(ref), norm_chars(hyp)
    if not r:
        return 0, 0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(h)], len(r)


def post(path, data, ctype="application/octet-stream"):
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    with urllib.request.urlopen(BASE + "/health", timeout=60) as r:
        h = json.loads(r.read())
    print("=== /health ===")
    for k, v in h.items():
        print(f"  {k}: {v}")

    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    print(f"\n=== POST /transcribe?language=<|zh|> ({len(manifest)} clips) ===")

    e_raw = n_raw = e_tr = n_tr = 0
    lat, hyps = [], []
    for m in manifest:
        raw = (AUDIO / m["file"]).read_bytes()
        d = post("/transcribe?language=%3C%7Czh%7C%3E", raw)
        hyp = d["text"]
        hyps.append(hyp)
        lat.append(d["_ms"])
        a, b = cer(m["reference"], hyp)
        c, dd = cer(to_traditional(m["reference"]), to_traditional(hyp))
        e_raw += a
        n_raw += b
        e_tr += c
        n_tr += dd
        print(f"  {m['file']}  {d['duration_s']:5.2f}s  {d['_ms']:7.1f}ms  "
              f"rtf={d['rtf']}  CER_raw={a}/{b}  CER_trad={c}/{dd}")
        print(f"     ref: {m['reference']}")
        print(f"     hyp: {hyp}")

    sv = script_verdict(hyps)
    print(f"\n  CER_raw  = {e_raw/n_raw:.4f} ({e_raw}/{n_raw} chars)")
    print(f"  CER_trad = {e_tr/n_tr:.4f} ({e_tr}/{n_tr} chars)")
    print(f"  server latency p50 = {statistics.median(lat):.1f}ms")
    print(f"  emitted script: {sv['verdict']} (simp={sv['simp']} trad={sv['trad']})")

    ok = e_tr / n_tr < 0.35
    print(f"\nVERDICT: {'PASS' if ok else 'CHECK NEEDED'} "
          f"(threshold CER_trad < 0.35)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
