"""
test_prompt_fair.py — honest version of the initial_prompt test.

The first run put the clips' own target words in `hotwords`
(開源服務 松楓橋 認養 華頓 議題 結構) and scored CER 0.1486. That leaks the answer:
of course accuracy improves when you hand the model the words it is being scored on.
That number must not be reported as a general result.

This version tests conditions that carry NO information about the clip contents:

  A. baseline
  B. orthography-only prompt   (a Traditional sentence about transcription style)
  C. generic Traditional hotwords (common words absent from all 10 references)
  D. domain-mismatched hotwords (deliberately wrong topic — control for
     "any hotwords help" vs "these hotwords help")

Only B and C are legitimate deployment settings. D exists to check that gains are
not an artifact of merely passing a non-empty string.

Runs on CPU: NPU rejects both parameters with
'Check *roi_end <= *max_dim failed' (static-shape decoder cannot take a prefix).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from script_check import script_verdict, to_traditional

AUDIO = Path("audio_zhtw")
MODEL = os.environ.get("INIZ_TEST_MODEL", "../models/whisper-small-int8-ov")
DEVICE = os.environ.get("INIZ_TEST_DEVICE", "CPU")
PUNCT = "，。！？、；：「」『』（）《》〈〉…—～·,.!?;:\"'()[]{}<>-– "

STYLE_PROMPT = "以下是繁體中文的逐字稿，請使用臺灣正體字書寫。"
# Common Traditional words, none of which appear in any of the 10 references.
GENERIC_HOTWORDS = "臺灣 繁體 環境 經濟 團隊 應該 覺得 電腦 網路 醫院"
# Deliberately wrong domain — control condition.
MISMATCH_HOTWORDS = "恐龍 火山 潛水艇 微積分 交響樂 冰川 疫苗 隕石"


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
            c = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
        prev = cur
    return prev[len(h)], len(r)


def leaks(hotwords, refs):
    """True if any hotword token appears in any reference — an unfair test."""
    for w in hotwords.split():
        for r in refs:
            if w and w in r:
                return w
    return None


def main():
    import soundfile as sf
    import openvino_genai as ov_genai

    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    refs = [m["reference"] for m in manifest]
    clips = []
    for m in manifest:
        arr, _ = sf.read(AUDIO / m["file"], dtype="float32")
        clips.append((m, arr.tolist()))

    conditions = [
        ("A_baseline", {}),
        ("B_style_prompt", {"initial_prompt": STYLE_PROMPT}),
        ("C_generic_hotwords", {"hotwords": GENERIC_HOTWORDS}),
        ("D_mismatch_hotwords", {"hotwords": MISMATCH_HOTWORDS}),
    ]

    # fail loudly if a "fair" condition is actually leaking
    for name, extra in conditions:
        hw = extra.get("hotwords")
        if hw:
            bad = leaks(hw, refs)
            if bad:
                print(f"ABORT: {name} hotwords leak reference word {bad!r}")
                return 1
    print("leak check: no condition contains any reference word", flush=True)

    print(f"model={MODEL} device={DEVICE} clips={len(clips)}", flush=True)
    pipe = ov_genai.WhisperPipeline(MODEL, DEVICE)
    pipe.generate(clips[0][1], max_new_tokens=4)

    results = {}
    for name, extra in conditions:
        print(f"\n=== {name} ===", flush=True)
        e_raw = n_raw = e_tr = n_tr = 0
        hyps, lat = [], []
        err = None
        for m, audio in clips:
            try:
                t0 = time.time()
                res = pipe.generate(audio, language="<|zh|>", task="transcribe",
                                    **extra)
                lat.append((time.time() - t0) * 1000)
            except Exception as ex:
                err = f"{type(ex).__name__}: {str(ex)[:90]}"
                break
            hyp = str(res).strip()
            hyps.append(hyp)
            a, b = cer(m["reference"], hyp)
            c, d = cer(to_traditional(m["reference"]), to_traditional(hyp))
            e_raw += a
            n_raw += b
            e_tr += c
            n_tr += d
            print(f"  {m['file']}  CER_raw={a}/{b}   {hyp}", flush=True)

        if err:
            print(f"  UNSUPPORTED: {err}", flush=True)
            results[name] = {"supported": False, "error": err}
            continue

        sv = script_verdict(hyps)
        results[name] = {
            "supported": True, "cer_raw": round(e_raw / n_raw, 4),
            "cer_trad": round(e_tr / n_tr, 4),
            "simp": sv["simp"], "trad": sv["trad"], "script": sv["verdict"],
            "p50_ms": round(sorted(lat)[len(lat) // 2], 1), "transcripts": hyps,
        }
        r = results[name]
        print(f"  -> CER_raw={r['cer_raw']} CER_trad={r['cer_trad']} "
              f"script={r['script']} (s={r['simp']} t={r['trad']}) "
              f"p50={r['p50_ms']}ms", flush=True)

    tag = f"{DEVICE}_{Path(MODEL).name}"
    Path(f"prompt_fair_{tag}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'condition':22s}{'CER_raw':>9s}{'CER_trad':>10s}{'simp':>6s}"
          f"{'trad':>6s}{'p50 ms':>9s}")
    for name, r in results.items():
        if r.get("supported"):
            print(f"{name:22s}{r['cer_raw']:>9.4f}{r['cer_trad']:>10.4f}"
                  f"{r['simp']:>6d}{r['trad']:>6d}{r['p50_ms']:>9.1f}")
        else:
            print(f"{name:22s} UNSUPPORTED")
    print(f"\nwrote prompt_fair_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
