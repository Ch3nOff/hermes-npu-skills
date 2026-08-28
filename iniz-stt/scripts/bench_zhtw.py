"""
bench_zhtw.py — Whisper on Traditional-Chinese audio, NPU vs CPU.

Reports THREE accuracy numbers per device, because for Chinese a single number is
misleading:

  CER_raw        — character error rate against the reference as-is
  CER_trad       — reference and hypothesis both forced to Traditional (opencc s2twp),
                   so a correct transcription written in Simplified is not punished
  script verdict — what orthography Whisper actually emitted

Whisper's zh training data is overwhelmingly Simplified, so it commonly transcribes
Traditional audio correctly but writes Simplified characters. CER_raw counts that as
a wall of errors; CER_trad separates "wrong words" from "wrong orthography".

CER (not WER) is the right metric here: Chinese has no whitespace word boundaries,
so splitting on spaces would produce one giant "word".
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from script_check import script_verdict, to_traditional

AUDIO = Path("audio_zhtw")
PUNCT = "，。！？、；：「」『』（）《》〈〉…—～·,.!?;:\"'()[]{}<>-– "


def norm_chars(s):
    """Strip punctuation and whitespace, keep characters. Returns a list of chars."""
    s = re.sub(r"\s+", "", s)
    return [c for c in s if c not in PUNCT]


def cer(ref, hyp):
    """Levenshtein at character level. Returns (errors, ref_len)."""
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


def load_clips():
    import soundfile as sf
    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for m in manifest:
        arr, sr = sf.read(AUDIO / m["file"], dtype="float32")
        assert sr == 16000, f"{m['file']} is {sr} Hz"
        clips.append((m, arr))
    return clips


def run_device(model, device, clips, language):
    import openvino_genai as ov_genai

    print(f"\n=== {device} (language={language}) ===", flush=True)
    t0 = time.time()
    try:
        pipe = ov_genai.WhisperPipeline(str(model), device)
    except Exception as e:
        print(f"COMPILE FAILED: {type(e).__name__}: {e}")
        return {"device": device, "compile_ok": False, "error": str(e)}
    compile_s = time.time() - t0
    print(f"compile: {compile_s:.2f}s", flush=True)

    short = min(clips, key=lambda c: len(c[1]))
    pipe.generate(short[1].tolist(), max_new_tokens=8)

    lat, rtf, hyps = [], [], []
    e_raw = n_raw = e_trad = n_trad = 0
    for m, arr in clips:
        t0 = time.time()
        res = pipe.generate(arr.tolist(), language=language, task="transcribe")
        el = time.time() - t0
        hyp = str(res).strip()
        hyps.append(hyp)
        lat.append(el * 1000)
        rtf.append(el / m["duration_s"])

        a, b = cer(m["reference"], hyp)
        e_raw += a
        n_raw += b
        c, d = cer(to_traditional(m["reference"]), to_traditional(hyp))
        e_trad += c
        n_trad += d

        print(f"  {m['file']}  {m['duration_s']:5.2f}s  {el*1000:7.0f}ms  "
              f"CER_raw={a}/{b}  CER_trad={c}/{d}", flush=True)
        print(f"     ref: {m['reference']}", flush=True)
        print(f"     hyp: {hyp}", flush=True)

    sv = script_verdict(hyps)
    out = {
        "device": device, "compile_ok": True, "compile_s": round(compile_s, 2),
        "language": language, "clips": len(clips),
        "latency_p50_ms": round(statistics.median(lat), 1),
        "latency_p90_ms": round(sorted(lat)[max(0, int(0.9 * len(lat)) - 1)], 1),
        "rtf_median": round(statistics.median(rtf), 4),
        "cer_raw": round(e_raw / n_raw, 4) if n_raw else None,
        "cer_raw_errors": e_raw, "cer_raw_chars": n_raw,
        "cer_trad": round(e_trad / n_trad, 4) if n_trad else None,
        "cer_trad_errors": e_trad, "cer_trad_chars": n_trad,
        "hyp_script": sv,
        "audio_seconds_total": round(sum(m["duration_s"] for m, _ in clips), 2),
        "transcripts": hyps,
    }
    print(f"  -> p50={out['latency_p50_ms']}ms  RTF={out['rtf_median']}  "
          f"CER_raw={out['cer_raw']}  CER_trad={out['cer_trad']}", flush=True)
    print(f"  -> Whisper emitted: {sv['verdict']} "
          f"(simp={sv['simp']} trad={sv['trad']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/whisper-base-int8-ov")
    ap.add_argument("--devices", default="NPU,CPU")
    ap.add_argument("--language", default="<|zh|>")
    ap.add_argument("--out", default="whisper_bench_zhtw.json")
    args = ap.parse_args()

    clips = load_clips()
    refs = [m["reference"] for m, _ in clips]
    rv = script_verdict(refs)
    print(f"clips: {len(clips)}, audio {sum(m['duration_s'] for m,_ in clips):.1f}s")
    print(f"reference script: {rv['verdict']} (simp={rv['simp']} trad={rv['trad']})")
    if rv["verdict"] != "TRADITIONAL":
        print("ABORT: references are not Traditional")
        return 1

    results = [run_device(args.model, d.strip(), clips, args.language)
               for d in args.devices.split(",")]
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")

    ok = [r for r in results if r.get("compile_ok")]
    print(f"\n{'device':8s}{'p50 ms':>9s}{'RTF':>8s}{'CER_raw':>10s}{'CER_trad':>10s}"
          f"{'emitted':>14s}")
    for r in ok:
        print(f"{r['device']:8s}{r['latency_p50_ms']:>9.1f}{r['rtf_median']:>8.3f}"
              f"{r['cer_raw']:>10.4f}{r['cer_trad']:>10.4f}"
              f"{r['hyp_script']['verdict']:>14s}")
    if len(ok) > 1:
        same = sum(1 for a, b in zip(ok[0]["transcripts"], ok[1]["transcripts"])
                   if a == b)
        print(f"\ntranscript agreement {ok[0]['device']} vs {ok[1]['device']}: "
              f"{same}/{len(clips)} byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
