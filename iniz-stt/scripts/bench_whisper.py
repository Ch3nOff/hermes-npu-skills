"""
bench_whisper.py — measure Whisper on NPU vs CPU vs iGPU with REAL audio.

Reports, per device:
  * whether it compiles at all, and how long
  * p50/p90 latency per clip
  * RTF (real-time factor) = processing time / audio duration; lower is better
  * WER against LibriSpeech ground truth (Whisper-style text normalization)
  * whether transcripts agree across devices

WER is computed on normalized text (uppercase, punctuation stripped) because
LibriSpeech references have no punctuation while Whisper emits it — comparing raw
strings would report a fake ~80% WER.
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

AUDIO = Path("audio")


def normalize(s: str) -> str:
    s = s.upper()
    s = s.replace("-", " ")
    s = re.sub(r"[^A-Z' ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer(ref: str, hyp: str):
    """Levenshtein on word level. Returns (errors, ref_len)."""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)], len(r)


def load_clips():
    manifest = json.loads((AUDIO / "manifest.json").read_text())
    import soundfile as sf
    clips = []
    for m in manifest:
        arr, sr = sf.read(AUDIO / m["file"], dtype="float32")
        assert sr == 16000, f"{m['file']} is {sr} Hz, expected 16000"
        clips.append((m, arr))
    return clips


def run_device(model_path, device, clips, extra=None):
    import openvino_genai as ov_genai

    print(f"\n=== {device} ===", flush=True)
    t0 = time.time()
    try:
        kwargs = dict(extra or {})
        pipe = ov_genai.WhisperPipeline(str(model_path), device, **kwargs)
    except Exception as e:
        print(f"COMPILE FAILED: {type(e).__name__}: {e}")
        return {"device": device, "compile_ok": False,
                "error": f"{type(e).__name__}: {e}"}
    compile_s = time.time() - t0
    print(f"compile/load: {compile_s:.2f}s")

    # warmup on the shortest clip so the first timed clip is not paying init cost
    short = min(clips, key=lambda c: len(c[1]))
    pipe.generate(short[1].tolist(), max_new_tokens=8)

    lat, rtf, texts = [], [], []
    err_total, ref_total = 0, 0
    for m, arr in clips:
        t0 = time.time()
        res = pipe.generate(arr.tolist())
        el = time.time() - t0
        text = str(res)
        lat.append(el * 1000)
        rtf.append(el / m["duration_s"])
        texts.append(text)
        e, n = wer(m["reference"], text)
        err_total += e
        ref_total += n
        print(f"  {m['file']}  {m['duration_s']:5.2f}s audio  {el*1000:7.0f}ms  "
              f"RTF={el/m['duration_s']:.3f}  WER={e}/{n}")
        print(f"     hyp: {text.strip()[:78]!r}")

    out = {
        "device": device, "compile_ok": True, "compile_s": round(compile_s, 2),
        "clips": len(clips),
        "latency_p50_ms": round(statistics.median(lat), 1),
        "latency_p90_ms": round(sorted(lat)[max(0, int(0.9 * len(lat)) - 1)], 1),
        "rtf_median": round(statistics.median(rtf), 4),
        "rtf_mean": round(sum(rtf) / len(rtf), 4),
        "wer": round(err_total / ref_total, 4),
        "wer_errors": err_total, "wer_ref_words": ref_total,
        "audio_seconds_total": round(sum(m["duration_s"] for m, _ in clips), 2),
        "transcripts": [t.strip() for t in texts],
    }
    print(f"  -> p50={out['latency_p50_ms']}ms  RTF_median={out['rtf_median']}  "
          f"WER={out['wer']:.4f} ({err_total}/{ref_total})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/whisper-base-int8-ov")
    ap.add_argument("--devices", default="NPU,CPU,GPU.0")
    ap.add_argument("--out", default="whisper_bench.json")
    args = ap.parse_args()

    import openvino as ov
    print("available devices:", ov.Core().available_devices)

    clips = load_clips()
    total = sum(m["duration_s"] for m, _ in clips)
    print(f"clips: {len(clips)}, total audio {total:.1f}s")

    results = []
    for dev in args.devices.split(","):
        dev = dev.strip()
        results.append(run_device(args.model, dev, clips))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

    ok = [r for r in results if r.get("compile_ok")]
    print(f"\n{'device':8s}{'compile':>9s}{'p50 ms':>9s}{'RTF':>8s}{'WER':>8s}")
    for r in ok:
        print(f"{r['device']:8s}{r['compile_s']:>9.2f}{r['latency_p50_ms']:>9.1f}"
              f"{r['rtf_median']:>8.3f}{r['wer']:>8.4f}")
    for r in results:
        if not r.get("compile_ok"):
            print(f"{r['device']:8s} FAILED: {r['error'][:70]}")

    # cross-device transcript agreement
    if len(ok) > 1:
        base = ok[0]
        print(f"\ntranscript agreement vs {base['device']}:")
        for r in ok[1:]:
            same = sum(1 for a, b in zip(base["transcripts"], r["transcripts"])
                       if normalize(a) == normalize(b))
            print(f"  {r['device']:8s} {same}/{len(clips)} identical (normalized)")


if __name__ == "__main__":
    main()
