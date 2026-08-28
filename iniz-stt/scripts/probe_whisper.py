"""
probe_whisper.py — find out WHERE Whisper-on-NPU is slow or hanging.

The full benchmark ran >7 minutes with no output (piped through tail, so buffered).
This probes one stage at a time with unbuffered prints and a hard per-stage timer,
so the actual bottleneck is visible instead of guessed.

Stages:
  1. import openvino_genai
  2. construct WhisperPipeline on the given device  (suspect #1: NPU compile)
  3. generate on 1s of silence                      (suspect #2: first inference)
  4. generate on the shortest real clip
"""

import json
import sys
import time
from pathlib import Path

AUDIO = Path("audio")


def stage(name, fn):
    print(f"[{time.strftime('%H:%M:%S')}] START {name}", flush=True)
    t0 = time.time()
    try:
        r = fn()
        el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] DONE  {name}  {el:.2f}s", flush=True)
        return r, el
    except Exception as e:
        el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] FAIL  {name}  {el:.2f}s  "
              f"{type(e).__name__}: {e}", flush=True)
        raise


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    model = Path(sys.argv[2] if len(sys.argv) > 2 else "../models/whisper-base-int8-ov")
    print(f"device={device} model={model}", flush=True)

    import soundfile as sf
    ov_genai, _ = stage("import openvino_genai", lambda: __import__("openvino_genai"))

    pipe, compile_s = stage(f"WhisperPipeline({device})",
                            lambda: ov_genai.WhisperPipeline(str(model), device))

    _, warm_s = stage("generate 1s silence",
                      lambda: pipe.generate([0.0] * 16000, max_new_tokens=4))

    manifest = json.loads((AUDIO / "manifest.json").read_text())
    short = min(manifest, key=lambda m: m["duration_s"])
    arr, sr = sf.read(AUDIO / short["file"], dtype="float32")
    print(f"shortest clip: {short['file']} {short['duration_s']}s "
          f"({len(arr)} samples @ {sr}Hz)", flush=True)

    lst, tolist_s = stage("arr.tolist()", lambda: arr.tolist())
    res, gen_s = stage(f"generate {short['duration_s']}s clip",
                       lambda: pipe.generate(lst))
    print(f"\ntext: {str(res).strip()[:90]!r}", flush=True)
    print(f"RTF: {gen_s / short['duration_s']:.3f}", flush=True)

    print(f"\n=== timing summary ({device}) ===")
    print(f"  pipeline construct : {compile_s:7.2f}s")
    print(f"  warmup (1s silence): {warm_s:7.2f}s")
    print(f"  tolist()           : {tolist_s:7.2f}s")
    print(f"  generate clip      : {gen_s:7.2f}s  (RTF {gen_s/short['duration_s']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
