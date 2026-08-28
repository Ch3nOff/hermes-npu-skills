"""
fetch_audio_zhtw.py — Traditional-Chinese (Taiwanese Mandarin) test audio.

Source: JacobLinCool/common_voice_16_1_zh_TW_clean (CC0-1.0), Common Voice 16.1
zh-TW filtered for clean recordings. Chosen after measuring script usage: this
dataset's references are Traditional (simp=0 / trad=99 on a 100-row sample), while
CJY/Chinese-Dialogue-180k is Simplified (980 vs 4) and has no ASR ground truth.

Downloads ONE test shard (~341 MB) rather than the full 3.1 GB, then picks clips
long enough to be meaningful (>= 2.0 s) so accuracy is not dominated by two-word
utterances.

Output: audio_zhtw/*.wav (16 kHz mono float32) + audio_zhtw/manifest.json
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/JacobLinCool/"
       "common_voice_16_1_zh_TW_clean/resolve/main/data/test-00000-of-00003.parquet")
OUT = Path("audio_zhtw")
TARGET_SR = 16000
N_SAMPLES = 10
MIN_DUR = 2.0


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    OUT.mkdir(exist_ok=True)
    pq_path = OUT / "_cv_zhtw_test.parquet"
    if not pq_path.exists():
        print(f"downloading {URL.split('/')[-1]} (~341 MB) ...", flush=True)
        urllib.request.urlretrieve(URL, pq_path)
    print(f"parquet: {pq_path.stat().st_size / 1e6:.1f} MB", flush=True)

    pf = pq.ParquetFile(pq_path)
    print(f"row groups: {pf.num_row_groups}, rows: {pf.metadata.num_rows}", flush=True)
    print(f"columns: {pf.schema_arrow.names}", flush=True)

    # read only the first row group — enough for 10 clips, avoids loading 341 MB
    table = pf.read_row_group(0, columns=["audio", "sentence"])
    rows = table.to_pylist()
    print(f"row group 0: {len(rows)} rows", flush=True)

    manifest = []
    for row in rows:
        if len(manifest) >= N_SAMPLES:
            break
        audio = row["audio"]
        raw = audio["bytes"] if isinstance(audio, dict) else audio
        if not raw:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception as e:
            print(f"  skip (decode): {type(e).__name__}", flush=True)
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        dur = len(arr) / sr
        if dur < MIN_DUR:
            continue

        i = len(manifest)
        name = f"zhtw_{i:02d}.wav"
        sf.write(OUT / name, arr, sr)
        ref = row["sentence"]
        manifest.append({
            "file": name, "duration_s": round(dur, 3), "sample_rate": sr,
            "reference": ref, "id": f"cv_zhtw_{i}",
        })
        print(f"  {name}  {dur:5.2f}s  {ref}", flush=True)

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(m["duration_s"] for m in manifest)
    print(f"\nwrote {len(manifest)} files, {total:.1f}s total -> {OUT}/manifest.json")

    # measured script check — never assume the dataset card
    from script_check import script_verdict
    v = script_verdict([m["reference"] for m in manifest])
    print(f"reference script: simp={v['simp']} trad={v['trad']} -> {v['verdict']}")
    if v["verdict"] != "TRADITIONAL":
        print("WARNING: references are not Traditional — do not use for this test")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
