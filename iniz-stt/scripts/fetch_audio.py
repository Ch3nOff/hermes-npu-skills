"""
fetch_audio.py — download REAL speech audio with ground-truth transcripts.

Synthetic audio (sine waves, TTS) would not prove anything about WER. This pulls
LibriSpeech samples that ship with reference transcripts, so accuracy can actually
be measured instead of asserted.

Reads the parquet directly with pyarrow instead of datasets' audio decoder — the
`datasets` Audio feature now requires torchcodec, which pulls in a heavy dependency
we do not need just to read WAV bytes.

Output: work/audio/*.wav (16 kHz mono float32) + work/audio/manifest.json
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
       "resolve/main/clean/validation-00000-of-00001.parquet")
OUT = Path("audio")
TARGET_SR = 16000
N_SAMPLES = 8


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    OUT.mkdir(exist_ok=True)
    pq_path = OUT / "_librispeech.parquet"
    if not pq_path.exists():
        print(f"downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, pq_path)
    print(f"parquet: {pq_path.stat().st_size / 1e6:.1f} MB")

    table = pq.read_table(pq_path)
    print("columns:", table.column_names)
    rows = table.to_pylist()
    print(f"rows: {len(rows)}")

    manifest = []
    n = min(N_SAMPLES, len(rows))
    for i in range(n):
        row = rows[i]
        audio = row["audio"]
        # parquet stores {'bytes': <flac/wav>, 'path': ...}
        raw = audio["bytes"] if isinstance(audio, dict) else audio
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        name = f"ls_{i:02d}.wav"
        sf.write(OUT / name, arr, sr)
        dur = len(arr) / sr
        ref = row.get("text") or row.get("sentence") or ""
        manifest.append({
            "file": name,
            "duration_s": round(dur, 3),
            "sample_rate": sr,
            "reference": ref,
            "id": str(row.get("id", f"sample_{i}")),
        })
        print(f"  {name}  {dur:5.2f}s  {ref[:62]!r}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(m["duration_s"] for m in manifest)
    print(f"\nwrote {len(manifest)} files, {total:.1f}s total audio -> {OUT}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
