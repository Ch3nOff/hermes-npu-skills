"""
stt_server.py — Iniz STT (offline speech-to-text on Intel NPU)

Same shape as guard_server.py: a persistent HTTP process that loads a quantized
model once onto the NPU and serves many requests. Only the pipeline differs —
openvino_genai.WhisperPipeline instead of a raw CompiledModel.

Unlike the guard, this one IS generative (encoder-decoder), so WhisperPipeline is
the correct API here. Do not carry the guard's "never use a GenAI pipeline" rule
over to this file; that rule was about a discriminative checkpoint with no lm_head.

API:
  GET  /health
    -> status, device, model, script_mode, compile_s, warmup_ms, requests_served,
       latency_p50_ms
  POST /transcribe
    body: raw audio bytes (wav/flac/ogg — anything soundfile can read)
          or JSON {"path": "/abs/path.wav", "language": "<|en|>", "task": "transcribe"}
    query: ?language=<|en|>&task=translate&timestamps=1
    -> {"text", "language", "duration_s", "rtf", "_ms", "_device", "_model", "chunks"?}

Set INIZ_STT_SCRIPT=trad to force Chinese output to Traditional (opencc, char-level).
Whisper emits MIXED orthography for zh-TW audio and has no token to control it, so
this post-processing step is the only way to guarantee Traditional. The response then
carries _script_mode, _script_converted, and _text_raw when a conversion happened.

Security: binds to 127.0.0.1 with NO authentication. Do NOT expose on 0.0.0.0
without adding auth — audio is sensitive input.
"""

import io
import json
import os
import statistics
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "INIZ_STT_MODEL", str(HERE.parent / "models" / "whisper-base-int8-ov")))
DEVICE = os.environ.get("INIZ_STT_DEVICE", "NPU")
PORT = int(os.environ.get("INIZ_STT_PORT", "8010"))
TARGET_SR = 16000
MAX_UPLOAD = int(os.environ.get("INIZ_STT_MAX_BYTES", str(64 * 1024 * 1024)))
# 'trad' converts Chinese output to Traditional via char-level opencc; 'off' leaves it
# alone. Whisper emits MIXED orthography for zh-TW audio, so this is the only way
# to guarantee Traditional output — the model has no token for it.
SCRIPT_MODE = os.environ.get("INIZ_STT_SCRIPT", "off").lower()


def convert_script(text: str) -> tuple:
    """Return (converted_text, changed_bool). No-op unless SCRIPT_MODE == 'trad'.

    Uses CHARACTER-level opencc s2t, not phrase-level s2twp. s2twp rewrites text that
    is already Traditional: measured case 說明了 -> 說明瞭, which turned a perfect
    whisper-medium transcription into an error. Char-level s2t is idempotent on
    Traditional input, which matters because Whisper output is MIXED.
    """
    if SCRIPT_MODE != "trad" or not text:
        return text, False
    try:
        from opencc import OpenCC
    except ImportError:
        return text, False
    cc = OpenCC("s2t")
    out = []
    for ch in text:
        conv = cc.convert(ch)
        out.append(conv if len(conv) == 1 else ch)
    result = "".join(out)
    return result, result != text


def decode_audio(raw: bytes):
    """Decode arbitrary audio bytes -> (float32 mono @16kHz, duration_s)."""
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    # WhisperPipeline expects values near [-1, 1]
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    return arr, len(arr) / sr


class Engine:
    """Load the Whisper IR once, serve repeatedly. Lock-guarded (NPU is single-request)."""

    def __init__(self, model_dir: Path, device: str):
        import openvino as ov
        import openvino_genai as ov_genai

        if not (model_dir / "openvino_encoder_model.xml").exists():
            raise FileNotFoundError(f"Whisper IR not found in {model_dir}")

        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(
                f"device {device} not available. present: {core.available_devices}")

        print(f"[stt] compiling {model_dir.name} -> {device} ...", flush=True)
        t0 = time.time()
        self.pipe = ov_genai.WhisperPipeline(str(model_dir), device)
        self.compile_s = time.time() - t0
        self.device = device
        self.model_dir = model_dir
        self.lock = Lock()
        print(f"[stt] loaded in {self.compile_s:.2f}s")

        # warmup with 1s of silence so the first real request is not paying init cost
        t0 = time.time()
        with self.lock:
            self.pipe.generate([0.0] * TARGET_SR, max_new_tokens=4)
        self.warmup_ms = (time.time() - t0) * 1000
        print(f"[stt] warmup {self.warmup_ms:.1f}ms")
        self.latencies = []

    def transcribe(self, arr, duration_s, language=None, task=None, timestamps=False):
        cfg = {}
        if language:
            cfg["language"] = language
        if task:
            cfg["task"] = task
        if timestamps:
            cfg["return_timestamps"] = True

        t0 = time.time()
        with self.lock:
            res = self.pipe.generate(arr.tolist(), **cfg)
        ms = (time.time() - t0) * 1000

        self.latencies.append(ms)
        if len(self.latencies) > 500:
            self.latencies = self.latencies[-500:]

        raw_text = str(res).strip()
        text, converted = convert_script(raw_text)
        out = {
            "text": text,
            "duration_s": round(duration_s, 3),
            "rtf": round((ms / 1000) / duration_s, 4) if duration_s else None,
            "_ms": round(ms, 1),
            "_device": self.device,
            "_model": self.model_dir.name,
        }
        if SCRIPT_MODE == "trad":
            out["_script_mode"] = "trad"
            out["_script_converted"] = converted
            if converted:
                out["_text_raw"] = raw_text
        # language / chunks are only present on some builds — never assume
        lang = getattr(res, "language", None)
        if lang:
            out["language"] = lang
        chunks = getattr(res, "chunks", None)
        if timestamps and chunks:
            out["chunks"] = [
                {"start": getattr(c, "start_ts", None),
                 "end": getattr(c, "end_ts", None),
                 "text": convert_script(str(getattr(c, "text", "")).strip())[0]}
                for c in chunks
            ]
        return out


class STTHandler(BaseHTTPRequestHandler):
    engine: Engine = None

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path != "/health":
            self._json(404, {"error": "unknown endpoint"})
            return
        e = self.engine
        lat = e.latencies
        self._json(200, {
            "status": "ok", "device": e.device, "model": e.model_dir.name,
            "script_mode": SCRIPT_MODE,
            "model_dir": str(e.model_dir), "sample_rate": TARGET_SR,
            "compile_s": round(e.compile_s, 2), "warmup_ms": round(e.warmup_ms, 1),
            "requests_served": len(lat),
            "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
        })

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/transcribe":
            self._json(404, {"error": "unknown endpoint"})
            return

        q = parse_qs(parsed.query)
        language = (q.get("language") or [None])[0]
        task = (q.get("task") or [None])[0]
        timestamps = (q.get("timestamps") or ["0"])[0] not in ("0", "", "false")

        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            self._json(400, {"error": "empty body: send audio bytes or JSON {\"path\": ...}"})
            return
        if n > MAX_UPLOAD:
            self._json(413, {"error": f"body too large ({n} bytes, max {MAX_UPLOAD})"})
            return
        raw = self.rfile.read(n)

        # JSON body -> read a local file instead of an upload
        if (self.headers.get("Content-Type") or "").startswith("application/json"):
            try:
                data = json.loads(raw)
            except Exception as ex:
                self._json(400, {"error": f"invalid JSON: {ex}"})
                return
            p = data.get("path")
            if not p:
                self._json(400, {"error": "JSON body requires 'path'"})
                return
            p = Path(p)
            if not p.exists():
                self._json(400, {"error": f"file not found: {p}"})
                return
            raw = p.read_bytes()
            language = data.get("language", language)
            task = data.get("task", task)
            timestamps = bool(data.get("timestamps", timestamps))

        try:
            arr, dur = decode_audio(raw)
        except Exception as ex:
            self._json(400, {"error": f"cannot decode audio: {type(ex).__name__}: {ex}"})
            return
        if dur <= 0:
            self._json(400, {"error": "audio has zero length"})
            return

        try:
            self._json(200, self.engine.transcribe(
                arr, dur, language=language, task=task, timestamps=timestamps))
        except Exception as ex:
            traceback.print_exc()
            self._json(500, {"error": f"{type(ex).__name__}: {ex}"})

    def log_message(self, *a):
        pass


def main():
    STTHandler.engine = Engine(MODEL_DIR, DEVICE)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), STTHandler)
    print(f"[stt] listening on http://127.0.0.1:{PORT}  "
          f"(POST /transcribe, GET /health)")
    print(f"[stt] device={DEVICE} model={MODEL_DIR} script={SCRIPT_MODE}")
    print("[stt] SECURITY NOTICE: binds to 127.0.0.1 with NO authentication. "
          "Do NOT expose on 0.0.0.0 without adding auth — audio is sensitive input.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stt] stopped.")


if __name__ == "__main__":
    main()
