"""
aux_server.py — Iniz Aux (summarize / sentiment / extract on Qwen2.5-0.5B)

Same process shape as guard_server.py and stt_server.py: load a quantized model once,
serve many requests, lock-guarded inference, /health with real counters.

DEVICE DEFAULT IS CPU, DELIBERATELY. Measured on this machine, the NPU runs this model
20-23x SLOWER than the CPU at identical accuracy:

  task        NPU p50      CPU p50    ratio
  summarize   14181.6 ms    704.9 ms   20.1x
  sentiment    1029.8 ms     43.8 ms   23.5x
  extract      2726.3 ms    119.4 ms   22.8x

The NPU IS genuinely executing it (adapter at 98.37% mean, verified via LUID
counters) — it is just badly suited to autoregressive decode, where each token is a
separate graph execution and the win from a static-shape accelerator disappears.
Set INIZ_AUX_DEVICE=NPU only when the goal is keeping cores free (CPU 4.5% vs 22.5%
during load), not speed.

API:
  GET  /health
    -> status, device, model, requests_served, latency_p50_ms per task
  POST /summarize   {"text": "...", "max_new_tokens": 120}
    -> {"summary", "_ms", "_device", "_model"}
  POST /sentiment   {"text": "..."}
    -> {"sentiment": "positive"|"negative"|"unparsed", "raw", "_ms", ...}
  POST /extract     {"text": "...", "question": "What is the filename?"}
    -> {"answer", "grounded": bool, "raw", "_ms", ...}

`grounded` reports whether the extracted string actually occurs in the input. It is a
substring check, not a correctness guarantee: measured accuracy is 7/10 exact on
INT8, and the misses are wrong-span picks (e.g. "torch==2.9.1" when asked for the
version), which ARE grounded. Treat extraction output as a candidate to verify.

Security: binds to 127.0.0.1 with NO authentication.
"""

import json
import os
import re
import statistics
import time
import traceback
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "INIZ_AUX_MODEL", str(HERE.parent / "models" / "qwen2.5-0.5b-instruct-int8-ov")))
DEVICE = os.environ.get("INIZ_AUX_DEVICE", "CPU")
PORT = int(os.environ.get("INIZ_AUX_PORT", "8011"))
MAX_CHARS = int(os.environ.get("INIZ_AUX_MAX_CHARS", "8000"))

SYS_SUMMARIZE = "You are a summarizer. Reply with 2-3 short sentences and nothing else."
SYS_SENTIMENT = "Classify sentiment. Answer with exactly one word: positive or negative."
SYS_EXTRACT = ("Extract the exact string the user asks for. Copy it verbatim from "
               "the text. Reply with only that string.")


def chat_prompt(system, user):
    """Qwen2.5 template written out explicitly.

    openvino_genai's Tokenizer.apply_chat_template has a different signature from
    transformers' (no `tokenize` kwarg); calling it the transformers way raises
    TypeError and invites a silent fallback. Writing the template avoids the issue.
    """
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def norm_answer(s):
    s = s.strip().strip("`\"'“”‘’ ")
    s = re.sub(r"^(the\s+)?(filename|file|path|version|email|identifier|answer)"
               r"\s*(is|:)\s*", "", s, flags=re.I)
    return s.strip().rstrip(".,;:").strip("`\"'")


class Engine:
    def __init__(self, model_dir: Path, device: str):
        import openvino as ov
        import openvino_genai as ov_genai

        if not (model_dir / "openvino_model.xml").exists():
            raise FileNotFoundError(f"LLM IR not found in {model_dir}")
        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(
                f"device {device} not available. present: {core.available_devices}")

        print(f"[aux] compiling {model_dir.name} -> {device} ...", flush=True)
        t0 = time.time()
        self.pipe = ov_genai.LLMPipeline(str(model_dir), device)
        self.compile_s = time.time() - t0
        self.genai = ov_genai
        self.device = device
        self.model_dir = model_dir
        self.lock = Lock()
        self.latencies = defaultdict(list)
        print(f"[aux] loaded in {self.compile_s:.2f}s", flush=True)

        t0 = time.time()
        with self.lock:
            self._raw(chat_prompt("You are terse.", "Say ok."), 4)
        self.warmup_ms = (time.time() - t0) * 1000
        print(f"[aux] warmup {self.warmup_ms:.1f}ms", flush=True)

    def _raw(self, prompt, max_new_tokens):
        cfg = self.genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.do_sample = False       # greedy: same input -> same output
        return str(self.pipe.generate(prompt, cfg)).strip()

    def run(self, task, system, user, max_new_tokens):
        t0 = time.time()
        with self.lock:
            raw = self._raw(chat_prompt(system, user), max_new_tokens)
        ms = (time.time() - t0) * 1000
        lat = self.latencies[task]
        lat.append(ms)
        if len(lat) > 300:
            self.latencies[task] = lat[-300:]
        return raw, ms

    def meta(self, ms):
        return {"_ms": round(ms, 1), "_device": self.device,
                "_model": self.model_dir.name}


class AuxHandler(BaseHTTPRequestHandler):
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
        per_task = {t: {"requests": len(v),
                        "latency_p50_ms": round(statistics.median(v), 1)}
                    for t, v in e.latencies.items() if v}
        self._json(200, {
            "status": "ok", "device": e.device, "model": e.model_dir.name,
            "model_dir": str(e.model_dir),
            "compile_s": round(e.compile_s, 2),
            "warmup_ms": round(e.warmup_ms, 1),
            "requests_served": sum(len(v) for v in e.latencies.values()),
            "tasks": per_task,
            "note": ("NPU runs this model ~20x slower than CPU at equal accuracy; "
                     "CPU is the default for a reason"),
        })

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return None, "empty body: send JSON {\"text\": ...}"
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw)
        except Exception as ex:
            return None, f"invalid JSON: {ex}"
        if not isinstance(data, dict):
            return None, "body must be a JSON object"
        text = (data.get("text") or "").strip()
        if not text:
            return None, "field 'text' is required"
        if len(text) > MAX_CHARS:
            return None, f"text too long ({len(text)} chars, max {MAX_CHARS})"
        return data, None

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/summarize", "/sentiment", "/extract"):
            self._json(404, {"error": "unknown endpoint"})
            return

        data, err = self._read_json()
        if err:
            self._json(400, {"error": err})
            return
        text = data["text"].strip()
        e = self.engine

        try:
            if path == "/summarize":
                mnt = int(data.get("max_new_tokens", 120))
                raw, ms = e.run("summarize", SYS_SUMMARIZE,
                                f"Summarize this text:\n\n{text}", mnt)
                out = {"summary": raw}
                out.update(e.meta(ms))
                self._json(200, out)

            elif path == "/sentiment":
                raw, ms = e.run("sentiment", SYS_SENTIMENT, text, 8)
                low = raw.lower()
                label = ("positive" if "positive" in low else
                         "negative" if "negative" in low else "unparsed")
                out = {"sentiment": label, "raw": raw}
                out.update(e.meta(ms))
                self._json(200, out)

            else:  # /extract
                q = (data.get("question") or "").strip()
                if not q:
                    self._json(400, {"error": "field 'question' is required "
                                              "for /extract"})
                    return
                raw, ms = e.run("extract", SYS_EXTRACT,
                                f"{q}\n\nText: {text}", 40)
                ans = norm_answer(raw)
                out = {"answer": ans, "raw": raw,
                       "grounded": bool(ans) and ans in text,
                       "_warning": ("grounded means the string occurs in the input, "
                                    "not that it answers the question; measured "
                                    "7/10 exact on INT8 — verify before use")}
                out.update(e.meta(ms))
                self._json(200, out)

        except Exception as ex:
            traceback.print_exc()
            self._json(500, {"error": f"{type(ex).__name__}: {ex}"})

    def log_message(self, *a):
        pass


def main():
    AuxHandler.engine = Engine(MODEL_DIR, DEVICE)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), AuxHandler)
    print(f"[aux] listening on http://127.0.0.1:{PORT}  "
          f"(POST /summarize /sentiment /extract, GET /health)")
    print(f"[aux] device={DEVICE} model={MODEL_DIR}")
    if DEVICE.upper().startswith("NPU"):
        print("[aux] WARNING: NPU is ~20x slower than CPU for this model "
              "(measured). Use it only to keep CPU cores free.")
    print("[aux] SECURITY NOTICE: binds to 127.0.0.1 with NO authentication.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[aux] stopped.")


if __name__ == "__main__":
    main()
