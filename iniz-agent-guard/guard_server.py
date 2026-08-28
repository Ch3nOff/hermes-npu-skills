"""
guard_server.py — Iniz Agent Guard (NPU security engine, fine-tuned classifier)

MAJOR CHANGE vs the previous version:
  The old version used openvino_genai.LLMPipeline and asked a 0.5B generative
  model to invent JSON scores — that approach has been abandoned. The
  fine-tuned checkpoint (lora_adapter, checkpoint-2634) has NO lm_head: it is
  a Qwen2Model plus 3 discriminative heads. So this server calls
  ov.CompiledModel directly and reads the score tensors — no text generation,
  no JSON parsing, no max_new_tokens.

Architecture of the served model:
  Qwen2Model (24 layers, hidden 896, LoRA r=8 q/k/v/o already merged)
    -> last-non-pad pooling
    -> inj_head   (Linear 896->1)  injection score regression
    -> shell_head (Linear 896->1)  shell-risk score regression
    -> action_head(Linear 896->4)  action classification

API:
  POST /scan   {"text": "...", "context": "..."}
    -> {"injection_score", "shell_risk_score", "action", "confidence",
        "action_probs", "_npu_ms", "_device", "_model"}
  GET  /health -> status, device, warmup latency
"""

import json
import os
import statistics
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "INIZ_GUARD_MODEL", str(HERE.parent / "models" / "iniz-guard-int8-ov")))
DEVICE = os.environ.get("INIZ_GUARD_DEVICE", "NPU")
PORT = int(os.environ.get("INIZ_GUARD_PORT", "8009"))

# Injection gate threshold. The default value comes from an empirical sweep in
# work/eval_ov_final.json (not a made-up number) — see SKILL.md.
THRESHOLD = float(os.environ.get("INIZ_GUARD_THRESHOLD", "0.30"))

# The keyword backstop is deliberately kept: shell_head is weak because the
# training dataset only had 12/14036 code_execution samples. A keyword hit
# RAISES the shell score.
_SHELL_KW = ["rm -rf", "curl ", "wget ", "nc -e", "netcat", "base64 -d", "eval(",
             "chmod 777", "chown root", "/etc/passwd", "/etc/shadow", "exfil",
             "reverse shell", "bind shell", "> /dev/tcp", "os.system", "subprocess",
             "dd if=", "mkfs", ":(){", "powershell -enc", "invoke-webrequest"]


class Engine:
    """Load the IR once, infer repeatedly. Thread-safe via a lock (NPU handles a single request)."""

    def __init__(self, model_dir: Path, device: str):
        import openvino as ov
        from transformers import AutoTokenizer

        meta_path = model_dir / "guard_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.seq_len = self.meta.get("seq_len", 192)
        self.device = device
        self.lock = Lock()

        xml = model_dir / "guard.xml"
        if not xml.exists():
            raise FileNotFoundError(f"IR not found: {xml}")

        print(f"[guard] tokenizer ...", flush=True)
        tok_src = str(model_dir) if (model_dir / "tokenizer.json").exists() \
            else self.meta.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        print(f"[guard] compiling {xml.name} -> {device} (seq_len={self.seq_len}) ...", flush=True)
        t0 = time.time()
        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f"device {device} is not available. available: {core.available_devices}")
        self.compiled = core.compile_model(core.read_model(str(xml)), device)
        self.req = self.compiled.create_infer_request()
        self.compile_s = time.time() - t0
        print(f"[guard] compiled in {self.compile_s:.2f}s")

        # warmup + verify against the reference numbers
        t0 = time.time()
        w = self.raw("warmup probe")
        self.warmup_ms = (time.time() - t0) * 1000
        print(f"[guard] warmup {self.warmup_ms:.1f}ms  inj={w[0]:.4f} shell={w[1]:.4f}")
        self.latencies = []

    def raw(self, text: str):
        enc = self.tok([text], return_tensors="np", padding="max_length",
                       truncation=True, max_length=self.seq_len)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        with self.lock:
            r = self.req.infer({"input_ids": ids, "attention_mask": mask})
        v = list(r.values())
        inj = float(np.array(v[0]).flatten()[0])
        sh = float(np.array(v[1]).flatten()[0])
        logits = np.array(v[2]).flatten().astype(np.float64)
        return inj, sh, logits

    def scan(self, text: str, context: str = ""):
        payload = f"{text}\nContext: {context}" if context else text
        t0 = time.time()
        inj, sh, logits = self.raw(payload)
        ms = (time.time() - t0) * 1000
        self.latencies.append(ms)
        if len(self.latencies) > 500:
            self.latencies = self.latencies[-500:]

        e = np.exp(logits - logits.max())
        probs = e / e.sum()

        # clamp the regression heads to [0,1] — a linear head without a sigmoid
        # can produce values outside that range
        inj_c = float(np.clip(inj, 0.0, 1.0))
        sh_c = float(np.clip(sh, 0.0, 1.0))

        # keyword backstop: shell_head is undertrained, do not rely on it alone
        low = payload.lower()
        hits = [k for k in _SHELL_KW if k in low]
        kw_shell = min(0.4 * len(hits), 1.0)
        shell_final = max(sh_c, kw_shell)

        action = ACTIONS[int(probs.argmax())]
        # gate: if the action head says PASS but a score crosses the threshold, escalate
        if action == "PASS" and max(inj_c, shell_final) >= THRESHOLD:
            action = "ISOLATE_FILE" if shell_final > inj_c else "PAUSE_AGENTS"
        # conversely: very low scores and no keyword hits -> PASS
        if max(inj_c, shell_final) < THRESHOLD and not hits:
            action = "PASS"

        mx = max(inj_c, shell_final)
        conf = "high" if mx > 0.6 or mx < 0.1 else ("medium" if mx > 0.35 else "low")

        return {
            "injection_score": round(inj_c, 4),
            "shell_risk_score": round(shell_final, 4),
            "action": action,
            "confidence": conf,
            "action_probs": {a: round(float(p), 4) for a, p in zip(ACTIONS, probs)},
            "shell_keyword_hits": hits,
            "_raw_injection": round(inj, 4),
            "_raw_shell": round(sh, 4),
            "_npu_ms": round(ms, 2),
            "_device": self.device,
            "_model": f"iniz-guard-{MODEL_DIR.name}",
        }


class GuardHandler(BaseHTTPRequestHandler):
    engine: Engine = None

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._json(404, {"error": "unknown endpoint"})
            return
        e = self.engine
        lat = e.latencies
        self._json(200, {
            "status": "ok", "device": e.device, "seq_len": e.seq_len,
            "model_dir": str(MODEL_DIR), "threshold": THRESHOLD,
            "compile_s": round(e.compile_s, 2), "warmup_ms": round(e.warmup_ms, 1),
            "requests_served": len(lat),
            "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
            "actions": ACTIONS,
        })

    def do_POST(self):
        if self.path != "/scan":
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
        except Exception as ex:
            self._json(400, {"error": f"invalid JSON: {ex}"})
            return
        text = data.get("text", "")
        if not text:
            self._json(400, {"error": "field 'text' is required"})
            return
        try:
            self._json(200, self.engine.scan(text, data.get("context", "")))
        except Exception as ex:
            traceback.print_exc()
            self._json(500, {"error": f"{type(ex).__name__}: {ex}"})

    def log_message(self, *a):
        pass


def main():
    GuardHandler.engine = Engine(MODEL_DIR, DEVICE)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), GuardHandler)
    print(f"[guard] listening on http://127.0.0.1:{PORT}  (POST /scan, GET /health)")
    print(f"[guard] device={DEVICE} threshold={THRESHOLD} model={MODEL_DIR}")
    print("[guard] SECURITY NOTICE: this server binds to 127.0.0.1 with NO authentication. "
          "Do NOT expose it on 0.0.0.0 without adding auth first.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[guard] stopped.")


if __name__ == "__main__":
    main()
