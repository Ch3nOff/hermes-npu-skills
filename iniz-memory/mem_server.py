"""
mem_server.py — Iniz Memory (local semantic search over the repo docs, on CPU/NPU)

Same process shape as stt_server.py: load models once, serve many requests,
lock-guarded inference, /health with real counters.

DEVICE DEFAULT IS CPU, DELIBERATELY. Measured on e5-small INT8, seq_len=256:
  single query embed: CPU 10.3ms vs NPU 15.2ms  (CPU 1.5x faster)
  bulk index:         CPU 9.9ms/chunk vs NPU 17.8ms/chunk (batch=8 helps CPU,
                      does nothing for NPU — larger static shape, same speed)
Same INT8 weights both sides (26/26 top1 agreement), so this is purely a latency
verdict, not quality. NPU remains an option (INIZ_MEM_DEVICE=NPU) for CPU offload,
same argument as whisper-base.

NO RERANKER — measured, then rejected. mMiniLM cross-encoder over top-10 moved
recall@1 0.615 -> 0.577 on the 26-query set (fixed 2, broke 3) at ~232ms/query,
20x the bi-encoder query cost. On a corpus full of cross-referencing sibling
sections it adds noise, not signal. See test_rerank.py + rerank_CPU.json.

API:
  GET  /health
    -> status, device, model, chunks, requests_served, latency_p50_ms
  POST /search
    body: {"query": "...", "top_k": 5}
    -> {"hits": [{"id","file","heading","text","score"}], "_ms", "_device", "_model"}

Security: binds to 127.0.0.1 with NO authentication.
"""

import json
import os
import statistics
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "INIZ_MEM_MODEL", str(HERE.parent / "models" / "e5-small-int8-ov")))
CORPUS = Path(os.environ.get(
    "INIZ_MEM_CORPUS", str(HERE / "corpus" / "chunks.json")))
DEVICE = os.environ.get("INIZ_MEM_DEVICE", "CPU")
PORT = int(os.environ.get("INIZ_MEM_PORT", "8012"))
SEQ_LEN = 256


class Engine:
    """E5 bi-encoder, static [1, SEQ] reshape (NPU rejects dynamic shapes).
    Corpus embedded once at startup; per request only the query is encoded."""

    def __init__(self, model_dir: Path, corpus_path: Path, device: str):
        import openvino as ov
        from transformers import AutoTokenizer

        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(
                f"device {device} not available. present: {core.available_devices}")
        model = core.read_model(str(model_dir / "openvino_model.xml"))
        model.reshape({"input_ids": [1, SEQ_LEN],
                       "attention_mask": [1, SEQ_LEN],
                       "token_type_ids": [1, SEQ_LEN]})
        print(f"[mem] compiling {model_dir.name} -> {device} ...", flush=True)
        t0 = time.time()
        self.compiled = core.compile_model(model, device)
        self.compile_s = time.time() - t0
        try:
            print(f"[mem] EXECUTION_DEVICES = "
                  f"{self.compiled.get_property('EXECUTION_DEVICES')}", flush=True)
        except Exception as e:
            print(f"[mem] (no EXECUTION_DEVICES: {e})", flush=True)
        self.tok = AutoTokenizer.from_pretrained(str(model_dir),
                                                 local_files_only=True)
        self.device = device
        self.model_dir = model_dir
        self.lock = Lock()

        chunks = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.chunks = chunks
        t0 = time.time()
        with self.lock:
            self.C = np.stack(self._embed([c["text"] for c in chunks],
                                          "passage: "))
        self.index_ms = (time.time() - t0) * 1000
        print(f"[mem] indexed {len(chunks)} chunks in {self.index_ms:.0f}ms "
              f"({self.index_ms/len(chunks):.1f}ms/chunk)")
        self.latencies = []

    def _embed(self, texts, prefix):
        vecs = []
        for t in texts:
            enc = self.tok(prefix + t, return_tensors="np",
                           padding="max_length", truncation=True,
                           max_length=SEQ_LEN)
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                    "token_type_ids": np.zeros_like(enc["input_ids"])}
            h = next(iter(self.compiled(feed).values()))
            mask = enc["attention_mask"][0]
            v = (h[0] * mask[:, None]).sum(0) / mask.sum()
            vecs.append(v / (float((v ** 2).sum() ** 0.5) or 1.0))
        return vecs

    def search(self, query, top_k):
        t0 = time.time()
        with self.lock:
            qv = self._embed([query], "query: ")[0]
            sims = self.C @ qv
            idx = np.argsort(-sims)[:top_k]
            hits = [{"id": self.chunks[i]["id"],
                     "file": self.chunks[i]["file"],
                     "heading": self.chunks[i]["heading"],
                     "text": self.chunks[i]["text"][:500],
                     "score": round(float(sims[i]), 4)} for i in idx]
        ms = (time.time() - t0) * 1000
        self.latencies.append(ms)
        if len(self.latencies) > 500:
            self.latencies = self.latencies[-500:]
        return {"hits": hits, "_ms": round(ms, 1), "_device": self.device,
                "_model": self.model_dir.name}


class MemHandler(BaseHTTPRequestHandler):
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
            "model_dir": str(e.model_dir), "chunks": len(e.chunks),
            "compile_s": round(e.compile_s, 2),
            "index_ms": round(e.index_ms, 1),
            "requests_served": len(lat),
            "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
            "reranker": "none (measured r@1 0.615->0.577, rejected)",
        })

    def do_POST(self):
        if urlparse(self.path).path != "/search":
            self._json(404, {"error": "unknown endpoint"})
            return
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            self._json(400, {"error": "empty body: send JSON {\"query\": ...}"})
            return
        try:
            data = json.loads(self.rfile.read(n))
        except Exception as ex:
            self._json(400, {"error": f"invalid JSON: {ex}"})
            return
        q = (data.get("query") or "").strip()
        if not q:
            self._json(400, {"error": "field 'query' is required"})
            return
        try:
            top_k = max(1, min(int(data.get("top_k", 5)), 20))
        except (TypeError, ValueError):
            self._json(400, {"error": "field 'top_k' must be an integer 1..20"})
            return
        try:
            self._json(200, self.engine.search(q, top_k))
        except Exception as ex:
            traceback.print_exc()
            self._json(500, {"error": f"{type(ex).__name__}: {ex}"})

    def log_message(self, *a):
        pass


def main():
    MemHandler.engine = Engine(MODEL_DIR, CORPUS, DEVICE)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), MemHandler)
    print(f"[mem] listening on http://127.0.0.1:{PORT}  "
          f"(POST /search, GET /health)")
    print(f"[mem] device={DEVICE} model={MODEL_DIR} chunks={len(MemHandler.engine.chunks)}")
    print("[mem] SECURITY NOTICE: binds to 127.0.0.1 with NO authentication.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[mem] stopped.")


if __name__ == "__main__":
    main()
