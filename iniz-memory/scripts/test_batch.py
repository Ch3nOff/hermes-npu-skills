"""
test_batch.py — does batching flip the NPU-vs-CPU verdict for bulk indexing?

Serving verdict is already settled (single query: CPU 10.3ms beats NPU 15.2ms).
But corpus indexing is bulk work: reshape to [B, SEQ] and compare per-chunk
throughput at B=1 vs B=8 on both devices. If the NPU amortizes its fixed overhead
over a batch, bulk indexing belongs on the NPU even though serving does not.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from bench_mem import load_all, SEQ_LEN

MODEL = "../models/e5-small-int8-ov"


def build_batched(device, batch):
    import openvino as ov
    from transformers import AutoTokenizer

    core = ov.Core()
    model = core.read_model(str(Path(MODEL) / "openvino_model.xml"))
    model.reshape({"input_ids": [batch, SEQ_LEN],
                   "attention_mask": [batch, SEQ_LEN],
                   "token_type_ids": [batch, SEQ_LEN]})
    compiled = core.compile_model(model, device)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    def embed_batch(texts):
        enc = tok(texts, return_tensors="np", padding="max_length",
                  truncation=True, max_length=SEQ_LEN)
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": np.zeros_like(enc["input_ids"])}
        out = compiled(feed)
        h = next(iter(out.values()))
        mask = enc["attention_mask"]
        v = (h * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    return compiled, embed_batch


def main():
    chunks, _ = load_all()
    texts = ["passage: " + c["text"] for c in chunks]
    print(f"chunks: {len(texts)}")

    for device in ("NPU", "CPU"):
        for batch in (1, 8):
            try:
                t0 = time.time()
                _, emb = build_batched(device, batch)
                compile_s = time.time() - t0
                # warmup
                emb(texts[:batch])
                t0 = time.time()
                for i in range(0, len(texts), batch):
                    blk = texts[i:i + batch]
                    if len(blk) < batch:
                        blk = blk + ["" ] * (batch - len(blk))
                    emb(blk)
                el = time.time() - t0
                print(f"{device} batch={batch}: compile={compile_s:.2f}s "
                      f"corpus={el*1000:.0f}ms ({el*1000/len(texts):.1f}ms/chunk)",
                      flush=True)
            except Exception as e:
                print(f"{device} batch={batch} FAILED: "
                      f"{type(e).__name__}: {str(e)[:100]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
