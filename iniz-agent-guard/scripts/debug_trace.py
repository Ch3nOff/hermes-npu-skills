"""
debug_trace.py — isolate export failures: torch.jit.trace vs torch.export vs ONNX.

Goal: find the conversion path that actually works for the Qwen2Model backbone
with its 3 heads, before handing it over to ov.convert_model.
"""

import traceback
import torch

from guard_model import load_checkpoint

S = 192


def build():
    import torch.nn as nn
    guard, _ = load_checkpoint("ckpt/model.safetensors", pooling="last_nonpad", verbose=False)
    merged = guard.base.merge_and_unload()
    merged.config._attn_implementation = "eager"
    merged.config.use_cache = False

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = merged
            self.inj_head = guard.inj_head
            self.shell_head = guard.shell_head
            self.action_head = guard.action_head

        def forward(self, input_ids, attention_mask):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            idx = (attention_mask.sum(dim=1).to(torch.int64) - 1).clamp(min=0)
            pooled = h[torch.arange(h.shape[0]), idx]
            return (self.inj_head(pooled).squeeze(-1),
                    self.shell_head(pooled).squeeze(-1),
                    self.action_head(pooled))

    return W().eval()


def attempt(name, fn):
    print(f"\n########## {name} ##########", flush=True)
    try:
        r = fn()
        print(f"[OK] {name} -> {r}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
        return False


def main():
    print(f"torch {torch.__version__}")
    import transformers
    print(f"transformers {transformers.__version__}")
    w = build()
    ids = torch.ones(1, S, dtype=torch.int64)
    mask = torch.ones(1, S, dtype=torch.int64)
    with torch.no_grad():
        ref = w(ids, mask)
    print(f"eager ref: {[float(x.flatten()[0]) for x in ref]}")

    # 1. jit.trace directly
    def t1():
        with torch.no_grad():
            m = torch.jit.trace(w, (ids, mask), strict=False, check_trace=False)
        return type(m).__name__
    ok_trace = attempt("torch.jit.trace(strict=False)", t1)

    # 2. torch.export (dynamo)
    def t2():
        with torch.no_grad():
            ep = torch.export.export(w, (ids, mask), strict=False)
        return type(ep).__name__
    ok_export = attempt("torch.export.export(strict=False)", t2)

    # 3. ONNX (dynamo=True)
    def t3():
        torch.onnx.export(w, (ids, mask), "guard_dynamo.onnx",
                          input_names=["input_ids", "attention_mask"],
                          output_names=["injection", "shell", "action_logits"],
                          dynamo=True, opset_version=18)
        import os
        return f"{os.path.getsize('guard_dynamo.onnx')/1e6:.1f} MB"
    ok_onnx_dynamo = attempt("torch.onnx.export(dynamo=True)", t3)

    # 4. ONNX (legacy TorchScript exporter)
    def t4():
        torch.onnx.export(w, (ids, mask), "guard_ts.onnx",
                          input_names=["input_ids", "attention_mask"],
                          output_names=["injection", "shell", "action_logits"],
                          dynamo=False, opset_version=17)
        import os
        return f"{os.path.getsize('guard_ts.onnx')/1e6:.1f} MB"
    ok_onnx_ts = attempt("torch.onnx.export(dynamo=False)", t4)

    print("\n=== SUMMARY ===")
    for n, v in [("jit.trace", ok_trace), ("torch.export", ok_export),
                 ("onnx dynamo", ok_onnx_dynamo), ("onnx torchscript", ok_onnx_ts)]:
        print(f"  {n:20s} {'OK' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
