"""
verify_npu.py — bukti bahwa IR hasil export benar-benar compile & jalan di NPU,
plus perbandingan numerik terhadap referensi PyTorch dan pengukuran latensi nyata.

Menguji tiap device yang tersedia (NPU, CPU, GPU.0) dan melaporkan:
  * compile time
  * latensi per-request (p50 / p90) atas beban nyata dari dataset
  * kesesuaian numerik: max abs diff vs PyTorch, dan action agreement
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]


def load_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tok, texts, S):
    enc = tok(texts, return_tensors="np", padding="max_length", truncation=True, max_length=S)
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/iniz-guard-int8-ov/guard.xml")
    ap.add_argument("--devices", default="NPU,CPU")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="npu_verify.json")
    args = ap.parse_args()

    import openvino as ov
    core = ov.Core()
    print("available:", core.available_devices)

    meta_path = Path(args.model).parent / "guard_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    S = meta.get("seq_len", 192)
    print(f"model={args.model} seq_len={S}")

    model = core.read_model(args.model)
    print("input shapes:", [(i.get_any_name(), str(i.get_partial_shape())) for i in model.inputs])

    tok = load_tok()
    import pandas as pd
    df = pd.read_parquet("data/full-validation.parquet").head(args.n)
    texts = df["text"].tolist()
    ids, mask = encode(tok, texts, S)

    # --- referensi PyTorch (fp32, sumber kebenaran) ---
    print("\n[ref] menghitung referensi PyTorch ...", flush=True)
    import torch
    from guard_model import load_checkpoint
    ref_model, _ = load_checkpoint("ckpt/model.safetensors",
                                   pooling=meta.get("pooling", "last_nonpad"), verbose=False)
    ref_inj, ref_sh, ref_act = [], [], []
    with torch.no_grad():
        for i in range(len(texts)):
            o = ref_model(torch.from_numpy(ids[i:i + 1]), torch.from_numpy(mask[i:i + 1]))
            ref_inj.append(float(o["injection"]))
            ref_sh.append(float(o["shell"]))
            ref_act.append(int(o["action_logits"].argmax()))
    ref_inj, ref_sh, ref_act = np.array(ref_inj), np.array(ref_sh), np.array(ref_act)
    del ref_model
    print(f"[ref] done. action dist: {np.bincount(ref_act, minlength=4).tolist()}")

    results = []
    for dev in args.devices.split(","):
        dev = dev.strip()
        if dev not in core.available_devices:
            print(f"\n=== {dev}: TIDAK TERSEDIA, skip ===")
            continue
        print(f"\n=== {dev} ===", flush=True)
        try:
            t0 = time.time()
            compiled = core.compile_model(model, dev)
            ct = time.time() - t0
            print(f"compile: {ct:.2f}s")
        except Exception as e:
            print(f"COMPILE FAILED: {type(e).__name__}: {e}")
            results.append({"device": dev, "compile_ok": False, "error": f"{type(e).__name__}: {e}"})
            continue

        req = compiled.create_infer_request()
        inj_o, sh_o, act_o, lat = [], [], [], []
        # warmup
        req.infer({"input_ids": ids[0:1], "attention_mask": mask[0:1]})
        for i in range(len(texts)):
            t0 = time.time()
            r = req.infer({"input_ids": ids[i:i + 1], "attention_mask": mask[i:i + 1]})
            lat.append((time.time() - t0) * 1000)
            vals = list(r.values())
            inj_o.append(float(np.array(vals[0]).flatten()[0]))
            sh_o.append(float(np.array(vals[1]).flatten()[0]))
            act_o.append(int(np.array(vals[2]).flatten().argmax()))
        inj_o, sh_o, act_o = np.array(inj_o), np.array(sh_o), np.array(act_o)

        d_inj = float(np.abs(inj_o - ref_inj).max())
        d_sh = float(np.abs(sh_o - ref_sh).max())
        agree = float((act_o == ref_act).mean())
        p50, p90 = statistics.median(lat), sorted(lat)[int(0.9 * len(lat)) - 1]
        print(f"latency p50={p50:.1f}ms p90={p90:.1f}ms min={min(lat):.1f} max={max(lat):.1f}")
        print(f"vs torch: max|d_inj|={d_inj:.5f} max|d_shell|={d_sh:.5f} action_agreement={agree:.4f}")
        print(f"action dist: {np.bincount(act_o, minlength=4).tolist()}")
        results.append({"device": dev, "compile_ok": True, "compile_s": round(ct, 2),
                        "p50_ms": round(p50, 1), "p90_ms": round(p90, 1),
                        "min_ms": round(min(lat), 1), "max_ms": round(max(lat), 1),
                        "max_abs_diff_injection": d_inj, "max_abs_diff_shell": d_sh,
                        "action_agreement_vs_torch": agree,
                        "n": len(texts), "model": args.model})

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    for r in results:
        if r.get("compile_ok"):
            print(f"  {r['device']:6s} p50={r['p50_ms']}ms  agree={r['action_agreement_vs_torch']:.3f}")
        else:
            print(f"  {r['device']:6s} FAILED: {r['error'][:80]}")


if __name__ == "__main__":
    main()
