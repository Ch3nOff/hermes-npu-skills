"""
export_guard_ov.py — export GuardModel (checkpoint fine-tuned) ke OpenVINO IR
untuk dijalankan di Intel NPU.

Kunci desain untuk NPU:
  * NPU butuh SHAPE STATIS. Export dengan [1, SEQ_LEN] tetap (default 192).
  * Input right-padded; pooling memakai Gather pada indeks sum(attention_mask)-1
    yang dihitung DI DALAM graph.
  * Tiga output: injection (f32 [1]), shell (f32 [1]), action_logits (f32 [1,4]).
  * LoRA di-merge ke base weight (merge_and_unload) supaya graph tidak memuat
    cabang lora_A/lora_B terpisah.

PITFALL PENTING (terverifikasi lewat debug_trace.py di mesin ini):
  transformers 4.57 + torch 2.9 => `torch.jit.trace` dan `torch.onnx.export`
  (dynamo=False) SELALU gagal pada Qwen2Model dengan
      RuntimeError: invalid unordered_map<K, T> key
  yang berasal dari walrus operator di masking_utils
  (`if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0`).
  Jalur yang BERHASIL: `torch.export.export(..., strict=False)` lalu serahkan
  ExportedProgram ke `ov.convert_model`. Jangan buang waktu di TorchScript.

Output:
  models/iniz-guard-fp16-ov/guard.xml|.bin
  models/iniz-guard-int8-ov/guard.xml|.bin   (weight compression via nncf)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from guard_model import load_checkpoint, ACTIONS


class ExportWrapper(nn.Module):
    """LoRA sudah di-merge, pooling di dalam graph, 3 output tensor."""

    def __init__(self, guard):
        super().__init__()
        merged = guard.base.merge_and_unload()
        # eager attention: sdpa/flash path tidak bisa di-trace maupun di-export bersih
        merged.config._attn_implementation = "eager"
        merged.config.use_cache = False
        self.backbone = merged
        self.inj_head = guard.inj_head
        self.shell_head = guard.shell_head
        self.action_head = guard.action_head

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        idx = (attention_mask.sum(dim=1).to(torch.int64) - 1).clamp(min=0)
        pooled = hidden[torch.arange(hidden.shape[0]), idx]
        return (self.inj_head(pooled).squeeze(-1),
                self.shell_head(pooled).squeeze(-1),
                self.action_head(pooled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/model.safetensors")
    ap.add_argument("--pooling", default="last_nonpad")
    ap.add_argument("--seq-len", type=int, default=128,
                    help="WAJIB sama dengan MAX_LENGTH saat training (128). "
                         "Nilai lain tidak error, hanya menambah latensi tanpa "
                         "manfaat kualitas (192 = +40%% latensi, kualitas sama).")
    ap.add_argument("--out-root", default="../models")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    print(f"[export] loading checkpoint (pooling={args.pooling}) ...", flush=True)
    guard, _ = load_checkpoint(args.ckpt, pooling=args.pooling, verbose=False)
    wrapper = ExportWrapper(guard).eval()

    S = args.seq_len
    ex_ids = torch.ones(1, S, dtype=torch.int64)
    ex_mask = torch.ones(1, S, dtype=torch.int64)

    with torch.no_grad():
        ref = wrapper(ex_ids, ex_mask)
    ref_vals = [float(ref[0]), float(ref[1])] + [float(x) for x in ref[2].flatten()]
    print(f"[export] torch ref (all-ones input): inj={ref_vals[0]:.5f} "
          f"shell={ref_vals[1]:.5f} action={ACTIONS[int(ref[2].argmax())]}")

    print("[export] torch.export.export(strict=False) ...", flush=True)
    with torch.no_grad():
        ep = torch.export.export(wrapper, (ex_ids, ex_mask), strict=False)

    import openvino as ov
    print(f"[export] ov.convert_model from ExportedProgram (static [1, {S}]) ...", flush=True)
    ov_model = ov.convert_model(ep)

    for i, name in enumerate(["input_ids", "attention_mask"]):
        ov_model.inputs[i].get_tensor().set_names({name})
    for i, name in enumerate(["injection", "shell", "action_logits"]):
        ov_model.outputs[i].get_tensor().set_names({name})

    # PITFALL: torch.export menghasilkan dimensi dinamis (?, ?) walaupun example
    # input-nya statis. NPU MENOLAK shape dinamis -> wajib reshape eksplisit.
    ov_model.reshape({0: ov.PartialShape([1, S]), 1: ov.PartialShape([1, S])})
    print("[export] inputs :", [(i.get_any_name(), str(i.get_partial_shape()),
                                 i.get_element_type().get_type_name()) for i in ov_model.inputs])
    print("[export] outputs:", [(o.get_any_name(), str(o.get_partial_shape()))
                                for o in ov_model.outputs])

    fp_dir = Path(args.out_root) / "iniz-guard-fp16-ov"
    fp_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(fp_dir / "guard.xml"), compress_to_fp16=True)
    print(f"[export] saved FP16 -> {fp_dir}  "
          f"({os.path.getsize(fp_dir/'guard.bin')/1e6:.1f} MB)")

    dirs = [fp_dir]
    if not args.skip_int8:
        import nncf
        print("[export] nncf compress_weights INT8_SYM ...", flush=True)
        comp = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)
        int8_dir = Path(args.out_root) / "iniz-guard-int8-ov"
        int8_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(comp, str(int8_dir / "guard.xml"))
        print(f"[export] saved INT8 -> {int8_dir}  "
              f"({os.path.getsize(int8_dir/'guard.bin')/1e6:.1f} MB)")
        dirs.append(int8_dir)

    for d in dirs:
        for f in ["tokenizer.json", "tokenizer_config.json"]:
            src = Path("ckpt") / f
            if src.exists():
                shutil.copy(src, d / f)
        # PITFALL: checkpoint ditulis oleh transformers 5.5.4 yang menyimpan
        # extra_special_tokens sebagai LIST. transformers 4.x mengharap DICT dan
        # gagal dengan: AttributeError: 'list' object has no attribute 'keys'.
        # Normalisasi di sini supaya tokenizer bisa dimuat lintas versi.
        cfg_path = d / "tokenizer_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            est = cfg.get("extra_special_tokens")
            if isinstance(est, list):
                cfg["extra_special_tokens"] = {}
                cfg["additional_special_tokens"] = est
                cfg_path.write_text(json.dumps(cfg, indent=2))
                print(f"[export] normalized extra_special_tokens (list->dict) in {cfg_path}")
        with open(d / "guard_meta.json", "w") as fh:
            json.dump({"seq_len": S, "pooling": args.pooling, "actions": ACTIONS,
                       "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                       "checkpoint": "lora_adapter checkpoint-2634",
                       "torch_reference_all_ones": ref_vals}, fh, indent=2)
    print("[export] done")


if __name__ == "__main__":
    main()
