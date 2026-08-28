"""
dryrun_finetune.py — menguji apakah finetune_local.py dari paket
iniz-agent-guard-fixed.zip BENAR-BENAR jalan, bukan cuma "terlihat benar".

Paket itu sendiri mengakui: "skrip ini BELUM dijalankan end-to-end".
Skrip ini menjalankannya dengan modifikasi minimal:
  * dataset dipotong ke N kecil (default 64 train / 32 val)
  * max_steps=3 supaya trainer.train() benar-benar dieksekusi tapi cepat
  * CPU (mesin ini punya torch CPU-only di venv), bf16/fp16 dimatikan
  * gradient_checkpointing dimatikan (butuh grad pada input embed; di CPU
    tanpa GPU ini hanya memperlambat)

Yang diuji: import, label mapping, dataset class, GuardHeadModel.forward,
loss backward, dan trainer.train() sampai selesai tanpa exception.
"""

import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_TOKEN", "dryrun-not-a-real-token")

import pandas as pd
import torch

INCOMING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "incoming", "iniz-agent-guard")
sys.path.insert(0, INCOMING)

print("=" * 70)
print("DRY RUN: finetune_local.py dari paket incoming")
print("=" * 70)

# ---- import modul tanpa menjalankan main() ----
import importlib.util
spec = importlib.util.spec_from_file_location(
    "incoming_finetune", os.path.join(INCOMING, "finetune_local.py"))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print("[1/6] import modul + guard HF_TOKEN: OK")
except Exception as e:
    print(f"[1/6] GAGAL import: {type(e).__name__}: {e}")
    raise

# ---- label mapping pada data lokal (tanpa akses jaringan) ----
try:
    train = pd.read_parquet("data/full-train.parquet").head(64).reset_index(drop=True)
    val = pd.read_parquet("data/full-validation.parquet").head(32).reset_index(drop=True)
    train[["inj", "shell", "action"]] = train.apply(mod.map_labels, axis=1)
    val[["inj", "shell", "action"]] = val.apply(mod.map_labels, axis=1)
    print(f"[2/6] map_labels: OK  dist={train['action'].value_counts().to_dict()}")
    print(f"      shell values: {sorted(train['shell'].unique())}")
except Exception as e:
    print(f"[2/6] GAGAL map_labels: {type(e).__name__}: {e}")
    raise

# ---- tokenizer + dataset ----
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

tok = AutoTokenizer.from_pretrained(mod.BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
train_ds = mod.GuardDataset(train, tok)
val_ds = mod.GuardDataset(val, tok)
b = train_ds[0]
print(f"[3/6] GuardDataset: OK  keys={sorted(b.keys())} "
      f"input_ids={tuple(b['input_ids'].shape)}")

# ---- model 3-head ----
dtype = torch.float32
backbone = AutoModel.from_pretrained(mod.BASE_MODEL, dtype=dtype)
backbone.config.use_cache = False
lora_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="FEATURE_EXTRACTION", bias="none")
backbone = get_peft_model(backbone, lora_cfg)
guard = mod.GuardHeadModel(backbone, len(mod.ACTIONS), dtype)
n_train = sum(p.numel() for p in guard.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in guard.parameters())
print(f"[4/6] GuardHeadModel: OK  trainable={n_train:,} / {n_all:,} "
      f"({100*n_train/n_all:.3f}%)")

# ---- forward + backward manual ----
try:
    batch = mod.guard_collator([train_ds[i] for i in range(2)])
    out = guard(**batch)
    print(f"[5/6] forward: OK  loss={out['loss'].item():.4f} "
          f"inj={out['injection'].tolist()} act={tuple(out['action'].shape)}")
    out["loss"].backward()
    grads = [p.grad is not None for n, p in guard.named_parameters() if p.requires_grad]
    print(f"      backward: OK  param dgn grad = {sum(grads)}/{len(grads)}")
except Exception as e:
    print(f"[5/6] GAGAL forward/backward: {type(e).__name__}: {e}")
    raise

# ---- trainer.train() 3 step ----
try:
    guard.zero_grad(set_to_none=True)
    args = TrainingArguments(
        output_dir="dryrun_out",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        bf16=False, fp16=False,
        max_steps=3,
        learning_rate=2e-4,
        logging_steps=1, logging_first_step=True,
        report_to="none",
        eval_strategy="no",
        save_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    trainer = Trainer(model=guard, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=mod.guard_collator)
    res = trainer.train()
    print(f"[6/6] trainer.train(): OK  steps={res.global_step} "
          f"train_loss={res.training_loss:.4f}")
except Exception as e:
    print(f"[6/6] GAGAL trainer.train(): {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc(limit=8)
    raise

print("\n" + "=" * 70)
print("VERDICT: finetune_local.py dari paket incoming JALAN end-to-end")
print("(dengan max_steps=3, CPU fp32, gradient_checkpointing off)")
print("=" * 70)
