"""
verify_arch_match.py — is finetune_local.py from the incoming package really the
script that produced checkpoint-2634?

Test: build GuardHeadModel exactly as finetune_local.py does, then compare its
state_dict KEY SET against the trained model.safetensors.
If they are 100% identical, that script is indeed the correct architecture.
"""

import os
import sys
import struct
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_TOKEN", "dryrun-not-a-real-token")

import torch
import importlib.util

INCOMING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "incoming", "iniz-agent-guard")
spec = importlib.util.spec_from_file_location(
    "incoming_finetune", os.path.join(INCOMING, "finetune_local.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

print("building GuardHeadModel exactly as finetune_local.py does ...", flush=True)
backbone = AutoModel.from_pretrained(mod.BASE_MODEL, dtype=torch.float32)
backbone.config.use_cache = False
lora_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="FEATURE_EXTRACTION", bias="none")
backbone = get_peft_model(backbone, lora_cfg)
guard = mod.GuardHeadModel(backbone, len(mod.ACTIONS), torch.float32)

built = set(guard.state_dict().keys())

with open("ckpt/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
ckpt = {k for k in hdr if k != "__metadata__"}

only_built = sorted(built - ckpt)
only_ckpt = sorted(ckpt - built)

print(f"\nkeys in built model          : {len(built)}")
print(f"keys in checkpoint           : {len(ckpt)}")
print(f"intersection                 : {len(built & ckpt)}")
print(f"only in built model          : {len(only_built)}")
for k in only_built[:12]:
    print("   -", k)
print(f"only in checkpoint           : {len(only_ckpt)}")
for k in only_ckpt[:12]:
    print("   +", k)

if not only_built and not only_ckpt:
    print("\nVERDICT: 100% IDENTICAL. The incoming finetune_local.py is the")
    print("         architecture that actually produced checkpoint-2634.")
else:
    print("\nVERDICT: NOT identical — see the differences above.")

# also compare the training hyperparameters
ta = torch.load("ckpt/training_args.bin", weights_only=False)
d = ta.to_dict() if hasattr(ta, "to_dict") else vars(ta)
print("\nhyperparameters: checkpoint vs finetune_local.py")
expected = {
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
    "num_train_epochs": 3, "learning_rate": 2e-4, "weight_decay": 0.01,
    "warmup_ratio": 0.05, "lr_scheduler_type": "cosine", "logging_steps": 20,
    "eval_steps": 200, "save_steps": 500, "optim": "adamw_torch",
}
match = 0
for k, want in expected.items():
    got = d.get(k)
    got_s = str(got).split(".")[-1] if k == "lr_scheduler_type" else got
    ok = str(got_s) == str(want)
    match += ok
    print(f"  {'OK ' if ok else 'DIFF'} {k:32s} ckpt={got_s}  script={want}")
print(f"\n{match}/{len(expected)} hyperparameters match")
