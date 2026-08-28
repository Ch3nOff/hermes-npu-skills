"""
finetune_local.py — Iniz Agent Guard, local fine-tuning entry point.

PERBAIKAN dari versi sebelumnya:
  1. TOKEN HUGGINGFACE HARDCODED DIHAPUS. Token sebelumnya di file ini
     sudah TERKIRIM keluar dari mesin lokal dan HARUS dianggap bocor —
     revoke di https://huggingface.co/settings/tokens sebelum
     melanjutkan apa pun dengan skrip ini. Token sekarang dibaca dari
     environment variable, tidak pernah ditulis langsung ke source.
  2. Arsitektur diganti dari AutoModelForCausalLM + generative LoRA
     fine-tuning menjadi AutoModel (base) + 3 custom head (injection
     regression, shell regression, action classification) — supaya
     KONSISTEN dengan apa yang benar-benar diserve guard_server.py.
     Versi sebelumnya melatih model generatif biasa yang formatnya
     tidak cocok dengan cara guard_server.py memanggil dan mem-parsing
     output model.
  3. Skema dataset diperbaiki. Field 'target'/'input' yang dipakai
     versi sebelumnya TIDAK ADA di neuralchemy/Prompt-injection-dataset
     — skema aslinya adalah category/severity/label/text/tags (sudah
     diverifikasi langsung ke dataset card). Mapping label memakai
     kategori yang benar-benar ada, bukan agent_manipulation/
     code_execution/instruction_override yang tidak eksis.
  4. Semua perbaikan dari sesi debugging notebook sebelumnya
     diikutsertakan: CUDA_VISIBLE_DEVICES sebelum import torch,
     hidden_states[-1] alih-alih last_hidden_state (karena AutoModel
     base MEMANG punya last_hidden_state secara sah — beda dari
     AutoModelForCausalLM yang tidak), device eksplisit sebelum
     forward pass, batch size kecil dengan gradient accumulation.

STATUS: skrip ini BELUM dijalankan end-to-end di sesi ini. Struktur
label-mapping dan arsitektur head sudah divalidasi terpisah (lihat
notebooks/05_finetune_guard_model.ipynb versi terbaru, bukan versi
lama yang masih ada di paket lama — lihat catatan di SKILL.md).
Jalankan dan laporkan traceback apa pun yang muncul; jangan asumsikan
ini otomatis bebas bug hanya karena sudah melalui sesi debugging lain.

Prasyarat sebelum menjalankan:
    export HF_TOKEN="hf_xxxxx"   # token BARU, setelah yang lama direvoke
    pip install transformers datasets peft accelerate optimum[openvino] nncf --break-system-packages
"""

# ============================================================
# 0. LIMIT KE SATU GPU — HARUS sebelum import torch.
# Trainer HuggingFace otomatis membungkus model dengan DataParallel
# begitu >1 GPU terlihat, walau tidak diminta eksplisit di
# TrainingArguments — ini yang menyebabkan OOM di sesi debugging
# sebelumnya meski model 0.5B secara matematis butuh <2GB.
# ============================================================
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# 1. TOKEN — dibaca dari environment, TIDAK PERNAH hardcoded.
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN tidak ditemukan di environment. Set dulu:\n"
        "  export HF_TOKEN='hf_xxxxx'  (Linux/Mac)\n"
        "  $env:HF_TOKEN='hf_xxxxx'    (PowerShell)\n"
        "Gunakan token BARU — jika ini terkait token lama yang pernah\n"
        "tertulis langsung di source file ini, token itu sudah harus\n"
        "direvoke di https://huggingface.co/settings/tokens sebelum\n"
        "token baru dibuat."
    )

import gc
import pandas as pd
import torch

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# ============================================================
# 2. KONFIGURASI
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 128
OUTPUT_DIR_ADAPTER = "./iniz-guard-lora"
OUTPUT_DIR_MERGED = "./iniz-guard-merged"
OUTPUT_DIR_OV = "./models/iniz-guard-int8-ov"
HUB_REPO_ID = "CH3NDev/iniz-agent-guard-int8"  # sesuaikan jika berbeda

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}

# ============================================================
# 3. LABEL MAPPING — skema kategori ASLI dataset, sudah diverifikasi
# langsung ke dataset card neuralchemy/Prompt-injection-dataset.
# Kategori agent_manipulation/code_execution/instruction_override di
# versi sebelumnya TIDAK ADA di dataset ini — jangan dipakai lagi.
# ============================================================
SEVERITY_MAP = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}

OVERRIDE_CATEGORIES = {"direct_injection", "jailbreak", "persona_replacement", "many_shot", "crescendo"}
OBFUSCATION_CATEGORIES = {"encoding_obfuscation", "token_smuggling", "indirect_injection", "context_overflow"}
EXTRACTION_CATEGORIES = {"system_extraction", "prompt_leaking"}

SHELL_RISK_KEYWORDS = [
    "eval", "exec", "atob", "bash", "shell", "subprocess",
    "os.system", "js_eval", "code_execution", "command", "__import__",
]


def compute_shell_risk_proxy(row) -> float:
    """
    Heuristik kasar, BUKAN label asli — dataset ini tidak punya contoh
    command-execution/shell-injection sungguhan. Baca hasil evaluasi
    head shell_risk dengan skeptisisme lebih besar dibanding head
    injection dan action.
    """
    if int(row["label"]) == 0:
        return 0.0
    raw_tags = row["tags"]
    tag_list = [] if raw_tags is None else [str(t) for t in list(raw_tags)]
    haystack = (str(row["text"]) + " " + " ".join(tag_list)).lower()
    if any(kw in haystack for kw in SHELL_RISK_KEYWORDS):
        return 0.6
    return 0.1


def map_labels(row):
    category = row["category"]
    severity = row["severity"]
    label = int(row["label"])
    injection = 0.0
    action = "PASS"
    if label == 1:
        injection = SEVERITY_MAP.get(severity, 0.5)
        if category in OVERRIDE_CATEGORIES:
            action = "PAUSE_AGENTS"
        elif category in OBFUSCATION_CATEGORIES:
            action = "ISOLATE_FILE"
        elif category in EXTRACTION_CATEGORIES:
            action = "USER_CONFIRMATION"
        else:
            action = "PAUSE_AGENTS"
    shell = compute_shell_risk_proxy(row)
    return pd.Series({"inj": injection, "shell": shell, "action": action})


# ============================================================
# 4. DATASET CLASS
# ============================================================
class GuardDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]), truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "inj_target": torch.tensor(float(row["inj"]), dtype=torch.float32),
            "shell_target": torch.tensor(float(row["shell"]), dtype=torch.float32),
            "action_target": torch.tensor(ACTION2IDX[row["action"]], dtype=torch.long),
        }


def guard_collator(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "inj_target": torch.stack([b["inj_target"] for b in batch]),
        "shell_target": torch.stack([b["shell_target"] for b in batch]),
        "action_target": torch.stack([b["action_target"] for b in batch]),
    }


# ============================================================
# 5. MODEL 3-HEAD — AutoModel (base), BUKAN AutoModelForCausalLM.
# last_hidden_state VALID di sini karena AutoModel base memang
# mengembalikan BaseModelOutputWithPast, bukan CausalLMOutputWithPast.
# ============================================================
class GuardHeadModel(torch.nn.Module):
    def __init__(self, base, actions_dim, dtype):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.inj_head = torch.nn.Linear(hidden, 1).to(dtype=dtype)
        self.shell_head = torch.nn.Linear(hidden, 1).to(dtype=dtype)
        self.action_head = torch.nn.Linear(hidden, actions_dim).to(dtype=dtype)
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.base.gradient_checkpointing_enable()
            else:
                self.base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def forward(self, input_ids, attention_mask, inj_target=None, shell_target=None, action_target=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Ambil token valid TERAKHIR (bukan [:, -1, :] mentah), karena
        # padding="max_length" bisa membuat posisi terakhir adalah
        # padding token, bukan token konten sesungguhnya.
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_idx, last_idx]
        pooled = pooled.to(dtype=self.inj_head.weight.dtype)

        inj_logits = self.inj_head(pooled).squeeze(-1)
        shell_logits = self.shell_head(pooled).squeeze(-1)
        action_logits = self.action_head(pooled)

        loss = None
        if inj_target is not None and shell_target is not None and action_target is not None:
            loss = (
                self.mse(inj_logits.float(), inj_target.float())
                + self.mse(shell_logits.float(), shell_target.float())
                + self.ce(action_logits.float(), action_target)
            )
        return {"loss": loss, "injection": inj_logits, "shell": shell_logits, "action": action_logits}


def main():
    print("=" * 70)
    print("INIZ AGENT GUARD — LOCAL FINE-TUNING")
    print("=" * 70)

    print("\nLoading dataset...")
    ds_raw = load_dataset("neuralchemy/Prompt-injection-dataset", "full")
    train = ds_raw["train"].to_pandas()
    val = ds_raw["validation"].to_pandas()
    test = ds_raw["test"].to_pandas()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    print("\nGenerating labels...")
    train[["inj", "shell", "action"]] = train.apply(map_labels, axis=1)
    val[["inj", "shell", "action"]] = val.apply(map_labels, axis=1)
    test[["inj", "shell", "action"]] = test.apply(map_labels, axis=1)
    print("Action distribution (train):", train["action"].value_counts().to_dict())

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = GuardDataset(train, tokenizer)
    val_ds = GuardDataset(val, tokenizer)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"\nModel dtype: {dtype}")

    print("\nLoading backbone (AutoModel, base — bukan ForCausalLM)...")
    backbone = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=dtype, token=HF_TOKEN)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="FEATURE_EXTRACTION", bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    guard_model = GuardHeadModel(backbone, len(ACTIONS), dtype)
    guard_model.gradient_checkpointing_enable()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_ADAPTER,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        logging_first_step=True,
        report_to="none",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=guard_model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=guard_collator,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_DIR_ADAPTER)

    print("\nMerging LoRA into base and exporting to OpenVINO INT8...")
    print(
        "CATATAN: langkah merge + export OpenVINO BELUM diotomasi di\n"
        "fungsi ini secara eksplisit untuk 3-head model — head kustom\n"
        "(inj_head/shell_head/action_head) TIDAK ikut dalam format save\n"
        "PEFT/OpenVINO standar. Dokumentasikan cara memuat kembali ketiga\n"
        "head ini sebelum upload ke Hub, atau restrukturisasi menjadi\n"
        "custom PEFT head yang PEFT-aware, supaya orang yang download\n"
        "model dari Hub benar-benar dapat model utuh, bukan cuma backbone."
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nDONE (training + save adapter). Upload ke Hub dilakukan manual")
    print("setelah verifikasi head kustom bisa dimuat ulang dengan benar.")


if __name__ == "__main__":
    main()
