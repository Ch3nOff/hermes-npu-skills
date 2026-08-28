"""
guard_model.py — rekonstruksi arsitektur model hasil fine-tuning
(lora_adapter.zip / checkpoint-2634).

CATATAN: rekonstruksi ini sudah DIVERIFIKASI identik dengan GuardHeadModel di
finetune_local.py — 488 kunci state_dict sama persis, 11/11 hyperparameter cocok
(lihat scripts/verify_arch_match.py). Untuk melatih dari awal pakai
finetune_local.py; modul ini untuk memuat checkpoint yang sudah ada.

Struktur state_dict yang ditemukan di model.safetensors:

    base.base_model.model.embed_tokens.weight                       -> Qwen2Model
    base.base_model.model.layers.{0..23}.self_attn.{q,k,v,o}_proj.base_layer.*
    base.base_model.model.layers.{0..23}.self_attn.{q,k,v,o}_proj.lora_{A,B}.default.weight
    base.base_model.model.layers.{0..23}.mlp.{gate,up,down}_proj.weight   (tanpa LoRA)
    base.base_model.model.norm.weight
    inj_head.{weight,bias}      [1, 896]
    shell_head.{weight,bias}    [1, 896]
    action_head.{weight,bias}   [4, 896]

Artinya:
  * backbone = Qwen2Model (AutoModel, TANPA lm_head)  -> keluaran last_hidden_state
  * dibungkus PeftModel(LoraModel(Qwen2Model)) dengan target q/k/v/o_proj, r=8
  * 3 linear head di atas hidden state (hidden_size=896)

Tidak ada lm_head -> ini classifier/regressor, BUKAN generative model.
Karena itu ia tidak bisa dilayani lewat openvino_genai.LLMPipeline.
"""

import torch
import torch.nn as nn

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_backbone(dtype=torch.float32, lora_r=8, lora_alpha=32):
    """Qwen2Model + LoRA (q,k,v,o) — struktur identik dengan checkpoint."""
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    backbone = AutoModel.from_pretrained(BASE_MODEL, dtype=dtype)
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=None,          # bukan CAUSAL_LM: tidak ada lm_head
    )
    return get_peft_model(backbone, cfg)


class GuardModel(nn.Module):
    """
    Multi-task guard: 2 regresi (injection, shell) + 1 klasifikasi (action, 4 kelas).

    pooling:
      'last_nonpad' -> hidden state token terakhir yang bukan padding
      'last'        -> hidden[:, -1]
      'mean'        -> rata-rata dengan mask
      'first'       -> hidden[:, 0]
    Strategi pooling yang benar tidak tersimpan di checkpoint, jadi dipilih
    secara empiris (lihat eval_guard.py).
    """

    def __init__(self, hidden_size=896, n_actions=4, pooling="last_nonpad", dtype=torch.float32):
        super().__init__()
        self.base = build_backbone(dtype=dtype)
        self.inj_head = nn.Linear(hidden_size, 1, dtype=dtype)
        self.shell_head = nn.Linear(hidden_size, 1, dtype=dtype)
        self.action_head = nn.Linear(hidden_size, n_actions, dtype=dtype)
        self.pooling = pooling

    def pool(self, hidden, attention_mask):
        if self.pooling == "last":
            return hidden[:, -1]
        if self.pooling == "first":
            return hidden[:, 0]
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        # last_nonpad
        idx = attention_mask.sum(1).long() - 1
        idx = idx.clamp(min=0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        pooled = self.pool(hidden, attention_mask)
        return {
            "injection": self.inj_head(pooled).squeeze(-1),
            "shell": self.shell_head(pooled).squeeze(-1),
            "action_logits": self.action_head(pooled),
        }


def load_checkpoint(path, pooling="last_nonpad", dtype=torch.float32, verbose=True):
    """Load model.safetensors hasil training ke GuardModel. Return (model, report)."""
    from safetensors.torch import load_file

    sd = load_file(path)
    model = GuardModel(pooling=pooling, dtype=dtype)
    sd = {k: v.to(dtype) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Qwen2.5 tie_word_embeddings: lm_head tidak ada di backbone AutoModel,
    # jadi tidak ada kunci yang wajar hilang selain rotary buffer non-persistent.
    report = {"missing": list(missing), "unexpected": list(unexpected)}
    if verbose:
        print(f"[load] tensors in file : {len(sd)}")
        print(f"[load] missing keys    : {len(missing)}")
        for k in list(missing)[:10]:
            print("        -", k)
        print(f"[load] unexpected keys : {len(unexpected)}")
        for k in list(unexpected)[:10]:
            print("        +", k)
    model.eval()
    return model, report
