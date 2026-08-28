"""
guard_model.py — reconstruction of the fine-tuned model architecture
(lora_adapter.zip / checkpoint-2634).

NOTE: this reconstruction has been VERIFIED identical to GuardHeadModel in
finetune_local.py — all 488 state_dict keys match exactly, 11/11 hyperparameters
agree (see scripts/verify_arch_match.py). To train from scratch use
finetune_local.py; this module is for loading the existing checkpoint.

state_dict structure found in model.safetensors:

    base.base_model.model.embed_tokens.weight                       -> Qwen2Model
    base.base_model.model.layers.{0..23}.self_attn.{q,k,v,o}_proj.base_layer.*
    base.base_model.model.layers.{0..23}.self_attn.{q,k,v,o}_proj.lora_{A,B}.default.weight
    base.base_model.model.layers.{0..23}.mlp.{gate,up,down}_proj.weight   (no LoRA)
    base.base_model.model.norm.weight
    inj_head.{weight,bias}      [1, 896]
    shell_head.{weight,bias}    [1, 896]
    action_head.{weight,bias}   [4, 896]

Which means:
  * backbone = Qwen2Model (AutoModel, WITHOUT lm_head)  -> emits last_hidden_state
  * wrapped in PeftModel(LoraModel(Qwen2Model)) targeting q/k/v/o_proj, r=8
  * 3 linear heads on top of the hidden state (hidden_size=896)

There is no lm_head -> this is a classifier/regressor, NOT a generative model.
That is why it cannot be served through openvino_genai.LLMPipeline.
"""

import torch
import torch.nn as nn

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_backbone(dtype=torch.float32, lora_r=8, lora_alpha=32):
    """Qwen2Model + LoRA (q,k,v,o) — structure identical to the checkpoint."""
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    backbone = AutoModel.from_pretrained(BASE_MODEL, dtype=dtype)
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=None,          # not CAUSAL_LM: there is no lm_head
    )
    return get_peft_model(backbone, cfg)


class GuardModel(nn.Module):
    """
    Multi-task guard: 2 regressions (injection, shell) + 1 classification
    (action, 4 classes).

    pooling:
      'last_nonpad' -> hidden state of the last non-padding token
      'last'        -> hidden[:, -1]
      'mean'        -> mask-weighted average
      'first'       -> hidden[:, 0]
    The correct pooling strategy is not stored in the checkpoint, so it is chosen
    empirically (see eval_guard.py).
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
    """Load the trained model.safetensors into GuardModel. Returns (model, report)."""
    from safetensors.torch import load_file

    sd = load_file(path)
    model = GuardModel(pooling=pooling, dtype=dtype)
    sd = {k: v.to(dtype) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Qwen2.5 tie_word_embeddings: lm_head is absent from the AutoModel backbone,
    # so no key should legitimately be missing apart from non-persistent rotary
    # buffers.
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
