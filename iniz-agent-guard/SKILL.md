---
tags: [mlops, openvino, npu, quantization, security, classifier, local-ai, offline]
---

# Skill: Iniz Agent Guard — Fine-tuned Multi-Head Classifier on Intel NPU

**Trigger:** When the task involves the Iniz Agent Guard model (prompt-injection /
shell-risk scoring), or more generally exporting a **fine-tuned discriminative**
transformer (backbone + regression/classification heads) to OpenVINO IR and serving
it on an Intel Core Ultra NPU.

**Behavior:** Reconstruct the head architecture from the checkpoint, validate the
label mapping against the model itself, evaluate on held-out data, export via
`torch.export` → OpenVINO IR with **static shapes**, prove the NPU executes it, and
serve through a direct `CompiledModel` call — never `openvino_genai.LLMPipeline`.

**Environment:** `openvino` + `nncf` ≥ 2026.3, `torch` ≥ 2.9, `transformers` 4.57,
`peft` 0.18, `datasets`. The NPU must appear in `ov.Core().available_devices`.

---

## Scope: TWO different patterns, do not mix them

This repo contains two paths that are easy to confuse:

| | Generative (predecessor) | **Discriminative (current production)** |
|---|---|---|
| Model | plain Qwen2.5-0.5B-Instruct | fine-tuned checkpoint-2634 |
| Has `lm_head`? | Yes | **No** |
| Serving API | `openvino_genai.LLMPipeline` | `core.compile_model` + read tensors |
| Output | JSON text that must be parsed | 3 score tensors directly |
| `max_new_tokens` | relevant | **concept does not exist here** |
| Latency | 2.7–6.6 s | **35 ms** |
| File | `~/npu-provider/scripts/npu_server.py` (outside repo) | `guard_server.py` |

Pitfalls specific to the generative path (`apply_chat_template` signature, truncated
`max_new_tokens`, INT4 garbage on rigid formats) live in
`references/tokenizer-signature-pitfall.md` and **do not apply** to the
discriminative path. Do not carry them into the guard server.

## Model Architecture (verified 100% against the checkpoint)

Checkpoint: `lora_adapter.zip` → `checkpoint-2634` (3 epochs, 2634 steps, final
loss ≈ 0.0094).

```
Qwen2Model (AutoModel, NO lm_head) — 24 layers, hidden 896, vocab 151936
  └─ LoRA r=8 α=32 on q_proj,k_proj,v_proj,o_proj  (192 lora_A/lora_B tensors)
     task_type="FEATURE_EXTRACTION"
  └─ pooling: hidden state of the last non-pad token
       ├─ inj_head    Linear(896 → 1)   injection score (regression)
       ├─ shell_head  Linear(896 → 1)   shell-risk score (regression, PROXY target)
       └─ action_head Linear(896 → 4)   action classification
```

488 tensors, **0 missing / 0 unexpected** on `load_state_dict`.

`finetune_local.py` in this repo is verified to be the correct script: the
`GuardHeadModel` it builds produces **488 state_dict keys identical** to the
checkpoint, and **11/11 hyperparameters** match `training_args.bin` (batch 1,
grad accum 16, 3 epochs, lr 2e-4, cosine, warmup 0.05, adamw_torch).
A 3-step CPU dry-run completed `trainer.train()` successfully
(`scripts/dryrun_finetune.py`, `scripts/verify_arch_match.py`).

`MAX_LENGTH = 128` in `finetune_local.py` is the **correct seq_len** — the production
IR must be exported with `--seq-len 128`, not 192. See the seq_len section below.

Actions: `["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]` (indices 0–3).

## Label Mapping — never guess, always validate

The mapping lives in `scripts/guard_labels.py`. It is the **single source of truth**:

```python
SEVERITY_MAP = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}
OVERRIDE_CATEGORIES    = {direct_injection, jailbreak, persona_replacement,
                          many_shot, crescendo}        -> PAUSE_AGENTS
OBFUSCATION_CATEGORIES = {encoding_obfuscation, token_smuggling,
                          indirect_injection, context_overflow}  -> ISOLATE_FILE
EXTRACTION_CATEGORIES  = {system_extraction, prompt_leaking}      -> USER_CONFIRMATION
other label==1 -> PAUSE_AGENTS ;  label==0 -> PASS
shell = 0.0 (benign) / 0.6 (keyword proxy hit) / 0.1 (attack without keyword)
```

This mapping was **proven** by making the model the judge
(`scripts/verify_incoming_mapping.py`) — not trusted from a document:

| Metric | wrong mapping (fake categories) | **correct mapping** | delta |
|---|---|---|---|
| action accuracy (test) | 0.7410 | **0.9660** | +0.225 |
| action macro-F1 (test) | 0.4264 | **0.8176** | +0.391 |
| shell MAE (test) | 0.0963 | **0.0465** | −0.050 |

If action metrics suddenly collapse to ~0.74 accuracy and ~0.43 macro-F1, the
mapping is almost certainly wrong, not the model.

Two categories in the mapping **do not exist** in the dataset (`context_overflow`,
`prompt_leaking`) — leave them in, so the mapping stays identical to what training used.

## Measured Numbers (not estimates)

Machine: Core Ultra 9 275HX + Intel AI Boost NPU, OpenVINO 2026.3, IR INT8_SYM 495 MB.

### seq_len 128 vs 192 — 128 is correct, and faster

`MAX_LENGTH=128` was used during training, so the production IR must be `[1,128]`.

| seq_len | p50 | p90 | compile | action acc (test) | macro-F1 | inj MAE |
|---|---|---|---|---|---|---|
| **128** (correct) | **34.4 ms** | 34.8 ms | 11.4 s | 0.9650 | 0.8171 | 0.0651 |
| 192 (wrong) | 48.4 ms | 48.7 ms | 3.5 s | 0.9660 | 0.8176 | 0.0643 |

Quality is effectively identical (difference <0.001) but **seq_len 128 is ~29% faster**.
Use 128. The seq_len 192 IR is kept as `models/iniz-guard-*-ov-seq192` for comparison.

### Latency per device (batch 1, seq_len 128)

| Device | EXECUTION_DEVICES | p50 |
|---|---|---|
| **NPU** | `NPU` | **34.4 ms** |
| CPU | `['CPU']` | ~94 ms (measured at seq 192) |
| GPU.0 iGPU | `['GPU.0']` | 86.6 ms (seq 128; acc 0.9575, one sample off NPU) |

### Quality (IR INT8 @ NPU, seq_len 128, validated mapping)

| Metric | validation (941) | test (942) |
|---|---|---|
| action accuracy | **0.9586** | **0.9650** |
| macro-F1 (classes present) | 0.5727 | **0.8171** |
| ROC-AUC injection → attack | 0.9670 | 0.9709 |
| ROC-AUC shell → proxy target | 0.9648 | **0.9731** |
| injection MAE | 0.0637 | 0.0651 |
| shell MAE | 0.0479 | 0.0462 |
| binary gate P / R / F1 | 0.962 / 0.985 / 0.973 | 0.968 / 0.984 / 0.976 |

Per-class (test): PASS F1 0.966 (n=390) · PAUSE_AGENTS F1 0.971 (n=541) ·
ISOLATE_FILE F1 0.667 (n=9) · USER_CONFIRMATION F1 0.667 (n=2).
The last two have very small support — their numbers are not a reliable indicator.

INT8 fidelity vs PyTorch fp32: `max|Δinj| = 0.045`, `max|Δshell| = 0.036`,
**action agreement 1.000** (40 samples).

### `injection_score` threshold
F1 peaks at **0.25–0.30** (F1 0.973–0.976). Above 0.5 recall falls off a cliff
(th=0.6 → recall 0.39). Server default: `0.30`.

## `shell_head`: well trained, BUT its target is a proxy

My initial conclusion ("shell_head undertrained, only 12 `code_execution` samples")
was **wrong** — an artifact of using the wrong label mapping. With the correct
mapping: ROC-AUC shell_head → proxy-hit = **0.973**. This head is well trained.

The real problem is subtler: **the target is a keyword proxy**, not actual shell
danger. So shell_head is a learned keyword detector, inheriting the weaknesses of
that keyword list. Measured in `scripts/diag_shell_head2.py`:

| Test group | shell_mean |
|---|---|
| dangerous shell **with** proxy keywords (`bash`, `eval`, `os.system`, …) | **0.445** |
| dangerous shell **without** proxy keywords (`rm -rf /`, `dd if=`, `mkfs`, fork bomb) | **0.165** |
| benign text that happens to contain a keyword ("What is eval() used for?") | −0.007 |
| pure injection with no shell element | 0.059 |

The 0.28 gap between the first two groups proves it. Practical consequence:
**the server's keyword backstop MUST use a broader list than the training proxy** —
`rm -rf`, `dd if=`, `mkfs`, `chmod 777`, `> /dev/tcp`, fork bombs, etc.
The good news: group 3 shows the head is not naive — benign text containing the words
`eval`/`bash` still scores ~0.

## Pitfalls (all verified empirically on this machine)

1. **A checkpoint without `lm_head` cannot be used with `LLMPipeline`.** Inspect the
   `model.safetensors` keys first. If there are `*_head.weight` entries and no
   `lm_head`, it is a classifier: serve it via `core.compile_model(...)` and read the
   output tensors.
2. **`torch.jit.trace` and `torch.onnx.export(dynamo=False)` ALWAYS fail on
   Qwen2Model** (transformers 4.57 + torch 2.9):
   `RuntimeError: invalid unordered_map<K, T> key`, originating from a walrus operator
   in `masking_utils`. `ov.convert_model(model, example_input=...)` fails too, because
   internally it goes `TorchScriptPythonDecoder` → `jit.trace`.
   The path that WORKS: **`torch.export.export(model, args, strict=False)`** followed
   by `ov.convert_model(exported_program)`.
3. **`attn_implementation` must be `eager` before export**
   (`config._attn_implementation = "eager"`, `use_cache = False`).
4. **`torch.export` produces DYNAMIC dims (`?,?`) even with a static example input.**
   The NPU rejects dynamic shapes. You must call
   `ov_model.reshape({0: PartialShape([1,S]), 1: PartialShape([1,S])})` before saving.
5. **`extra_special_tokens` list vs dict.** Checkpoints written by transformers 5.x
   store it as a **list**; transformers 4.x blows up with
   `AttributeError: 'list' object has no attribute 'keys'`. Normalize to
   `extra_special_tokens = {}` + `additional_special_tokens = <list>`.
6. **IR `seq_len` must equal training `MAX_LENGTH`.** A wrong value throws no error at
   all — it just costs latency for no quality gain (192 vs 128: +40% latency, same
   quality).
7. **Windows reports the NPU INSIDE the `\GPU Engine(*)` counter set.** There is no
   `NPU` counter set on this Windows 11 build. "GPU Engine activity exists" is **not**
   evidence of a fallback. Map the LUIDs first (this machine):
   | LUID | Device |
   |---|---|
   | `0x00000000_0x00011cf3` | **NPU — Intel(R) AI Boost** (engtype `compute`, max 102.75%) |
   | `0x00000000_0x00010480` | iGPU — Intel(R) Graphics |
   | `0x00000000_0x0001099d` | dGPU — NVIDIA RTX 5060 |
   Loading device `CPU` → **zero** GPU-Engine instances for that pid.
   Re-run `scripts/luid_attribution.py <device>` on any other machine.
8. **`compiled.get_property("EXECUTION_DEVICES")` is the cheapest evidence.** The NPU
   returns the string `NPU`; CPU/GPU return a list.
9. **A wrong label mapping masquerades as a bad model.** 0.74 accuracy / 0.43 macro-F1
   vs 0.97 / 0.82 — purely because of the mapping. Validate the mapping with the model
   as judge before concluding anything about model quality.
10. **`shell_head` optimizes for a keyword proxy, not real danger** (see section above).
11. **Linear regression heads without a sigmoid can leave the [0,1] range** (small
    negative values on benign input). Clamp on the server side.
12. **The multi-device benchmark phase contaminates counter evidence.** Compiling to
    `GPU.0` makes the process hold a GPU context for its entire lifetime. Run the
    counter proof in a separate process: `prove_npu.py --npu-only`.
13. **`gradient_checkpointing=True` + LoRA on CPU** slows things down for no benefit and
    requires gradients on the input embedding. Disable it for CPU dry-runs.

## Workflow

Scripts live in `scripts/`; run them from `~/npu-provider/work/`.

```bash
cd ~/npu-provider/work
V=../.venv/Scripts/python.exe   # separate venv: the default python is the Hermes venv (no torch)

# 1. inspect the checkpoint: any *_head.weight? any lm_head?
# 2. reconstruct + smoke test      -> guard_model.py (0 missing / 0 unexpected)
# 3. verify the training script     -> $V verify_arch_match.py
# 4. 3-step training dry-run        -> $V dryrun_finetune.py
# 5. validate the label mapping     -> $V verify_incoming_mapping.py
# 6. pick pooling empirically       -> $V eval_guard.py --pooling all --limit 240
# 7. diagnose the heads             -> $V diag_shell_head2.py
# 8. export static IR + INT8        -> $V export_guard_ov.py --seq-len 128
# 9. NPU proof + LUID attribution   -> $V prove_npu.py --npu-only
#                                      $V luid_attribution.py NPU|GPU.0|CPU
# 10. final production eval         -> $V eval_final.py --device NPU
# 11. serve + HTTP smoke test
INIZ_GUARD_DEVICE=NPU INIZ_GUARD_THRESHOLD=0.30 $V ../guard_server.py
$V smoke_client.py
```

## Server

`guard_server.py` — direct `CompiledModel`, no text generation.

```
GET  /health -> device, seq_len, threshold, compile_s, warmup_ms, latency_p50_ms
POST /scan   {"text": "...", "context": "..."}
  -> injection_score, shell_risk_score, action, confidence, action_probs,
     shell_keyword_hits, _raw_injection, _raw_shell, _npu_ms, _device, _model
```

Env: `INIZ_GUARD_MODEL`, `INIZ_GUARD_DEVICE` (NPU), `INIZ_GUARD_PORT` (8009),
`INIZ_GUARD_THRESHOLD` (0.30).

Measured (seq_len 128): `_npu_ms` p50 **35.4 ms**, client round-trip p50 52 ms.
Smoke test 9/10 matched coarse expectations; 1 false positive
(*"Summarize this quarterly report in three bullet points"* → inj 0.648).

**Security:** binds to `127.0.0.1` **without authentication**. Do not move it to
`0.0.0.0` without adding auth.

## Validation Criteria

An NPU deployment is correct only when **all** of these hold:

- (A) `compiled.get_property("EXECUTION_DEVICES")` == `NPU`
- (B) GPU-Engine counters for the server pid touch only the NPU LUID, CPU `_Total` < 15%
- (C) p50 ≈ 35 ms at `seq_len=128`. Far above 90 ms → likely CPU fallback
- (D) IR input shape is static `[1,128]`, not `[?,?]`
- (E) IR vs PyTorch fp32 action agreement == 1.000 over ≥40 samples
- (F) action accuracy ≥ 0.95 using `guard_labels.py`. If it is ~0.74 → wrong mapping

## What is still open

- `shell_head` optimizes for a keyword proxy; needs a real shell-command dataset to retrain.
- `ISOLATE_FILE` (support 9) and `USER_CONFIRMATION` (support 2) are too rare in the
  test set to judge reliably — not necessarily broken, just unproven.
- False positives on benign business text (~2% FP rate).
- Threshold 0.30 comes from a test-split sweep, not validated on real traffic.
- Automatic LoRA merge + export for the 3 custom heads is not in `finetune_local.py`
  (must go through `export_guard_ov.py` separately).

## See Also
- `scripts/` — guard_labels.py, guard_model.py, export_guard_ov.py, eval_final.py,
  verify_incoming_mapping.py, verify_arch_match.py, dryrun_finetune.py,
  diag_shell_head2.py, prove_npu.py, luid_attribution.py, smoke_client.py,
  upload_hf.py, verify_hf_download.py
- `notebooks/01_train_guard.ipynb` — 3-head training (produces checkpoint-2634)
- `notebooks/02_verify_export_deploy.ipynb` — verify → export → NPU proof → deploy
- `results/` — eval_final.json, eval_final_seq128.json, eval_final_gpu0.json,
  mapping_verdict.json,
  npu_verify_int8.json, npu_proof_npuonly.json, luid_{NPU,GPU_0,CPU}.json
- `references/npu-export-pitfalls.md` — real tracebacks for every failing export path
- `references/tokenizer-signature-pitfall.md` — GENERATIVE-path pitfall (not the guard)
- `references/npu-gpu-util-proof.md` — Task Manager evidence methodology (generative path)

## Model Weights

| Artifact | Location | Size |
|---|---|---|
| OpenVINO IR INT8 (NPU-ready, `seq_len=128`) | [`CH3NDev/iniz-agent-guard-int8`](https://huggingface.co/CH3NDev/iniz-agent-guard-int8) → `openvino/` | 495 MB |
| 3-head checkpoint + tokenizer (retrain / re-export) | same repo → `checkpoint/` | 992 MB |

```bash
# ready-to-use IR
hf download CH3NDev/iniz-agent-guard-int8 --include "openvino/*" \
  --local-dir ~/npu-provider/models/iniz-guard-int8-ov

# raw checkpoint (for re-export / retraining)
hf download CH3NDev/iniz-agent-guard-int8 --include "checkpoint/*" \
  --local-dir ~/npu-provider/work/ckpt
```

The HF artifacts are **verified intact**: re-downloaded from the Hub, compiled on the
NPU (`EXECUTION_DEVICES = NPU`), and inference output matched the local model with
`max|delta| = 0.0000000000` and identical actions — see `scripts/verify_hf_download.py`.
