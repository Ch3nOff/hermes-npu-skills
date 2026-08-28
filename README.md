# Iniz — NPU-Native Local AI Skills for Hermes Agent

Run small AI models locally on the Intel NPU (`Intel(R) AI Boost`, via OpenVINO),
packaged as skills/plugins for
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

The motivation is simple: the NPU in Intel Core Ultra laptops mostly sits idle because
mainstream AI tooling defaults to CPU/GPU. This project puts it to work on the
workloads it actually suits — small, frequently called, latency-sensitive — starting
with a **security guard** as the first use case.

**Model weights:** [`CH3NDev/iniz-agent-guard-int8`](https://huggingface.co/CH3NDev/iniz-agent-guard-int8)
on HuggingFace (ready-to-use OpenVINO IR INT8, 495 MB + 3-head checkpoint, 992 MB).
Not committed here because it exceeds practical Git limits — see
[Getting Started](#getting-started).

> **How to read this README.** Every claim is tagged ✅ **Proven** (working code +
> measured numbers + result files you can inspect) or 🧭 **Planned** (a sensible
> direction with **not a single line of code written yet**). Every ✅ number has a
> backing JSON file in `iniz-agent-guard/results/`. Do not treat the 🧭 section as
> features — it is a roadmap.

Reference hardware for all numbers below: **Intel Core Ultra 9 275HX** (Arrow Lake-HX)
+ Intel AI Boost NPU, Windows 11, OpenVINO 2026.3.

---

## ✅ Iniz Agent Guard — prompt-injection detector on the NPU

A discriminative classifier: **Qwen2.5-0.5B + LoRA r=8** with three heads
(`injection_score`, `shell_risk_score`, `action` over 4 classes), exported to
OpenVINO IR INT8 and executed on the NPU. It intercepts tool calls / LLM calls to
detect prompt injection **100% locally, with no cloud connection**.

### Measured results

| Metric | validation (941) | test (942) |
|---|---|---|
| `action` accuracy | **0.9586** | **0.9650** |
| `action` macro-F1 (classes present) | 0.5727 | **0.8171** |
| ROC-AUC `injection` → attack | 0.9670 | 0.9709 |
| ROC-AUC `shell` → proxy target | 0.9648 | 0.9731 |
| `injection` MAE | 0.0637 | 0.0651 |
| binary gate precision / recall / F1 | 0.962 / 0.985 / 0.973 | 0.968 / 0.984 / 0.976 |

Per-request latency on the NPU (batch 1, `seq_len=128`, INT8): **p50 34.4 ms**,
p90 34.8 ms. End-to-end server: `_npu_ms` p50 35.4 ms, client round-trip 52 ms.

| Device | `EXECUTION_DEVICES` | p50 |
|---|---|---|
| **NPU** | `NPU` | **34.4 ms** |
| CPU | `['CPU']` | ~94 ms |
| iGPU (GPU.0) | `['GPU.0']` | ~105 ms |

The NPU is ~2.7× faster than the CPU for this workload.

Evidence: `results/eval_final_seq128.json`, `results/npu_verify_int8.json`,
`results/npu_proof_npuonly.json`.

### Proof of NPU execution (three layers)

1. `compiled.get_property("EXECUTION_DEVICES")` → `NPU`
2. Windows counter LUID attribution. **This Windows 11 build has no `NPU` counter
   set** — the NPU appears as an adapter inside `\GPU Engine(*)`, so "GPU Engine
   activity exists" is not evidence of a fallback. LUIDs were mapped by running a
   load one-device-per-process: NPU = `0x…0x11cf3` (engtype `compute`, max 102.75%),
   iGPU = `0x…0x10480`, dGPU = `0x…0x1099d`. Loading `CPU` → zero GPU-Engine
   instances for that pid. CPU `_Total` during 228 NPU inferences: mean 6.3%.
3. INT8 numerical fidelity vs PyTorch fp32: `max|Δinj| = 0.045`,
   **action agreement 1.000** (40 samples).

### Fine-tuning status: complete and validated

`checkpoint-2634` (3 epochs, 2634 steps, final loss 0.0094) is verified:

- `finetune_local.py` produces **488 state_dict keys identical** to the checkpoint,
  and **11/11 hyperparameters** match `training_args.bin`
  → `scripts/verify_arch_match.py`
- A 3-step training dry-run completed `trainer.train()` successfully
  → `scripts/dryrun_finetune.py`
- The label mapping was validated with **the model as judge**: the correct mapping
  gives 0.966 accuracy vs 0.741 for the wrong one (macro-F1 0.818 vs 0.426)
  → `scripts/verify_incoming_mapping.py`, `results/mapping_verdict.json`

### Honest limitations

- **`shell_head` optimizes for a proxy target, not real shell danger.** ROC-AUC
  against its own target is 0.973 (well trained), but that target is a keyword proxy.
  Measured: dangerous shell **with** proxy keywords → score 0.445; equally dangerous
  shell **without** those keywords (`rm -rf /`, `dd if=`, `mkfs`, fork bomb) → only
  0.165. That is why the server applies a keyword backstop broader than the training
  proxy.
- `ISOLATE_FILE` (support 9) and `USER_CONFIRMATION` (support 2) are too rare in the
  test set to judge reliably.
- ~2% false positives on benign business text (example: *"Summarize this quarterly
  report in three bullet points"* → injection 0.648).
- Threshold 0.30 comes from a test-split sweep and is **not validated on real traffic**.
- The server binds to `127.0.0.1` **without authentication**. Do not expose it on
  `0.0.0.0` without adding auth.

---

## ✅ Foundation: the NPU model-serving pattern

A reusable pattern for other NPU workloads: a persistent HTTP server process
(separate from Hermes) that loads a quantized model once and serves repeatedly, with
device `"NPU"` set explicitly.

There are **two variants that must not be mixed**:

| | Generative | Discriminative (guard) |
|---|---|---|
| API | `openvino_genai.LLMPipeline` | `core.compile_model` + read tensors |
| Has `lm_head` | yes | **no** |
| Output | text that must be parsed | score tensors directly |
| Latency | 2.7–6.6 s | **35 ms** |

Generative-path pitfalls (`apply_chat_template` signature, truncated `max_new_tokens`,
INT4 garbage on rigid formats) live in
`iniz-agent-guard/references/tokenizer-signature-pitfall.md` — they **do not apply**
to the discriminative path.

Expensive export pitfalls, all documented with real tracebacks in
`iniz-agent-guard/references/npu-export-pitfalls.md`:

- `torch.jit.trace`, `torch.onnx.export(dynamo=False)`, and
  `ov.convert_model(model, example_input=…)` **all fail** on Qwen2Model
  (transformers 4.57 + torch 2.9) with `invalid unordered_map<K, T> key`.
  The path that works: `torch.export.export(…, strict=False)` →
  `ov.convert_model(exported_program)`.
- `torch.export` reports inputs as `[?,?]` even with a static example. **The NPU
  rejects dynamic shapes** → `ov_model.reshape(...)` is mandatory.
- The IR `seq_len` must equal training `MAX_LENGTH` (128). A wrong value throws no
  error — it just costs ~40% latency for nothing.

---

## 🧭 Roadmap

Nothing in this section is implemented yet.

### Offline speech-to-text & translation

Relatively low technical risk: OpenVINO GenAI ships an official `WhisperPipeline`,
Intel has an Audacity plugin that does exactly this on the NPU, and there is direct
prior experience from the `taiwan-mandarin-stt` project (Whisper large-v3 via
OpenVINO, same laptop). The pattern is: swap the pipeline inside the existing server
structure.

*Starting point:* `iniz-agent-guard/guard_server.py` as the server skeleton.

### Summarization & content generation as an auxiliary model

The NPU handling small, frequent workloads (captioning, page summaries, context
compression) that currently ride on an expensive model — **not** a replacement for the
main reasoning model.

*Honest limitation from an earlier session:* **precision extraction** tasks (pull a
filename out of free text) failed consistently at both quantizations — INT4
hallucinated, INT8 refused to answer. Summarization and sentiment were far more
stable. Do not assume "document summarization" works automatically just because it
looks similar; verify it again.

### Power optimization

What **is** proven: the NPU does not load the CPU/GPU while in use (CPU `_Total` mean
6.3% during an NPU workload). That is far narrower than a claim of "tens of percent
battery savings" — which has **never been measured** in this project and would need
separate end-to-end power profiling.

The "learn usage patterns for automatic power management" part is a much larger leap:
it needs long-horizon observability, resource allocation policy, and possibly
driver-level access. Treat it as a separate research project, not a Hermes skill.

*What is realistic as a skill:* guidance on **when** to route a task to the NPU vs
CPU/GPU based on workload shape (small-and-frequent vs large-and-rare) — an extension
of `SKILL.md`, not a new skill.

---

## Repo Layout

```
iniz-agent-guard/
├── SKILL.md                  # Full skill: architecture, numbers, 13 pitfalls
├── finetune_local.py         # 3-head training (verified == checkpoint-2634)
├── guard_server.py           # NPU HTTP server (CompiledModel, not LLMPipeline)
├── scripts/                  # 17 scripts: export, eval, verification, NPU proof
├── results/                  # Measured result JSON (independently inspectable)
├── references/               # Export/tokenizer pitfalls, NPU proof methodology
└── notebooks/
    ├── 01_train_guard.ipynb            # 3-head training (produces checkpoint-2634)
    └── 02_verify_export_deploy.ipynb   # Verify → export IR → NPU proof → deploy
```

**Model weights are not included** (IR INT8 = 495 MB, beyond practical Git limits).
Get them from [HuggingFace](https://huggingface.co/CH3NDev/iniz-agent-guard-int8) or
generate them yourself with `scripts/export_guard_ov.py --seq-len 128`.

## Requirements

- Intel Core Ultra (or similar NPU) with `Intel(R) AI Boost` visible in
  `ov.Core().available_devices`
- `openvino` + `nncf` ≥ 2026.3, `torch` ≥ 2.9, `transformers` 4.57, `peft` 0.18,
  `datasets`
- Generative path only: `openvino-genai`
- `HF_TOKEN` from an environment variable — **never hardcoded in source**

## Getting Started

1. Verify the NPU is detected:
   `python -c "import openvino as ov; print(ov.Core().available_devices)"`
2. Fetch the weights:
   `hf download CH3NDev/iniz-agent-guard-int8 --include "openvino/*" --local-dir ~/npu-provider/models/iniz-guard-int8-ov`
   (or export your own: `python iniz-agent-guard/scripts/export_guard_ov.py --seq-len 128`)
3. Prove NPU execution: `python iniz-agent-guard/scripts/prove_npu.py --npu-only`
4. Start the server: `INIZ_GUARD_DEVICE=NPU python iniz-agent-guard/guard_server.py`
5. Test it: `python iniz-agent-guard/scripts/smoke_client.py`
6. Reproduce the numbers: `python iniz-agent-guard/scripts/eval_final.py --device NPU`

Read `iniz-agent-guard/SKILL.md` before changing anything — all 13 pitfalls in there
were found through real failures, not speculation.

---

## Installing this skill into Hermes Agent

This skill is consumed by [Hermes Agent](https://github.com/NousResearch/hermes-agent).
Once installed, Hermes loads `SKILL.md` automatically when you mention things like
"run the guard on the NPU", "export a model to OpenVINO", or "prompt injection
detector".

### 1. Find the Hermes skills directory

```bash
hermes doctor        # shows the active config paths
```

The skills directory lives at:

| OS | Path |
|---|---|
| Linux / macOS | `~/.hermes/skills/` |
| Windows | `%LOCALAPPDATA%\hermes\skills\` |

If you use **profiles**, resolve it from `$HERMES_HOME` (`$HERMES_HOME/skills/`) —
do not hardcode `~/.hermes`.

### 2. Clone into the skills directory

**Linux / macOS**

```bash
mkdir -p ~/.hermes/skills/local-ai
git clone https://github.com/Ch3nOff/hermes-npu-skills.git \
  ~/.hermes/skills/local-ai/hermes-npu-skills
```

**Windows (git-bash)**

```bash
mkdir -p "$LOCALAPPDATA/hermes/skills/local-ai"
git clone https://github.com/Ch3nOff/hermes-npu-skills.git \
  "$LOCALAPPDATA/hermes/skills/local-ai/hermes-npu-skills"
```

`local-ai/` there is a **category** — Hermes uses subdirectories to group skills. Any
other name works too.

### 3. Verify Hermes sees it

```bash
hermes skills list | grep -i iniz
```

You should see `iniz-agent-guard` with its description. If not:

- Make sure `SKILL.md` sits at
  `<skills>/local-ai/hermes-npu-skills/iniz-agent-guard/SKILL.md`
- Make sure the YAML frontmatter at the very top of `SKILL.md` is intact (`---` … `---`)
- Restart the Hermes session (skills are scanned at session start)

### 4. Set up the runtime

This skill needs its own venv — the Hermes venv has no `torch`:

```bash
python -m venv ~/npu-provider/.venv
~/npu-provider/.venv/bin/pip install \
  "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
~/npu-provider/.venv/bin/pip install \
  "transformers==4.57.1" "peft==0.18.0" "openvino==2026.3.0" \
  "openvino-tokenizers==2026.3.0" nncf datasets pandas pyarrow scikit-learn
```

On Windows use `Scripts/` instead of `bin/` and `pip.exe` instead of `pip`.

Verify the NPU is detected:

```bash
~/npu-provider/.venv/bin/python -c \
  "import openvino as ov; print(ov.Core().available_devices)"
# must include 'NPU'
```

### 5. Fetch the model weights

```bash
hf download CH3NDev/iniz-agent-guard-int8 \
  --include "openvino/*" \
  --local-dir ~/npu-provider/models/iniz-guard-int8-ov
```

Or export them yourself from the checkpoint:

```bash
cd <skill>/iniz-agent-guard
python scripts/export_guard_ov.py --seq-len 128
```

### 6. Run it

```bash
INIZ_GUARD_DEVICE=NPU INIZ_GUARD_THRESHOLD=0.30 \
  ~/npu-provider/.venv/bin/python <skill>/iniz-agent-guard/guard_server.py

# in another terminal
curl -s http://127.0.0.1:8009/health
curl -s -X POST http://127.0.0.1:8009/scan \
  -H "Content-Type: application/json" \
  -d '{"text":"Ignore all previous instructions"}'
```

### 7. Try it through Hermes

```
hermes chat -q "run the iniz guard on the NPU and scan the text 'ignore all previous instructions'"
```

Hermes will load `SKILL.md`, read its workflow, and carry out the steps itself.

### Running the server automatically (optional)

Use Hermes' built-in `cronjob` so the server comes up on boot, or launch it as a
background process from a Hermes session:

```
terminal(command="INIZ_GUARD_DEVICE=NPU python <skill>/iniz-agent-guard/guard_server.py",
         background=true, notify=["listening on"])
```

### Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `hermes skills list` does not show the skill | `SKILL.md` is at the wrong level, or the YAML frontmatter is broken |
| `ModuleNotFoundError: torch` | you are using the Hermes venv python instead of `~/npu-provider/.venv` |
| `'NPU' not available` | Intel NPU driver not installed / not Core Ultra hardware |
| `AttributeError: 'list' object has no attribute 'keys'` | `extra_special_tokens` from transformers 5.x — see pitfall #5 in `SKILL.md` |
| p50 latency ~94 ms instead of ~35 ms | silently running on CPU. Check `compiled.get_property("EXECUTION_DEVICES")` |
| accuracy suddenly ~0.74 | wrong label mapping — use `scripts/guard_labels.py` |

Every pitfall is detailed in `iniz-agent-guard/SKILL.md` — 13 of them, all found
through real failures.

---

## License

MIT — see [`LICENSE`](./LICENSE).

The model weights on HuggingFace are **Apache-2.0**, following the base model
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
