# Iniz — NPU-Native Local AI Skills for Hermes Agent

Run small AI models locally on the Intel NPU (`Intel(R) AI Boost`, via OpenVINO),
packaged as skills/plugins for
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

The motivation is simple: the NPU in Intel Core Ultra laptops mostly sits idle because
mainstream AI tooling defaults to CPU/GPU. This project puts it to work on the
workloads it actually suits — small, frequently called, latency-sensitive — starting
with a **security guard** as the first use case, plus **offline speech-to-text**, a
**small auxiliary LLM**, and **real energy measurement** after it. One of those workloads
turned out to be a poor fit for the NPU, and that result is reported as prominently as
the wins.

**Model weights:** [`CH3NDev/iniz-agent-guard-int8`](https://huggingface.co/CH3NDev/iniz-agent-guard-int8)
on HuggingFace (ready-to-use OpenVINO IR INT8, 495 MB + 3-head checkpoint, 992 MB).
Not committed here because it exceeds practical Git limits — see
[Getting Started](#getting-started).

> **How to read this README.** Every claim is tagged ✅ **Proven** (working code +
> measured numbers + result files you can inspect) or 🧭 **Planned** (a sensible
> direction with **not a single line of code written yet**). Every ✅ number has a
> backing JSON file in `iniz-agent-guard/results/`, `iniz-stt/results/`,
> `iniz-aux/results/`, or `iniz-power/results/`. Do not treat the 🧭 section as
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

## ✅ Iniz STT — offline speech-to-text on the NPU

`whisper-base` exported to OpenVINO IR INT8 (81 MB) and served on the NPU through
`openvino_genai.WhisperPipeline`. Fully offline — audio never leaves the machine.
English uses `whisper-base`; **Traditional Chinese needs `whisper-medium`** (748 MB) +
opencc post-processing — `base` is unusable there and `large-v3` does not run on this
NPU at all.

### Measured results

8 LibriSpeech clips, 86.34 s of audio, ground-truth transcripts, WER on normalized
text (English, `whisper-base`):

| Device | compile | p50 latency | p90 | RTF (median) | WER | CPU load |
|---|---|---|---|---|---|---|
| **NPU** | 0.66 s | 169.9 ms | 232.5 ms | 0.0196 | **0.0900** | **7.5 % mean / 10.6 % max** |
| **CPU** | 0.58 s | 167.8 ms | 207.7 ms | 0.0189 | **0.0900** | 27.0 % mean / 49.2 % max |

RTF ≈ 0.02 means both run about **50× faster than real time**. Transcripts are
**8/8 identical** between devices.

### The honest headline: NPU is not faster here (at `base`)

For `whisper-base` on English, the NPU and CPU are within 1 % on latency and identical
on accuracy. **The reason to use the NPU is CPU offload, not speed** — 7.5 % versus
27.0 % mean CPU while transcribing. This flips at `whisper-small` (see Chinese below),
where the NPU wins by 1.33×.

### Proof of NPU execution

`WhisperPipeline` does not expose `EXECUTION_DEVICES` (it wraps encoder + decoder),
and two near-identical latency columns are exactly what a silent fallback looks like.
So device attribution was proven with per-pid GPU-Engine counters instead:

| Requested | NPU adapter `0x00000000_0x00011cf3` | iGPU | CPU `_Total` | Verdict |
|---|---|---|---|---|
| `NPU` | **active, max 93.79 %, mean 89.06 %** | idle | 7.5 % | genuinely NPU |
| `CPU` | idle | idle | 27.0 % / 49.2 % max | genuinely CPU |

93 transcriptions in ~13 s on the NPU. Evidence in
`iniz-stt/results/whisper_proof_{NPU,CPU}.json`.

### Traditional Chinese (Taiwanese Mandarin)

10 Common Voice 16.1 zh-TW clips (39.7 s), references verified Traditional
(`simp=0 / trad=11`) before use. CER, not WER — Chinese has no word spacing.

| Model | IR | NPU p50 | CPU p50 | CER_raw | CER_trad | Emitted script |
|---|---|---|---|---|---|---|
| `whisper-base` | 81 MB | 101.8 ms | 118.4 ms | 0.5135 | 0.4865 | MIXED |
| `whisper-small` | 245 MB | 297.1 ms | 394.1 ms | 0.2027 | 0.1622 | MIXED |
| **`whisper-medium`** | 748 MB | **849.7 ms** | 1182.4 ms | **0.1081** | **0.0811** | TRADITIONAL |
| `whisper-large-v3` | 1.5 GB | **FAILS** | 2136.1 ms | 0.2703 | 0.0811 | SIMPLIFIED |

**`whisper-base` is unusable for Chinese** — half the characters wrong, two clips
returned romanized nonsense (`土地認養案例` → `thoody learn yang andi`).
**`whisper-medium` is the pick:** `CER_trad` 0.0811, and it is the largest Whisper that
runs on this NPU at all.

**`large-v3` compiles on the NPU (208.97 s) then fails every inference** with
`ZE_RESULT_ERROR_UNINITIALIZED — driver is not initialized`, reproduced twice. The NPU
reports `Status: OK` and `medium` works in the same session, so this is a plugin size
ceiling. On CPU `large-v3` only *matches* `medium`'s `CER_trad` while being 2.5×
slower, and its `CER_raw` is worse (0.2703) because it emits Simplified more often.

**At `small`/`medium` the NPU is genuinely faster** (1.33× and 1.39×). The "NPU ≈ CPU"
result above is a `base`-sized finding, not a general one — NPU fixed overhead
amortizes as the model grows.

### Guaranteeing Traditional output

Whisper emits MIXED orthography and has no token to control it. `INIZ_STT_SCRIPT=trad`
post-processes through opencc: emitted script becomes **TRADITIONAL (simp=0)** and
`CER_raw` drops (0.2027 → 0.1757 on `small`) at no latency cost. Responses carry
`_text_raw` and `_script_converted` so the conversion is auditable.

Use **char-level `s2t`, never phrase-level `s2twp`** — `s2twp` rewrites text that is
already Traditional (`說明了` → `說明瞭`), adding errors to correct transcriptions.

`initial_prompt` and `hotwords` exist in `WhisperGenerationConfig` but **fail on the
NPU** (`Check '*roi_end <= *max_dim' failed`). On CPU a Traditional style prompt
reaches exactly the same 0.1757, so opencc is strictly better here.

At `medium` + opencc, `CER_raw` equals `CER_trad` (0.0811) — every remaining error is a
real misrecognition (`土地認養案例` → `土地任陽案例`), not orthography.

Device proof under Chinese load: `small` NPU max 105.10 % / mean 96.95 %; `medium` NPU
max 97.11 % / mean 91.60 %, CPU 6.6 % mean. Over HTTP with `language=<|zh|>`:
`medium` compile 102.87 s, p50 901.7 ms, `VERDICT: PASS`.

### Server

`iniz-stt/stt_server.py` — `POST /transcribe` (raw audio bytes or JSON `{"path":…}`),
`GET /health`. Over HTTP on the NPU: compile 1.01 s, warmup 151.2 ms, p50 167.5 ms
across 11 requests. Smoke test passes with WER 0.0900 and all five error cases
returning 4xx.

### Honest limitations

- **Traditional output relies on opencc post-processing**, not on the model. No
  in-model control works on the NPU.
- **Only 10 zh clips (39.7 s), short read speech.** 74 reference characters is a small
  sample — one clip moves `CER_trad` by ~0.01.
- **`large-v3` unusable on this NPU**; whether a newer driver lifts the ceiling is
  untested.
- **`CER_trad` 0.0811 ≈ 1 wrong character in 12.** Fine for search or rough notes, not
  verbatim transcription without review.
- **Translation still unvalidated** — `task=translate` was only run on English audio,
  where returning English tests nothing.
- **`GPU.0` never benchmarked** in any language.
- **No long-form audio** (longest 29.4 s) and **no streaming**.
- The `taiwan-mandarin-stt` prior experience cited in the old roadmap **does not exist
  on this machine** — this was built from scratch.
- The server binds to `127.0.0.1` **without authentication**. Audio is sensitive
  input — do not expose on `0.0.0.0` without adding auth.

---

## ✅ Iniz Aux — small generative model for cheap, frequent work

Qwen2.5-0.5B-Instruct (INT4 327 MB / INT8 494 MB) for summaries, sentiment, and field
extraction. Measured against ground truth, not eyeballed.

### The NPU is the wrong device for this, by 20×

| Task | NPU p50 | CPU p50 | NPU penalty |
|---|---|---|---|
| summarize (120 tok) | 14181.6 ms | **704.9 ms** | **20.1× slower** |
| sentiment (8 tok) | 1029.8 ms | **43.8 ms** | **23.5× slower** |
| extract (40 tok) | 2726.3 ms | **119.4 ms** | **22.8× slower** |

Accuracy is identical across devices (ROUGE-1 0.3179 NPU / 0.3218 CPU; sentiment
0.9250 both). The NPU **is** executing it — adapter at **98.37 % mean**, proven with
LUID counters — it is just bad at autoregressive decode, where each token is a separate
graph execution and a static-shape accelerator has nothing to amortize.

`aux_server.py` defaults to **CPU**. Use `INIZ_AUX_DEVICE=NPU` only to keep cores free
(CPU 4.5 % vs 22.5 % during load), never for latency.

This is the opposite of the Whisper result above, and both are real: a fixed-shape
encoder pass suits the NPU; 60+ tiny sequential decode steps do not.

### The earlier "precision extraction fails" warning does not reproduce

The old roadmap warned that INT4 hallucinated and INT8 refused. Re-tested on 10 items
with exact known answers:

| Model | Device | exact | refusal | hallucination | wrong span |
|---|---|---|---|---|---|
| INT8 | NPU | **7/10** | **0** | **0** | 3 |
| INT4 | NPU | 6/10 | **0** | **0** | 4 |
| INT8 | CPU | 6/10 | **0** | **0** | 4 |

**Zero refusals and zero hallucinations in 30 attempts.** Every failure was a wrong
*span* — a real substring, just not the requested one: `torch==2.9.1` instead of
`2.9.1`, `prod-media-eu-west-1/thumbnails/` with the `s3://` dropped. That is a
boundary-selection problem, fixable with a regex filter, not fabrication. The stale
warning would have killed a task that works ~65 % of the time and fails safely.

### Measured quality

12 CNN/DailyMail articles vs human highlights, 40 balanced SST-2 sentences, greedy
decoding:

| Model | Device | ROUGE-1 | ROUGE-2 | ROUGE-L | Sentiment | Extract |
|---|---|---|---|---|---|---|
| INT8 | NPU | 0.3179 | 0.1299 | 0.2321 | **0.9250** | 7/10 |
| INT4 | NPU | 0.3155 | 0.1041 | 0.2425 | 0.9000 | 6/10 |
| INT8 | CPU | 0.3218 | 0.1336 | 0.2404 | **0.9250** | 6/10 |

**Sentiment is the one to actually deploy:** 0.9250 accuracy, 0 unparsed outputs across
40 items, 43.8 ms on CPU. INT4 is 4.2× faster than INT8 on NPU for ~equal ROUGE-1, but
ROUGE-2 drops 20 % — phrasing drifts further.

### ROUGE hides a real failure mode

`cnn_10` (ROUGE-1 0.089) produced fluent output that moved *Roseanne Barr's* booing and
President Bush's "disgraceful" quote onto *Vince Neil*. Every entity is real and
present in the article, so substring grounding checks pass. Extrinsic-entity audit:
INT8 3/66 entities absent from source (4.5 %), INT4 0/47 (0.0 %) — but INT4 cites 29 %
fewer entities, so that 0 % is partly terseness, not fidelity.

**Misattribution is not measured** — it needs NLI or a human. Treat 0.5B summaries as
drafts.

### Honest limitations

- **Summarization is mediocre** (ROUGE-1 0.32 vs 0.40+ reference-grade) and
  inconsistent (per-item 0.089–0.531).
- **Misattribution rate unknown**; one confirmed case in 12 is a floor, not a rate.
- **Extraction ~65 % exact**, no post-filter or retry implemented.
- **Only 12 summarization items** — wide error bars.
- **No long-context test** (articles capped at 3500 chars), so "context compression"
  remains unproven.
- **`GPU.0` never benchmarked**; **INT4 on CPU never benchmarked**.
- **No batching** — requests serialize under a lock.
- The server binds to `127.0.0.1` **without authentication**; free-text input to a
  generative model should not be exposed without auth.

---

## ✅ Iniz Power — real energy per task, not utilization vibes

Intel RAPL package-power measurement (`\Energy Meter(*)\Power`) while hammering each
workload in a tight loop, idle subtracted, median-based. Marginal energy per work unit:

| Workload | NPU | CPU | Winner per task |
|---|---|---|---|
| guard inference | 0.248 / 0.056 mWh | 2.18 / 1.73 mWh | **NPU, 8.8–30.8× less** |
| Whisper-base transcription | 1.22 / 0.43 mWh | 4.36 / 3.34 mWh | **NPU, 3.6–7.8× less** |
| Qwen2.5-0.5B generation (60 tok) | 72.6 / 18.7 mWh | 10.0 / 7.2 mWh | **CPU, 2.6–7.2× less** |

Each pair is quiet-floor / elevated-floor assumption — the idle floor moved 7 W → 27 W
mid-campaign (cause undetermined, no benchmark process alive), so ratios are ranges.
No verdict flips between them.

The finding that matters: the NPU draws **less power** on aux (33.6 W vs 76.1 W) but
takes 19× longer, spending more energy per task. **Watts ≠ energy.** Anyone routing by
watts or CPU % alone would pick the NPU and drain the battery faster.

Scope, stated plainly: RAPL package only (not wall power; NPU-tile coverage
unverified, so NPU wins are lower bounds), tight-loop saturation (not sporadic use),
`power_analysis.json` marks comparisons below baseline drift as UNRESOLVED rather than
printing them.

### Honest limitations

- **Floor drift is the dominant uncertainty** (±20 W vs signals of 25–95 W). Re-run
  with the machine left alone and baseline immediately before AND after.
- **~86 Wh battery translations are illustrative**, not promises (display/radio/dGPU
  excluded).
- **Linux not covered** (needs RAPL sysfs path instead of perf counters).

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

Nothing in this section is implemented yet. (Speech-to-text, the auxiliary model, and
SoC-package energy measurement **moved out** of this section — all three are
implemented and measured above.)

### Power optimization

What **is** now measured: marginal SoC-package energy per task for all three workloads
(see ✅ Iniz Power above) — guard 8.8–30.8× cheaper on NPU, Whisper 3.6–7.8× cheaper
on NPU, aux generation 2.6–7.2× cheaper on CPU.

What is **still not measured**: wall power (display, dGPU, losses), and therefore any
"hours of battery saved" claim. That needs a battery-discharge run or an external
meter — RAPL package scope cannot produce it.

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

iniz-stt/
├── SKILL.md                  # Full skill: size ladder, NPU ceiling, 14 pitfalls
├── stt_server.py             # NPU HTTP server (WhisperPipeline + opencc script mode)
├── scripts/                  # export/fetch audio, probe, bench, NPU proof, client
│                             #   + zh: script_check, fetch_audio_zhtw, bench_zhtw,
│                             #         test_prompt_fair, stt_client_zhtw
└── results/                  # whisper_bench{,_zhtw*}.json + LUID proofs + prompt test

iniz-aux/
├── SKILL.md                  # Full skill: CPU-beats-NPU verdict, 8 pitfalls
├── aux_server.py             # HTTP server (LLMPipeline, CPU default by measurement)
├── scripts/                  # fetch data, bench, grounding audit, NPU proof, client
└── results/                  # aux_bench_*, aux_*_grounding, aux_proof_* JSON

iniz-power/
├── SKILL.md                  # Full skill: RAPL method, ranges, 7 pitfalls
├── scripts/                  # power_probe.py, analyze_power.py
└── results/                  # power_{guard,stt,aux}_{NPU,CPU}.json + 5 baselines
```

**Model weights are not included** (guard IR INT8 = 495 MB, beyond practical Git
limits). Get them from
[HuggingFace](https://huggingface.co/CH3NDev/iniz-agent-guard-int8) or generate them
yourself with `scripts/export_guard_ov.py --seq-len 128`. The STT IRs are not shipped
either — regenerate `whisper-base` in ~3 minutes (English) or `whisper-medium` in
~7 minutes (Chinese) with
`optimum-cli export openvino --model openai/whisper-<size> --weight-format int8 models/whisper-<size>-int8-ov`.

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

For offline speech-to-text:

1. Export the IR (~3 min):
   `optimum-cli export openvino --model openai/whisper-base --weight-format int8 models/whisper-base-int8-ov`
2. Fetch test audio with ground truth: `python iniz-stt/scripts/fetch_audio.py`
3. Prove NPU execution: `python iniz-stt/scripts/prove_whisper_npu.py NPU`
4. Start the server: `INIZ_STT_DEVICE=NPU python iniz-stt/stt_server.py`
5. Test it: `python iniz-stt/scripts/stt_client.py`

For Traditional Chinese, export `whisper-medium` instead and run the zh scripts:

```bash
optimum-cli export openvino --model openai/whisper-medium --weight-format int8 \
  models/whisper-medium-int8-ov
python iniz-stt/scripts/fetch_audio_zhtw.py       # verifies refs are Traditional
python iniz-stt/scripts/bench_zhtw.py --model ../models/whisper-medium-int8-ov
INIZ_STT_SCRIPT=trad INIZ_STT_MODEL=C:/path/to/models/whisper-medium-int8-ov \
  python iniz-stt/stt_server.py                  # allow ~103 s for NPU compile
python iniz-stt/scripts/stt_client_zhtw.py
```

`iniz-stt/scripts/script_check.py` measures whether any Chinese text is Simplified or
Traditional — run it on a dataset before trusting its card.

For the auxiliary model (summaries / sentiment / extraction):

```bash
python iniz-aux/scripts/fetch_aux_data.py          # CNN/DailyMail + SST-2 + extraction
python iniz-aux/scripts/bench_aux.py --model ../models/qwen2.5-0.5b-instruct-int8-ov --device CPU
python iniz-aux/scripts/prove_aux_npu.py NPU       # confirms NPU runs it, just slowly
INIZ_AUX_DEVICE=CPU python iniz-aux/aux_server.py
python iniz-aux/scripts/aux_client.py
```

Read `iniz-aux/SKILL.md` before switching that server to the NPU — it is 20× slower
there for the same accuracy.

Read `iniz-stt/SKILL.md` first — it opens with a **negative result** (the NPU is not
faster than CPU for `whisper-base`) that changes how you should deploy it.

To reproduce the energy numbers (Windows, machine left alone):

```bash
python iniz-power/scripts/power_probe.py baseline   # idle floor; repeat 2-3x
python iniz-power/scripts/power_probe.py guard NPU
python iniz-power/scripts/power_probe.py guard CPU
python iniz-power/scripts/analyze_power.py          # verdict table
```

Read `iniz-power/SKILL.md` before quoting any mWh figure — the floor-drift caveat
changes single numbers into ranges.

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
