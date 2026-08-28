---
name: iniz-stt
description: Offline speech-to-text on Intel NPU via OpenVINO Whisper.
version: 0.1.0
author: Matthew Chen (CH3NDev), Hermes Agent
license: MIT
platforms: [windows, linux]
metadata:
  hermes:
    tags: [openvino, npu, whisper, speech-to-text, offline, local-ai]
    related_skills: [iniz-agent-guard]
---

# Skill: Iniz STT — Offline Speech-to-Text on Intel NPU

**Trigger:** When the task involves local/offline transcription, or serving a
generative encoder-decoder (Whisper) on an Intel Core Ultra NPU through
`openvino_genai.WhisperPipeline`.

**Behavior:** Export Whisper to OpenVINO IR INT8 with `optimum-cli`, benchmark it
against real audio with ground-truth transcripts, prove which device actually
executes it, then serve it over HTTP with the same process shape as
`iniz-agent-guard/guard_server.py`.

**Environment:** `openvino-genai` ≥ 2026.3, `optimum[openvino]`, `librosa`,
`soundfile`, `pyarrow`. The NPU must appear in `ov.Core().available_devices`.

---

## The headline result is a negative one — read this first

On this machine, **the NPU is not faster than the CPU for `whisper-base`**:

| Device | compile | p50 latency | p90 | RTF (median) | WER | CPU load during run |
|---|---|---|---|---|---|---|
| **NPU** | 0.66 s | **169.9 ms** | 232.5 ms | 0.0196 | 0.0900 | **7.5 % mean / 10.6 % max** |
| **CPU** | 0.58 s | **167.8 ms** | 207.7 ms | 0.0189 | 0.0900 | 27.0 % mean / 49.2 % max |

Same 8 LibriSpeech clips (86.34 s of audio), same 18/200 word errors, and
**8/8 normalized transcripts identical** between devices.

The NPU wins on **CPU offload, not speed**: 7.5 % versus 27.0 % mean CPU while
transcribing. That is the actual reason to deploy this — transcription that does not
compete with the rest of the machine for cores. If you need raw wall-clock speed for
`whisper-base`, use CPU.

RTF 0.02 means both devices run roughly **50× faster than real time**, so latency is
not the binding constraint at this model size anyway.

---

## Do not skip the device proof

`WhisperPipeline` wraps several models and does **not** expose
`EXECUTION_DEVICES`, so the guard skill's cheapest device check is unavailable here.
Two nearly identical latency columns are exactly what a silent fallback looks like,
so this was verified independently with per-pid GPU-Engine counters
(`scripts/prove_whisper_npu.py`):

| Requested device | NPU adapter (`0x00000000_0x00011cf3`) | iGPU | CPU `_Total` | Verdict |
|---|---|---|---|---|
| `NPU` | **active, max 93.79 %, mean 89.06 %** (`engtype=compute`) | idle | 7.5 % | genuinely on the NPU |
| `CPU` | idle | idle | 27.0 % mean / 49.2 % max | genuinely on the CPU |

93 transcriptions in ~13 s on NPU, 87 on CPU. The NPU numbers are real, not a
fallback that happens to match.

---

## Scope: how this differs from Iniz Agent Guard

The guard skill's central rule is "never use a GenAI pipeline." **That rule does not
apply here**, and carrying it over would be the wrong call:

| | Iniz Agent Guard | **Iniz STT** |
|---|---|---|
| Model kind | discriminative (backbone + 3 heads, no `lm_head`) | generative encoder-decoder |
| Correct API | `core.compile_model` + read tensors | **`openvino_genai.WhisperPipeline`** |
| Export path | `torch.export` → `ov.convert_model` (manual) | **`optimum-cli export openvino`** |
| Static shapes | mandatory, `reshape()` by hand | handled by the pipeline |
| Device check | `compiled.get_property('EXECUTION_DEVICES')` | **not exposed — use LUID counters** |
| NPU vs CPU | NPU 2.7× faster | **NPU ≈ CPU; win is CPU offload** |

The one thing that carries over unchanged is the **server shape**: load the model once
into a long-lived process, guard inference with a lock, expose `/health` with real
counters, and never trust a device string without proof.

---

## Procedure

Run everything from the runtime dir with its venv (`$V` below is
`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux).

1. **Export the IR.** Completion: `openvino_encoder_model.xml` and
   `openvino_decoder_model.xml` exist and the dir is ~81 MB.
   ```
   terminal(command="optimum-cli export openvino --trust-remote-code \
     --model openai/whisper-base --weight-format int8 models/whisper-base-int8-ov")
   ```
   NNCF reports `int8_asym, per-channel 100% (38/38)` for the encoder and
   `100% (62/62)` for the decoder.

2. **Fetch real audio with ground truth.** Completion: `audio/manifest.json` lists 8
   clips totalling 86.3 s, each with a `reference` string.
   ```
   terminal(command="$V scripts/fetch_audio.py")
   ```
   Synthetic audio cannot produce a meaningful WER — this pulls LibriSpeech
   `validation-00000-of-00001.parquet` and decodes it with `soundfile`.

3. **Probe one device before benchmarking all of them.** Completion: a per-stage
   timing table prints in under ~5 s.
   ```
   terminal(command="$V -u scripts/probe_whisper.py NPU")
   ```
   Do this first — it separates "pipeline construction is slow" from "inference is
   slow" and costs seconds instead of minutes.

4. **Benchmark with WER.** Completion: `whisper_bench.json` holds p50/p90, RTF, and
   WER per device.
   ```
   terminal(command="$V -u scripts/bench_whisper.py --devices NPU,CPU")
   ```

5. **Prove the device.** Completion: `whisper_proof_NPU.json` has
   `"npu_active": true`, and the CPU run has it `false`.
   ```
   terminal(command="$V -u scripts/prove_whisper_npu.py NPU")
   terminal(command="$V -u scripts/prove_whisper_npu.py CPU")
   ```

6. **Serve and smoke test.** Completion: `VERDICT: PASS`, overall WER < 0.25, all
   five error cases return 4xx.
   ```
   terminal(command="INIZ_STT_DEVICE=NPU $V scripts/stt_server.py", background=True)
   terminal(command="$V -u scripts/stt_client.py")
   ```

---

## Server

`stt_server.py` — 238 lines, threaded HTTP, model loaded once, lock-guarded
inference (the NPU serves one request at a time).

```
GET  /health
  -> {"status","device","model","compile_s","warmup_ms","requests_served","latency_p50_ms"}

POST /transcribe
  body: raw audio bytes (wav/flac/ogg — anything soundfile reads)
        or JSON {"path": "<abs path>", "language": "<|en|>", "task": "transcribe"}
  query: ?language=<|en|>&task=translate&timestamps=1
  -> {"text","duration_s","rtf","_ms","_device","_model","chunks"?}
```

Environment: `INIZ_STT_MODEL`, `INIZ_STT_DEVICE` (default `NPU`),
`INIZ_STT_PORT` (default `8010`), `INIZ_STT_MAX_BYTES` (default 64 MB).

**Security:** binds to `127.0.0.1` with **no authentication**. Audio is sensitive
input — do not move it to `0.0.0.0` without adding auth first. The server prints this
warning on every start.

Measured over HTTP on the NPU: `compile 1.01 s`, `warmup 151.2 ms`,
`p50 167.5 ms` across 11 requests, client-side p50 210 ms. Per-clip RTF ranged
0.0153 (29.4 s clip) to 0.0425 (5.9 s clip) — longer audio amortizes better.

---

## Pitfalls (all hit on this machine)

1. **`datasets` cannot decode audio without `torchcodec`.** `load_dataset(...)` on an
   audio dataset raises `ImportError: To support decoding audio data, please install
   'torchcodec'`. Read the parquet directly with `pyarrow` and decode the `bytes`
   field with `soundfile` — `scripts/fetch_audio.py` does this.

2. **Piping a long benchmark through `tail` hides all progress.** The first full
   benchmark ran >7 minutes with zero visible output because `| tail -70` buffered
   everything; it looked like a hang. Two independent checks said otherwise: the
   process had `CPU_seconds=0` and no GPU-Engine activity. Always run these with
   `python -u` and no pipe.

3. **`WhisperPipeline` has no `EXECUTION_DEVICES`.** It wraps encoder + decoder, so
   `get_property` is unavailable. Use the LUID counter method
   (`scripts/prove_whisper_npu.py`), not `assert device == 'NPU'`.

4. **Comparing raw strings gives a fake ~80 % WER.** LibriSpeech references are
   uppercase with no punctuation; Whisper emits `"Mr. Quilter's ..."`. Normalize
   (uppercase, strip punctuation, collapse spaces) before scoring. WER 0.09 vs 0.80
   is entirely this.

5. **`GPU.0` was never benchmarked.** Only NPU and CPU are measured here. The iGPU
   column is absent, not zero.

6. **`task=translate` on English audio returns English.** Not a bug — verified
   working, but it proves nothing about translation quality. No non-English audio was
   tested, so translation is **unvalidated**.

---

## What is still open

- **No non-English audio tested.** Language detection and `task=translate` are
  exercised but not validated for quality. The `taiwan-mandarin-stt` prior experience
  referenced in the roadmap **does not exist on this machine** — no local project, no
  Whisper in the HF cache. This was built from scratch.
- **Only `whisper-base`.** `small` / `large-v3` are untested; the NPU-vs-CPU verdict
  may flip at larger sizes where the NPU's fixed overhead amortizes.
- **No long-form audio.** Longest clip is 29.4 s. Whisper's 30 s window means
  chunking behaviour past that is unverified.
- **`timestamps=1` returned 1 chunk** for a 4.8 s clip. Correct but trivially so; not
  a real test of segmentation.
- **No streaming / partial results.** Requests are whole-file only.

---

## Verification

```
$V -u scripts/probe_whisper.py NPU          # stage timings, ~5 s
$V -u scripts/bench_whisper.py --devices NPU,CPU
$V -u scripts/prove_whisper_npu.py NPU      # expect npu_active: true
$V -u scripts/stt_client.py                 # expect VERDICT: PASS
```

Accept only if: `npu_active` is `true` for the NPU run and `false` for CPU, WER
≤ 0.10 on the LibriSpeech clips, NPU/CPU transcript agreement is 8/8, and the client
prints `VERDICT: PASS` with all five error cases returning 4xx.

## See Also

- `results/whisper_bench.json` — per-device latency, RTF, WER, and all transcripts
- `results/whisper_proof_NPU.json`, `results/whisper_proof_CPU.json` — LUID evidence
- `iniz-agent-guard/SKILL.md` — the discriminative counterpart and the shared
  server/proof discipline
