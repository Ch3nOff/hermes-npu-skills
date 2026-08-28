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
`soundfile`, `pyarrow`, `opencc-python-reimplemented` (Chinese only). The NPU must
appear in `ov.Core().available_devices`.

---

## Pick the model by language, not by habit

| Language | Model | Why |
|---|---|---|
| English | `whisper-base` | WER 0.0900, 169.9 ms — `small` buys nothing measurable |
| **Chinese** | **`whisper-small`** | `base` gives CER 0.51 (unusable); `small` gives 0.20 |

The English and Chinese sections below reach **opposite conclusions about the NPU**,
and both are correct: at `base` the NPU ties the CPU, at `small` it wins by 1.33×.

---

## The headline result is a negative one — read this first

On this machine, **the NPU is not faster than the CPU for `whisper-base`** on English:

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

## Traditional Chinese: model size decides this, not the device

Measured on 10 Common Voice 16.1 zh-TW clips (39.7 s), references verified
Traditional (`simp=0 / trad=11`) before use. CER, not WER — Chinese has no
whitespace word boundaries.

| Model | Device | p50 | RTF | CER_raw | CER_trad | Exact clips |
|---|---|---|---|---|---|---|
| `whisper-base` | NPU | 101.8 ms | 0.026 | **0.5135** | 0.4865 | 3/10 |
| `whisper-base` | CPU | 118.4 ms | 0.031 | 0.5135 | 0.4865 | 3/10 |
| **`whisper-small`** | **NPU** | 297.1 ms | 0.074 | **0.2027** | **0.1622** | **4/10** |
| `whisper-small` | CPU | 394.1 ms | 0.106 | 0.2027 | 0.1622 | 4/10 |

**`whisper-base` is not usable for Chinese.** Half the characters are wrong, and two
clips came back as romanized nonsense (`土地認養案例` → `thoody learn yang andi`).
`whisper-small` cuts CER 2.5× for 2.9× the latency — that trade is worth it here, and
unlike English there is a real quality reason to pay it.

At `whisper-small` the NPU also becomes **genuinely faster**: 297.1 ms vs 394.1 ms
(1.33×), where at `base` the two devices were within 1 %. The NPU's fixed overhead
amortizes as the model grows, so the English section's "NPU ≈ CPU" verdict is a
`base`-sized finding, not a general one.

### Two CER numbers, because one would lie

Whisper's zh training data is overwhelmingly Simplified, so it transcribes Traditional
audio correctly but sometimes writes Simplified characters. `開源服務` → `开源服务`
is a **perfect** transcription in the wrong orthography: `CER_raw` scores it 2/4,
`CER_trad` (both sides forced Traditional via opencc `s2twp`) scores it 0/4.

Emitted script was **MIXED** on both models (`simp=3 trad=9` at `small`) — Whisper
does not honour a Traditional preference, and there is no `language` token for it.
If you need guaranteed Traditional output, post-process with opencc; the model will
not do it for you.

### Device proof under Chinese load

`whisper-small` + zh audio on NPU: NPU adapter `0x00000000_0x00011cf3` at
**max 105.10 %, mean 96.95 %**, CPU `_Total` 4.7 % mean — 42 transcriptions in ~13 s.
Evidence in `results/whisper_proof_NPU_small_zhtw.json`.

Over HTTP with `language=<|zh|>`: p50 301.0 ms, CER_trad 0.1622, `VERDICT: PASS`.
Note `whisper-small` compile takes **11.64 s** on the NPU versus 1.01 s for `base` —
budget for that at server start.

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

7. **For Chinese, repeat with the zh scripts and `whisper-small`.** Completion:
   `fetch_audio_zhtw.py` prints `reference script: ... -> TRADITIONAL`, and the bench
   reports both `CER_raw` and `CER_trad`.
   ```
   terminal(command="$V -u scripts/fetch_audio_zhtw.py")
   terminal(command="$V -u scripts/bench_zhtw.py --model ../models/whisper-small-int8-ov")
   terminal(command="$V -u scripts/stt_client_zhtw.py")
   ```
   `scripts/script_check.py` is the Simplified/Traditional detector — run it on any
   new dataset **before** trusting its card. Pass an absolute forward-slash path to
   `INIZ_STT_MODEL`; `$HOME/...` becomes `\c\Users\...` under MSYS and the server
   cannot find the IR.

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
   working, but it proves nothing about translation quality. Cross-language
   translation remains **unvalidated**.

7. **Dataset cards lie about orthography — measure it.** `CJY/Chinese-Dialogue-180k`
   looked like a Chinese test set but is **Simplified** (980 Simplified-only chars vs
   4 Traditional in a 92-row sample) and has no ASR ground truth (it is
   speech-to-speech dialogue, 126 GB). `scripts/script_check.py` settles this in
   seconds; `JacobLinCool/common_voice_16_1_zh_TW_clean` measured `simp=0 / trad=99`.

8. **`datasets` audio decode needs `torchcodec`, and parquet row groups save time.**
   `pq.ParquetFile(...).read_row_group(0)` pulls 100 rows out of a 341 MB shard
   without materializing the whole file.

9. **MSYS mangles `$HOME` for native programs.** `INIZ_STT_MODEL="$HOME/..."` reached
   the server as `\c\Users\Matthew Chen\...` and it raised
   `FileNotFoundError: Whisper IR not found`. Use `C:/Users/...` forward-slash paths.

10. **`whisper-small` compile is 11.64 s on NPU** (vs 1.01 s for `base`). It looks
    like a hang if you expect base-like startup.

---

## What is still open

- **Traditional-Chinese output is not guaranteed.** Whisper emits MIXED orthography;
  opencc post-processing would be needed for consistent Traditional. Not implemented.
- **Only 10 zh clips (39.7 s), all short** (2.8–5.3 s) and all read speech. Common
  Voice is not conversational audio.
- **`whisper-large-v3` untested.** `small` fixed most of the Chinese CER; `large-v3`
  may fix the rest, but its NPU compile time and memory are unmeasured here.
- **Translation still unvalidated.** `task=translate` was only run on English audio,
  where it correctly returns English — that tests nothing.
- **No long-form audio.** Longest clip anywhere is 29.4 s (English). Whisper's 30 s
  window means chunking past that is unverified.
- **`GPU.0` never benchmarked** in any language.
- **No streaming / partial results.** Requests are whole-file only.
- The `taiwan-mandarin-stt` prior experience cited in the repo roadmap **does not
  exist on this machine** — no local project, no Whisper in the HF cache. Everything
  here was built from scratch.

---

## Verification

```
$V -u scripts/probe_whisper.py NPU          # stage timings, ~5 s
$V -u scripts/bench_whisper.py --devices NPU,CPU
$V -u scripts/prove_whisper_npu.py NPU      # expect npu_active: true
$V -u scripts/stt_client.py                 # expect VERDICT: PASS

# Traditional Chinese
$V -u scripts/fetch_audio_zhtw.py           # expect -> TRADITIONAL
$V -u scripts/bench_zhtw.py --model ../models/whisper-small-int8-ov
$V -u scripts/stt_client_zhtw.py            # expect VERDICT: PASS
```

Accept only if: `npu_active` is `true` for the NPU run and `false` for CPU, English
WER ≤ 0.10 with 8/8 transcript agreement, Chinese `CER_trad` ≤ 0.20 on
`whisper-small`, and both clients print `VERDICT: PASS`.

## See Also

- `results/whisper_bench.json` — English per-device latency, RTF, WER, transcripts
- `results/whisper_bench_zhtw.json`, `results/whisper_bench_zhtw_small.json` —
  Chinese CER_raw / CER_trad / emitted script per device
- `results/whisper_proof_NPU.json`, `whisper_proof_CPU.json`,
  `whisper_proof_NPU_small_zhtw.json` — LUID evidence
- `iniz-agent-guard/SKILL.md` — the discriminative counterpart and the shared
  server/proof discipline
