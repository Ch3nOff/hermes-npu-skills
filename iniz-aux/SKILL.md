---
name: iniz-aux
description: Small-LLM summarize/sentiment/extract, CPU beats NPU here.
version: 0.1.0
author: Matthew Chen (CH3NDev), Hermes Agent
license: MIT
platforms: [windows, linux]
metadata:
  hermes:
    tags: [openvino, npu, qwen2.5, summarization, sentiment, local-ai]
    related_skills: [iniz-agent-guard, iniz-stt]
---

# Skill: Iniz Aux — Small Generative Model for Cheap, Frequent Work

**Trigger:** When offloading small frequent LLM work (page summaries, captions,
sentiment, field extraction) to a local quantized model, or when deciding whether an
Intel NPU is the right device for an autoregressive workload.

**Behavior:** Benchmark Qwen2.5-0.5B-Instruct INT4/INT8 against ground truth on three
tasks, prove which device executes it, then serve it over HTTP with the same process
shape as `guard_server.py` — with the device default set by measurement, not habit.

**Environment:** `openvino-genai` ≥ 2026.3, `pyarrow`. The NPU must appear in
`ov.Core().available_devices` if you intend to test it.

---

## Headline: the NPU is the wrong device for this, by 20×

| Task | NPU p50 | CPU p50 | NPU penalty |
|---|---|---|---|
| summarize (120 tok) | 14181.6 ms | **704.9 ms** | **20.1× slower** |
| sentiment (8 tok) | 1029.8 ms | **43.8 ms** | **23.5× slower** |
| extract (40 tok) | 2726.3 ms | **119.4 ms** | **22.8× slower** |

Accuracy is *identical* across devices (ROUGE-1 0.3179 NPU vs 0.3218 CPU; sentiment
0.9250 both). The NPU is genuinely executing the model — adapter at **98.37 % mean**,
verified with LUID counters — it is simply bad at autoregressive decode, where every
token is a separate graph execution and a static-shape accelerator has nothing to
amortize.

iGPU spot-check (sentiment only): `GPU.0` p50 **89.8 ms** at acc 0.9250 — 11.5× faster
than the NPU but still 2× slower than the CPU. The speculation that the iGPU might
beat both for decode is now measured and wrong for this task; summarize/extract on
`GPU.0` remain unmeasured because nothing suggests a flip.

**`aux_server.py` therefore defaults to `CPU`.** Use `INIZ_AUX_DEVICE=NPU` only to keep
cores free (CPU 4.5 % vs 22.5 % during load) — never for latency.

This is the opposite of the Whisper result in `iniz-stt`, and both are real: Whisper's
encoder is one big fixed-shape pass the NPU likes; a decoder-only LLM is 60+ tiny
sequential passes it does not.

---

## The earlier "precision extraction fails" claim does NOT reproduce

The roadmap carried a warning from an earlier session: *INT4 hallucinated, INT8 refused
to answer* on precision extraction. Re-tested on 10 items with exact known answers:

| Model | Device | exact | refusal | hallucination | wrong span |
|---|---|---|---|---|---|
| INT8 | NPU | **7/10** | **0** | **0** | 3 |
| INT4 | NPU | 6/10 | **0** | **0** | 4 |
| INT8 | CPU | 6/10 | **0** | **0** | 4 |

Zero refusals and zero hallucinations in 30 attempts. Every failure was a **wrong
span** — a real substring of the input, just not the requested one:

- asked for the version, got `torch==2.9.1` instead of `2.9.1`
- asked for the URL, got `prod-media-eu-west-1/thumbnails/` (dropped the `s3://`)
- asked for the event id, got `signature errors`
- INT8@CPU picked `4.12.0` where the answer was `4.11.2` — both versions appear

That is a **boundary/selection** problem, not fabrication — and it turned out to be
fixable with a regex post-filter. The old claim would have led to abandoning a
task that works about 65 % of the time raw and fails safely.

Reproduced twice on INT8@NPU (7/10 both runs) before reporting.

### Post-filter: 7/10 → 10/10 (INT8@NPU)

`scripts/extract_filter.py` applies one generic rule per answer kind, never per item:

| Kind | Rule | Fixes |
|---|---|---|
| version | pull the first `v?\d+\.\d+(\.\d+)*` out of the answer | `torch==2.9.1` → `2.9.1` |
| path | expand the answer's span in the source over the URL charset | restores dropped `s3://` |
| identifier | answer must be one token (len≥6, digit or uppercase); else use the source's sole matching token | `signature errors` → `evt_1Nx8Kd2eZvKYlo2C` |
| email / filename | token regex on the answer | idempotent on correct answers |

Measured on recorded model outputs (no re-runs needed — the filter is deterministic):

| Run | raw exact | filtered exact | Still wrong (documented, not attempted) |
|---|---|---|---|
| INT8@NPU | 7/10 | **10/10** | — |
| INT4@NPU | 6/10 | **9/10** | ext_00 `config`: wrong pick, no extension to anchor on |
| INT8@CPU | 6/10 | **9/10** | ext_01 `4.12.0`: genuinely ambiguous, two versions in source |

Acceptance gate: the filter must not break any previously-exact answer (idempotence).
Verified across all three runs — 19 exact answers in, 19 unchanged out. One real bug
caught during this: the identifier rule first absorbed trailing sentence punctuation
(`...lo2C.`); candidates are now stripped of `.,;:` before matching.

`/extract` applies the filter live (`kind` from an optional request field, else
inferred from question keywords; unknown kind → raw answer, old behavior) and reports
`filter_rule` per response plus `_text_raw`-style audit via the existing `raw` field.
Verified over HTTP: all three previously-failing cases now return exact answers,
one exact case confirmed unchanged (`filter_rule: null`).

OVERFITTING CAVEAT, stated plainly: rules were tuned and tested on the SAME n=10 set
(the only labeled set). 10/10 measures "the filter implements what we saw", not
"extraction works in general". The rules contain no item-specific constants (no
literal answers, no indices) — the most that can honestly be claimed at n=10.

---

## Measured quality

12 CNN/DailyMail test articles (avg 1917 chars) vs human highlights; 40 balanced SST-2
sentences; 10 extraction items. Greedy decoding (`do_sample=False`) so numbers repeat.

| Model | Device | ROUGE-1 | ROUGE-2 | ROUGE-L | Sentiment acc | Extract |
|---|---|---|---|---|---|---|
| INT8 | NPU | 0.3179 | 0.1299 | 0.2321 | **0.9250** | 7/10 |
| INT4 | NPU | 0.3155 | 0.1041 | 0.2425 | 0.9000 | 6/10 |
| INT8 | CPU | 0.3218 | 0.1336 | 0.2404 | **0.9250** | 6/10 |

**Sentiment is the standout: 0.9250 with 0 unparsed outputs across 40 items.** It is
cheap (8 tokens), the output space is two words, and it needs no post-processing. If
you deploy one thing from this skill, deploy that.

**INT4 buys speed, not much loss.** ROUGE-1 within 0.002 of INT8 and 4.2× faster on
NPU (3384.5 ms vs 14181.6 ms). ROUGE-2 is the exception — 0.1041 vs 0.1299, a 20 %
drop, meaning INT4 phrasing drifts further from the reference even when content
overlaps.

ROUGE-1 ≈ 0.32 is **mediocre**: reference-quality summarization scores 0.40+. Per-item
spread was 0.089 → 0.531, so quality is inconsistent, not uniformly middling.

---

## ROUGE hides a failure mode you must know about

`cnn_10` scored ROUGE-1 0.089 and its output was fluent, plausible, and **factually
wrong in a way no automatic metric here catches**:

- Article: *Roseanne Barr's* rendition was booed; President Bush called **her**
  performance "disgraceful". Vince Neil was criticised separately.
- Output: *"Vince Neil's performance ... with the crowd booing him and President Bush
  calling it 'disgraceful.'"*

Every entity is real and present in the source. A substring check passes. ROUGE only
notices because the wording diverges — had the model misattributed using the
reference's exact words, ROUGE would have gone *up*.

`scripts/check_summary_grounding.py` measures what is mechanically checkable and is
explicit about what is not:

| Model | Entities cited | Extrinsic (absent from article) | Summaries with ≥1 |
|---|---|---|---|
| INT8 @ NPU | 66 | 3 (4.5 %) | 2/12 |
| INT4 @ NPU | 47 | **0 (0.0 %)** | **0/12** |

INT4 looks cleaner here, but partly because it cites **29 % fewer entities** (47 vs
66) — a terser summary has less to get wrong. Do not read 0 % as "INT4 is more
faithful".

**Misattribution is NOT measured** — detecting it needs NLI or a human. The script says
so in its output rather than letting a clean extrinsic score imply faithfulness. Treat
summaries from a 0.5B model as drafts, and never let one make a factual claim
unreviewed.

---

## Procedure

`$V` is `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux).

1. **Fetch ground-truth data.** Completion: `aux_data/{summarize,sentiment,extract}.json`
   exist with 12 / 40 / 10 items.
   ```
   terminal(command="$V -u scripts/fetch_aux_data.py")
   ```
   Reads CNN/DailyMail and SST-2 parquet directly with `pyarrow` — `datasets` is not
   required.

2. **Benchmark each quantization on the NPU.** Completion: `aux_bench_*.json` per
   model with ROUGE, accuracy, and an extraction outcome breakdown.
   ```
   terminal(command="$V -u scripts/bench_aux.py --model ../models/qwen2.5-0.5b-instruct-int8-ov --device NPU")
   terminal(command="$V -u scripts/bench_aux.py --model ../models/qwen2.5-0.5b-instruct-int4-ov --device NPU")
   ```

3. **Benchmark the CPU — do not skip this.** It is what revealed the 20× gap.
   ```
   terminal(command="$V -u scripts/bench_aux.py --model ../models/qwen2.5-0.5b-instruct-int8-ov --device CPU")
   ```

4. **Check grounding.** Completion: an extrinsic-entity count per model.
   ```
   terminal(command="$V -u scripts/check_summary_grounding.py aux_bench_qwen2.5-0.5b-instruct-int8-ov_NPU.json")
   ```

5. **Prove the device.** Completion: `npu_active: true` for the NPU run, `false` for CPU.
   ```
   terminal(command="$V -u scripts/prove_aux_npu.py NPU")
   terminal(command="$V -u scripts/prove_aux_npu.py CPU")
   ```
   Needed because two very different latencies could still mean a silent fallback —
   here it confirmed the NPU is real and just slow.

6. **Serve and smoke test.** Completion: `VERDICT: PASS`, all five error cases 4xx.
   ```
   terminal(command="INIZ_AUX_DEVICE=CPU $V scripts/aux_server.py", background=True)
   terminal(command="$V -u scripts/aux_client.py")
   ```

---

## Server

`aux_server.py` — 255 lines, threaded HTTP, model loaded once, lock-guarded inference.

```
GET  /health      -> device, model, compile_s, warmup_ms, per-task p50
POST /summarize   {"text": "...", "max_new_tokens": 120} -> {"summary", "_ms", ...}
POST /sentiment   {"text": "..."} -> {"sentiment": positive|negative|unparsed, "raw"}
POST /extract     {"text": "...", "question": "..."} -> {"answer", "grounded", "raw"}
```

Environment: `INIZ_AUX_MODEL`, `INIZ_AUX_DEVICE` (**default `CPU`**),
`INIZ_AUX_PORT` (8011), `INIZ_AUX_MAX_CHARS` (8000).

`/extract` returns `grounded` — whether the answer occurs in the input — plus a
`_warning` that grounded ≠ correct. All four misses in the smoke test were
*grounded-but-wrong*, so the flag alone must not be treated as validation.

Measured over HTTP on CPU: compile 1.29 s, warmup 160.5 ms, summarize p50 944.7 ms,
sentiment p50 46.6 ms (12/12 correct), extract p50 122.2 ms (6/10 exact), 25 requests,
`VERDICT: PASS`.

**Security:** binds to `127.0.0.1` with **no authentication**. Free-text input to a
generative model — do not expose it without auth.

---

## Pitfalls

1. **Do not assume the NPU helps because it helped elsewhere.** It is 20–23× slower
   here than CPU at equal accuracy. Measure per workload shape: fixed-shape encoder
   passes (Whisper) win, autoregressive decode loses.

2. **Do not inherit a failure claim without re-testing it.** The "INT4 hallucinates,
   INT8 refuses" warning did not reproduce at all — 0/30 refusals or hallucinations.
   Stale negative findings cost more than stale positive ones because they stop work
   that would succeed.

3. **`grounded` is not `correct`.** Every extraction miss returned a genuine substring
   of the input. A substring check cannot catch a wrong-span answer.

4. **ROUGE cannot see misattribution.** `cnn_10` moved one person's booing and a
   president's quote onto a different musician using only in-article entities.
   Fluent + grounded + wrong is the dangerous combination.

5. **Write the Qwen2.5 chat template by hand.** `openvino_genai`'s
   `Tokenizer.apply_chat_template` has a different signature from transformers' (no
   `tokenize` kwarg); calling it the transformers way raises `TypeError` and invites a
   silent fallback to an unformatted prompt.

6. **Set `do_sample=False`.** With sampling on, every benchmark run gives different
   numbers and nothing is comparable.

7. **A terser model looks more faithful than it is.** INT4 scored 0 extrinsic entities
   partly by citing 29 % fewer entities than INT8. Normalize by entity count before
   concluding anything.

8. **NPU compile time for this model varies 3.55 s – 16.10 s** across runs of the same
   IR. Do not treat a single compile measurement as the number.

---

## What is still open

- **Summarization quality is mediocre** (ROUGE-1 0.32 vs 0.40+ for reference-grade)
  and inconsistent (per-item 0.089–0.531). Usable for rough gisting, not for anything
  a reader will rely on unreviewed.
- **Misattribution rate is unmeasured.** Needs an NLI model or human review; one
  confirmed case in 12 summaries is a floor, not a rate.
- **Extraction is 10/10 with the post-filter (7/10 raw), n=10.** The remaining risk
  is overfitting to the tiny labeled set, not span boundaries. Retry logic is still
  unimplemented.
- **Only 12 summarization items.** ROUGE on 12 articles has wide error bars.
- **No long-context test.** Articles were capped at 3500 chars; context compression on
  genuinely long input is untested.
- **iGPU spot-checked on sentiment only** (89.8 ms — between CPU 43.8 and NPU
  1029.8, same 0.9250); summarize/extract on `GPU.0` unmeasured.
- **INT4 on CPU never benchmarked** (only INT4@NPU and INT8@CPU/NPU).
- **No batching.** Frequent small calls are served one at a time under a lock.

## Verification

```
$V -u scripts/fetch_aux_data.py
$V -u scripts/bench_aux.py --model ../models/qwen2.5-0.5b-instruct-int8-ov --device CPU
$V -u scripts/test_filter.py                # expect 10/10 INT8@NPU, gate PASS
$V -u scripts/prove_aux_npu.py NPU      # expect npu_active: true
$V -u scripts/aux_client.py             # expect VERDICT: PASS
```

Accept only if: sentiment accuracy ≥ 0.90 with 0 unparsed, extraction exact 10/10
on INT8@NPU after filtering (≥ 7/10 raw) with the idempotence gate passing,
with 0 hallucinations, `npu_active` true on NPU and false on CPU, and the client prints
`VERDICT: PASS`.

## See Also

- `results/aux_bench_*_{NPU,CPU}.json` — per-item outputs, ROUGE, accuracy, outcomes
- `results/aux_bench_gpu0.json` — sentiment-only iGPU spot-check (89.8 ms, 0.9250)
- `scripts/extract_filter.py`, `scripts/test_filter.py` — kind-specific post-filter
  (7/10 → 10/10 INT8@NPU) with the idempotence gate
- `results/aux_bench_*_grounding.json` — extrinsic-entity audit
- `results/aux_proof_*_{NPU,CPU}.json` — LUID device evidence
- `iniz-stt/SKILL.md` — where the NPU *does* win, and why the shape differs
- `iniz-agent-guard/SKILL.md` — the discriminative pattern (35 ms, NPU 2.7× faster)
