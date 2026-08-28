# NPU Utilization Evidence — Iniz Agent Guard

Status: fills in the TODO previously flagged in this skill package.
This evidence was gathered during the `hermes-npu-provider` debugging
session (the predecessor of this project, a generative
Qwen2.5-0.5B-Instruct model — not the fine-tuned 3-head version), but
the verification methodology applies equally to confirming that
`guard_server.py` really runs on the NPU rather than silently falling
back to CPU/GPU.

## Evidence summary

Three independent sources were collected and they agree with each other:

1. **Visual Task Manager** — three screenshots show the NPU graph
   jumping to a plateau pattern (a sharp rise when inference starts,
   flat at the top for the duration, a sharp drop when it finishes)
   exactly within the time window in which the benchmark ran, while
   GPU 0 (Intel iGPU) and GPU 1 (NVIDIA RTX discrete) stayed flat and
   low over the same window.

2. **CSV performance sampler** (`perf_sampler_gpu_detail.ps1` in this
   `references/` folder) — records `gpu_util_sum_pct` and
   `cpu_util_pct` per timestamp. The CPU peaked at roughly 4.7% during
   the inference window, far below what you would expect if the
   workload were really running on the CPU.

3. **Per-process GPU attribution** — the per-PID breakdown shows that
   the GPU activity in that same window came from UI processes (Hermes
   UI itself, the Windows DWM compositor) — NOT from the Python/server
   process running model inference.

## ⚠️ An honest note on the scope of this evidence

- **This evidence was gathered for a plain 0.5B generative model**
  (`hermes-npu-provider`, before the 3-head fine-tuning), not for the
  current `guard_server.py` directly. The methodology is identical and
  the device (`NPU`) in `guard_server.py` has not changed, but if you
  want evidence that is 100% specific to the fine-tuned 3-head model,
  repeat this Task Manager capture + CSV sampler AFTER the fine-tuned
  model has actually been exported to OpenVINO INT8 and served through
  `guard_server.py`.
- **`gpu_util_sum_pct` in the raw CSV did at one point show 17-41%**
  in the same window as NPU inference — this looked suspicious AT
  FIRST, but the per-process breakdown (point 3 above) explains that it
  came from the Hermes UI rendering streaming output in real time, not
  from the inference process itself. Noted here so that it is not
  misread as contradictory evidence by someone re-reading the raw CSV
  without this context.
- **The measured latency** (~2.7–6.6 seconds depending on INT4/INT8
  quantization) is far above the blueprint's original target (<15ms) —
  a fact anyone reading this proof-of-utilization needs to know so they
  do not conflate "the NPU is being used" with "the NPU is fast". NPU
  usage is confirmed; its absolute performance is a separate question,
  and the current generation of NPU hardware (Meteor Lake/Arrow Lake)
  still has high variance even for small LLM workloads.

## How to replicate this verification yourself

1. Run `scripts/guard_server.py`.
2. Open Task Manager > Performance > NPU **before** firing any request
   at the server.
3. Send a few test requests (see the curl examples in the Quick
   Reference section of `SKILL.md`).
4. Observe: the NPU graph must rise in the same time window in which
   the requests are processed; the discrete GPU graph (if any) must
   stay flat over that same window.
5. Run `references/perf_sampler_gpu_detail.ps1` in parallel to obtain a
   CSV record that can be re-examined, rather than just a fleeting
   visual observation.

Genuinely convincing evidence requires all three of the above to line
up within one and the same time window — not just a single source on
its own.
