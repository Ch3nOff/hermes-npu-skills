"""
prove_aux_npu.py — is the generative model really on the NPU, and how loaded is it?

The CPU beat the NPU by ~20x on summarization (704.9 ms vs 14181.6 ms p50), which is
the opposite of the Whisper result. Two explanations must be separated:

  A. the NPU is genuinely running it, just badly (decode-bound, one token per graph
     execution, no KV-cache benefit)
  B. it silently fell back, and the "NPU" number is really something else

LLMPipeline does not expose EXECUTION_DEVICES, so this uses per-pid GPU-Engine
counters, same method as prove_whisper_npu.py.

LUID map for this machine:
  0x00000000_0x00011cf3 = NPU  (Intel AI Boost)
  0x00000000_0x00010480 = iGPU
  0x00000000_0x0001099d = dGPU RTX 5060
"""

import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

NPU_LUID = "0x00000000_0x00011cf3"
IGPU_LUID = "0x00000000_0x00010480"
PROMPT = ("<|im_start|>system\nYou are a summarizer. Reply with 2-3 short "
          "sentences.<|im_end|>\n<|im_start|>user\nSummarize: The quarterly report "
          "shows revenue up 12 percent on stronger subscription renewals, while "
          "hardware sales declined slightly in the same period.<|im_end|>\n"
          "<|im_start|>assistant\n")


def sample(duration_s, out_path, pid):
    ps = f"""
$deadline = (Get-Date).AddSeconds({duration_s})
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("ts,instance,util,cpu")
while ((Get-Date) -lt $deadline) {{
  try {{
    $cpu = (Get-Counter '\\Processor(_Total)\\% Processor Time' -ErrorAction Stop).CounterSamples[0].CookedValue
    $smp = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop).CounterSamples |
      Where-Object {{ $_.InstanceName -like "pid_{pid}_*" -and $_.CookedValue -gt 0.2 }}
    $ts = (Get-Date).ToString("HH:mm:ss.fff")
    if ($smp.Count -eq 0) {{ $lines.Add("$ts,(none),0,$([math]::Round($cpu,1))") }}
    foreach ($s in $smp) {{ $lines.Add("$ts,$($s.InstanceName),$([math]::Round($s.CookedValue,2)),$([math]::Round($cpu,1))") }}
  }} catch {{ }}
  Start-Sleep -Milliseconds 300
}}
$lines | Out-File -FilePath "{out_path}" -Encoding utf8
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    model = os.environ.get("INIZ_AUX_MODEL",
                           "../models/qwen2.5-0.5b-instruct-int8-ov")
    tag = os.environ.get("INIZ_AUX_TAG", f"{Path(model).name}_{device}")
    pid = os.getpid()
    print(f"pid={pid} device={device} model={Path(model).name}", flush=True)

    import openvino_genai as ov_genai

    t0 = time.time()
    pipe = ov_genai.LLMPipeline(model, device)
    print(f"compile: {time.time()-t0:.2f}s", flush=True)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 60
    cfg.do_sample = False
    pipe.generate(PROMPT, cfg)  # warmup

    csv = Path(tempfile.gettempdir()) / f"aux_proof_{tag}.csv"
    dur = 18
    t = threading.Thread(target=sample, args=(dur, str(csv), pid), daemon=True)
    t.start()
    time.sleep(1.0)

    n, toks, end = 0, 0, time.time() + dur - 3
    while time.time() < end:
        out = pipe.generate(PROMPT, cfg)
        n += 1
        toks += len(str(out).split())
    print(f"ran {n} generations (~{toks} words) in ~{dur-3}s", flush=True)
    t.join(timeout=25)

    if not csv.exists():
        print("counter CSV not created")
        return 1

    util = collections.defaultdict(list)
    cpus = []
    for line in csv.read_text(encoding="utf-8-sig").splitlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 4:
            try:
                if p[1] != "(none)":
                    util[p[1]].append(float(p[2]))
                cpus.append(float(p[3]))
            except ValueError:
                pass

    print(f"\nGPU-Engine instances for pid {pid} during {device} load:")
    hit_npu = hit_igpu = False
    if not util:
        print("  (none at all)")
    for k, v in sorted(util.items(), key=lambda kv: -max(kv[1])):
        luid = k.split("_luid_")[1].split("_phys")[0] if "_luid_" in k else "?"
        eng = k.split("engtype_")[1] if "engtype_" in k else "?"
        label = "NPU" if luid == NPU_LUID else ("iGPU" if luid == IGPU_LUID else luid)
        hit_npu |= luid == NPU_LUID
        hit_igpu |= luid == IGPU_LUID
        print(f"  {label:5s} engtype={eng:<10s} n={len(v):>3d} "
              f"max={max(v):>7.2f}% mean={sum(v)/len(v):>7.2f}%")
    if cpus:
        print(f"CPU _Total: mean={sum(cpus)/len(cpus):.1f}% max={max(cpus):.1f}%")

    print(f"\nVERDICT for device={device}:")
    if hit_npu:
        print("  NPU adapter WAS active -> genuinely on the NPU (just slow)")
    elif hit_igpu:
        print("  iGPU active, NPU idle -> fell back to the integrated GPU")
    else:
        print("  NO accelerator activity -> running on CPU despite device string")

    out = {"device_requested": device, "model": Path(model).name, "pid": pid,
           "generations": n, "npu_active": hit_npu, "igpu_active": hit_igpu,
           "cpu_mean_pct": round(sum(cpus)/len(cpus), 1) if cpus else None,
           "cpu_max_pct": round(max(cpus), 1) if cpus else None,
           "instances": {k: {"max": max(v), "mean": sum(v)/len(v), "n": len(v)}
                         for k, v in util.items()}}
    Path(f"aux_proof_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote aux_proof_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
