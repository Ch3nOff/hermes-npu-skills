"""
prove_whisper_npu.py — is Whisper ACTUALLY running on the NPU?

Why this script exists: the benchmark measured NPU p50 169.9 ms and CPU p50
167.8 ms — a 1% difference. For the guard classifier the NPU was 2.7x faster than
CPU. Identical numbers are the exact signature of a silent device fallback, which
pitfall #7/#8 of the guard skill warns about.

WhisperPipeline does not expose EXECUTION_DEVICES (it wraps several models), so
this uses the LUID attribution method instead: hammer the pipeline in a tight loop
while sampling per-pid GPU-Engine counters, then check which adapter lit up.

LUID map for this machine (from luid_attribution.py):
  0x00000000_0x00011cf3 = NPU  (Intel AI Boost)
  0x00000000_0x00010480 = iGPU (Intel Graphics)
  0x00000000_0x0001099d = dGPU (RTX 5060)

A real NPU run must light 0x11cf3. If nothing lights and CPU sits high, it is
running on CPU.
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

AUDIO = Path(os.environ.get("INIZ_PROOF_AUDIO", "audio"))
NPU_LUID = "0x00000000_0x00011cf3"
IGPU_LUID = "0x00000000_0x00010480"


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
    model = os.environ.get("INIZ_PROOF_MODEL", "../models/whisper-base-int8-ov")
    tag = os.environ.get("INIZ_PROOF_TAG", device.replace(".", "_"))
    pid = os.getpid()
    print(f"pid={pid} device={device} model={model} audio={AUDIO}", flush=True)

    import soundfile as sf
    import openvino_genai as ov_genai

    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    clip = min(manifest, key=lambda m: m["duration_s"])
    arr, _ = sf.read(AUDIO / clip["file"], dtype="float32")
    audio = arr.tolist()
    print(f"clip: {clip['file']} {clip['duration_s']}s", flush=True)

    pipe = ov_genai.WhisperPipeline(model, device)
    pipe.generate(audio)  # warmup

    csv = Path(tempfile.gettempdir()) / f"whisper_proof_{tag}.csv"
    dur = 16
    t = threading.Thread(target=sample, args=(dur, str(csv), pid), daemon=True)
    t.start()
    time.sleep(1.0)

    n, end = 0, time.time() + dur - 3
    while time.time() < end:
        pipe.generate(audio)
        n += 1
    print(f"ran {n} transcriptions in ~{dur-3}s", flush=True)
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
    if not util:
        print("  (none at all)")
    hit_npu = hit_igpu = False
    for k, v in sorted(util.items(), key=lambda kv: -max(kv[1])):
        luid = k.split("_luid_")[1].split("_phys")[0] if "_luid_" in k else "?"
        eng = k.split("engtype_")[1] if "engtype_" in k else "?"
        dev_label = "NPU" if luid == NPU_LUID else ("iGPU" if luid == IGPU_LUID else luid)
        if luid == NPU_LUID:
            hit_npu = True
        if luid == IGPU_LUID:
            hit_igpu = True
        print(f"  {dev_label:5s} engtype={eng:<10s} n={len(v):>3d} "
              f"max={max(v):>7.2f}% mean={sum(v)/len(v):>7.2f}%")

    if cpus:
        print(f"CPU _Total: mean={sum(cpus)/len(cpus):.1f}% max={max(cpus):.1f}%")

    print(f"\nVERDICT for device={device}:")
    if hit_npu:
        print("  NPU adapter WAS active -> genuinely running on the NPU")
    elif hit_igpu:
        print("  iGPU active, NPU idle -> fell back to the integrated GPU")
    else:
        print("  NO accelerator activity -> running on CPU despite device string")

    out = {"device_requested": device, "model": model, "audio_dir": str(AUDIO),
           "pid": pid, "transcriptions": n,
           "npu_active": hit_npu, "igpu_active": hit_igpu,
           "cpu_mean_pct": round(sum(cpus)/len(cpus), 1) if cpus else None,
           "cpu_max_pct": round(max(cpus), 1) if cpus else None,
           "instances": {k: {"max": max(v), "mean": sum(v)/len(v), "n": len(v)}
                         for k, v in util.items()}}
    Path(f"whisper_proof_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote whisper_proof_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
