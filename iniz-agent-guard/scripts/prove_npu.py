"""
prove_npu.py — hard evidence that inference really runs on the NPU and does not
silently fall back to the CPU.

Three layers of proof:
  1. compiled_model.get_property("EXECUTION_DEVICES") — the OpenVINO runtime's own
     report of the device that ACTUALLY executes the graph.
  2. NPU vs CPU vs GPU load: run N inferences while sampling the Windows
     GPU Engine Utilization Percentage and Processor(_Total) counters.
     The NPU has no counter set of its own in this Windows build (verified:
     Get-Counter -ListSet only has GPU*) — the NPU instead shows up as an adapter
     inside GPU Engine. That is why the LUID must first be mapped via
     luid_attribution.py.
  3. Explicit NPU vs CPU throughput comparison — if the NPU were secretly the CPU,
     its latency would be identical to device="CPU".
"""

import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

MODEL = "../models/iniz-guard-int8-ov/guard.xml"
N_ITER = 60


def sample_counters(duration_s, out_path):
    ps = f"""
$deadline = (Get-Date).AddSeconds({duration_s})
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("timestamp,instance,util_pct,cpu_total")
while ((Get-Date) -lt $deadline) {{
  try {{
    $cpu = (Get-Counter '\\Processor(_Total)\\% Processor Time' -ErrorAction Stop).CounterSamples[0].CookedValue
    $smp = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop).CounterSamples |
      Where-Object {{ $_.CookedValue -gt 0.5 }} | Sort-Object CookedValue -Descending | Select-Object -First 6
    $ts = (Get-Date).ToString("HH:mm:ss.fff")
    if ($smp.Count -eq 0) {{ $lines.Add("$ts,(no-gpu-activity),0,$([math]::Round($cpu,1))") }}
    foreach ($s in $smp) {{ $lines.Add("$ts,$($s.InstanceName),$([math]::Round($s.CookedValue,2)),$([math]::Round($cpu,1))") }}
  }} catch {{ }}
  Start-Sleep -Milliseconds 400
}}
$lines | Out-File -FilePath "{out_path}" -Encoding utf8
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True)


def bench(core, model, device, ids, mask, n=N_ITER):
    import openvino as ov
    t0 = time.time()
    compiled = core.compile_model(model, device)
    compile_s = time.time() - t0
    try:
        exec_dev = compiled.get_property("EXECUTION_DEVICES")
    except Exception as e:
        exec_dev = f"<unavailable: {type(e).__name__}>"
    req = compiled.create_infer_request()
    req.infer({"input_ids": ids, "attention_mask": mask})
    lat = []
    for _ in range(n):
        t0 = time.time()
        req.infer({"input_ids": ids, "attention_mask": mask})
        lat.append((time.time() - t0) * 1000)
    return {"device": device, "execution_devices": str(exec_dev),
            "compile_s": round(compile_s, 2),
            "p50_ms": round(statistics.median(lat), 2),
            "p90_ms": round(sorted(lat)[int(.9 * len(lat)) - 1], 2),
            "min_ms": round(min(lat), 2), "n": n}


def main():
    import openvino as ov
    import os
    core = ov.Core()
    model = core.read_model(MODEL)
    S = json.loads((Path(MODEL).parent / "guard_meta.json").read_text())["seq_len"]
    ids = np.ones((1, S), dtype=np.int64)
    mask = np.ones((1, S), dtype=np.int64)
    npu_only = "--npu-only" in sys.argv
    print(f"pid={os.getpid()}  seq_len={S}  npu_only={npu_only}")
    print(f"available: {core.available_devices}\n")

    results = []
    if not npu_only:
        print("=== LAYER 1+3: EXECUTION_DEVICES + throughput per device ===")
        for dev in ["NPU", "CPU", "GPU.0"]:
            if dev not in core.available_devices:
                print(f"  {dev}: not available")
                continue
            try:
                r = bench(core, model, dev, ids, mask)
                results.append(r)
                print(f"  {dev:6s} exec={r['execution_devices']:<12s} compile={r['compile_s']:>6.2f}s "
                      f"p50={r['p50_ms']:>7.2f}ms p90={r['p90_ms']:>7.2f}ms")
            except Exception as e:
                print(f"  {dev:6s} FAILED: {type(e).__name__}: {e}")
                results.append({"device": dev, "error": f"{type(e).__name__}: {e}"})
        print("\nNOTE: the phase above ALSO COMPILES the model on GPU.0, so this "
              "process holds a GPU context. For a clean layer 2, re-run with "
              "--npu-only.")

    print("\n=== LAYER 2: Windows counters during NPU load ===")
    csv_path = Path(tempfile.gettempdir()) / ("npu_proof_counters_npuonly.csv"
                                              if npu_only else "npu_proof_counters.csv")
    dur = 14
    t = threading.Thread(target=sample_counters, args=(dur, str(csv_path)), daemon=True)
    t.start()
    time.sleep(1.5)
    compiled = core.compile_model(model, "NPU")
    try:
        print(f"  EXECUTION_DEVICES = {compiled.get_property('EXECUTION_DEVICES')}")
    except Exception:
        pass
    req = compiled.create_infer_request()
    n_load = 0
    t_end = time.time() + dur - 3
    while time.time() < t_end:
        req.infer({"input_ids": ids, "attention_mask": mask})
        n_load += 1
    print(f"  ran {n_load} NPU inferences over ~{dur-3}s")
    t.join(timeout=20)

    own_report = None
    if csv_path.exists():
        rows = [l.strip() for l in csv_path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        print(f"  counter samples: {len(rows)-1}")
        import collections
        inst = collections.Counter()
        util = collections.defaultdict(list)
        cpus = []
        for l in rows[1:]:
            p = l.split(",")
            if len(p) >= 4:
                inst[p[1]] += 1
                try:
                    util[p[1]].append(float(p[2]))
                    cpus.append(float(p[3]))
                except ValueError:
                    pass
        print("  top GPU-engine instances during the load:")
        for k, v in inst.most_common(8):
            mx = max(util[k]) if util[k] else 0
            print(f"    {v:>4d}x  max_util={mx:>6.2f}%  {k}")
        if cpus:
            print(f"  CPU _Total: mean={statistics.mean(cpus):.1f}% max={max(cpus):.1f}%")
        pid = str(os.getpid())
        own = [k for k in inst if f"pid_{pid}" in k]
        own_report = own
        if own:
            print(f"  GPU-engine instances owned by this pid ({pid}):")
            for k in own:
                print(f"    max_util={max(util[k]):.2f}%  {k}")
        else:
            print(f"  GPU-engine instances owned by this pid ({pid}): NONE "
                  f"-> the process does not use the GPU at all")
    else:
        print("  counter CSV was not created")

    out = {"npu_only_mode": npu_only, "execution_device_report": results,
           "counters_csv": str(csv_path), "npu_inferences_under_load": n_load,
           "own_pid_gpu_instances": own_report}
    Path("npu_proof_npuonly.json" if npu_only else "npu_proof.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote {'npu_proof_npuonly.json' if npu_only else 'npu_proof.json'}")


if __name__ == "__main__":
    sys.exit(main())
