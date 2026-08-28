"""
luid_attribution.py — determine which LUID in the GPU Engine counters is the
NPU and which are the iGPU/dGPU.

Windows reports the NPU (MCDM adapter) INSIDE the 'GPU Engine' counter set, so
merely seeing 'there is GPU Engine activity' proves NOTHING. This experiment
runs a workload on exactly one device per process, then records the LUID +
engtype that shows up for that pid. That way every LUID can be labeled.

Usage:
  python luid_attribution.py NPU
  python luid_attribution.py GPU.0
  python luid_attribution.py CPU
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

import numpy as np

MODEL = "../models/iniz-guard-int8-ov/guard.xml"


def sample(duration_s, out_path, pid):
    ps = f"""
$deadline = (Get-Date).AddSeconds({duration_s})
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("timestamp,instance,util")
while ((Get-Date) -lt $deadline) {{
  try {{
    $smp = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop).CounterSamples |
      Where-Object {{ $_.InstanceName -like "pid_{pid}_*" }}
    $ts = (Get-Date).ToString("HH:mm:ss.fff")
    foreach ($s in $smp) {{ $lines.Add("$ts,$($s.InstanceName),$([math]::Round($s.CookedValue,2))") }}
  }} catch {{ }}
  Start-Sleep -Milliseconds 350
}}
$lines | Out-File -FilePath "{out_path}" -Encoding utf8
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    import openvino as ov
    core = ov.Core()
    model = core.read_model(MODEL)
    S = json.loads((Path(MODEL).parent / "guard_meta.json").read_text())["seq_len"]
    ids = np.ones((1, S), dtype=np.int64)
    mask = np.ones((1, S), dtype=np.int64)
    pid = os.getpid()
    print(f"pid={pid} device={device}")

    compiled = core.compile_model(model, device)
    print(f"EXECUTION_DEVICES = {compiled.get_property('EXECUTION_DEVICES')}")
    req = compiled.create_infer_request()
    req.infer({"input_ids": ids, "attention_mask": mask})

    csv = Path(tempfile.gettempdir()) / f"luid_{device.replace('.','_')}.csv"
    dur = 12
    t = threading.Thread(target=sample, args=(dur, str(csv), pid), daemon=True)
    t.start()
    time.sleep(1.0)
    n = 0
    end = time.time() + dur - 2.5
    while time.time() < end:
        req.infer({"input_ids": ids, "attention_mask": mask})
        n += 1
    print(f"{n} inferences run on {device}")
    t.join(timeout=20)

    if not csv.exists():
        print("CSV was not created")
        return 1
    util = collections.defaultdict(list)
    for line in csv.read_text(encoding="utf-8-sig").splitlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 3:
            try:
                util[p[1]].append(float(p[2]))
            except ValueError:
                pass
    print(f"\nGPU-Engine instances for pid {pid} during the {device} workload:")
    if not util:
        print("  (no instances at all)")
    rows = []
    for k, v in sorted(util.items(), key=lambda kv: -max(kv[1])):
        luid = k.split("_luid_")[1].split("_phys")[0] if "_luid_" in k else "?"
        eng = k.split("engtype_")[1] if "engtype_" in k else "?"
        print(f"  luid={luid}  engtype={eng:<10s} samples={len(v):>3d} "
              f"max={max(v):>7.2f}% mean={sum(v)/len(v):>7.2f}%")
        rows.append({"device_under_test": device, "luid": luid, "engtype": eng,
                     "samples": len(v), "max_util": max(v), "mean_util": sum(v) / len(v)})
    out = Path(f"luid_{device.replace('.','_')}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
