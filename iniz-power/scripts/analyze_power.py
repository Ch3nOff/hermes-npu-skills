"""
analyze_power.py — turn power_probe.py JSON files into honest comparisons.

Uses MEDIAN package power (robust to the 13-18W background spikes), with the
spread across all baseline runs as the error bar. Prints:

  * idle floor: median range + mean +/- sd across every power_baseline*.json
  * per workload: throughput, median pkg power, marginal vs idle median,
    total + marginal energy, energy per work unit
  * NPU vs CPU verdict per task on ENERGY PER UNIT (not watts, not latency)

A marginal smaller than the baseline run-to-run spread is reported as
UNRESOLVED, not as a number.
"""

import glob
import json
import statistics
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    base_files = sorted(glob.glob("power_baseline*.json"))
    # power_baseline_final.json is a copy of the final baseline, not new data
    base_files = [f for f in base_files if "final" not in f]
    bases = [load(f) for f in base_files
             if load(f).get("label") == "baseline"]
    meds = [b["rails"]["rapl_package0_pkg"]["median_mw"] for b in bases]
    means = [b["rails"]["rapl_package0_pkg"]["mean_mw"] for b in bases]
    idle_med = statistics.median(meds)
    spread = max(meds) - min(meds)

    print(f"idle floor from {len(bases)} baselines:")
    print(f"  median range : {min(meds):.1f} .. {max(meds):.1f} mW "
          f"(spread {spread:.1f} mW <- error bar)")
    print(f"  means        : {min(means):.1f} .. {max(means):.1f} mW")
    print(f"  adopted idle : {idle_med:.1f} mW (median of medians)")

    print(f"\n{'task':22s}{'units':>7s}{'thr/s':>8s}"
          f"{'pkg_med':>10s}{'marg':>9s}{'mWh/unit':>11s}  verdict")
    results = {}
    for f in sorted(glob.glob("power_*.json")):
        d = load(f)
        if d.get("label", "").startswith("baseline") or "rails" not in d:
            continue
        r = d["rails"]["rapl_package0_pkg"]
        units = d["work_units"] or 0
        thr = units / d["elapsed_s"] if units else 0
        marg = r["median_mw"] - idle_med
        mwh = (r["median_mw"] * d["elapsed_s"] / 3600.0 / units) if units else 0
        mmwh = (marg * d["elapsed_s"] / 3600.0 / units) if units else 0
        ok = abs(marg) >= spread and units > 0
        tag = "RESOLVED" if ok else ("UNRESOLVED" if units else "no units")
        print(f"{d['label']:22s}{units:>7d}{thr:>8.2f}"
              f"{r['median_mw']:>10.1f}{marg:>9.1f}{mmwh:>11.5f}  {tag}")
        results[d["label"]] = {
            "units": units, "thr_per_s": round(thr, 2),
            "pkg_med_mw": r["median_mw"], "marginal_med_mw": round(marg, 1),
            "marginal_mwh_per_unit": round(mmwh, 5) if units else None,
            "resolved": ok,
        }

    print("\n--- NPU vs CPU energy per work unit (marginal mWh) ---")
    for task in ("guard", "stt", "aux"):
        n, c = results.get(f"{task}_NPU"), results.get(f"{task}_CPU")
        if not (n and c):
            print(f"{task}: MISSING a device run")
            continue
        if not (n["resolved"] and c["resolved"]):
            print(f"{task}: cannot compare, a side is UNRESOLVED")
            continue
        ratio = n["marginal_mwh_per_unit"] / c["marginal_mwh_per_unit"]
        winner = "CPU" if ratio > 1 else "NPU"
        print(f"{task}: NPU {n['marginal_mwh_per_unit']:.5f} vs "
              f"CPU {c['marginal_mwh_per_unit']:.5f}  -> "
              f"{winner} wins by {max(ratio, 1/ratio):.2f}x per task")

    Path("power_analysis.json").write_text(
        json.dumps({"idle_median_mw": idle_med, "idle_spread_mw": spread,
                    "results": results}, indent=2))
    print("\nwrote power_analysis.json")


if __name__ == "__main__":
    main()
