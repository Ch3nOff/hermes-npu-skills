---
name: iniz-power
description: Measure real energy per AI task with Intel RAPL.
version: 0.1.0
author: Matthew Chen (CH3NDev), Hermes Agent
license: MIT
platforms: [windows]
metadata:
  hermes:
    tags: [openvino, npu, power, energy, rapl, measurement, local-ai]
    related_skills: [iniz-agent-guard, iniz-stt, iniz-aux]
---

# Skill: Iniz Power — Real Energy per Task on Intel (RAPL)

**Trigger:** When someone claims a device "saves power" for an AI workload, or when
deciding NPU vs CPU on battery. Utilization percentages and latency ratios are not
energy — measure it.

**Behavior:** Sample Intel RAPL package power via the Windows `Energy Meter` counters
while hammering a workload in a tight loop, subtract an idle baseline, and report
**marginal energy per work unit**. Every number ships with its error bar, and effects
smaller than baseline drift are reported as UNRESOLVED instead of as numbers.

**Environment:** Windows with `\Energy Meter(*)\Power` counters (present on this
Intel Core Ultra 9 275HX, driver 32.0.100.4512). No extra installs — sampling is a
PowerShell one-liner driven from Python. Linux needs a different path (RAPL sysfs);
this skill does not cover it.

---

## Headline results (Core Ultra 9 275HX, OpenVINO 2026.3)

Marginal SoC-package energy per work unit, idle subtracted, median-based. The idle
floor moved mid-campaign (7 W → 27 W, see below), so every ratio is reported as a
**range**: quiet-floor assumption first, elevated-floor assumption second. No verdict
flips between them.

| Workload | NPU | CPU | Winner per task |
|---|---|---|---|
| guard inference (Qwen2.5-0.5B + heads, INT8) | 0.248 / 0.056 mWh | 2.18 / 1.73 mWh | **NPU, 8.8–30.8× less** |
| Whisper-base transcription (4.8 s clip, INT8) | 1.22 / 0.43 mWh | 4.36 / 3.34 mWh | **NPU, 3.6–7.8× less** |
| Qwen2.5-0.5B generation (60 tokens, INT8) | 72.6 / 18.7 mWh | 10.0 / 7.2 mWh | **CPU, 2.6–7.2× less** |

Throughput in the same runs: guard 28.4/s NPU vs 12.1/s CPU; STT 6.9/s vs 5.3/s;
aux 0.10/s NPU vs 1.90/s CPU.

Scaled to a full charge (quiet-floor figures; assumes an ~86 Wh battery from
`Win32_Battery` `RemainingCapacity`, ignoring everything else in the system):

| Workload | NPU | CPU |
|---|---|---|
| guard inferences | ~347,000 | ~39,000 |
| transcriptions | ~70,000 | ~20,000 |
| generations (60 tok) | ~1,200 | ~8,500 |

The battery column is an illustration, not a promise — display, radio, and dGPU are
outside RAPL package scope.

---

## The lesson that justifies this skill: watts ≠ energy

The aux case is why utilization reasoning fails. The NPU draws **less power**
(33.6 W vs 76.1 W package median) but takes **19× longer** per generation
(9.9 s vs 0.5 s), so it spends **2.6–7.2× more energy** per task depending on floor
assumption. Anyone watching only watts, or only CPU %, would route this workload to
the NPU and drain the battery faster. The guard and STT cases go the other way (NPU
wins 8.8–30.8× and 3.6–7.8×), which is exactly why per-workload measurement beats a
blanket rule.

---

## Scope: what these numbers are and are not

- **RAPL package (`rapl_package0_pkg`), not wall power.** Display, dGPU, chargers,
  and PSU losses are excluded. Do not compare these mWh figures with wall-plug
  measurements.
- **Whether the NPU tile sits inside package0 on Arrow Lake-HX is NOT verified
  here.** If any NPU consumption falls outside the package domain, NPU-side figures
  are UNDER-counted and the NPU looks better than it is. Treat NPU wins as a lower
  bound, not a settled figure.
- **Tight-loop saturation, not sporadic use.** Both devices were pegged; a single
  request against an idle machine costs less (race-to-idle dominates). Energy per
  unit under saturation is the correct A/B metric, but do not quote it as "one
  inference drains X from your battery while you browse".
- **Median-based.** Idle shows 13–18 W background spikes, so all comparisons use
  medians. Means are in the JSON for inspection.
- **Coarse sampling.** `Get-Counter` yields ~1 sample/s despite the 250 ms sleep, so
  a 25 s run is n≈22. Adequate for 25–90 s windows, useless for single requests.

## The baseline moved mid-campaign — read this

Idle floor before the campaign: 6.6 / 6.8 / 7.5 W medians across three 30 s runs.
Idle floor right after: **26.6 W**, sustained (min 24.2 W), with no benchmark process
alive — verified zero `npu-provider` python processes and per-process CPU under
0.5 cores each. The desktop was active (Edge, Hermes, Roblox Studio, Discord), so the
likely cause is user-side activity, but that is an inference, not a measurement.

What this means for the numbers above:

- Workload signals (33–102 W) dwarf the drift (≤20 W). NPU-vs-CPU **ratios** are
  approximately robust; absolute marginals carry a ±20 W package-level uncertainty
  that no statistics can remove retroactively.
- Marginals were computed against the pre-campaign floor (7.2 W, median of medians —
  robust to the outlier by construction).
- `results/` keeps all five baselines (`run1–3`, `final`, `check2`) so anyone can see
  the drift instead of taking a single floor on faith.
- If you re-run: do it with the machine left alone, and take a baseline immediately
  before AND after. The script refuses to run workloads without a baseline file, but
  it cannot refuse a stale one — that check is on you.

---

## Procedure

Run from `npu-provider/work/` with the project venv (`$V` = `.venv/Scripts/python.exe`).

1. **Idle baseline, machine untouched.** Completion: `power_baseline.json` exists with
   ~30 s of samples. Repeat 2–3× and check the spread before continuing — if medians
   move by more than your expected effect size, stop; the machine is too busy.
   ```
   terminal(command="$V -u scripts/power_probe.py baseline")   # INIZ_POWER_SECONDS=30 default
   ```

2. **One workload, one device, sequentially.** Never parallelize — two loads share the
   package and both numbers become garbage.
   ```
   terminal(command="$V -u scripts/power_probe.py guard NPU")
   terminal(command="$V -u scripts/power_probe.py guard CPU")
   terminal(command="$V -u scripts/power_probe.py stt NPU")
   terminal(command="$V -u scripts/power_probe.py aux CPU")
   ```
   Compile/warmup runs before sampling starts, so load cost is excluded. Slow workloads
   need longer windows: aux on NPU managed 10 units in 90 s (`INIZ_POWER_SECONDS=90`).

3. **Analyze.** Completion: `power_analysis.json` with per-task verdicts.
   ```
   terminal(command="$V scripts/analyze_power.py")
   ```
   Effects smaller than the baseline run-to-run spread print as UNRESOLVED.

4. **Closing baseline.** Proves the floor did not move under you. If it did, say so in
   the writeup with both floors shown.

---

## Pitfalls (all hit on this machine)

1. **`Win32_Battery DischargeRate reads 0 on AC.** `PowerLineStatus=Online`,
   `DischargeRate=0` — the battery sensor is blind while plugged in. RAPL works
   either way, which is why this skill uses it.

2. **`Get-Counter` on Energy Meter is ~1 sample/s, not 4.** The 250 ms sleep does not
   set the rate; each counter read costs ~1.1 s. Size windows accordingly (≥25 s) and
   do not try to profile single requests this way.

3. **The idle floor is the measurement, not a formality.** Three quiet runs agreed
   within 0.9 W; the post-campaign run was 19 W higher with no benchmark alive. A
   single baseline taken on faith would have silently biased every marginal.

4. **Tight loops pin all cores on CPU.** CPU-side package power (76–102 W) is a 275HX
   at full turbo, not "one user asking one question". Correct for A/B, misleading as
   an absolute ("the CPU uses 100 W to transcribe").

5. **Warmup matters more than usual here.** NPU compile runs 1–103 s depending on
   model; sampling through it would charge one-time cost to every unit. The script
   compiles and warms up before the sampler starts.

6. **No benchmark orphans — verify, don't assume.** After the campaign, two `python`
   processes with high cumulative CPU turned out to be the user's Hermes CLI servers
   (started hours earlier), not leaked samplers. `Win32_Process.CommandLine` settles
   it in seconds; killing them would have broken the user's setup.

7. **Sequential or nothing.** Background runs share the package. The campaign ran six
   workloads plus baselines in one shell chain precisely so nothing overlapped.

## Verification

```
$V -u scripts/power_probe.py baseline            # floor, machine untouched
$V -u scripts/power_probe.py guard NPU           # EXECUTION_DEVICES must print NPU
$V -u scripts/power_probe.py guard CPU
$V scripts/analyze_power.py                      # expect RESOLVED verdicts
```

Accept only if: the device line prints the requested device, `power_analysis.json`
marks the comparison RESOLVED, and a closing baseline is within a few W of the
opening one — otherwise report the drift, not just the headline.

## See Also

- `results/power_analysis.json` — the verdict table
- `results/power_{guard,stt,aux}_{NPU,CPU}.json` — per-run rails, units, throughput
- `results/power_baseline_{run1,run2,run3,final,check2}.json` — the floor, including
  the drift
- `iniz-aux/SKILL.md` — the 20× latency gap that motivated the energy question
- `iniz-stt/SKILL.md`, `iniz-agent-guard/SKILL.md` — the workloads measured
