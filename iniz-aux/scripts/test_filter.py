"""
test_filter.py — apply extract_filter.refine to RECORDED model outputs.

No model re-runs: reads the three bench JSONs (INT8@NPU, INT4@NPU, INT8@CPU),
refines every row's recorded `output`, and reports exact-before/after per run
plus which rule fired. A rule that breaks any previously-exact answer fails
the run loudly — idempotence on correct answers is the acceptance gate.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_filter import refine

# Same resolution story as aux_client.py: fetched data, env wins, cwd default.
_cands = ([Path(os.environ["INIZ_AUX_DATA"])]
          if os.environ.get("INIZ_AUX_DATA") else []) + [Path("aux_data")]
DATA = next((p for p in _cands if p.exists()), None)
if DATA is None:
    raise FileNotFoundError(
        "aux_data/ not found — run scripts/fetch_aux_data.py from "
        "~/npu-provider/work/ or set INIZ_AUX_DATA.")
RUNS = [
    "aux_bench_int8_NPU_extract.json",
    "aux_bench_int4_NPU_extract.json",
    "aux_bench_qwen2.5-0.5b-instruct-int8-ov_CPU.json",
    "aux_bench_int4_CPU.json",
]


def main():
    _items = {it["id"]: it for it in json.loads(
        (DATA / "extract.json").read_text(encoding="utf-8"))}
    all_ok, broke = True, []
    summary = {}
    for f in RUNS:
        p = Path(f)
        if not p.exists():
            print(f"SKIP {f} (missing)")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d["extract"]["rows"] if "extract" in d else d.get("rows", [])
        before = sum(1 for r in rows if r.get("outcome") == "exact"
                     or r.get("parsed") == r.get("gold"))
        after, detail = 0, []
        for r in rows:
            it = _items[r["id"]]
            new, rule = refine(it["kind"], r.get("output", ""), it["text"])
            hit = new == it["answer"]
            after += hit
            was = r.get("parsed", r.get("output")) == it["answer"]
            if was and not hit:
                all_ok = False
                broke.append((f, r["id"], r.get("output"), new))
            if rule or (not was and hit) or (not was and not hit):
                detail.append((r["id"], was, hit, rule, r.get("output"), new))
        summary[f] = (before, after, len(rows))
        print(f"{f}: exact {before}/{len(rows)} -> {after}/{len(rows)}")
        for i, was, hit, rule, old, new in detail:
            print(f"  {i} was_exact={was} now_exact={hit} rule={rule}")
            print(f"    model={old!r}")
            print(f"    final={new!r}")
    print()
    if broke:
        print("BROKE previously-exact answers (gate FAILED):")
        for f, i, old, new in broke:
            print(f"  {f} {i}: {old!r} -> {new!r}")
        return 1
    print("idempotence gate: no exact answer was broken. PASS" if all_ok else "")
    print("note: tuned+tested on the same n=10 set — see module caveat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
