"""
eval_final.py — evaluasi produksi FINAL memakai:
  * IR INT8 di NPU
  * mapping label TERVALIDASI dari guard_labels.py (mapping paket incoming,
    terbukti +0.39 macro-F1 vs mapping notebook lama)

Menghasilkan per-split:
  * metrik action (accuracy, macro-F1, per-class P/R/F1, confusion)
  * MAE + ROC-AUC untuk head injection & shell
  * sweep threshold injection untuk gate biner attack-vs-benign
  * latensi nyata p50/p90
  * crosstab kategori x action prediksi
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from guard_labels import ACTIONS, ACTION2IDX, apply_labels


def per_class(y_true, y_pred, n):
    rows = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"action": ACTIONS[c], "support": support, "tp": tp,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
    present = [r["f1"] for r in rows if r["support"] > 0]
    return rows, float(np.mean(present)) if present else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/iniz-guard-int8-ov/guard.xml")
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--out", default="eval_final.json")
    args = ap.parse_args()

    import openvino as ov
    from transformers import AutoTokenizer
    from sklearn.metrics import roc_auc_score

    meta = json.loads((Path(args.model).parent / "guard_meta.json").read_text())
    S = meta["seq_len"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    core = ov.Core()
    t0 = time.time()
    compiled = core.compile_model(core.read_model(args.model), args.device)
    print(f"[{args.device}] compiled in {time.time()-t0:.2f}s  seq_len={S}")
    try:
        print(f"[{args.device}] EXECUTION_DEVICES = {compiled.get_property('EXECUTION_DEVICES')}")
    except Exception:
        pass
    req = compiled.create_infer_request()

    out = {"model": args.model, "device": args.device, "seq_len": S,
           "label_mapping": "guard_labels.py (tervalidasi)", "splits": {}}

    for split in args.splits.split(","):
        split = split.strip()
        df = apply_labels(pd.read_parquet(f"data/full-{split}.parquet"))
        texts = df["text"].tolist()
        enc = tok(texts, return_tensors="np", padding="max_length",
                  truncation=True, max_length=S)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)

        req.infer({"input_ids": ids[0:1], "attention_mask": mask[0:1]})
        inj_p, sh_p, act_i, lat = [], [], [], []
        for i in range(len(texts)):
            s = time.time()
            r = req.infer({"input_ids": ids[i:i + 1], "attention_mask": mask[i:i + 1]})
            lat.append((time.time() - s) * 1000)
            v = list(r.values())
            inj_p.append(float(np.array(v[0]).flatten()[0]))
            sh_p.append(float(np.array(v[1]).flatten()[0]))
            act_i.append(int(np.array(v[2]).flatten().argmax()))
            if i % 300 == 0:
                print(f"  {split} {i}/{len(texts)}", flush=True)
        inj_p, sh_p = np.array(inj_p), np.array(sh_p)
        act_p = np.array(act_i)

        act_t = np.array([ACTION2IDX[a] for a in df["action"]])
        inj_t = df["inj"].to_numpy(float)
        sh_t = df["shell"].to_numpy(float)
        lab = df["label"].to_numpy(int)

        print(f"\n########## {split} (n={len(df)}) ##########")
        p50 = statistics.median(lat)
        p90 = sorted(lat)[int(.9 * len(lat)) - 1]
        print(f"latency p50={p50:.1f}ms p90={p90:.1f}ms")

        acc = float((act_p == act_t).mean())
        rows, mf1 = per_class(act_t, act_p, 4)
        print(f"\naction: accuracy={acc:.4f}  macro-F1(kelas ada)={mf1:.4f}")
        print(f"{'action':<20}{'support':>9}{'prec':>9}{'recall':>9}{'f1':>9}")
        for r in rows:
            print(f"{r['action']:<20}{r['support']:>9}{r['precision']:>9.4f}"
                  f"{r['recall']:>9.4f}{r['f1']:>9.4f}")
        print("\nconfusion (rows=true, cols=pred)")
        print("              " + "".join(f"{a[:9]:>11s}" for a in ACTIONS))
        for i, a in enumerate(ACTIONS):
            row = [int(((act_t == i) & (act_p == j)).sum()) for j in range(4)]
            print(f"{a[:11]:11s}   " + "".join(f"{v:>11d}" for v in row))

        mae_i = float(np.abs(inj_p - inj_t).mean())
        mae_s = float(np.abs(sh_p - sh_t).mean())
        auc_i = float(roc_auc_score(lab, inj_p))
        # AUC shell: apakah shell head bisa membedakan proxy-hit (0.6) vs bukan
        sh_bin = (sh_t >= 0.6).astype(int)
        auc_s = float(roc_auc_score(sh_bin, sh_p)) if sh_bin.sum() else float("nan")
        print(f"\ninjection: MAE={mae_i:.4f}  ROC-AUC(attack)={auc_i:.4f}")
        print(f"shell    : MAE={mae_s:.4f}  ROC-AUC(proxy-hit)={auc_s:.4f}  "
              f"n_proxy_hit={int(sh_bin.sum())}")

        pred_bad = (act_p != 0).astype(int)
        tp = int(((pred_bad == 1) & (lab == 1)).sum()); fp = int(((pred_bad == 1) & (lab == 0)).sum())
        fn = int(((pred_bad == 0) & (lab == 1)).sum()); tn = int(((pred_bad == 0) & (lab == 0)).sum())
        gp = tp / (tp + fp) if tp + fp else 0
        gr = tp / (tp + fn) if tp + fn else 0
        print(f"\ngate via action!=PASS: P={gp:.4f} R={gr:.4f} "
              f"F1={2*gp*gr/(gp+gr) if gp+gr else 0:.4f} (TP={tp} FP={fp} FN={fn} TN={tn})")

        sweep, best = [], None
        for th in np.arange(0.05, 0.95, 0.05):
            pb = (inj_p >= th).astype(int)
            t_p = int(((pb == 1) & (lab == 1)).sum()); f_p = int(((pb == 1) & (lab == 0)).sum())
            f_n = int(((pb == 0) & (lab == 1)).sum())
            pr = t_p / (t_p + f_p) if t_p + f_p else 0
            rc = t_p / (t_p + f_n) if t_p + f_n else 0
            f1b = 2 * pr * rc / (pr + rc) if pr + rc else 0
            e = {"th": round(float(th), 2), "P": round(pr, 4), "R": round(rc, 4),
                 "F1": round(f1b, 4), "FP": f_p, "FN": f_n}
            sweep.append(e)
            if best is None or f1b > best["F1"]:
                best = e
        print(f"threshold terbaik: {best}")

        df2 = df.assign(p_action=[ACTIONS[i] for i in act_p])
        ct = pd.crosstab(df2["category"], df2["p_action"])
        print(f"\nkategori x action prediksi:\n{ct.to_string()}")

        out["splits"][split] = {
            "n": len(df), "latency_p50_ms": round(p50, 1), "latency_p90_ms": round(p90, 1),
            "acc_action": acc, "macro_f1_present_classes": mf1, "per_class": rows,
            "mae_injection": mae_i, "mae_shell": mae_s,
            "roc_auc_injection_attack": auc_i, "roc_auc_shell_proxy": auc_s,
            "gate_via_action": {"precision": gp, "recall": gr,
                                "tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "threshold_sweep": sweep, "best_threshold": best,
            "pred_action_dist": {ACTIONS[i]: int((act_p == i).sum()) for i in range(4)},
        }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
