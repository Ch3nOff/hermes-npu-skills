"""
SUPERSEDED: this is an early script; use scripts/eval_final.py instead
(this one relies on an outdated label mapping).

eval_guard.py — REAL evaluation of the fine-tuned checkpoint on the
validation/test split.

Metrics:
  - action: accuracy, macro-F1, confusion matrix
  - injection / shell: MAE against the notebook's label mapping
  - binary gate: does "not PASS" detect label=1 (precision/recall/F1)

It also compares pooling strategies (last_nonpad / mean / last / first),
because the checkpoint does not store that information.

Usage:
  python eval_guard.py --split validation --pooling last_nonpad --limit 0
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from guard_model import load_checkpoint, ACTIONS, ACTION2IDX

SEVERITY_MAP = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}

# Label mapping from notebook 05 (cell 3). Categories outside the explicit
# lists fall back to USER_CONFIRMATION.
PAUSE_CATS = {"direct_injection", "prompt_extraction", "system_extraction", "agent_manipulation"}
ISOLATE_CATS = {"code_execution", "instruction_override"}


def map_labels(row):
    cat, sev, label = row["category"], row["severity"], int(row["label"])
    inj, sh, act = 0.0, 0.0, "PASS"
    if label == 1:
        inj = SEVERITY_MAP.get(sev, 0.5)
        if cat == "code_execution":
            sh, act = 0.9, "ISOLATE_FILE"
        elif cat == "instruction_override":
            sh, act = 0.5, "ISOLATE_FILE"
        elif cat in PAUSE_CATS:
            act = "PAUSE_AGENTS"
        else:
            act = "USER_CONFIRMATION"
    return pd.Series({"inj": inj, "shell": sh, "action": act})


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
        elif tp == 0:
            continue
        else:
            p, r = tp / (tp + fp), tp / (tp + fn)
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def run(model, tok, df, batch_size=16, max_len=192):
    inj_p, sh_p, act_p = [], [], []
    t0 = time.time()
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])
        inj_p.append(out["injection"].float().numpy())
        sh_p.append(out["shell"].float().numpy())
        act_p.append(out["action_logits"].float().argmax(-1).numpy())
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  {done}/{len(texts)}  ({time.time()-t0:.0f}s)", flush=True)
    return (np.concatenate(inj_p), np.concatenate(sh_p), np.concatenate(act_p),
            time.time() - t0)


def report(df, inj_p, sh_p, act_p, elapsed, tag):
    inj_t = df["inj"].to_numpy(dtype=float)
    sh_t = df["shell"].to_numpy(dtype=float)
    act_t = np.array([ACTION2IDX[a] for a in df["action"]])
    lab = df["label"].to_numpy(dtype=int)

    acc = float((act_p == act_t).mean())
    f1 = macro_f1(act_t, act_p, len(ACTIONS))
    mae_i = float(np.abs(inj_p - inj_t).mean())
    mae_s = float(np.abs(sh_p - sh_t).mean())

    # binary gate: a non-PASS prediction means "dangerous"
    pred_bad = (act_p != 0).astype(int)
    tp = int(((pred_bad == 1) & (lab == 1)).sum())
    fp = int(((pred_bad == 1) & (lab == 0)).sum())
    fn = int(((pred_bad == 0) & (lab == 1)).sum())
    tn = int(((pred_bad == 0) & (lab == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    bf1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\n===== {tag}  (n={len(df)}, {elapsed:.0f}s, {len(df)/elapsed:.1f} sample/s) =====")
    print(f"action accuracy   : {acc:.4f}")
    print(f"action macro-F1   : {f1:.4f}")
    print(f"injection MAE     : {mae_i:.4f}")
    print(f"shell MAE         : {mae_s:.4f}")
    print(f"binary gate       : precision={prec:.4f} recall={rec:.4f} F1={bf1:.4f}")
    print(f"                    TP={tp} FP={fp} FN={fn} TN={tn}")
    print("\nconfusion matrix (rows=true, cols=pred)")
    print("            " + "".join(f"{a[:9]:>11s}" for a in ACTIONS))
    for i, a in enumerate(ACTIONS):
        row = [int(((act_t == i) & (act_p == j)).sum()) for j in range(len(ACTIONS))]
        print(f"{a[:11]:11s} " + "".join(f"{v:>11d}" for v in row))

    return {"tag": tag, "n": len(df), "acc_action": acc, "f1_macro_action": f1,
            "mae_injection": mae_i, "mae_shell": mae_s,
            "gate_precision": prec, "gate_recall": rec, "gate_f1": bf1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "eval_seconds": round(elapsed, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/model.safetensors")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--pooling", default="all",
                    help="last_nonpad|mean|last|first|all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    df = pd.read_parquet(f"data/full-{args.split}.parquet")
    df[["inj", "shell", "action"]] = df.apply(map_labels, axis=1)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
    print(f"split={args.split} n={len(df)}")
    print("true action dist:", df["action"].value_counts().to_dict())

    poolings = ["last_nonpad", "mean", "last", "first"] if args.pooling == "all" else [args.pooling]
    results = []
    for p in poolings:
        print(f"\n--- loading model (pooling={p}) ---", flush=True)
        model, _ = load_checkpoint(args.ckpt, pooling=p, verbose=False)
        inj_p, sh_p, act_p, el = run(model, tok, df, args.batch_size)
        results.append(report(df, inj_p, sh_p, act_p, el, f"{args.split}/{p}"))
        del model

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    best = max(results, key=lambda r: r["acc_action"])
    print(f"BEST pooling: {best['tag']}  acc={best['acc_action']:.4f} f1={best['f1_macro_action']:.4f}")


if __name__ == "__main__":
    main()
