"""
verify_incoming_mapping.py — tests whether the label mapping in the
iniz-agent-guard-fixed.zip package is the mapping ACTUALLY used during training.

That package claims the mapping:
    OVERRIDE_CATEGORIES    -> PAUSE_AGENTS
    OBFUSCATION_CATEGORIES -> ISOLATE_FILE
    EXTRACTION_CATEGORIES  -> USER_CONFIRMATION
    everything else (label=1) -> PAUSE_AGENTS
    label=0                -> PASS
plus shell = keyword proxy (0.0 / 0.1 / 0.6), NOT 0.9 as in the old notebook.

This is compared head-to-head with the old notebook mapping. The trained model is
the judge: the correct mapping will yield far higher accuracy & macro-F1, and a
lower shell MAE.

Run through the INT8 IR on the NPU (production numbers).
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
SEVERITY_MAP = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}

# ---------- mapping A: old notebook (what I used previously) ----------
A_PAUSE = {"direct_injection", "prompt_extraction", "system_extraction", "agent_manipulation"}


def map_old(row):
    cat, sev, label = row["category"], row["severity"], int(row["label"])
    inj, sh, act = 0.0, 0.0, "PASS"
    if label == 1:
        inj = SEVERITY_MAP.get(sev, 0.5)
        if cat == "code_execution":
            sh, act = 0.9, "ISOLATE_FILE"
        elif cat == "instruction_override":
            sh, act = 0.5, "ISOLATE_FILE"
        elif cat in A_PAUSE:
            act = "PAUSE_AGENTS"
        else:
            act = "USER_CONFIRMATION"
    return pd.Series({"inj": inj, "shell": sh, "action": act})


# ---------- mapping B: incoming package (iniz-agent-guard-fixed.zip) ----------
B_OVERRIDE = {"direct_injection", "jailbreak", "persona_replacement", "many_shot", "crescendo"}
B_OBFUSCATION = {"encoding_obfuscation", "token_smuggling", "indirect_injection", "context_overflow"}
B_EXTRACTION = {"system_extraction", "prompt_leaking"}
B_SHELL_KW = ["eval", "exec", "atob", "bash", "shell", "subprocess",
              "os.system", "js_eval", "code_execution", "command", "__import__"]


def map_incoming(row):
    cat, sev, label = row["category"], row["severity"], int(row["label"])
    inj, act = 0.0, "PASS"
    if label == 1:
        inj = SEVERITY_MAP.get(sev, 0.5)
        if cat in B_OVERRIDE:
            act = "PAUSE_AGENTS"
        elif cat in B_OBFUSCATION:
            act = "ISOLATE_FILE"
        elif cat in B_EXTRACTION:
            act = "USER_CONFIRMATION"
        else:
            act = "PAUSE_AGENTS"
    # shell proxy
    if int(row["label"]) == 0:
        sh = 0.0
    else:
        tags = row["tags"]
        tl = [] if tags is None else [str(t) for t in list(tags)]
        hay = (str(row["text"]) + " " + " ".join(tl)).lower()
        sh = 0.6 if any(k in hay for k in B_SHELL_KW) else 0.1
    return pd.Series({"inj": inj, "shell": sh, "action": act})


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp == 0:
            if fp or fn:
                f1s.append(0.0)
            continue
        p, r = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def score_split(compiled, tok, df, S):
    req = compiled.create_infer_request()
    texts = df["text"].tolist()
    enc = tok(texts, return_tensors="np", padding="max_length", truncation=True, max_length=S)
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    req.infer({"input_ids": ids[0:1], "attention_mask": mask[0:1]})
    inj, sh, act = [], [], []
    for i in range(len(texts)):
        r = req.infer({"input_ids": ids[i:i + 1], "attention_mask": mask[i:i + 1]})
        v = list(r.values())
        inj.append(float(np.array(v[0]).flatten()[0]))
        sh.append(float(np.array(v[1]).flatten()[0]))
        act.append(int(np.array(v[2]).flatten().argmax()))
        if i % 300 == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return np.array(inj), np.array(sh), np.array(act)


def report(name, df, inj_p, sh_p, act_p, mapper):
    lab = df.apply(mapper, axis=1)
    act_t = np.array([ACTION2IDX[a] for a in lab["action"]])
    inj_t = lab["inj"].to_numpy(float)
    sh_t = lab["shell"].to_numpy(float)
    acc = float((act_p == act_t).mean())
    f1 = macro_f1(act_t, act_p, 4)
    mae_i = float(np.abs(inj_p - inj_t).mean())
    mae_s = float(np.abs(sh_p - sh_t).mean())
    print(f"\n  ### mapping {name}")
    print(f"    accuracy action : {acc:.4f}")
    print(f"    macro-F1 action : {f1:.4f}")
    print(f"    MAE injection   : {mae_i:.4f}")
    print(f"    MAE shell       : {mae_s:.4f}")
    print("    confusion (rows=true, cols=pred)")
    print("              " + "".join(f"{a[:9]:>11s}" for a in ACTIONS))
    for i, a in enumerate(ACTIONS):
        row = [int(((act_t == i) & (act_p == j)).sum()) for j in range(4)]
        print(f"    {a[:9]:9s} " + "".join(f"{v:>11d}" for v in row))
    return {"mapping": name, "acc": acc, "macro_f1": f1,
            "mae_injection": mae_i, "mae_shell": mae_s}


def main():
    import openvino as ov
    from transformers import AutoTokenizer

    model_xml = "../models/iniz-guard-int8-ov/guard.xml"
    S = json.loads((Path(model_xml).parent / "guard_meta.json").read_text())["seq_len"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    core = ov.Core()
    compiled = core.compile_model(core.read_model(model_xml), "NPU")
    print(f"NPU compiled, seq_len={S}")

    out = {}
    for split in ["validation", "test"]:
        df = pd.read_parquet(f"data/full-{split}.parquet")
        print(f"\n########## {split} (n={len(df)}) ##########", flush=True)
        t0 = time.time()
        inj_p, sh_p, act_p = score_split(compiled, tok, df, S)
        print(f"  scored in {time.time()-t0:.0f}s")
        ra = report("A (old notebook)", df, inj_p, sh_p, act_p, map_old)
        rb = report("B (incoming zip)", df, inj_p, sh_p, act_p, map_incoming)
        winner = "B" if rb["macro_f1"] > ra["macro_f1"] else "A"
        print(f"\n  >>> WINNER {split}: mapping {winner} "
              f"(macro-F1 {max(ra['macro_f1'], rb['macro_f1']):.4f} vs "
              f"{min(ra['macro_f1'], rb['macro_f1']):.4f})")
        out[split] = {"A": ra, "B": rb, "winner": winner}

    Path("mapping_verdict.json").write_text(json.dumps(out, indent=2))
    print("\nwrote mapping_verdict.json")


if __name__ == "__main__":
    main()
