"""
infer_mapping.py — reverse-engineer label mapping yang SEBENARNYA dipakai saat
training checkpoint (lora_adapter), dengan cara membandingkan prediksi model
terhadap kategori/severity asli dataset.

Alasan: notebook 05 (cell 3) menuliskan satu mapping, tapi confusion matrix hasil
eval menunjukkan kelas USER_CONFIRMATION tidak pernah diprediksi — indikasi
mapping saat training berbeda. Model adalah sumber kebenaran di sini.

Output: tabel kategori x (mean inj, mean shell, modus action prediksi).
"""

import numpy as np
import pandas as pd
import torch

from guard_model import load_checkpoint, ACTIONS

BATCH = 16
MAXLEN = 192


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model, _ = load_checkpoint("ckpt/model.safetensors", pooling="last_nonpad", verbose=False)

    # ambil train split: mapping training tercermin paling kuat di data latih
    df = pd.read_parquet("data/full-train.parquet")
    # sampel proporsional per kategori, minimal 8 / maksimal 40 per kategori
    parts = []
    for cat, g in df.groupby("category"):
        n = int(np.clip(len(g), 0, 40))
        n = min(n, len(g))
        parts.append(g.sample(n=n, random_state=0))
    sub = pd.concat(parts).reset_index(drop=True)
    print(f"sampled {len(sub)} rows across {sub['category'].nunique()} categories", flush=True)

    inj, sh, act = [], [], []
    texts = sub["text"].tolist()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN)
        with torch.no_grad():
            o = model(enc["input_ids"], enc["attention_mask"])
        inj.append(o["injection"].float().numpy())
        sh.append(o["shell"].float().numpy())
        act.append(o["action_logits"].float().argmax(-1).numpy())
        if (i // BATCH) % 15 == 0:
            print(f"  {min(i+BATCH,len(texts))}/{len(texts)}", flush=True)

    sub["p_inj"] = np.concatenate(inj)
    sub["p_shell"] = np.concatenate(sh)
    sub["p_action"] = [ACTIONS[i] for i in np.concatenate(act)]

    print("\n=== per-kategori (prediksi model) ===")
    g = sub.groupby("category").agg(
        n=("text", "size"),
        inj=("p_inj", "mean"),
        shell=("p_shell", "mean"),
    )
    modus = sub.groupby("category")["p_action"].agg(lambda s: s.value_counts().index[0])
    purity = sub.groupby("category")["p_action"].agg(
        lambda s: s.value_counts().iloc[0] / len(s))
    g["action_modus"] = modus
    g["purity"] = purity
    print(g.sort_values("n", ascending=False).round(3).to_string())

    print("\n=== per-severity ===")
    gs = sub.groupby("severity").agg(n=("text", "size"), inj=("p_inj", "mean"),
                                     shell=("p_shell", "mean"))
    gs["action_modus"] = sub.groupby("severity")["p_action"].agg(
        lambda s: s.value_counts().index[0])
    print(gs.round(3).to_string())

    print("\n=== distribusi action prediksi keseluruhan ===")
    print(sub["p_action"].value_counts().to_dict())

    print("\n=== apakah shell head aktif? kategori dgn shell tertinggi ===")
    print(g.sort_values("shell", ascending=False)[["n", "shell", "inj"]].head(8).round(3).to_string())

    sub[["text", "label", "category", "severity", "p_inj", "p_shell", "p_action"]].to_csv(
        "mapping_probe.csv", index=False)
    print("\nwrote mapping_probe.csv")


if __name__ == "__main__":
    main()
