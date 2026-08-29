"""
bench_aux.py — is Qwen2.5-0.5B on the NPU actually usable as an auxiliary model?

Tests three tasks against ground truth, on both quantizations and both devices:

  summarize  — ROUGE-1/2/L F1 vs CNN/DailyMail human highlights
  sentiment  — accuracy vs SST-2 labels (parses the answer, does not eyeball it)
  extract    — exact-substring match against a known answer, plus a HALLUCINATION
               check: did the model return a string that is not in the source at all?

The extraction task exists to re-test an earlier session's claim that INT4
hallucinated and INT8 refused. Both failure modes are counted explicitly:
  refusal      — empty or non-committal output
  hallucination— an answer not present in the source text
so "it failed" is broken down instead of asserted.

ROUGE is implemented here (unigram/bigram/LCS F1) to avoid pulling in a package for
three numbers. It matches rouge-score's F-measure on whitespace-tokenized lowercase
text, which is all that is needed for a relative comparison.
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

DATA = Path("aux_data")
REFUSAL_MARKERS = [
    "i cannot", "i can't", "cannot determine", "not mentioned", "no filename",
    "unable to", "i'm sorry", "sorry,", "as an ai", "there is no",
    "not specified", "not provided", "unclear",
]


# ---------- metrics ----------

def toks(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def _f1(match, n_hyp, n_ref):
    if n_hyp == 0 or n_ref == 0 or match == 0:
        return 0.0
    p, r = match / n_hyp, match / n_ref
    return 2 * p * r / (p + r)


def rouge_n(ref, hyp, n):
    import collections
    rt, ht = toks(ref), toks(hyp)
    rg = collections.Counter(tuple(rt[i:i+n]) for i in range(len(rt) - n + 1))
    hg = collections.Counter(tuple(ht[i:i+n]) for i in range(len(ht) - n + 1))
    match = sum((rg & hg).values())
    return _f1(match, sum(hg.values()), sum(rg.values()))


def rouge_l(ref, hyp):
    rt, ht = toks(ref), toks(hyp)
    if not rt or not ht:
        return 0.0
    prev = [0] * (len(ht) + 1)
    for i in range(1, len(rt) + 1):
        cur = [0] * (len(ht) + 1)
        for j in range(1, len(ht) + 1):
            cur[j] = prev[j-1] + 1 if rt[i-1] == ht[j-1] else max(prev[j], cur[j-1])
        prev = cur
    return _f1(prev[len(ht)], len(ht), len(rt))


def norm_answer(s):
    """Strip quotes/backticks/trailing punctuation so formatting is not scored."""
    s = s.strip().strip("`\"'“”‘’ ")
    s = re.sub(r"^(the\s+)?(filename|file|path|version|email|identifier|answer)"
               r"\s*(is|:)\s*", "", s, flags=re.I)
    return s.strip().rstrip(".,;:").strip("`\"'")


def is_refusal(s):
    low = s.lower()
    return (not s.strip()) or any(m in low for m in REFUSAL_MARKERS)


# ---------- runner ----------

def make_pipe(model, device):
    import openvino_genai as ov_genai
    t0 = time.time()
    pipe = ov_genai.LLMPipeline(str(model), device)
    return pipe, time.time() - t0


def gen(pipe, prompt, max_new_tokens):
    import openvino_genai as ov_genai
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False          # greedy: repeatable numbers
    t0 = time.time()
    out = pipe.generate(prompt, cfg)
    return str(out).strip(), (time.time() - t0) * 1000


def chat(pipe, system, user, max_new_tokens):
    """Qwen2.5 chat template, written explicitly — apply_chat_template signatures
    differ between openvino_genai and transformers (a known pitfall)."""
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    return gen(pipe, prompt, max_new_tokens)


def run_summarize(pipe, items, limit):
    sysmsg = "You are a summarizer. Reply with 2-3 short sentences and nothing else."
    r1s, r2s, rls, lat, rows = [], [], [], [], []
    for it in items[:limit]:
        user = f"Summarize this news article:\n\n{it['article']}"
        hyp, ms = chat(pipe, sysmsg, user, 120)
        r1, r2, rl = (rouge_n(it["reference"], hyp, 1),
                      rouge_n(it["reference"], hyp, 2),
                      rouge_l(it["reference"], hyp))
        r1s.append(r1); r2s.append(r2); rls.append(rl); lat.append(ms)
        rows.append({"id": it["id"], "rouge1": round(r1, 4), "rouge2": round(r2, 4),
                     "rougeL": round(rl, 4), "ms": round(ms, 1), "output": hyp})
        print(f"    {it['id']}  R1={r1:.3f} R2={r2:.3f} RL={rl:.3f}  {ms:6.0f}ms",
              flush=True)
    return {
        "n": len(rows),
        "rouge1": round(statistics.mean(r1s), 4),
        "rouge2": round(statistics.mean(r2s), 4),
        "rougeL": round(statistics.mean(rls), 4),
        "p50_ms": round(statistics.median(lat), 1),
        "rows": rows,
    }


def run_sentiment(pipe, items, limit):
    sysmsg = ("Classify sentiment. Answer with exactly one word: "
              "positive or negative.")
    ok = unparsed = 0
    lat, rows = [], []
    for it in items[:limit]:
        hyp, ms = chat(pipe, sysmsg, it["text"], 8)
        low = hyp.lower()
        pred = ("positive" if "positive" in low else
                "negative" if "negative" in low else "UNPARSED")
        if pred == "UNPARSED":
            unparsed += 1
        ok += pred == it["label"]
        lat.append(ms)
        rows.append({"id": it["id"], "gold": it["label"], "pred": pred,
                     "ms": round(ms, 1), "output": hyp})
    acc = ok / len(rows)
    print(f"    accuracy {acc:.4f} ({ok}/{len(rows)}), unparsed {unparsed}", flush=True)
    return {"n": len(rows), "accuracy": round(acc, 4), "correct": ok,
            "unparsed": unparsed, "p50_ms": round(statistics.median(lat), 1),
            "rows": rows}


def run_extract(pipe, items, limit):
    sysmsg = ("Extract the exact string the user asks for. Copy it verbatim from "
              "the text. Reply with only that string.")
    exact = refused = halluc = wrong = 0
    lat, rows = [], []
    for it in items[:limit]:
        user = f"{it['question']}\n\nText: {it['text']}"
        raw, ms = chat(pipe, sysmsg, user, 40)
        ans = norm_answer(raw)
        # classification of the outcome
        if ans == it["answer"]:
            kind, exact = "exact", exact + 1
        elif is_refusal(raw):
            kind, refused = "refusal", refused + 1
        elif ans and ans not in it["text"]:
            kind, halluc = "hallucination", halluc + 1
        else:
            kind, wrong = "wrong_span", wrong + 1
        lat.append(ms)
        rows.append({"id": it["id"], "kind": it["kind"], "gold": it["answer"],
                     "parsed": ans, "outcome": kind, "ms": round(ms, 1),
                     "output": raw})
        print(f"    {it['id']}  {kind:14s} gold={it['answer']!r} got={ans!r}",
              flush=True)
    n = len(rows)
    print(f"    exact {exact}/{n}  refusal {refused}  hallucination {halluc}  "
          f"wrong_span {wrong}", flush=True)
    return {"n": n, "exact": exact, "exact_rate": round(exact / n, 4),
            "refusal": refused, "hallucination": halluc, "wrong_span": wrong,
            "p50_ms": round(statistics.median(lat), 1), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--tasks", default="summarize,sentiment,extract")
    ap.add_argument("--limit-sum", type=int, default=12)
    ap.add_argument("--limit-sent", type=int, default=40)
    ap.add_argument("--limit-ext", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = Path(args.model)
    tag = f"{model.name}_{args.device.replace('.', '_')}"
    out_path = Path(args.out or f"aux_bench_{tag}.json")

    print(f"model={model.name} device={args.device}", flush=True)
    try:
        pipe, compile_s = make_pipe(model, args.device)
    except Exception as e:
        print(f"COMPILE FAILED: {type(e).__name__}: {e}")
        Path(out_path).write_text(json.dumps(
            {"model": model.name, "device": args.device, "compile_ok": False,
             "error": f"{type(e).__name__}: {e}"}, indent=2), encoding="utf-8")
        return 1
    print(f"compile: {compile_s:.2f}s", flush=True)

    results = {"model": model.name, "device": args.device, "compile_ok": True,
               "compile_s": round(compile_s, 2)}
    tasks = [t.strip() for t in args.tasks.split(",")]

    if "summarize" in tasks:
        print("\n  == summarize (ROUGE vs human highlights) ==", flush=True)
        items = json.loads((DATA / "summarize.json").read_text(encoding="utf-8"))
        results["summarize"] = run_summarize(pipe, items, args.limit_sum)
    if "sentiment" in tasks:
        print("\n  == sentiment (SST-2 accuracy) ==", flush=True)
        items = json.loads((DATA / "sentiment.json").read_text(encoding="utf-8"))
        results["sentiment"] = run_sentiment(pipe, items, args.limit_sent)
    if "extract" in tasks:
        print("\n  == precision extraction (exact match) ==", flush=True)
        items = json.loads((DATA / "extract.json").read_text(encoding="utf-8"))
        results["extract"] = run_extract(pipe, items, args.limit_ext)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")

    s, se, e = (results.get("summarize"), results.get("sentiment"),
                results.get("extract"))
    print(f"\nSUMMARY {model.name} @ {args.device}")
    if s:
        print(f"  summarize  R1={s['rouge1']:.4f} R2={s['rouge2']:.4f} "
              f"RL={s['rougeL']:.4f}  p50={s['p50_ms']}ms")
    if se:
        print(f"  sentiment  acc={se['accuracy']:.4f} "
              f"unparsed={se['unparsed']}  p50={se['p50_ms']}ms")
    if e:
        print(f"  extract    exact={e['exact']}/{e['n']} refusal={e['refusal']} "
              f"halluc={e['hallucination']} wrong={e['wrong_span']}  "
              f"p50={e['p50_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
