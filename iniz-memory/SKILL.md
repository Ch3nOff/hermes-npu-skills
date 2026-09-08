---
name: iniz-memory
description: Local semantic search over repo docs, NPU-served.
version: 0.1.0
author: Matthew Chen (CH3NDev), Hermes Agent
license: MIT
platforms: [windows, linux]
metadata:
  hermes:
    tags: [openvino, npu, embeddings, retrieval, rag, memory, local-ai]
    related_skills: [iniz-agent-guard, iniz-stt, iniz-aux]
---

# Skill: Iniz Memory — Local Semantic Search on Multilingual Embeddings

**Trigger:** When Hermes needs to find things in local documents by meaning rather
than keywords — prior decisions, measured numbers, procedures — or when deciding
where an embedding workload should run.

**Behavior:** Chunk the repo docs, embed with `multilingual-e5-base` INT8, rank by
cosine similarity against 39 hand-written test queries (English + Indonesian) with
gold chunk IDs. Serve over HTTP with the same process shape as the sibling servers.

**Environment:** `openvino` ≥ 2026.3, `transformers` (tokenizer only). NPU is the
measured default (see below); CPU remains a flag flip away.

---

## Headline results

113 chunks from the repo's own 8 markdown files; 39 queries (20 EN + 19 ID, including
cross-lingual Indonesian queries over English chunks).

### Model ladder (recall, 39 queries)

| Model | Size INT8 | overall r@1/r@3/r@5 | EN r@3 | ID r@3 |
|---|---|---|---|---|
| e5-small | 140 MB | 0.590 / 0.795 / 0.897 | 0.900 | 0.684 |
| **e5-base** | **293 MB** | **0.590 / 0.846 / 0.923** | **0.850** | **0.842** |

e5-base wins overall (+0.051 r@3), and cross-lingual ID jumps +0.158 (0.684 → 0.842)
— the base model is what makes Indonesian queries over English docs work reliably.
EN drops a hair (0.900 → 0.850); the ladder trades a little monolingual for a lot
of cross-lingual. **e5-base is the recommended model.**

### Device verdict depends on model size (same pattern as Whisper)

| Model | Axis | NPU | CPU |
|---|---|---|---|
| e5-small | single query | 15.2 ms | **10.3 ms** |
| e5-small | bulk index | 16.7 ms/chunk | **11.1 ms/chunk** |
| **e5-base** | **single query** | **25.5 ms** | 35.6 ms |
| **e5-base** | **bulk index (113)** | **27.2 ms/chunk** | 36.0 ms/chunk |

At small size the NPU's fixed overhead dominates (CPU faster everywhere, batch=8
helps CPU only). At base size the verdict flips: NPU 1.4× faster per query and bulk.
**`mem_server.py` therefore defaults to e5-base on NPU** (compile 9.7 s vs 0.9 s CPU
— one-time cost, stated so nobody mistakes startup for a hang).

Recall@3/@5 are device-identical on base (0.846/0.923 both); r@1 ties flip on device
numerics (22 NPU vs 23 CPU hits of 39) — reported, not hidden. Top-1 agreement
34/39: the 5 flips are all near-tie reorderings, never a miss turning into a hit or
vice versa at r@3.

The misses are all near-misses: every gold chunk for the misses@3 ranks 4–9.
Cross-lingual ID at n=19 (r@3 0.842) is now a real measurement, not a smoke signal —
though still hand-written by one author (see pitfall 6).

---

## Reranker measured twice, rejected twice

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, covers Indonesian, 140 MB
INT8) over bi-encoder top-10:

| Bi-encoder | recall@1 before → after | fixed / broken | cost |
|---|---|---|---|
| e5-small (26 queries) | 0.615 → 0.577 | 2 / 3 | ~232 ms/query |
| e5-base (39 queries) | 0.590 → 0.590 | 3 / 3 | ~230 ms/query |

Net zero or negative at ~20× the bi-encoder query cost, breaking previously correct
answers both times. On a corpus of cross-referencing sibling sections the
cross-encoder adds noise, not signal. Not shipped; `scripts/test_rerank.py` (takes
`INIZ_RERANK_BI_MODEL`) + `results/rerank_CPU.json` (small) and
`results/rerank_base_CPU.json` (base) preserve both negative results. Quality
verdicts are device-independent (same weights), so CPU-only rerank tests settle it
for the NPU too.

---

## Procedure

`$V` is `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux). Work runs
from `npu-provider/work/`; the skill ships the scripts plus a frozen copy of the
corpus and results.

1. **Chunk the corpus.** Completion: `corpus/chunks.json` with ~113 chunks, avg
   ~765 chars, listed per file.
   ```
   terminal(command="$V scripts/build_corpus.py")
   ```
   Splits each `.md` by `##` headings, keeps file+heading provenance per chunk,
   drops stubs under 200 chars, splits sections over 1500 chars by paragraph.

2. **Export the model.** Completion: `openvino_model.xml/.bin` + tokenizer, INT8
   75/75 layers, ~293 MB for base (~140 MB for small).
   ```
   terminal(command="optimum-cli export openvino --model intfloat/multilingual-e5-base --task feature-extraction --weight-format int8 models/e5-base-int8-ov")
   ```
   Note: the e5-small IR declares `token_type_ids`, the e5-base IR does not.
   `bench_mem.py` and `mem_server.py` feed only the inputs the IR declares — a
   hardcoded three-input feed breaks on base with no useful error.

3. **Benchmark.** Completion: `mem_bench_base.json` with recall splits + latency
   per device (`--model` selects the IR; default e5-base).
   ```
   terminal(command="$V -u scripts/bench_mem.py --devices NPU,CPU")
   ```
   39 hand-written queries (`corpus/queries.json`, documented limits below). E5
   prefixes are applied inside the script (`query:` / `passage:`) — they are part of
   the model's training contract, not decoration.

4. **Batch check before claiming anything about indexing.** Completion: per-chunk
   ms at batch 1 and 8, both devices.
   ```
   terminal(command="$V -u scripts/test_batch.py")
   ```

5. **Reranker test (optional, kept for the record).** Completion: `rerank_base_CPU.json`
   with before/after r@1.
   ```
   terminal(command="INIZ_RERANK_BI_MODEL=../models/e5-base-int8-ov $V -u scripts/test_rerank.py CPU")
   ```

6. **Serve and smoke test.** Completion: `VERDICT: PASS` with served r@3/r@5 exactly
   matching the offline bench (33/39 and 36/39, both devices) and r@1 in the tie
   range {22, 23}.
   ```
   terminal(command="$V scripts/mem_server.py", background=True)
   terminal(command="$V -u scripts/mem_client.py")
   ```

---

## Server

`mem_server.py` — 208 lines, threaded HTTP, corpus embedded once at startup (~3.1 s
on NPU for e5-base), lock-guarded per-query encode.

```
GET  /health      -> status, device, model, chunks, compile_s, index_ms, p50
POST /search      {"query": "...", "top_k": 5} -> hits[{id,file,heading,text,score}]
```

Environment: `INIZ_MEM_MODEL` (default e5-base), `INIZ_MEM_CORPUS` (explicit env,
else `corpus/chunks.json` next to the server, else `../work/corpus/chunks.json` —
any further guessing would hide config errors), `INIZ_MEM_DEVICE`
(**default `NPU`**), `INIZ_MEM_PORT` (8012).

Measured over HTTP on NPU (e5-base): index 3066 ms for 113 chunks, query p50
26.4 ms, served r@3/r@5 = 33/39 and 36/39 with r@1 = 22/39 (the NPU side of the tie
range), all four error cases 4xx, 39/39 requests served, `VERDICT: PASS`.

**Security:** binds to `127.0.0.1` with **no authentication**.

---

## Pitfalls (all hit on this machine)

1. **The IR wants `token_type_ids`, the tokenizer does not emit it — on small
   only.** `KeyError: 'token_type_ids'` on first embed with e5-small. Fix: feed
   zeros (correct — single segment, no sentence pair). The e5-base IR dropped the
   input entirely, so both scripts now feed only the inputs the IR declares; a
   hardcoded feed breaks on one model or the other with no useful error.

2. **Dynamic `[?,?]` inputs must be reshaped before NPU compile** — same pitfall as
   the guard skill. Static `[1, 256]` here (chunks average ~765 chars ≈ ~190 tokens;
   truncation at 256 is a measured fit, not a guess).

3. **E5 prefixes are load-bearing.** `query:` / `passage:` are training contract. An
   early draft without them ran fine and scored worse for no visible reason.

4. **Batching does not always amortize NPU overhead** (measured on e5-small).
   Batch=8 on the NPU: same per-chunk speed, 2× compile. Only keep batching if you
   measure a gain — and re-measure per model, since the base verdict flipped.

5. **A reranker can subtract accuracy — twice measured.** −0.038 r@1 on small,
   net zero on base, 3 newly broken queries each time. "Rerank top-k" is not a free
   upgrade — test it on your own queries or do not ship it.

6. **39 hand-written queries are a solid smoke signal, not a benchmark.** Same
   author wrote queries and gold labels; phrasing bias is unavoidable. ID split at
   n=19 (r@3 0.842) is quotable with its n attached, never as a precise capability
   claim.

7. **The corpus rots as docs change.** The server embeds whatever `chunks.json` holds
   at startup. After editing any skill doc, re-run `build_corpus.py` and restart —
   there is no watcher, by design (deterministic startup beats magic).

---

## What is still open

- **The Obsidian vault is missing.** `Clevates-m` is registered in `obsidian.json`
  but absent from disk (searched Documents, Desktop, Downloads, OneDrive). This skill
  was therefore built and measured on the repo docs per explicit user choice. Point
  `build_corpus.py`'s `REPO` at the real vault when it reappears and re-run — nothing
  in the pipeline assumes English-only or repo-shaped input except the queries.
- **Cross-lingual ID (r@3 0.842, n=19)** improved with base but remains below EN
  (0.850 — nearly closed). Next trigger for a bigger model (`bge-m3`) would be ID
  lagging again at larger n, not curiosity.
- **No incremental indexing.** Full re-embed at startup (~3.1 s NPU / ~4.1 s CPU for
  113 chunks; grows linearly). Fine for hundreds of notes, not thousands.
- **No hybrid search.** Pure dense cosine; BM25 fallback for exact terms (filenames,
  error codes, hex IDs) is unimplemented.
- **Chunking is heading-based.** Tables split across part boundaries lose row context;
  unmeasured how much this costs.

## Verification

```
$V scripts/build_corpus.py          # expect ~113 chunks from 8 files
$V -u scripts/bench_mem.py --devices NPU,CPU
                                    # expect r@3≈0.85 overall, r@3/r@5 device-identical
$V -u scripts/mem_client.py         # expect VERDICT: PASS, r@3=33/39 r@5=36/39
```

Accept only if: served r@3/r@5 equal the offline bench exactly, r@1 falls in the
tie range {22, 23} (device numerics flip near-ties — a wider gap means a pipeline
bug), and the client prints `VERDICT: PASS`.

## See Also

- `corpus/chunks.json`, `corpus/queries.json` — the measured corpus and query set
  (39 queries since the ID expansion)
- `results/mem_bench_base.json` — e5-base per-query rows, splits, latency per device
- `results/mem_bench_v2.json` — e5-small on the same 39 queries (the ladder's lower rung)
- `results/mem_bench.json` — e5-small on the original 26 queries (superseded)
- `results/rerank_base_CPU.json`, `results/rerank_CPU.json` — both rejected reranker runs
- `iniz-agent-guard/SKILL.md` — the reshape/static-shape discipline reused here
- `iniz-stt/SKILL.md` — where the NPU *does* win at larger sizes
