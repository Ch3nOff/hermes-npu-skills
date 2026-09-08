---
name: iniz-memory
description: Local semantic search over repo docs, CPU-served.
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

**Behavior:** Chunk the repo docs, embed with `multilingual-e5-small` INT8, rank by
cosine similarity against 26 hand-written test queries (English + Indonesian) with
gold chunk IDs. Serve over HTTP with the same process shape as the sibling servers.

**Environment:** `openvino` ≥ 2026.3, `transformers` (tokenizer only). NPU optional —
and, measured, not recommended here (see below).

---

## Headline results

113 chunks from the repo's own 8 markdown files; 26 queries (20 EN + 6 ID, including
cross-lingual Indonesian queries over English chunks). Same INT8 weights on both
devices, 26/26 top-1 agreement — recall is device-independent, so the device verdict
is pure latency.

| Split | n | recall@1 | recall@3 | recall@5 |
|---|---|---|---|---|
| overall | 26 | 0.615 | 0.808 | 0.962 |
| monolingual EN | 20 | 0.650 | 0.900 | 1.000 |
| cross-lingual ID | 6 | 0.500 | 0.500 | 0.833 |
| hard (paraphrased) | 5 | 0.800 | 1.000 | 1.000 |

| Axis | NPU | CPU |
|---|---|---|
| compile | 7.1 s | 0.9 s |
| single query | 15.2 ms | **10.3 ms** |
| bulk index (113 chunks) | 16.7 ms/chunk | **11.1 ms/chunk** |
| bulk index, batch=8 | 17.8 ms/chunk | **9.9 ms/chunk** |

**`mem_server.py` defaults to CPU.** At e5-small size (118 MB INT8) the NPU's fixed
overhead dominates each inference and batching does not amortize it (larger static
shape, 2× compile, same speed). Same verdict shape as whisper-base: NPU stays an
offload option, never a latency win at this size. This may flip for larger embedding
models — untested, stated as untested.

The misses are all near-misses, not garbage: every gold chunk for the 5 misses@3
ranks 4–9. Cross-lingual ID is the weak spot (r@3 0.5), but n=6 is a smoke signal
with wide error bars, not a benchmark — it says "multilingual works, worse than
monolingual", nothing more precise.

---

## Reranker measured, then rejected

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, covers Indonesian, 140 MB
INT8) over bi-encoder top-10:

| | recall@1 | delta | cost |
|---|---|---|---|
| bi-encoder alone | 0.615 | — | ~10 ms/query |
| + cross-encoder top-10 | 0.577 | **−0.038** | ~232 ms/query (23 ms/pair) |

Fixed 2 queries, **broke 3 previously correct ones** (e.g. moved a correct
large-v3 answer to a sibling section). On a corpus full of cross-referencing sibling
sections the cross-encoder adds noise, not signal — at 20× the query cost. It is not
shipped; `scripts/test_rerank.py` + `results/rerank_CPU.json` preserve the negative
result. Quality verdicts are device-independent (same weights), so the CPU-only
rerank test settles it for the NPU too.

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
   75/75 layers, ~140 MB.
   ```
   terminal(command="optimum-cli export openvino --model intfloat/multilingual-e5-small --task feature-extraction --weight-format int8 models/e5-small-int8-ov")
   ```

3. **Benchmark.** Completion: `mem_bench.json` with recall splits + latency per device.
   ```
   terminal(command="$V -u scripts/bench_mem.py --devices NPU,CPU")
   ```
   26 hand-written queries (`corpus/queries.json`, documented limits below). E5
   prefixes are applied inside the script (`query:` / `passage:`) — they are part of
   the model's training contract, not decoration.

4. **Batch check before claiming anything about indexing.** Completion: per-chunk
   ms at batch 1 and 8, both devices.
   ```
   terminal(command="$V -u scripts/test_batch.py")
   ```

5. **Reranker test (optional, kept for the record).** Completion: `rerank_CPU.json`
   with before/after r@1.
   ```
   terminal(command="$V -u scripts/test_rerank.py CPU")
   ```

6. **Serve and smoke test.** Completion: `VERDICT: PASS` with served recall exactly
   matching the offline bench (16/21/25).
   ```
   terminal(command="INIZ_MEM_DEVICE=CPU $V scripts/mem_server.py", background=True)
   terminal(command="$V -u scripts/mem_client.py")
   ```

---

## Server

`mem_server.py` — 201 lines, threaded HTTP, corpus embedded once at startup (~1.3 s
on CPU), lock-guarded per-query encode.

```
GET  /health      -> status, device, model, chunks, compile_s, index_ms, p50
POST /search      {"query": "...", "top_k": 5} -> hits[{id,file,heading,text,score}]
```

Environment: `INIZ_MEM_MODEL`, `INIZ_MEM_CORPUS` (defaults to `corpus/chunks.json`
next to the server), `INIZ_MEM_DEVICE` (**default `CPU`**), `INIZ_MEM_PORT` (8012).

Measured over HTTP on CPU: index 1311 ms for 113 chunks, query p50 10.8 ms, served
recall r@1/r@3/r@5 = 0.6154/0.8077/0.9615 — bit-identical to the offline bench, all
four error cases 4xx, 26/26 requests served, `VERDICT: PASS`.

**Security:** binds to `127.0.0.1` with **no authentication**.

---

## Pitfalls (all hit on this machine)

1. **The IR wants `token_type_ids`, the tokenizer does not emit it.** `KeyError:
   'token_type_ids'` on first embed. Fix: feed zeros (correct — single segment, no
   sentence pair). Recorded in `bench_mem.py`, not worked around globally.

2. **Dynamic `[?,?]` inputs must be reshaped before NPU compile** — same pitfall as
   the guard skill. Static `[1, 256]` here (chunks average ~765 chars ≈ ~190 tokens;
   truncation at 256 is a measured fit, not a guess).

3. **E5 prefixes are load-bearing.** `query:` / `passage:` are training contract. An
   early draft without them ran fine and scored worse for no visible reason.

4. **Batching does not always amortize NPU overhead.** Batch=8 on the NPU: same
   per-chunk speed, 2× compile. Only keep batching if you measure a gain.

5. **A reranker can subtract accuracy.** Measured −0.038 r@1 with 3 newly broken
   queries. "Rerank top-k" is not a free upgrade — test it on your own queries or do
   not ship it.

6. **26 hand-written queries are a smoke signal.** Same author wrote queries and gold
   labels; phrasing bias is unavoidable. The cross-lingual split (n=6) especially:
   quote it as "0.5, wide bars", never as a precise capability claim.

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
- **Cross-lingual recall (ID r@3 0.5, n=6)** is the obvious next measurement with a
  bigger query set — and the trigger for revisiting larger multilingual models.
- **Larger embedding models untested** (`bge-m3`, `e5-base`) — the NPU-vs-CPU latency
  verdict may flip with size, same pattern as Whisper.
- **No incremental indexing.** Full re-embed at startup (~1.3 s for 113 chunks; grows
  linearly). Fine for hundreds of notes, not thousands.
- **No hybrid search.** Pure dense cosine; BM25 fallback for exact terms (filenames,
  error codes, hex IDs) is unimplemented.
- **Chunking is heading-based.** Tables split across part boundaries lose row context;
  unmeasured how much this costs.

## Verification

```
$V scripts/build_corpus.py          # expect ~113 chunks from 8 files
$V -u scripts/bench_mem.py --devices NPU,CPU
                                    # expect r@3≈0.81 overall, 26/26 top1 agreement
$V -u scripts/mem_client.py         # expect VERDICT: PASS, recall matches bench
```

Accept only if: served recall equals the offline bench exactly, NPU/CPU top-1
agreement is near-total (same weights — divergence means a pipeline bug, not a model
difference), and the client prints `VERDICT: PASS`.

## See Also

- `corpus/chunks.json`, `corpus/queries.json` — the measured corpus and query set
- `results/mem_bench.json` — per-query rows, splits, latency per device
- `results/rerank_CPU.json` — the rejected reranker experiment
- `iniz-agent-guard/SKILL.md` — the reshape/static-shape discipline reused here
- `iniz-stt/SKILL.md` — where the NPU *does* win at larger sizes
