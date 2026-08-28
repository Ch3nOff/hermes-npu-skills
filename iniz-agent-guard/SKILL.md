---
tags: [mlops, openvino, npu, quantization, security, classifier, local-ai, offline]
---

# Skill: Iniz Agent Guard — Fine-tuned Multi-Head Classifier on Intel NPU

## English

**Trigger:** When the task involves the Iniz Agent Guard model (prompt-injection /
shell-risk scoring), or more generally exporting a **fine-tuned discriminative**
transformer (backbone + regression/classification heads) to OpenVINO IR and serving
it on an Intel Core Ultra NPU.

**Behavior:** Reconstruct the head architecture from the checkpoint, validate the
label mapping against the model itself, evaluate on held-out data, export via
`torch.export` → OpenVINO IR with **static shapes**, prove the NPU executes it, and
serve through a direct `CompiledModel` call — never `openvino_genai.LLMPipeline`.

**Environment:** `openvino` + `nncf` ≥ 2026.3, `torch` ≥ 2.9, `transformers` 4.57,
`peft` 0.18, `datasets`. NPU must appear in `ov.Core().available_devices`.

---

## Bahasa Indonesia 🇮🇩

**Trigger:** Saat tugas menyentuh model Iniz Agent Guard (skoring prompt-injection /
shell-risk), atau secara umum meng-export transformer **diskriminatif** hasil
fine-tune (backbone + head regresi/klasifikasi) ke OpenVINO IR lalu melayaninya di
NPU Intel Core Ultra.

**Behavior:** Rekonstruksi arsitektur head dari checkpoint, validasi label mapping
dengan model sebagai hakim, evaluasi di data held-out, export lewat `torch.export` →
OpenVINO IR dengan **shape statis**, buktikan NPU yang mengeksekusi, lalu layani
lewat `CompiledModel` langsung — **jangan** `openvino_genai.LLMPipeline`.

---

## Ruang Lingkup: DUA pola berbeda, jangan dicampur

Repo ini punya dua jalur yang gampang tertukar:

| | Generatif (pendahulu) | **Diskriminatif (produksi sekarang)** |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct apa adanya | checkpoint-2634 fine-tuned |
| Punya `lm_head`? | Ya | **Tidak** |
| API serving | `openvino_genai.LLMPipeline` | `core.compile_model` + baca tensor |
| Output | teks JSON yang harus di-parse | 3 tensor skor langsung |
| `max_new_tokens` | relevan | **tidak ada konsep ini** |
| Latensi | 2,7–6,6 s | **35 ms** |
| Berkas |  `~/npu-provider/scripts/npu_server.py` (di luar repo) | `guard_server.py` |

Pitfall khas jalur generatif (signature `apply_chat_template`, `max_new_tokens`
kepotong, INT4 berantakan) ada di `references/tokenizer-signature-pitfall.md` dan
**tidak berlaku** untuk jalur diskriminatif. Jangan menerapkannya di guard server.

## Arsitektur Model (terverifikasi 100% dari checkpoint)

Checkpoint: `lora_adapter.zip` → `checkpoint-2634` (3 epoch, 2634 step, loss akhir ≈ 0,0094).

```
Qwen2Model (AutoModel, TANPA lm_head) — 24 layer, hidden 896, vocab 151936
  └─ LoRA r=8 α=32 pada q_proj,k_proj,v_proj,o_proj  (192 tensor lora_A/lora_B)
     task_type="FEATURE_EXTRACTION"
  └─ pooling: hidden state token terakhir non-pad
       ├─ inj_head    Linear(896 → 1)   regresi skor injection
       ├─ shell_head  Linear(896 → 1)   regresi skor shell-risk (target PROXY)
       └─ action_head Linear(896 → 4)   klasifikasi aksi
```

488 tensor, **0 missing / 0 unexpected** saat `load_state_dict`.

`finetune_local.py` di repo ini sudah diverifikasi sebagai skrip yang benar:
`GuardHeadModel` yang dibangunnya menghasilkan **488 kunci state_dict yang identik
100%** dengan checkpoint, dan **11/11 hyperparameter** cocok dengan `training_args.bin`
(batch 1, grad accum 16, 3 epoch, lr 2e-4, cosine, warmup 0.05, adamw_torch).
Dry-run 3 step di CPU berhasil sampai `trainer.train()` selesai
(`work/dryrun_finetune.py`, `work/verify_arch_match.py`).

`MAX_LENGTH = 128` di `finetune_local.py` adalah **seq_len yang benar** — IR produksi
harus di-export dengan `--seq-len 128`, bukan 192. Lihat bagian seq_len di bawah.

Actions: `["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]` (indeks 0–3).

## Label Mapping — jangan tebak, validasi

Mapping ada di `scripts/guard_labels.py`. Ini **satu-satunya sumber kebenaran**:

```python
SEVERITY_MAP = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}
OVERRIDE_CATEGORIES    = {direct_injection, jailbreak, persona_replacement,
                          many_shot, crescendo}        -> PAUSE_AGENTS
OBFUSCATION_CATEGORIES = {encoding_obfuscation, token_smuggling,
                          indirect_injection, context_overflow}  -> ISOLATE_FILE
EXTRACTION_CATEGORIES  = {system_extraction, prompt_leaking}      -> USER_CONFIRMATION
label==1 lainnya -> PAUSE_AGENTS ;  label==0 -> PASS
shell = 0.0 (benign) / 0.6 (keyword proxy hit) / 0.1 (serangan tanpa keyword)
```

Mapping ini **dibuktikan** dengan menjadikan model sebagai hakim
(`work/verify_incoming_mapping.py`) — bukan dipercaya dari dokumen:

| Metrik | mapping salah (kategori palsu) | **mapping benar** | delta |
|---|---|---|---|
| accuracy action (test) | 0,7410 | **0,9660** | +0,225 |
| macro-F1 action (test) | 0,4264 | **0,8176** | +0,391 |
| MAE shell (test) | 0,0963 | **0,0465** | −0,050 |

Kalau metrik action tiba-tiba anjlok ke ~0,74 dan macro-F1 ~0,43, hampir pasti
mapping-nya yang salah, bukan modelnya.

Dua kategori di mapping **tidak eksis** di dataset (`context_overflow`,
`prompt_leaking`) — biarkan saja agar identik dengan yang dipakai saat training.

## Angka Terukur (bukan estimasi)

Mesin: Core Ultra 9 275HX + Intel AI Boost NPU, OpenVINO 2026.3, IR INT8_SYM 495 MB.

### seq_len 128 vs 192 — 128 yang benar, dan lebih cepat

`MAX_LENGTH=128` dipakai saat training, jadi IR produksi harus `[1,128]`.

| seq_len | p50 | p90 | compile | acc action (test) | macro-F1 | MAE inj |
|---|---|---|---|---|---|---|
| **128** (benar) | **34,4 ms** | 34,8 ms | 11,4 s | 0,9650 | 0,8171 | 0,0651 |
| 192 (salah) | 48,4 ms | 48,7 ms | 3,5 s | 0,9660 | 0,8176 | 0,0643 |

Kualitas praktis identik (selisih <0,001) tapi **seq_len 128 ~29 % lebih cepat**.
Pakai 128. IR seq_len 192 disimpan sebagai `models/iniz-guard-*-ov-seq192` untuk
pembanding.

### Latensi per device (batch 1, seq_len 128)

| Device | EXECUTION_DEVICES | p50 |
|---|---|---|
| **NPU** | `NPU` | **34,4 ms** |
| CPU | `['CPU']` | ~94 ms (seq 192) |
| GPU.0 iGPU | `['GPU.0']` | ~105 ms (seq 192) |

### Kualitas (IR INT8 @ NPU, seq_len 128, mapping tervalidasi)

| Metrik | validation (941) | test (942) |
|---|---|---|
| accuracy action | **0,9586** | **0,9650** |
| macro-F1 (kelas yang ada) | 0,5727 | **0,8171** |
| ROC-AUC injection → serangan | 0,9670 | 0,9709 |
| ROC-AUC shell → proxy-hit | 0,9648 | **0,9731** |
| MAE injection | 0,0637 | 0,0651 |
| MAE shell | 0,0479 | 0,0462 |
| Gate biner P / R / F1 | 0,962 / 0,985 / 0,973 | 0,968 / 0,984 / 0,976 |

Per-kelas (test): PASS F1 0,966 (n=390) · PAUSE_AGENTS F1 0,971 (n=541) ·
ISOLATE_FILE F1 0,667 (n=9) · USER_CONFIRMATION F1 0,667 (n=2).
Dua kelas terakhir punya support sangat kecil — angkanya bukan indikator andal.

Kesetiaan INT8 vs PyTorch fp32: `max|Δinj| = 0,045`, `max|Δshell| = 0,036`,
**action agreement 1,000** (40 sampel).

### Threshold `injection_score`
Optimum F1 di **0,25–0,30** (F1 0,973–0,976). Di atas 0,5 recall jatuh bebas
(th=0,6 → recall 0,39). Default server: `0,30`.

## `shell_head`: terlatih baik, TAPI targetnya proxy

Kesimpulan awal saya ("shell_head undertrained, hanya 12 sampel `code_execution`")
**salah** — itu artefak dari mapping label yang salah. Dengan mapping benar:
ROC-AUC shell_head → proxy-hit = **0,973**. Head ini terlatih dengan baik.

Masalah sesungguhnya lebih halus: **targetnya adalah proxy keyword**, bukan bahaya
shell nyata. Jadi shell_head adalah detektor keyword yang dipelajari, dan mewarisi
kelemahan daftar keyword-nya. Diukur di `work/diag_shell_head2.py`:

| Grup uji | shell_mean |
|---|---|
| shell berbahaya **dengan** keyword proxy (`bash`, `eval`, `os.system`, …) | **0,445** |
| shell berbahaya **tanpa** keyword proxy (`rm -rf /`, `dd if=`, `mkfs`, `:(){ :|:& };:`) | **0,165** |
| benign yang kebetulan memuat keyword ("What is eval() used for?") | −0,007 |
| injeksi murni tanpa unsur shell | 0,059 |

Selisih 0,28 antara dua grup pertama membuktikannya. Konsekuensi praktis:
**keyword backstop di server WAJIB memakai daftar lebih luas dari proxy training** —
`rm -rf`, `dd if=`, `mkfs`, `chmod 777`, `> /dev/tcp`, fork-bomb, dll.
Kabar baiknya: grup 3 menunjukkan head ini **tidak** naif — teks benign yang memuat
kata `eval`/`bash` tetap diberi skor ~0.

## Pitfalls (semua terverifikasi empiris di mesin ini)

1. **Checkpoint tanpa `lm_head` ≠ bisa dipakai `LLMPipeline`.** Periksa key
   `model.safetensors` dulu. Ada `*_head.weight` dan tidak ada `lm_head` → classifier:
   layani lewat `core.compile_model(...)` + baca tensor output.
2. **`torch.jit.trace` dan `torch.onnx.export(dynamo=False)` SELALU gagal pada
   Qwen2Model** (transformers 4.57 + torch 2.9):
   `RuntimeError: invalid unordered_map<K, T> key`, berasal dari walrus operator di
   `masking_utils`. `ov.convert_model(model, example_input=...)` juga gagal karena
   internalnya `TorchScriptPythonDecoder` → `jit.trace`.
   Jalur yang BERHASIL: **`torch.export.export(model, args, strict=False)`** lalu
   `ov.convert_model(exported_program)`.
3. **`attn_implementation` wajib `eager` sebelum export**
   (`config._attn_implementation = "eager"`, `use_cache = False`).
4. **`torch.export` menghasilkan dimensi DINAMIS (`?,?`) walau example input statis.**
   NPU menolak shape dinamis. Wajib
   `ov_model.reshape({0: PartialShape([1,S]), 1: PartialShape([1,S])})` sebelum save.
5. **`extra_special_tokens` list vs dict.** Checkpoint dari transformers 5.x
   menyimpannya sebagai **list**; transformers 4.x meledak dengan
   `AttributeError: 'list' object has no attribute 'keys'`. Normalisasi:
   `extra_special_tokens = {}` + `additional_special_tokens = <list>`.
6. **`seq_len` IR harus == `MAX_LENGTH` saat training.** Salah nilai tidak melempar
   error apa pun — cuma latensi lebih besar tanpa manfaat (192 vs 128: +40 % latensi,
   kualitas sama).
7. **Windows melaporkan NPU DI DALAM counter `\GPU Engine(*)`.** Tidak ada counter set
   `NPU` di Windows 11 build ini. "Ada aktivitas GPU Engine" **bukan** bukti fallback.
   Petakan LUID dulu (mesin ini):
   | LUID | Perangkat |
   |---|---|
   | `0x00000000_0x00011cf3` | **NPU — Intel(R) AI Boost** (engtype `compute`, max 102,75 %) |
   | `0x00000000_0x00010480` | iGPU — Intel(R) Graphics |
   | `0x00000000_0x0001099d` | dGPU — NVIDIA RTX 5060 |
   Beban device `CPU` → **nol** instance GPU-Engine untuk pid itu.
   Ulangi `scripts/luid_attribution.py <device>` di mesin lain.
8. **`compiled.get_property("EXECUTION_DEVICES")` adalah bukti termurah.** NPU
   mengembalikan string `NPU`; CPU/GPU mengembalikan list.
9. **Label mapping yang salah menyamar sebagai model buruk.** Akurasi 0,74 /
   macro-F1 0,43 vs 0,97 / 0,82 — semata karena mapping. Validasi mapping dengan
   model sebagai hakim sebelum menyimpulkan apa pun soal kualitas model.
10. **`shell_head` mengejar proxy keyword, bukan bahaya nyata** (lihat bagian di atas).
11. **Head regresi linear tanpa sigmoid bisa keluar rentang [0,1]** (nilai negatif
    kecil pada input benign). Clamp di sisi server.
12. **Fase benchmark multi-device mengotori bukti counter.** Meng-compile ke `GPU.0`
    membuat proses memegang konteks GPU seumur hidupnya. Jalankan pembuktian counter
    di proses terpisah: `prove_npu.py --npu-only`.
13. **`gradient_checkpointing=True` + LoRA di CPU** memperlambat tanpa manfaat dan
    butuh grad pada input embedding. Matikan untuk dry-run CPU.

## Workflow

Skrip ada di `scripts/`; dijalankan dari `~/npu-provider/work/`.

```bash
cd ~/npu-provider/work
V=../.venv/Scripts/python.exe   # venv terpisah: python default = venv Hermes (tanpa torch)

# 1. inspeksi checkpoint: ada *_head.weight? ada lm_head?
# 2. rekonstruksi + smoke test        -> guard_model.py (0 missing / 0 unexpected)
# 3. verifikasi skrip training cocok  -> $V verify_arch_match.py
# 4. dry-run training 3 step          -> $V dryrun_finetune.py
# 5. validasi label mapping           -> $V verify_incoming_mapping.py
# 6. pilih pooling empiris            -> $V eval_guard.py --pooling all --limit 240
# 7. diagnosa head                    -> $V diag_shell_head2.py
# 8. export IR statis + INT8          -> $V export_guard_ov.py --seq-len 128
# 9. bukti NPU + atribusi LUID        -> $V prove_npu.py --npu-only
#                                        $V luid_attribution.py NPU|GPU.0|CPU
# 10. eval final produksi             -> $V eval_final.py --device NPU
# 11. serve + smoke test HTTP
INIZ_GUARD_DEVICE=NPU INIZ_GUARD_THRESHOLD=0.30 $V ../guard_server.py
$V smoke_client.py
```

## Server

`guard_server.py` — `CompiledModel` langsung, tanpa generasi teks.

```
GET  /health -> device, seq_len, threshold, compile_s, warmup_ms, latency_p50_ms
POST /scan   {"text": "...", "context": "..."}
  -> injection_score, shell_risk_score, action, confidence, action_probs,
     shell_keyword_hits, _raw_injection, _raw_shell, _npu_ms, _device, _model
```

Env: `INIZ_GUARD_MODEL`, `INIZ_GUARD_DEVICE` (NPU), `INIZ_GUARD_PORT` (8009),
`INIZ_GUARD_THRESHOLD` (0.30).

Terukur (seq_len 128): `_npu_ms` p50 **35,4 ms**, round-trip klien p50 52 ms.
Smoke test 9/10 sesuai ekspektasi kasar; 1 false positive
(*"Summarize this quarterly report in three bullet points"* → inj 0,648).

**Keamanan:** bind `127.0.0.1` **tanpa autentikasi**. Jangan pindah ke `0.0.0.0`
tanpa menambahkan auth.

## Kriteria Validasi

Deployment NPU benar bila **semua** terpenuhi:

- (A) `compiled.get_property("EXECUTION_DEVICES")` == `NPU`
- (B) Counter GPU-Engine untuk pid server hanya menyentuh LUID NPU, CPU `_Total` < 15 %
- (C) p50 ≈ 35 ms untuk `seq_len=128`. Jauh di atas 90 ms → kemungkinan fallback CPU
- (D) IR input shape statis `[1,128]`, bukan `[?,?]`
- (E) Action agreement IR vs PyTorch fp32 == 1,000 pada ≥40 sampel
- (F) accuracy action ≥ 0,95 dengan mapping `guard_labels.py`. Kalau ~0,74 → mapping salah

## Yang MASIH belum selesai

- `shell_head` mengejar proxy keyword; perlu dataset shell-command nyata untuk retrain.
- `ISOLATE_FILE` (support 9) dan `USER_CONFIRMATION` (support 2) terlalu jarang di
  test set untuk dinilai andal — bukan berarti mati, tapi belum terbukti.
- False positive pada teks bisnis benign (~2 % FP rate).
- Threshold 0,30 dari sweep test split, belum divalidasi trafik nyata.
- Merge LoRA + export otomatis untuk 3 head kustom belum masuk `finetune_local.py`
  (harus lewat `export_guard_ov.py` terpisah).

## See Also
- `scripts/` — guard_labels.py, guard_model.py, export_guard_ov.py, eval_final.py,
  verify_incoming_mapping.py, verify_arch_match.py, dryrun_finetune.py,
  diag_shell_head2.py, prove_npu.py, luid_attribution.py, smoke_client.py,
  upload_hf.py, verify_hf_download.py
- `notebooks/01_train_guard.ipynb` — training 3-head (menghasilkan checkpoint-2634)
- `notebooks/02_verify_export_deploy.ipynb` — verifikasi → export → bukti NPU → deploy
- `results/` — eval_final.json, eval_final_seq128.json, mapping_verdict.json,
  npu_verify_int8.json, npu_proof_npuonly.json, luid_{NPU,GPU_0,CPU}.json
- `references/npu-export-pitfalls.md` — traceback tiap jalur export yang gagal
- `references/tokenizer-signature-pitfall.md` — pitfall jalur GENERATIF (bukan guard)
- `references/npu-gpu-util-proof.md` — metodologi bukti Task Manager (jalur generatif)

## Bobot Model

| Artefak | Lokasi | Ukuran |
|---|---|---|
| IR OpenVINO INT8 (siap NPU, `seq_len=128`) | [`CH3NDev/iniz-agent-guard-int8`](https://huggingface.co/CH3NDev/iniz-agent-guard-int8) → `openvino/` | 495 MB |
| Checkpoint 3-head + tokenizer (untuk retrain/export ulang) | idem → `checkpoint/` | 992 MB |

```bash
# IR siap pakai
hf download CH3NDev/iniz-agent-guard-int8 --include "openvino/*" \
  --local-dir ~/npu-provider/models/iniz-guard-int8-ov

# checkpoint mentah (kalau mau export ulang / retrain)
hf download CH3NDev/iniz-agent-guard-int8 --include "checkpoint/*" \
  --local-dir ~/npu-provider/work/ckpt
```

Artefak HF sudah **diverifikasi utuh**: di-download ulang dari Hub, di-compile di NPU
(`EXECUTION_DEVICES = NPU`), dan hasil inferensinya `max|delta| = 0.0000000000`
dibanding model lokal dengan action identik — lihat `scripts/verify_hf_download.py`.
