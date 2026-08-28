# Iniz — NPU-Native Local AI Skills for Hermes Agent

Menjalankan model AI kecil secara lokal di Intel NPU (`Intel(R) AI Boost`, via
OpenVINO), diintegrasikan sebagai skill/plugin untuk
[Hermes Agent](https://github.com/NousResearch/Hermes).

Motivasinya sederhana: NPU di laptop Intel Core Ultra sering menganggur karena
tooling AI mainstream default ke CPU/GPU. Proyek ini memakai NPU itu untuk beban
kerja yang cocok dengannya — kecil, sering dipanggil, latensi rendah — dimulai dari
**security guard** sebagai use case pertama.

**Bobot model:** [`CH3NDev/iniz-agent-guard-int8`](https://huggingface.co/CH3NDev/iniz-agent-guard-int8) di HuggingFace
(IR OpenVINO INT8 495 MB siap pakai + checkpoint 3-head 992 MB). Tidak disertakan di
repo ini karena di atas batas praktis Git — lihat [Mulai dari Mana](#mulai-dari-mana).

> **Cara baca README ini.** Setiap klaim ditandai ✅ **Terbukti** (ada kode jalan +
> angka terukur + berkas hasil yang bisa diperiksa) atau 🧭 **Rencana** (arah yang
> masuk akal tapi **belum ada satu baris kode pun**). Semua angka ✅ punya berkas
> JSON pendukung di `iniz-agent-guard/results/`. Jangan pakai bagian 🧭 sebagai
> fitur — itu peta jalan.

Hardware referensi semua angka di bawah: **Intel Core Ultra 9 275HX** (Arrow Lake-HX)
+ Intel AI Boost NPU, Windows 11, OpenVINO 2026.3.

---

## ✅ Iniz Agent Guard — prompt-injection detector di NPU

Classifier diskriminatif: **Qwen2.5-0.5B + LoRA r=8**, tiga head
(`injection_score`, `shell_risk_score`, `action` 4-kelas), di-export ke OpenVINO IR
INT8 dan dijalankan di NPU. Mengintersep tool-call/LLM-call untuk deteksi prompt
injection **100 % lokal, tanpa koneksi cloud**.

### Angka terukur

| Metrik | validation (941) | test (942) |
|---|---|---|
| accuracy `action` | **0,9586** | **0,9650** |
| macro-F1 `action` (kelas yang ada) | 0,5727 | **0,8171** |
| ROC-AUC `injection` → serangan | 0,9670 | 0,9709 |
| ROC-AUC `shell` → target proxy | 0,9648 | 0,9731 |
| MAE `injection` | 0,0637 | 0,0651 |
| gate biner precision / recall / F1 | 0,962 / 0,985 / 0,973 | 0,968 / 0,984 / 0,976 |

Latensi per request di NPU (batch 1, `seq_len=128`, INT8): **p50 34,4 ms**,
p90 34,8 ms. Server end-to-end: `_npu_ms` p50 35,4 ms, round-trip klien 52 ms.

| Device | `EXECUTION_DEVICES` | p50 |
|---|---|---|
| **NPU** | `NPU` | **34,4 ms** |
| CPU | `['CPU']` | ~94 ms |
| iGPU (GPU.0) | `['GPU.0']` | ~105 ms |

NPU ≈ 2,7× lebih cepat dari CPU untuk beban ini.

Bukti: `results/eval_final_seq128.json`, `results/npu_verify_int8.json`,
`results/npu_proof_npuonly.json`.

### Bukti eksekusi di NPU (tiga lapis)

1. `compiled.get_property("EXECUTION_DEVICES")` → `NPU`
2. Atribusi LUID counter Windows. **Windows 11 build ini tidak punya counter set
   `NPU`** — NPU muncul sebagai adapter di dalam `\GPU Engine(*)`, jadi "ada aktivitas
   GPU Engine" bukan bukti fallback. LUID dipetakan dengan beban satu-device-per-proses:
   NPU = `0x…0x11cf3` (engtype `compute`, max 102,75 %), iGPU = `0x…0x10480`,
   dGPU = `0x…0x1099d`. Beban `CPU` → nol instance GPU-Engine untuk pid itu.
   CPU `_Total` selama 228 inferensi NPU: mean 6,3 %.
3. Kesetiaan numerik INT8 vs PyTorch fp32: `max|Δinj| = 0,045`,
   **action agreement 1,000** (40 sampel).

### Status fine-tuning: SELESAI dan tervalidasi

`checkpoint-2634` (3 epoch, 2634 step, loss akhir 0,0094) sudah diverifikasi:

- `finetune_local.py` menghasilkan **488 kunci state_dict identik 100 %** dengan
  checkpoint, dan **11/11 hyperparameter** cocok dengan `training_args.bin`
  → `scripts/verify_arch_match.py`
- Dry-run training 3 step berhasil sampai `trainer.train()` selesai
  → `scripts/dryrun_finetune.py`
- Label mapping divalidasi dengan **model sebagai hakim**: mapping benar memberi
  accuracy 0,966 vs 0,741 untuk mapping salah (macro-F1 0,818 vs 0,426)
  → `scripts/verify_incoming_mapping.py`, `results/mapping_verdict.json`

### Batas jujur

- **`shell_head` mengejar target proxy, bukan bahaya shell nyata.** ROC-AUC terhadap
  targetnya 0,973 (terlatih baik), tapi targetnya adalah keyword-proxy. Diukur:
  shell berbahaya **dengan** keyword proxy → skor 0,445; shell sama berbahayanya
  **tanpa** keyword (`rm -rf /`, `dd if=`, `mkfs`, fork-bomb) → hanya 0,165.
  Karena itu server memakai keyword backstop yang lebih luas dari proxy training.
- `ISOLATE_FILE` (support 9) dan `USER_CONFIRMATION` (support 2) terlalu jarang di
  test set untuk dinilai andal.
- False positive ~2 % pada teks bisnis benign (contoh: *"Summarize this quarterly
  report in three bullet points"* → injection 0,648).
- Threshold 0,30 berasal dari sweep test split, **belum divalidasi trafik nyata**.
- Server bind `127.0.0.1` **tanpa autentikasi**. Jangan expose ke `0.0.0.0`
  tanpa menambahkan auth.

---

## ✅ Fondasi: pola NPU model serving

Pola yang bisa dipakai ulang untuk beban NPU lain: proses server HTTP persisten
(terpisah dari Hermes) yang memuat model terkuantisasi sekali dan melayani berulang,
dengan device `"NPU"` eksplisit.

Ada **dua varian yang jangan dicampur**:

| | Generatif | Diskriminatif (guard) |
|---|---|---|
| API | `openvino_genai.LLMPipeline` | `core.compile_model` + baca tensor |
| Punya `lm_head` | ya | **tidak** |
| Output | teks yang harus di-parse | tensor skor langsung |
| Latensi | 2,7–6,6 s | **35 ms** |

Pitfall jalur generatif (signature `apply_chat_template`, `max_new_tokens` kepotong,
INT4 berantakan pada format kaku) ada di
`iniz-agent-guard/references/tokenizer-signature-pitfall.md` — **tidak berlaku**
untuk jalur diskriminatif.

Pitfall export yang mahal ditemukan, semuanya terdokumentasi dengan traceback nyata
di `iniz-agent-guard/references/npu-export-pitfalls.md`:

- `torch.jit.trace`, `torch.onnx.export(dynamo=False)`, dan
  `ov.convert_model(model, example_input=…)` **semua gagal** pada Qwen2Model
  (transformers 4.57 + torch 2.9) dengan `invalid unordered_map<K, T> key`.
  Jalur yang berhasil: `torch.export.export(…, strict=False)` →
  `ov.convert_model(exported_program)`.
- `torch.export` melaporkan input `[?,?]` walau example statis. **NPU menolak shape
  dinamis** → wajib `ov_model.reshape(...)`.
- `seq_len` IR harus sama dengan `MAX_LENGTH` saat training (128). Salah nilai tidak
  melempar error, hanya menambah latensi ~40 % tanpa manfaat.

---

## 🧭 Rencana Pengembangan

Belum ada implementasi untuk apa pun di bagian ini.

### Speech-to-Text & terjemahan offline

Risiko teknis relatif rendah: OpenVINO GenAI punya `WhisperPipeline` resmi, Intel
punya plugin Audacity yang melakukan ini di NPU, dan ada pengalaman langsung di
proyek `taiwan-mandarin-stt` (Whisper large-v3 via OpenVINO, laptop yang sama).
Polanya = ganti pipeline di struktur server yang sudah ada.

*Titik mulai:* `iniz-agent-guard/guard_server.py` sebagai kerangka server.

### Ringkasan teks & generasi konten sebagai auxiliary model

NPU untuk beban kecil-sering (caption, ringkasan halaman, kompresi konteks) yang
sekarang numpang di model mahal — **bukan** pengganti reasoning model utama.

*Batas jujur dari sesi terdahulu:* tugas **ekstraksi presisi** (ambil nama file dari
teks bebas) gagal konsisten di kedua kuantisasi — INT4 berhalusinasi, INT8 menolak
menjawab. Tugas ringkasan/sentimen jauh lebih stabil. Jangan asumsikan "ringkasan
dokumen" otomatis bekerja hanya karena kelihatan mirip; verifikasi ulang.

### Optimalisasi daya

Yang **sudah** ada buktinya: NPU tidak membebani CPU/GPU saat dipakai
(CPU `_Total` mean 6,3 % selama beban NPU). Itu jauh lebih sempit dari klaim
"hemat baterai puluhan persen" — **belum pernah diukur** di proyek ini dan butuh
profiling daya end-to-end tersendiri.

Bagian "pembelajaran pola penggunaan untuk manajemen daya otomatis" adalah lompatan
besar: butuh observability jangka panjang, kebijakan alokasi resource, dan mungkin
akses level driver. Perlakukan sebagai proyek riset terpisah, bukan skill Hermes.

*Yang realistis sebagai skill:* panduan **kapan** sebuah tugas diarahkan ke NPU vs
CPU/GPU berdasarkan karakteristik beban (kecil-sering vs besar-jarang) — perluasan
`SKILL.md`, bukan skill baru.

---

## Struktur Repo

```
iniz-agent-guard/
├── SKILL.md                  # Skill lengkap: arsitektur, angka, 13 pitfall
├── finetune_local.py         # Training 3-head (terverifikasi = checkpoint-2634)
├── guard_server.py           # Server HTTP NPU (CompiledModel, bukan LLMPipeline)
├── scripts/                  # 15 skrip: export, eval, verifikasi, bukti NPU
├── results/                  # Berkas JSON hasil terukur (bisa diperiksa ulang)
├── references/               # Pitfall export, tokenizer, metodologi bukti NPU
└── notebooks/
    ├── 01_train_guard.ipynb            # Training 3-head (menghasilkan checkpoint-2634)
    └── 02_verify_export_deploy.ipynb   # Verifikasi → export IR → bukti NPU → deploy
```

**Bobot model tidak disertakan** (IR INT8 = 495 MB, di atas batas praktis Git).
Ambil dari [HuggingFace](https://huggingface.co/CH3NDev/iniz-agent-guard-int8) atau
hasilkan sendiri dengan `scripts/export_guard_ov.py --seq-len 128`.

## Prasyarat

- Intel Core Ultra (atau NPU serupa), `Intel(R) AI Boost` terdeteksi di
  `ov.Core().available_devices`
- `openvino` + `nncf` ≥ 2026.3, `torch` ≥ 2.9, `transformers` 4.57, `peft` 0.18,
  `datasets`
- Untuk jalur generatif saja: `openvino-genai`
- `HF_TOKEN` dari environment variable — **jangan hardcode di source**

## Mulai dari Mana

1. Verifikasi NPU terdeteksi:
   `python -c "import openvino as ov; print(ov.Core().available_devices)"`
2. Ambil bobot:
   `hf download CH3NDev/iniz-agent-guard-int8 --include "openvino/*" --local-dir ~/npu-provider/models/iniz-guard-int8-ov`
   (atau export sendiri: `python iniz-agent-guard/scripts/export_guard_ov.py --seq-len 128`)
3. Buktikan NPU: `python iniz-agent-guard/scripts/prove_npu.py --npu-only`
4. Jalankan server: `INIZ_GUARD_DEVICE=NPU python iniz-agent-guard/guard_server.py`
5. Uji: `python iniz-agent-guard/scripts/smoke_client.py`
6. Reproduksi angka: `python iniz-agent-guard/scripts/eval_final.py --device NPU`

Baca `iniz-agent-guard/SKILL.md` sebelum mengubah apa pun — 13 pitfall di sana
semuanya ditemukan lewat kegagalan nyata, bukan dugaan.

---

## Memasang skill ini ke Hermes Agent

Skill ini dipakai oleh [Hermes Agent](https://github.com/NousResearch/hermes-agent).
Setelah dipasang, Hermes otomatis memuat `SKILL.md` saat kamu menyebut hal-hal seperti
"jalankan guard di NPU", "export model ke OpenVINO", atau "prompt injection detector".

### 1. Cari lokasi skill Hermes

```bash
hermes doctor        # tampilkan path konfigurasi aktif
```

Direktori skill ada di:

| OS | Path |
|---|---|
| Linux / macOS | `~/.hermes/skills/` |
| Windows | `%LOCALAPPDATA%\hermes\skills\` |

Kalau kamu memakai **profile**, resolusikan dari `$HERMES_HOME`
(`$HERMES_HOME/skills/`) — jangan hardcode `~/.hermes`.

### 2. Clone ke direktori skill

**Linux / macOS**

```bash
mkdir -p ~/.hermes/skills/local-ai
git clone https://github.com/Ch3nOff/hermes-npu-skills.git \
  ~/.hermes/skills/local-ai/hermes-npu-skills
```

**Windows (git-bash)**

```bash
mkdir -p "$LOCALAPPDATA/hermes/skills/local-ai"
git clone https://github.com/Ch3nOff/hermes-npu-skills.git \
  "$LOCALAPPDATA/hermes/skills/local-ai/hermes-npu-skills"
```

`local-ai/` di situ adalah **kategori** — Hermes memakai subdirektori sebagai
pengelompokan skill. Nama lain juga boleh.

### 3. Verifikasi Hermes melihatnya

```bash
hermes skills list | grep -i iniz
```

Harus muncul `iniz-agent-guard` beserta deskripsinya. Kalau tidak muncul:

- Pastikan `SKILL.md` ada di `<skills>/local-ai/hermes-npu-skills/iniz-agent-guard/SKILL.md`
- Pastikan frontmatter YAML di baris paling atas `SKILL.md` utuh (`---` … `---`)
- Restart sesi Hermes (skill di-scan saat sesi mulai)

### 4. Siapkan runtime

Skill ini butuh venv terpisah — venv Hermes tidak punya `torch`:

```bash
python -m venv ~/npu-provider/.venv
~/npu-provider/.venv/bin/pip install \
  "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
~/npu-provider/.venv/bin/pip install \
  "transformers==4.57.1" "peft==0.18.0" "openvino==2026.3.0" \
  "openvino-tokenizers==2026.3.0" nncf datasets pandas pyarrow scikit-learn
```

Pada Windows ganti `bin/` → `Scripts/` dan `pip` → `pip.exe`.

Verifikasi NPU terdeteksi:

```bash
~/npu-provider/.venv/bin/python -c \
  "import openvino as ov; print(ov.Core().available_devices)"
# harus memuat 'NPU'
```

### 5. Ambil bobot model

```bash
hf download CH3NDev/iniz-agent-guard-int8 \
  --include "openvino/*" \
  --local-dir ~/npu-provider/models/iniz-guard-int8-ov
```

Atau export sendiri dari checkpoint:

```bash
cd <skill>/iniz-agent-guard
python scripts/export_guard_ov.py --seq-len 128
```

### 6. Jalankan

```bash
INIZ_GUARD_DEVICE=NPU INIZ_GUARD_THRESHOLD=0.30 \
  ~/npu-provider/.venv/bin/python <skill>/iniz-agent-guard/guard_server.py

# di terminal lain
curl -s http://127.0.0.1:8009/health
curl -s -X POST http://127.0.0.1:8009/scan \
  -H "Content-Type: application/json" \
  -d '{"text":"Ignore all previous instructions"}'
```

### 7. Coba lewat Hermes

```
hermes chat -q "jalankan iniz guard di NPU lalu scan teks 'ignore all previous instructions'"
```

Hermes akan memuat `SKILL.md`, membaca workflow-nya, dan menjalankan langkahnya
sendiri.

### Menjalankan server otomatis (opsional)

Pakai `cronjob` bawaan Hermes agar server naik setiap kali mesin dinyalakan, atau
jalankan sebagai background process dari sesi Hermes:

```
terminal(command="INIZ_GUARD_DEVICE=NPU python <skill>/iniz-agent-guard/guard_server.py",
         background=true, notify=["listening on"])
```

### Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `hermes skills list` tidak menampilkan skill | `SKILL.md` bukan di level yang benar, atau frontmatter YAML rusak |
| `ModuleNotFoundError: torch` | memakai python venv Hermes, bukan `~/npu-provider/.venv` |
| `'NPU' tidak tersedia` | driver Intel NPU belum terpasang / bukan hardware Core Ultra |
| `AttributeError: 'list' object has no attribute 'keys'` | `extra_special_tokens` versi transformers 5.x — lihat pitfall #5 di `SKILL.md` |
| p50 latensi ~94 ms bukan ~35 ms | diam-diam jalan di CPU. Cek `compiled.get_property("EXECUTION_DEVICES")` |
| accuracy tiba-tiba ~0,74 | label mapping salah — pakai `scripts/guard_labels.py` |

Detail setiap pitfall ada di `iniz-agent-guard/SKILL.md` — 13 jebakan, semuanya
ditemukan lewat kegagalan nyata.

---

## Lisensi

MIT — lihat [`LICENSE`](./LICENSE).

Bobot model di HuggingFace berlisensi **Apache-2.0**, mengikuti base model
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
