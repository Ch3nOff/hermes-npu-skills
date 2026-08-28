# Bukti Utilisasi NPU — Iniz Agent Guard

Status: mengisi TODO yang sebelumnya ditandai di paket skill ini.
Bukti ini dikumpulkan selama sesi debugging `hermes-npu-provider`
(pendahulu proyek ini, model Qwen2.5-0.5B-Instruct generatif — bukan
versi 3-head fine-tuned), tapi metodologi verifikasinya berlaku sama
untuk memastikan `guard_server.py` benar-benar jalan di NPU, bukan
diam-diam fallback ke CPU/GPU.

## Ringkasan bukti

Tiga sumber independen dikumpulkan dan saling konsisten:

1. **Visual Task Manager** — tiga screenshot menunjukkan grafik NPU
   melonjak ke pola plateau (naik tajam saat inferensi dimulai, datar
   di puncak selama proses, turun tajam saat selesai) persis pada
   window waktu benchmark dijalankan, sementara GPU 0 (Intel iGPU) dan
   GPU 1 (NVIDIA RTX discrete) tetap landai di bawahnya pada window
   yang sama.

2. **CSV performance sampler** (`perf_sampler_gpu_detail.ps1` di
   folder `references/` ini) — mencatat `gpu_util_sum_pct` dan
   `cpu_util_pct` per timestamp. CPU tercatat maksimum ~4.7% selama
   window inferensi, jauh di bawah yang diharapkan kalau beban kerja
   itu benar-benar jalan di CPU.

3. **Atribusi per-proses GPU** — breakdown per-PID menunjukkan aktivitas
   GPU pada window yang sama berasal dari proses UI (Hermes UI sendiri,
   compositor DWM Windows) — BUKAN dari proses Python/server yang
   menjalankan inferensi model.

## ⚠️ Catatan jujur soal batas bukti ini

- **Bukti ini dikumpulkan untuk model generatif 0.5B biasa**
  (`hermes-npu-provider`, sebelum fine-tuning 3-head), bukan untuk
  `guard_server.py` versi terkini secara langsung. Metodologinya
  identik dan device (`NPU`) di `guard_server.py` tidak berubah, tapi
  jika ingin bukti yang 100% spesifik untuk model 3-head hasil
  fine-tuning, ulangi capture Task Manager + CSV sampler ini SETELAH
  model hasil fine-tuning benar-benar di-export ke OpenVINO INT8 dan
  diserve lewat `guard_server.py`.
- **`gpu_util_sum_pct` di CSV mentah pernah menunjukkan angka 17-41%**
  pada window yang sama dengan inferensi NPU — ini AWALNYA terlihat
  mencurigakan, tapi breakdown per-proses (poin 3 di atas) menjelaskan
  ini berasal dari Hermes UI yang me-render hasil streaming secara
  real-time, bukan dari proses inferensi itu sendiri. Dicatat di sini
  supaya tidak disalahartikan sebagai bukti yang bertentangan kalau
  seseorang membaca ulang CSV mentah tanpa konteks ini.
- **Latensi yang terukur** (~2.7–6.6 detik tergantung kuantisasi
  INT4/INT8) jauh di atas target awal blueprint (<15ms) — ini fakta
  yang perlu diketahui siapa pun yang membaca proof-of-utilization ini
  supaya tidak salah simpul "NPU dipakai" dengan "NPU cepat". NPU
  dipakai secara terkonfirmasi; performa absolutnya adalah pertanyaan
  terpisah, dan generasi hardware NPU saat ini (Meteor Lake/Arrow
  Lake) masih punya variansi tinggi untuk beban kerja LLM kecil
  sekalipun.

## Cara mereplikasi verifikasi ini sendiri

1. Jalankan `scripts/guard_server.py`.
2. Buka Task Manager > Performance > NPU **sebelum** menembak request
   apa pun ke server.
3. Kirim beberapa request test (lihat contoh curl di `SKILL.md`
   bagian Quick Reference).
4. Amati: grafik NPU harus naik pada window waktu yang sama dengan
   request diproses; grafik GPU discrete (kalau ada) harus tetap
   landai pada window yang sama.
5. Jalankan `references/perf_sampler_gpu_detail.ps1` secara paralel
   untuk mendapat catatan CSV yang bisa diperiksa ulang, bukan cuma
   observasi visual sesaat.

Bukti yang genuinely meyakinkan butuh ketiga hal di atas ditemukan
bersamaan pada satu window waktu yang sama — bukan cuma satu sumber
saja.
