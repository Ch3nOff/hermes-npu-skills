"""
03_compare_generation_modes.py — Bandingkan hasil benchmark antar
GENERATION_MODE (greedy_64, sample_64, greedy_32, sample_32) untuk
isolasi penyebab latensi 4.8s/request yang ditemukan di run v2.

CARA PAKAI:
  1. Set GENERATION_MODE = "greedy_64" di npu_server_v3.py, jalankan server.
  2. Jalankan skrip ini dengan --mode greedy_64 (di terminal lain).
  3. Ctrl+C server, ganti GENERATION_MODE = "sample_64", jalankan ulang.
  4. Jalankan skrip ini lagi dengan --mode sample_64.
  5. Ulangi untuk greedy_32 dan sample_32 jika perlu.
  6. Jalankan skrip ini SEKALI LAGI tanpa --mode untuk lihat perbandingan
     semua hasil yang sudah terkumpul.

Hasil tiap run disimpan ke results.json (di folder yang sama) supaya
perbandingan bisa dilakukan lintas sesi, tidak hilang saat terminal ditutup.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path


SERVER_URL = "http://127.0.0.1:8008/v1/chat/completions"
RESULTS_FILE = Path(__file__).parent / "results.json"

TEST_PROMPTS = [
    "Ringkas dalam satu kalimat: pengguna meminta bantuan membaca file config.",
    "Klasifikasikan sentimen pesan ini: 'server down lagi, tolong cek'",
    "Ekstrak nama file dari teks: 'edit provider.py lalu jalankan npu_server.py'",
]


def call_server(prompt: str) -> dict:
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
    wall_time_ms = (time.time() - start) * 1000

    return {
        "wall_time_ms": round(wall_time_ms, 2),
        "server_latency_ms": result.get("_npu_latency_ms"),
        "approx_tok_per_sec": result.get("_npu_approx_tok_per_sec"),
        "generation_config": result.get("_npu_generation_config"),
        "finish_reason": result.get("_npu_finish_reason"),
        "likely_hit_token_budget": result.get("_npu_likely_hit_token_budget"),
        "output": result["choices"][0]["message"]["content"],
    }


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def run_mode(mode_label: str):
    print(f"\n{'=' * 60}")
    print(f"MENJALANKAN BENCHMARK — label run: {mode_label}")
    print(f"{'=' * 60}")
    print(
        "PENTING: pastikan npu_server_v3.py yang SEDANG JALAN memang "
        f"di-set ke GENERATION_MODE yang sesuai dengan label '{mode_label}' "
        "ini. Skrip ini TIDAK bisa mengecek itu otomatis — cek manual log "
        "server saat start ('GENERATION_MODE aktif: ...')."
    )

    run_results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] {prompt[:50]}...")
        try:
            result = call_server(prompt)
            run_results.append(result)
            print(f"  Wall: {result['wall_time_ms']}ms | tok/s: {result['approx_tok_per_sec']}")
            print(f"  Config aktual dari server: {result['generation_config']}")
            print(f"  Kemungkinan habiskan token budget: {result['likely_hit_token_budget']}")
            print(f"  Output: {result['output'][:100]}")
        except Exception as e:
            print(f"  GAGAL: {e}")
            return

    all_results = load_results()
    all_results[mode_label] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requests": run_results,
        "avg_wall_time_ms": round(sum(r["wall_time_ms"] for r in run_results) / len(run_results), 2),
    }
    save_results(all_results)
    print(f"\nHasil disimpan ke {RESULTS_FILE} dengan label '{mode_label}'")


def show_comparison():
    all_results = load_results()
    if not all_results:
        print("Belum ada hasil tersimpan. Jalankan dengan --mode <label> dulu.")
        return

    print(f"\n{'=' * 70}")
    print("PERBANDINGAN ANTAR MODE")
    print(f"{'=' * 70}")
    print(f"{'Mode':<15} {'Avg wall (ms)':<15} {'Tercatat pada'}")
    print("-" * 70)
    for label, data in all_results.items():
        print(f"{label:<15} {data['avg_wall_time_ms']:<15} {data['timestamp']}")

    print(
        "\nCara baca:\n"
        "- Kalau sample_64 jauh lebih cepat dari greedy_64 -> do_sample=False\n"
        "  (constraint yang v2 asumsikan wajib) kemungkinan besar penyebab\n"
        "  utama lambatnya, BUKAN ukuran model atau NPU itu sendiri.\n"
        "- Kalau greedy_32 jauh lebih cepat dari greedy_64 (mendekati rasio\n"
        "  1:2) -> model memang menghabiskan hampir seluruh token budget\n"
        "  tiap kali (dugaan (a) di laporan kamu terkonfirmasi), turunkan\n"
        "  max_new_tokens permanen untuk tugas auxiliary singkat.\n"
        "- Kalau SEMUA mode sama-sama lambat (~4-5s, tidak ada yang beda\n"
        "  signifikan) -> penyebabnya BUKAN di parameter generate, kembali\n"
        "  ke dugaan (b) throttling daya/thermal — cek Windows Power Mode\n"
        "  (Best Performance vs Balanced vs Power Saver) saat benchmark."
    )

    # Cek finish_reason di semua run yang tersimpan — kalau field ini
    # konsisten "length" atau setara (bukan "stop"/"eos"), itu bukti
    # langsung dugaan (a), bukan lagi tebakan dari rasio tok/s.
    print(f"\n{'=' * 70}")
    print("CEK finish_reason (kalau library expose info ini)")
    print(f"{'=' * 70}")
    any_finish_reason_found = False
    for label, data in all_results.items():
        for i, req in enumerate(data["requests"], 1):
            fr = req.get("finish_reason", "unknown")
            if fr not in ("unknown", None):
                any_finish_reason_found = True
                print(f"  {label} #{i}: finish_reason={fr}")
    if not any_finish_reason_found:
        print(
            "  Tidak ada finish_reason yang berhasil ditangkap dari library\n"
            "  di semua run. Diagnosis harus mengandalkan proxy "
            "  'likely_hit_token_budget' saja (word count vs max_new_tokens),\n"
            "  yang kurang presisi dibanding finish_reason asli."
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--mode":
        run_mode(sys.argv[2])
    else:
        show_comparison()
