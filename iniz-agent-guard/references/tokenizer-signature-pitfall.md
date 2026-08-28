# Pitfall: Signature `apply_chat_template` Berbeda Antara `openvino_genai` dan `transformers`

Status: file ini direferensikan di `SKILL.md` (bagian Pitfalls) tapi
sebelumnya belum dibuat — ditulis sekarang berdasarkan bug yang
benar-benar ditemukan dan diperbaiki selama sesi debugging
`hermes-npu-provider` (proyek pendahulu `iniz-agent-guard`).

## Masalah

`transformers.AutoTokenizer.apply_chat_template()` menerima kwarg
`tokenize=False` untuk mengembalikan string prompt mentah (bukan token
id) — ini pola yang umum dipakai dan diharapkan banyak orang.

`openvino_genai`'s tokenizer, meski API-nya secara sengaja dibuat mirip
`transformers`, **tidak mendukung kwarg `tokenize=`** pada
`apply_chat_template()`-nya sendiri. Memanggil dengan `tokenize=False`
pada tokenizer `openvino_genai` melempar `TypeError`.

## Kenapa ini berbahaya secara diam-diam

Kalau kode ditulis dengan pola try/except yang menangkap `TypeError` lalu
fallback ke formatter prompt manual (mis. menggabungkan `<|role|>` secara
manual), bug ini **tidak pernah muncul sebagai error yang terlihat** —
program tetap jalan, tapi diam-diam memakai format prompt yang salah untuk
model. Ini persis yang terjadi di sesi debugging: log start server bilang
"Chat template dimuat" (karena deteksi awal berhasil), tapi tiap request
sebenarnya tetap jatuh ke fallback manual karena `TypeError` di runtime
tidak pernah ditangkap dan dilaporkan — akibatnya jawaban model untuk
kasus ekstraksi presisi salah total, dan butuh beberapa putaran
debugging untuk sadar akar masalahnya bukan di kuantisasi model,
melainkan di format prompt yang salah sejak awal.

## Solusi

Coba dua jalur secara eksplisit, dan **cetak/log jalur mana yang
benar-benar dipakai** — jangan biarkan fallback terjadi diam-diam:

```python
chat_template_fn = None
template_source = None

# Jalur 1: tokenizer bawaan openvino_genai (TANPA kwarg tokenize=)
try:
    tok = pipeline.get_tokenizer()
    if hasattr(tok, "apply_chat_template"):
        chat_template_fn = tok.apply_chat_template
        template_source = "openvino_genai tokenizer (bawaan)"
except Exception as e:
    print(f"Jalur tokenizer bawaan tidak tersedia: {e}")

# Jalur 2 (fallback): transformers.AutoTokenizer, YANG memang mendukung
# tokenize=False, dipakai HANYA untuk menyusun teks prompt — inferensi
# tetap lewat pipeline openvino_genai di atas.
if chat_template_fn is None:
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    chat_template_fn = hf_tok.apply_chat_template
    template_source = "transformers AutoTokenizer (fallback)"

print(f"Chat template dimuat via: {template_source}")

# Saat memanggil, coba TANPA tokenize= dulu (cocok openvino_genai),
# baru WITH tokenize=False sebagai fallback kedua (cocok transformers):
try:
    prompt = chat_template_fn(messages, add_generation_prompt=True)
except TypeError:
    prompt = chat_template_fn(messages, tokenize=False, add_generation_prompt=True)
```

## Cara mendeteksi kalau ini sedang terjadi di deployment Anda

Jangan percaya log start server saja ("chat template dimuat" bisa
menyesatkan jika deteksi awal berhasil tapi runtime call tetap gagal).
Tambahkan logging per-request yang mencetak prompt final yang benar-benar
dikirim ke model — kalau formatnya `<|role|>\ncontent` manual padahal
seharusnya format ChatML resmi (`<|im_start|>role\ncontent<|im_end|>`),
itu tanda fallback manual sedang aktif meski log start bilang sebaliknya.
