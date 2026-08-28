# NPU Export Pitfalls — traceback nyata per jalur

Semua hasil di bawah diukur di mesin ini (Core Ultra 9 275HX, Windows 11,
torch 2.9.1+cpu, transformers 4.57.1, openvino 2026.3.0) pada
`Qwen2Model` + 3 linear head, LoRA r=8 sudah di-merge, `seq_len=192`.

Skrip pembanding: `~/npu-provider/work/debug_trace.py`

## Ringkasan

| Jalur | Hasil |
|---|---|
| `torch.jit.trace(model, args, strict=False)` | **FAIL** |
| `torch.export.export(model, args, strict=False)` | **OK** |
| `torch.onnx.export(..., dynamo=True)` | FAIL (dependency) |
| `torch.onnx.export(..., dynamo=False)` | **FAIL** (sama seperti jit.trace) |

## 1. torch.jit.trace — FAIL

```
  if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
RuntimeError: invalid unordered_map<K, T> key
```

Sumber: `transformers/masking_utils.py`. Walrus operator di dalam kondisi yang
bergantung pada `attention_mask.shape` membuat TorchScript gagal membangun peta
tipe. Mengubah `attn_implementation` ke `eager` **tidak** menolong — error tetap
sama, karena masalahnya di kode masking, bukan di implementasi attention.

Jangan buang waktu mencoba:
- `strict=True` / `strict=False` — keduanya gagal
- `check_trace=False` — gagal di tahap sebelum pengecekan
- `example_input` dengan mask semua-1 — gagal
- `ov.convert_model(model, example_input=...)` — gagal, karena internal-nya
  memanggil `TorchScriptPythonDecoder` → `jit.trace`:

```
File "openvino/tools/ovc/moc_frontend/pytorch_frontend_utils.py", line 162, in get_pytorch_decoder
    decoder = TorchScriptPythonDecoder(...)
File "openvino/frontend/pytorch/ts_decoder.py", line 84, in __init__
    raise RuntimeError("Couldn't get TorchScript module by tracing.")
```

## 2. torch.export.export — OK (jalur yang dipakai)

```python
model.config._attn_implementation = "eager"
model.config.use_cache = False
with torch.no_grad():
    ep = torch.export.export(wrapper, (ex_ids, ex_mask), strict=False)
ov_model = ov.convert_model(ep)          # terima ExportedProgram langsung
ov_model.reshape({0: ov.PartialShape([1, S]), 1: ov.PartialShape([1, S])})
```

Dua langkah yang mudah terlewat:

1. **`eager` wajib.** Tanpa itu jalur sdpa ikut ter-export dan hasilnya tidak
   compile bersih di NPU.
2. **`reshape` wajib.** `torch.export` melaporkan input sebagai `[?,?]`
   walaupun example input `[1,192]`:

   ```
   [export] inputs : [('input_ids', '[?,?]', 'i64'), ('attention_mask', '[?,?]', 'i64')]   # sebelum reshape
   [export] inputs : [('input_ids', '[1,192]', 'i64'), ('attention_mask', '[1,192]', 'i64')] # sesudah
   ```

   NPU tidak menerima dimensi dinamis.

## 3. torch.onnx.export(dynamo=True) — FAIL (dependency)

```
ModuleNotFoundError: No module named 'onnxscript'
```

Bisa diperbaiki dengan `pip install onnxscript`, tapi tidak perlu: jalur
`torch.export` sudah berhasil dan lebih pendek (tanpa perantara ONNX).

## 4. torch.onnx.export(dynamo=False) — FAIL

Sama dengan jalur 1 — exporter legacy memakai TorchScript di belakang:

```
RuntimeError: invalid unordered_map<K, T> key
```

## Referensi angka export

| Artefak | Ukuran |
|---|---|
| `iniz-guard-fp16-ov/guard.bin` | 988.2 MB |
| `iniz-guard-int8-ov/guard.bin` | 495.3 MB |

`nncf.compress_weights(mode=INT8_SYM)`: 172/172 layer terkompresi (100 %),
per-channel, selesai dalam ~5 s.

## Pemetaan LUID counter Windows (mesin ini)

Windows 11 build ini **tidak punya** counter set `NPU`:

```powershell
(Get-Counter -ListSet * | Where-Object {$_.CounterSetName -match 'NPU|GPU'}).CounterSetName
# GPU Local Adapter Memory / GPU Engine / GPU Adapter Memory /
# GPU Process Memory / GPU Non Local Adapter Memory
```

NPU muncul sebagai adapter di `\GPU Engine(*)`. Pemetaan hasil
`work/luid_attribution.py`:

| LUID | Perangkat | Bukti |
|---|---|---|
| `0x00000000_0x00011cf3` | Intel(R) AI Boost (NPU) | beban NPU → engtype `compute` max 102.75 % |
| `0x00000000_0x00010480` | Intel(R) Graphics (iGPU) | beban GPU.0 → engtype `compute` max 100.07 % |
| `0x00000000_0x0001099d` | NVIDIA RTX 5060 (dGPU) | hanya engine idle 0 % |

Beban device `CPU` → **nol** instance GPU-Engine untuk pid tersebut.

PnP ID untuk konfirmasi:

```
Intel(R) AI Boost                  PCI\VEN_8086&DEV_AD1D
Intel(R) Graphics                  PCI\VEN_8086&DEV_7D67
NVIDIA GeForce RTX 5060 Laptop GPU PCI\VEN_10DE&DEV_2D59
```

## Jebakan pengukuran

Meng-compile model ke `GPU.0` dalam proses yang sama membuat proses itu memegang
konteks GPU sampai mati — counter GPU-Engine untuk pid tersebut jadi tidak bisa
dipakai sebagai bukti "tidak menyentuh GPU". Jalankan pembuktian counter di
proses terpisah: `python prove_npu.py --npu-only`.
