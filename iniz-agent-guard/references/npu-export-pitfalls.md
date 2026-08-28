# NPU Export Pitfalls — real tracebacks per path

All results below were measured on this machine (Core Ultra 9 275HX, Windows 11,
torch 2.9.1+cpu, transformers 4.57.1, openvino 2026.3.0) on
`Qwen2Model` + 3 linear heads, LoRA r=8 already merged, `seq_len=192`.

Comparison script: `~/npu-provider/work/debug_trace.py`

## Summary

| Path | Result |
|---|---|
| `torch.jit.trace(model, args, strict=False)` | **FAIL** |
| `torch.export.export(model, args, strict=False)` | **OK** |
| `torch.onnx.export(..., dynamo=True)` | FAIL (dependency) |
| `torch.onnx.export(..., dynamo=False)` | **FAIL** (same as jit.trace) |

## 1. torch.jit.trace — FAIL

```
  if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
RuntimeError: invalid unordered_map<K, T> key
```

Source: `transformers/masking_utils.py`. A walrus operator inside a condition
that depends on `attention_mask.shape` makes TorchScript fail to build the type
map. Changing `attn_implementation` to `eager` does **not** help — the error
stays the same, because the problem is in the masking code, not in the attention
implementation.

Do not waste time trying:
- `strict=True` / `strict=False` — both fail
- `check_trace=False` — fails at a stage before the check
- `example_input` with an all-ones mask — fails
- `ov.convert_model(model, example_input=...)` — fails, because internally it
  calls `TorchScriptPythonDecoder` → `jit.trace`:

```
File "openvino/tools/ovc/moc_frontend/pytorch_frontend_utils.py", line 162, in get_pytorch_decoder
    decoder = TorchScriptPythonDecoder(...)
File "openvino/frontend/pytorch/ts_decoder.py", line 84, in __init__
    raise RuntimeError("Couldn't get TorchScript module by tracing.")
```

## 2. torch.export.export — OK (the path actually used)

```python
model.config._attn_implementation = "eager"
model.config.use_cache = False
with torch.no_grad():
    ep = torch.export.export(wrapper, (ex_ids, ex_mask), strict=False)
ov_model = ov.convert_model(ep)          # accepts the ExportedProgram directly
ov_model.reshape({0: ov.PartialShape([1, S]), 1: ov.PartialShape([1, S])})
```

Two steps that are easy to miss:

1. **`eager` is required.** Without it the sdpa path gets exported too and the
   result does not compile cleanly on the NPU.
2. **`reshape` is required.** `torch.export` reports the inputs as `[?,?]`
   even though the example input is `[1,192]`:

   ```
   [export] inputs : [('input_ids', '[?,?]', 'i64'), ('attention_mask', '[?,?]', 'i64')]   # before reshape
   [export] inputs : [('input_ids', '[1,192]', 'i64'), ('attention_mask', '[1,192]', 'i64')] # after
   ```

   The NPU does not accept dynamic dimensions.

## 3. torch.onnx.export(dynamo=True) — FAIL (dependency)

```
ModuleNotFoundError: No module named 'onnxscript'
```

This can be fixed with `pip install onnxscript`, but there is no need: the
`torch.export` path already succeeds and is shorter (no ONNX intermediary).

## 4. torch.onnx.export(dynamo=False) — FAIL

Same as path 1 — the legacy exporter uses TorchScript underneath:

```
RuntimeError: invalid unordered_map<K, T> key
```

## Export size reference

| Artifact | Size |
|---|---|
| `iniz-guard-fp16-ov/guard.bin` | 988.2 MB |
| `iniz-guard-int8-ov/guard.bin` | 495.3 MB |

`nncf.compress_weights(mode=INT8_SYM)`: 172/172 layers compressed (100 %),
per-channel, finished in ~5 s.

## Windows LUID counter mapping (this machine)

This Windows 11 build **does not have** an `NPU` counter set:

```powershell
(Get-Counter -ListSet * | Where-Object {$_.CounterSetName -match 'NPU|GPU'}).CounterSetName
# GPU Local Adapter Memory / GPU Engine / GPU Adapter Memory /
# GPU Process Memory / GPU Non Local Adapter Memory
```

The NPU shows up as an adapter under `\GPU Engine(*)`. Mapping produced by
`work/luid_attribution.py`:

| LUID | Device | Evidence |
|---|---|---|
| `0x00000000_0x00011cf3` | Intel(R) AI Boost (NPU) | NPU load → engtype `compute` max 102.75 % |
| `0x00000000_0x00010480` | Intel(R) Graphics (iGPU) | GPU.0 load → engtype `compute` max 100.07 % |
| `0x00000000_0x0001099d` | NVIDIA RTX 5060 (dGPU) | idle engines only, 0 % |

A `CPU` device load → **zero** GPU-Engine instances for that pid.

PnP IDs for confirmation:

```
Intel(R) AI Boost                  PCI\VEN_8086&DEV_AD1D
Intel(R) Graphics                  PCI\VEN_8086&DEV_7D67
NVIDIA GeForce RTX 5060 Laptop GPU PCI\VEN_10DE&DEV_2D59
```

## Measurement pitfall

Compiling a model to `GPU.0` inside the same process makes that process hold a
GPU context until it exits — the GPU-Engine counters for that pid then cannot be
used as evidence of "never touched the GPU". Run the counter proof in a separate
process: `python prove_npu.py --npu-only`.
