# Pitfall: The `apply_chat_template` Signature Differs Between `openvino_genai` and `transformers`

Status: this file is referenced in `SKILL.md` (Pitfalls section) but had
not previously been written — it is written now based on a bug that was
actually found and fixed during the `hermes-npu-provider` debugging
session (the predecessor of `iniz-agent-guard`).

## The problem

`transformers.AutoTokenizer.apply_chat_template()` accepts the
`tokenize=False` kwarg to return the raw prompt string (instead of token
ids) — this is the common pattern that most people expect.

`openvino_genai`'s tokenizer, even though its API was intentionally made
to resemble `transformers`, **does not support the `tokenize=` kwarg**
on its own `apply_chat_template()`. Calling it with `tokenize=False` on
an `openvino_genai` tokenizer raises a `TypeError`.

## Why this is silently dangerous

If the code is written with a try/except pattern that catches
`TypeError` and then falls back to a manual prompt formatter (e.g.
concatenating `<|role|>` by hand), this bug **never surfaces as a
visible error** — the program keeps running, but silently uses the wrong
prompt format for the model. That is exactly what happened during the
debugging session: the server startup log said "Chat template loaded"
(because the initial detection succeeded), but every request actually
still fell through to the manual fallback because the runtime
`TypeError` was never caught and reported — as a result the model's
answers for precision-extraction cases were completely wrong, and it
took several rounds of debugging to realize the root cause was not model
quantization but a prompt format that had been wrong from the start.

## The solution

Try both paths explicitly, and **print/log which path is actually
used** — never let the fallback happen silently:

```python
chat_template_fn = None
template_source = None

# Path 1: the built-in openvino_genai tokenizer (WITHOUT the tokenize= kwarg)
try:
    tok = pipeline.get_tokenizer()
    if hasattr(tok, "apply_chat_template"):
        chat_template_fn = tok.apply_chat_template
        template_source = "openvino_genai tokenizer (built-in)"
except Exception as e:
    print(f"Built-in tokenizer path unavailable: {e}")

# Path 2 (fallback): transformers.AutoTokenizer, which DOES support
# tokenize=False, used ONLY to build the prompt text — inference still
# goes through the openvino_genai pipeline above.
if chat_template_fn is None:
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    chat_template_fn = hf_tok.apply_chat_template
    template_source = "transformers AutoTokenizer (fallback)"

print(f"Chat template loaded via: {template_source}")

# When calling, try WITHOUT tokenize= first (matches openvino_genai),
# then WITH tokenize=False as a second fallback (matches transformers):
try:
    prompt = chat_template_fn(messages, add_generation_prompt=True)
except TypeError:
    prompt = chat_template_fn(messages, tokenize=False, add_generation_prompt=True)
```

## How to detect whether this is happening in your deployment

Do not trust the server startup log alone ("chat template loaded" can be
misleading if the initial detection succeeds but the runtime call still
fails). Add per-request logging that prints the final prompt actually
sent to the model — if the format is the manual `<|role|>\ncontent` when
it should be the official ChatML format
(`<|im_start|>role\ncontent<|im_end|>`), that is a sign the manual
fallback is active even though the startup log says otherwise.
