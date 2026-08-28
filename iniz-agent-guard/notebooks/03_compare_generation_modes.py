"""
03_compare_generation_modes.py — Compare benchmark results across
GENERATION_MODE settings (greedy_64, sample_64, greedy_32, sample_32) to
isolate the cause of the 4.8s/request latency observed in run v2.

HOW TO USE:
  1. Set GENERATION_MODE = "greedy_64" in npu_server_v3.py, start the server.
  2. Run this script with --mode greedy_64 (in another terminal).
  3. Ctrl+C the server, switch to GENERATION_MODE = "sample_64", restart it.
  4. Run this script again with --mode sample_64.
  5. Repeat for greedy_32 and sample_32 if needed.
  6. Run this script ONE MORE TIME without --mode to see the comparison of
     every result collected so far.

Each run's results are stored in results.json (same folder) so comparisons can
span sessions and are not lost when the terminal is closed.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path


SERVER_URL = "http://127.0.0.1:8008/v1/chat/completions"
RESULTS_FILE = Path(__file__).parent / "results.json"

TEST_PROMPTS = [
    "Summarize in one sentence: the user asks for help reading a config file.",
    "Classify the sentiment of this message: 'server down again, please check'",
    "Extract the filenames from this text: 'edit provider.py then run npu_server.py'",
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
    print(f"RUNNING BENCHMARK — run label: {mode_label}")
    print(f"{'=' * 60}")
    print(
        "IMPORTANT: make sure the npu_server_v3.py instance CURRENTLY RUNNING "
        f"is actually set to the GENERATION_MODE matching this '{mode_label}' "
        "label. This script CANNOT check that automatically — inspect the "
        "server's startup log manually (it prints the active GENERATION_MODE)."
    )

    run_results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] {prompt[:50]}...")
        try:
            result = call_server(prompt)
            run_results.append(result)
            print(f"  Wall: {result['wall_time_ms']}ms | tok/s: {result['approx_tok_per_sec']}")
            print(f"  Server-reported config: {result['generation_config']}")
            print(f"  Likely hit token budget: {result['likely_hit_token_budget']}")
            print(f"  Output: {result['output'][:100]}")
        except Exception as e:
            print(f"  FAILED: {e}")
            return

    all_results = load_results()
    all_results[mode_label] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requests": run_results,
        "avg_wall_time_ms": round(sum(r["wall_time_ms"] for r in run_results) / len(run_results), 2),
    }
    save_results(all_results)
    print(f"\nResults stored in {RESULTS_FILE} under the label '{mode_label}'")


def show_comparison():
    all_results = load_results()
    if not all_results:
        print("No results stored yet. Run with --mode <label> first.")
        return

    print(f"\n{'=' * 70}")
    print("COMPARISON ACROSS MODES")
    print(f"{'=' * 70}")
    print(f"{'Mode':<15} {'Avg wall (ms)':<15} {'Recorded at'}")
    print("-" * 70)
    for label, data in all_results.items():
        print(f"{label:<15} {data['avg_wall_time_ms']:<15} {data['timestamp']}")

    print(
        "\nHow to read this:\n"
        "- If sample_64 is far faster than greedy_64 -> do_sample=False\n"
        "  (the constraint v2 assumed was required) is most likely the main\n"
        "  cause of the slowness, NOT the model size or the NPU itself.\n"
        "- If greedy_32 is far faster than greedy_64 (close to a 1:2 ratio)\n"
        "  -> the model really does burn almost the entire token budget every\n"
        "  time (hypothesis (a) in your report is confirmed); permanently\n"
        "  lower max_new_tokens for short auxiliary tasks.\n"
        "- If ALL modes are equally slow (~4-5s, no significant difference)\n"
        "  -> the cause is NOT the generate parameters; go back to hypothesis\n"
        "  (b), power/thermal throttling — check the Windows Power Mode\n"
        "  (Best Performance vs Balanced vs Power Saver) during benchmarks."
    )

    # Check finish_reason across all stored runs — if this field is
    # consistently "length" or equivalent (not "stop"/"eos"), that is direct
    # evidence for hypothesis (a), no longer a guess from the tok/s ratio.
    print(f"\n{'=' * 70}")
    print("finish_reason CHECK (if the library exposes this info)")
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
            "  No finish_reason could be captured from the library in any\n"
            "  run. The diagnosis has to rely on the "
            "  'likely_hit_token_budget' proxy alone (word count vs\n"
            "  max_new_tokens), which is less precise than a real\n"
            "  finish_reason."
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--mode":
        run_mode(sys.argv[2])
    else:
        show_comparison()
