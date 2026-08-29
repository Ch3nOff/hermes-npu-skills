"""
check_summary_grounding.py — does ROUGE hide factual errors?

Motivating case found in the INT8 run (cnn_10, ROUGE-1 0.089):

  article : Roseanne Barr's rendition was booed and President Bush called it
            "disgraceful". Vince Neil's performance was separately criticised.
  output  : "Vince Neil's performance ... with the crowd booing him and President
            Bush calling it 'disgraceful.'"

Every entity is real and present in the source, so a naive "are these words in the
article?" check passes. The summary is still wrong: it re-attributes Roseanne's
booing and Bush's quote to Vince Neil. ROUGE cannot see this, and neither can
substring matching.

This script measures two things that ARE mechanically checkable, and is explicit
about the third that is not:

  1. extrinsic entities — capitalised tokens / numbers in the summary that do NOT
     appear in the article at all. These are unambiguous fabrications.
  2. entity density     — how many source entities the summary reuses, as a proxy
                          for "grounded but possibly misattributed".
  3. misattribution     — NOT measured. Requires NLI or a human. Flagged as a known
                          blind spot rather than silently scored as correct.

Run after bench_aux.py; reads its output JSON.
"""

import json
import re
import sys
from pathlib import Path

DATA = Path("aux_data")
STOP = {
    "The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "This", "That",
    "But", "And", "If", "As", "For", "To", "Of", "With", "By", "From", "Just",
    "There", "When", "While", "After", "Before", "His", "Her", "Their", "Its",
    "CNN", "Sure", "Facts", "Ask", "Can", "Anyone", "One",
}


def entities(text):
    """Capitalised multi-token names plus standalone numbers/versions."""
    caps = re.findall(r"\b[A-Z][a-zA-Z'’]+(?:\s+[A-Z][a-zA-Z'’]+)*", text)
    out = set()
    for c in caps:
        parts = [p for p in c.split() if p not in STOP]
        if parts:
            out.add(" ".join(parts))
    out |= set(re.findall(r"\b\d+(?:[.-]\d+)+\b", text))
    return {e for e in out if len(e) > 2}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "aux_bench_qwen2.5-0.5b-instruct-int8-ov_NPU.json"
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summarize" not in d:
        print(f"{path} has no summarize section")
        return 1

    items = {i["id"]: i for i in
             json.loads((DATA / "summarize.json").read_text(encoding="utf-8"))}

    total_ext = 0
    total_ents = 0
    flagged = []
    print(f"grounding check: {d['model']} @ {d['device']}\n")
    for r in d["summarize"]["rows"]:
        art = items[r["id"]]["article"]
        hyp = r["output"]
        hyp_ents = entities(hyp)
        ext = sorted(e for e in hyp_ents if e not in art)
        total_ents += len(hyp_ents)
        total_ext += len(ext)
        status = "OK" if not ext else "EXTRINSIC"
        print(f"  {r['id']}  R1={r['rouge1']:.3f}  entities={len(hyp_ents):2d}  "
              f"{status}")
        if ext:
            print(f"     not in article: {ext}")
            flagged.append((r["id"], ext))

    n = len(d["summarize"]["rows"])
    print(f"\nsummaries: {n}")
    print(f"entities mentioned: {total_ents}")
    print(f"extrinsic (not in article): {total_ext} "
          f"({total_ext/max(total_ents,1):.1%})")
    print(f"summaries with >=1 extrinsic entity: {len(flagged)}/{n}")
    print("\nNOT MEASURED: misattribution (entities that ARE in the article but are")
    print("linked to the wrong subject). cnn_10 is a confirmed example — it moves")
    print("Roseanne Barr's booing and Bush's 'disgraceful' quote onto Vince Neil.")
    print("Detecting this needs NLI or a human reader; ROUGE and substring checks")
    print("both score it as grounded.")

    out = Path(path).with_name(Path(path).stem + "_grounding.json")
    out.write_text(json.dumps({
        "model": d["model"], "device": d["device"], "summaries": n,
        "entities_total": total_ents, "extrinsic_total": total_ext,
        "extrinsic_rate": round(total_ext / max(total_ents, 1), 4),
        "summaries_with_extrinsic": len(flagged),
        "flagged": [{"id": i, "extrinsic": e} for i, e in flagged],
        "not_measured": "misattribution of in-article entities (needs NLI/human)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
