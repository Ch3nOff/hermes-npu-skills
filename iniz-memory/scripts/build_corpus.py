"""
build_corpus.py — chunk repo docs into a retrieval corpus.

Splits every .md in the repo by ## headings. Each chunk keeps its file + heading
path so a query can name exactly which chunk(s) answer it. Code blocks are kept
(they carry real content: commands, numbers, transcripts); inline markdown markup
is stripped for cleaner embeddings.

Output: corpus/chunks.json  [{id, file, heading, text, chars}]
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(r"C:\Users\Matthew Chen\Documents\hermes-skills")
OUT = Path("corpus")
MIN_CHARS = 200       # skip stubs / TOC entries
MAX_CHARS = 1500      # split very long sections by paragraph


def clean(text):
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)          # images
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)          # links keep text
    text = re.sub(r"[*_`#>|-]{1,3}", "", text)                      # markup
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long(heading, body):
    if len(body) <= MAX_CHARS:
        return [(heading, body)]
    paras, cur, out, n = body.split("\n\n"), [], [], 1
    for p in paras:
        cur.append(p)
        if sum(map(len, cur)) >= MAX_CHARS - 200:
            out.append((f"{heading} (part {n})", "\n\n".join(cur)))
            cur, n = [], n + 1
    if cur:
        out.append((f"{heading} (part {n})", "\n\n".join(cur)))
    return out


def main():
    files = sorted(REPO.glob("*.md")) + sorted(REPO.glob("*/SKILL.md")) \
        + sorted(REPO.glob("*/references/*.md"))
    chunks, n = [], 0
    for f in files:
        if ".git" in f.parts:
            continue
        rel = f.relative_to(REPO).as_posix()
        text = f.read_text(encoding="utf-8")
        parts = re.split(r"(?m)^(#{1,3}\s+.+)$", text)
        # parts[0] = preamble, then (heading, body) pairs
        pre = clean(parts[0])
        if len(pre) >= MIN_CHARS:
            chunks.append({"id": f"c{n:03d}", "file": rel,
                           "heading": "(preamble)",
                           "text": f"{rel} :: {pre}"})
            n += 1
        for i in range(1, len(parts) - 1, 2):
            heading = clean(parts[i])
            body = clean(parts[i + 1])
            if len(body) < MIN_CHARS:
                continue
            for h2, b2 in split_long(heading, body):
                chunks.append({"id": f"c{n:03d}", "file": rel,
                               "heading": h2, "text": f"{h2}\n{b2}"})
                n += 1

    OUT.mkdir(exist_ok=True)
    (OUT / "chunks.json").write_text(
        json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"files: {len(files)}, chunks: {len(chunks)}")
    per_file = {}
    for c in chunks:
        per_file[c["file"]] = per_file.get(c["file"], 0) + 1
    for f, k in sorted(per_file.items()):
        print(f"  {k:3d}  {f}")
    avg = sum(len(c["text"]) for c in chunks) // max(len(chunks), 1)
    print(f"avg chunk: {avg} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
