"""
fetch_aux_data.py — real test data for the auxiliary-model claim.

Three tasks, three sources, all with ground truth so nothing has to be judged by eye:

  1. summarization  — CNN/DailyMail 3.0.0 test split (article + human highlights)
  2. sentiment      — SST-2 validation split (sentence + 0/1 label)
  3. extraction     — synthetic-but-verifiable: filenames/versions/paths embedded in
                      free text. The target string is known exactly, so a hallucinated
                      answer is detectable rather than a matter of opinion.

Task 3 is the one the earlier session reported failing (INT4 hallucinated, INT8
refused). It is included specifically to re-test that claim rather than inherit it.

Reads parquet directly with pyarrow — `datasets` is not needed and its audio/text
decoders add dependencies.

Output: aux_data/{summarize,sentiment,extract}.json
"""

import json
import sys
import urllib.request
from pathlib import Path

OUT = Path("aux_data")
CNN_URL = ("https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/"
           "3.0.0/test-00000-of-00001.parquet")
SST2_URL = ("https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/"
            "data/validation-00000-of-00001.parquet")

N_SUM = 12
N_SENT = 40
MAX_ARTICLE_CHARS = 3500   # keep prompts inside a 0.5B model's practical window


def fetch(url, dest):
    if not dest.exists():
        print(f"downloading {dest.name} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)


def build_summarize():
    import pyarrow.parquet as pq

    p = OUT / "_cnn_test.parquet"
    fetch(CNN_URL, p)
    pf = pq.ParquetFile(p)
    print(f"  row groups: {pf.num_row_groups}, rows: {pf.metadata.num_rows}")
    rows = pf.read_row_group(0, columns=["article", "highlights"]).to_pylist()

    out = []
    for r in rows:
        if len(out) >= N_SUM:
            break
        art = (r["article"] or "").strip()
        ref = (r["highlights"] or "").strip()
        # skip the very long ones; a 0.5B model cannot use 10k chars anyway
        if not (600 <= len(art) <= MAX_ARTICLE_CHARS) or len(ref) < 60:
            continue
        out.append({"id": f"cnn_{len(out):02d}", "article": art, "reference": ref,
                    "article_chars": len(art), "reference_chars": len(ref)})
    (OUT / "summarize.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summarize: {len(out)} items, "
          f"avg article {sum(o['article_chars'] for o in out)//len(out)} chars")
    return out


def build_sentiment():
    import pyarrow.parquet as pq

    p = OUT / "_sst2_val.parquet"
    fetch(SST2_URL, p)
    pf = pq.ParquetFile(p)
    rows = pf.read_row_group(0).to_pylist()
    print(f"  sst2 columns: {pf.schema_arrow.names}, rows: {len(rows)}")

    out = []
    pos = neg = 0
    for r in rows:
        if len(out) >= N_SENT:
            break
        s = (r.get("sentence") or "").strip()
        lab = r.get("label")
        if not s or lab not in (0, 1) or len(s) < 20:
            continue
        # keep it balanced so accuracy is not inflated by a skewed split
        if lab == 1 and pos >= N_SENT // 2:
            continue
        if lab == 0 and neg >= N_SENT // 2:
            continue
        pos += lab == 1
        neg += lab == 0
        out.append({"id": f"sst2_{len(out):02d}", "text": s,
                    "label": "positive" if lab == 1 else "negative"})
    (OUT / "sentiment.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"sentiment: {len(out)} items ({pos} positive / {neg} negative)")
    return out


def build_extract():
    """Precision extraction: the answer is a known exact string."""
    items = [
        ("The build failed after I renamed the config. Traceback points at "
         "line 42 of data_loader_v2.py, though the stack is noisy.",
         "data_loader_v2.py", "filename"),
        ("Deploy notes: rolled back to release 4.11.2 because the canary showed "
         "elevated p99 latency. 4.12.0 stays parked.",
         "4.11.2", "version"),
        ("Please pull the archive from /var/backups/nightly/2026-03-14.tar.gz "
         "before the retention job clears it.",
         "/var/backups/nightly/2026-03-14.tar.gz", "path"),
        ("Ticket assigned to maria.gonzalez@example.org after the on-call "
         "handoff; cc the platform list if it escalates.",
         "maria.gonzalez@example.org", "email"),
        ("The migration script lives in scripts/migrate_users_0042.sql and must "
         "run before the API restarts.",
         "scripts/migrate_users_0042.sql", "path"),
        ("Our stripe webhook is failing with signature errors; the endpoint "
         "secret starts whsec_ and the event id was evt_1Nx8Kd2eZvKYlo2C.",
         "evt_1Nx8Kd2eZvKYlo2C", "identifier"),
        ("I pinned torch==2.9.1 to dodge the export regression, but numpy "
         "stayed on 2.3.4 for now.",
         "2.9.1", "version"),
        ("Log rotation writes to app-2026-03-14.log.gz nightly; the previous "
         "day's file is deleted after upload.",
         "app-2026-03-14.log.gz", "filename"),
        ("Commit 9f4c2ab broke the CI cache; 3e81d07 is the last green one.",
         "9f4c2ab", "identifier"),
        ("Set the bucket to s3://prod-media-eu-west-1/thumbnails/ and leave the "
         "staging path alone.",
         "s3://prod-media-eu-west-1/thumbnails/", "path"),
    ]
    prompts = {
        "filename": "What is the filename mentioned in the text?",
        "version": "What is the version number mentioned in the text?",
        "path": "What is the file path or URL mentioned in the text?",
        "email": "What is the email address mentioned in the text?",
        "identifier": "What is the identifier mentioned in the text?",
    }
    out = [{"id": f"ext_{i:02d}", "text": t, "answer": a, "kind": k,
            "question": prompts[k]}
           for i, (t, a, k) in enumerate(items)]
    (OUT / "extract.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"extract: {len(out)} items across {len(set(k for _,_,k in items))} kinds")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    print("=== summarization (CNN/DailyMail) ===")
    build_summarize()
    print("\n=== sentiment (SST-2) ===")
    build_sentiment()
    print("\n=== precision extraction (known exact answers) ===")
    build_extract()
    print(f"\nwrote {OUT}/summarize.json, sentiment.json, extract.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
