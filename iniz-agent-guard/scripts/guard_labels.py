"""
guard_labels.py — SATU sumber kebenaran untuk label mapping Iniz Agent Guard.

Mapping ini BUKAN karangan: sudah divalidasi head-to-head terhadap model hasil
training lewat work/verify_incoming_mapping.py. Model adalah hakimnya.

Hasil validasi (IR INT8 di NPU, mapping ini vs mapping notebook lama):

              validation                test
  accuracy    0.9586  (vs 0.7301)       0.9660  (vs 0.7410)
  macro-F1    0.5727  (vs 0.4191)       0.8176  (vs 0.4264)
  MAE shell   0.0479  (vs 0.0877)       0.0465  (vs 0.0963)

Kenaikan accuracy +0.23 dan macro-F1 +0.39 (test) membuktikan mapping inilah
yang dipakai saat training checkpoint-2634.

CATATAN dua kategori yang TIDAK ADA di dataset (dibiarkan agar mapping tetap
identik dengan yang dipakai saat training, jangan dihapus):
  - OBFUSCATION: 'context_overflow' tidak eksis di dataset
  - EXTRACTION : 'prompt_leaking'  tidak eksis (yang ada 'prompt_leak')
"""

import pandas as pd

ACTIONS = ["PASS", "PAUSE_AGENTS", "ISOLATE_FILE", "USER_CONFIRMATION"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}

SEVERITY_MAP = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}

OVERRIDE_CATEGORIES = {"direct_injection", "jailbreak", "persona_replacement",
                       "many_shot", "crescendo"}
OBFUSCATION_CATEGORIES = {"encoding_obfuscation", "token_smuggling",
                          "indirect_injection", "context_overflow"}
EXTRACTION_CATEGORIES = {"system_extraction", "prompt_leaking"}

# shell_risk adalah PROXY keyword, bukan label ground-truth — dataset tidak punya
# contoh shell-injection sungguhan. Nilai: 0.0 benign / 0.6 keyword hit / 0.1 lain.
SHELL_RISK_KEYWORDS = [
    "eval", "exec", "atob", "bash", "shell", "subprocess",
    "os.system", "js_eval", "code_execution", "command", "__import__",
]


def compute_shell_risk_proxy(row) -> float:
    if int(row["label"]) == 0:
        return 0.0
    raw_tags = row.get("tags")
    tag_list = [] if raw_tags is None else [str(t) for t in list(raw_tags)]
    haystack = (str(row["text"]) + " " + " ".join(tag_list)).lower()
    return 0.6 if any(kw in haystack for kw in SHELL_RISK_KEYWORDS) else 0.1


def map_labels(row) -> pd.Series:
    """category/severity/label -> (inj, shell, action). Mapping tervalidasi."""
    category = row["category"]
    severity = row["severity"]
    label = int(row["label"])
    injection = 0.0
    action = "PASS"
    if label == 1:
        injection = SEVERITY_MAP.get(severity, 0.5)
        if category in OVERRIDE_CATEGORIES:
            action = "PAUSE_AGENTS"
        elif category in OBFUSCATION_CATEGORIES:
            action = "ISOLATE_FILE"
        elif category in EXTRACTION_CATEGORIES:
            action = "USER_CONFIRMATION"
        else:
            action = "PAUSE_AGENTS"
    return pd.Series({"inj": injection, "shell": compute_shell_risk_proxy(row),
                      "action": action})


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["inj", "shell", "action"]] = df.apply(map_labels, axis=1)
    return df
