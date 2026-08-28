"""
guard_labels.py — THE single source of truth for Iniz Agent Guard label mapping.

This mapping is not guesswork: it was validated head-to-head against the trained
model via work/verify_incoming_mapping.py. The model is the judge.

Validation results (INT8 IR on NPU, this mapping vs the old notebook mapping):

              validation                test
  accuracy    0.9586  (vs 0.7301)       0.9660  (vs 0.7410)
  macro-F1    0.5727  (vs 0.4191)       0.8176  (vs 0.4264)
  MAE shell   0.0479  (vs 0.0877)       0.0465  (vs 0.0963)

The +0.23 accuracy gain and +0.39 macro-F1 gain (test) prove this is the mapping
that was used when training checkpoint-2634.

NOTE on two categories that DO NOT EXIST in the dataset (kept so the mapping
stays identical to the one used during training — do not remove them):
  - OBFUSCATION: 'context_overflow' does not exist in the dataset
  - EXTRACTION : 'prompt_leaking'  does not exist (the dataset has 'prompt_leak')
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

# shell_risk is a keyword PROXY, not a ground-truth label — the dataset has no
# real shell-injection examples. Values: 0.0 benign / 0.6 keyword hit / 0.1 other.
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
    """category/severity/label -> (inj, shell, action). Validated mapping."""
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
