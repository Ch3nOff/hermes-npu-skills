"""
upload_hf.py — upload the Iniz Agent Guard artifacts to the HuggingFace Hub.

What gets uploaded:
  openvino/           IR INT8 seq_len=128 (ready to run on NPU)  ~495 MB
  checkpoint/         model.safetensors 3-head + tokenizer        ~992 MB
  README.md           model card with measured numbers

The token is read from the `hf auth login` cache (never hardcode it).
"""

import os
import sys
from pathlib import Path

REPO_ID = os.environ.get("INIZ_HF_REPO", "CH3NDev/iniz-agent-guard-int8")
HOME = Path.home()
IR = HOME / "npu-provider" / "models" / "iniz-guard-int8-ov"
CKPT = HOME / "npu-provider" / "work" / "ckpt"


def main():
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as: {who['name']}")

    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"repo ready: https://huggingface.co/{REPO_ID}")

    # 1. OpenVINO INT8 IR (ready to use)
    print(f"\nuploading IR from {IR} ...", flush=True)
    api.upload_folder(
        repo_id=REPO_ID, folder_path=str(IR), path_in_repo="openvino",
        commit_message="Add OpenVINO IR INT8 (seq_len=128, NPU-ready)",
    )
    print("IR OK")

    # 2. raw checkpoint (for retraining / re-export)
    print(f"\nuploading checkpoint from {CKPT} ...", flush=True)
    api.upload_folder(
        repo_id=REPO_ID, folder_path=str(CKPT), path_in_repo="checkpoint",
        allow_patterns=["model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "chat_template.jinja"],
        commit_message="Add 3-head checkpoint (Qwen2Model + LoRA r=8)",
    )
    print("checkpoint OK")

    print(f"\nDONE: https://huggingface.co/{REPO_ID}")
    for f in api.list_repo_files(REPO_ID):
        print("  ", f)


if __name__ == "__main__":
    sys.exit(main())
