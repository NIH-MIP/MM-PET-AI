#!/usr/bin/env python3
"""Upload an exported model folder to the Hugging Face Hub.

Log in first with `huggingface-cli login` (token with write access).

Usage:
    python tools/push_to_hub.py --folder hf_model/ --repo-id <user>/MM-PET-AI [--private] [--tag v1.0]

After the first upload, set DEFAULT_REPO_ID in mmpetai/weights.py to the repo id
so `mmpetai` downloads it without --model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REQUIRED = ("model.safetensors", "config.json", "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, type=Path, help="folder written by tools/export_weights.py")
    parser.add_argument("--repo-id", required=True, help="<user or org>/<name>")
    parser.add_argument("--private", action="store_true", help="create the repository as private")
    parser.add_argument("--tag", default=None, help="optional tag for this version, e.g. v1.0")
    parser.add_argument("--message", default="Upload MM-PET-AI model", help="commit message")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (args.folder / name).is_file()]
    if missing:
        raise SystemExit(f"{args.folder} is missing {missing}; run tools/export_weights.py first.")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id, repo_type="model", folder_path=str(args.folder),
        allow_patterns=list(REQUIRED), commit_message=args.message,
    )
    if args.tag:
        api.create_tag(args.repo_id, tag=args.tag, repo_type="model")
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    print(commit)


if __name__ == "__main__":
    main()
