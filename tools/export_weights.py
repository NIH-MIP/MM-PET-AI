#!/usr/bin/env python3
"""Convert an MM-PET-AI training checkpoint into a Hugging Face-ready model folder.

The folder contains model.safetensors, config.json, and README.md (model card).
Training-only parameters and DDP / torch.compile prefixes are removed, and the
result is verified to load into the inference network with strict=True.

Usage:
    python tools/export_weights.py \
        --checkpoint /path/to/fold0_model_best.pth \
        --out hf_model/

Then test locally and upload:
    mmpetai -pet pet.nii.gz -ct ct.nii.gz -o test.nii.gz --model hf_model/
    python tools/push_to_hub.py --folder hf_model/ --repo-id <user>/MM-PET-AI
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch

from mmpetai import __version__
from mmpetai.network import build_model
from mmpetai.weights import WEIGHTS_FILENAME, clean_state_dict, extract_state_dict, save_model_folder

MODEL_CARD_TEMPLATE = Path(__file__).with_name("MODEL_CARD.md")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="training checkpoint (.pth)")
    parser.add_argument("--out", required=True, type=Path, help="output model folder")
    parser.add_argument("--note", default="", help="free-text note stored in config.json")
    args = parser.parse_args()

    # Your own training checkpoint is trusted; it may contain non-tensor objects.
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state_dict, dropped = clean_state_dict(extract_state_dict(checkpoint))

    model = build_model()
    model.load_state_dict(state_dict, strict=True)

    metadata = {"source_checkpoint": args.checkpoint.name, "exported_with": f"mmpetai {__version__}"}
    if args.note:
        metadata["note"] = args.note
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("epoch"), int):
        metadata["epoch"] = checkpoint["epoch"]

    save_model_folder(model.state_dict(), args.out, metadata)
    if MODEL_CARD_TEMPLATE.is_file():
        shutil.copyfile(MODEL_CARD_TEMPLATE, args.out / "README.md")

    weights = args.out / WEIGHTS_FILENAME
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dropped {len(dropped)} training-only tensors.")
    print(f"Saved {args.out} ({n_params / 1e6:.1f} M parameters, {weights.stat().st_size / 1e6:.0f} MB)")
    print(f"{WEIGHTS_FILENAME} sha256: {_sha256(weights)}")


if __name__ == "__main__":
    main()
