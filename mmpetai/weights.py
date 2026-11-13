"""Locating, downloading, loading, and exporting MM-PET-AI weights.

A model is a folder (local or on the Hugging Face Hub) containing:

    model.safetensors   network weights
    config.json         architecture and processing settings (for validation)
    README.md           model card (Hub only)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from .config import (
    CT_WINDOW_HU,
    LESION_GATE_THRESHOLD,
    MODEL_KWARGS,
    PET_SUV_MAX,
    ROI_SIZE,
    SEGMENTATION_THRESHOLD,
    SW_MODE,
    SW_OVERLAP,
    TARGET_SPACING,
    TRACERS,
)

# Hugging Face Hub repository of the released model.
DEFAULT_REPO_ID = "zhanghaoyue/MM-PET-AI"
MODEL_ENV_VAR = "MMPETAI_MODEL"

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
MODEL_TYPE = "mmpetai"
FORMAT_VERSION = 1

# Keys present in training checkpoints that the inference network does not use.
TRAINING_ONLY_PREFIXES = (
    "ds1.",
    "ds2.",
    "organ_head.",
    "organ_ds1_head.",
    "organ_ds2_head.",
    "pet_local_aux_gate.",
    "pet_local_aux_head.",
    # Per-encoder UNETR skip blocks; their outputs were never consumed (the
    # model rebuilds skips from the cross-attended tokens via *_reenc*).
    "encoder_pet.encoder2.",
    "encoder_pet.encoder3.",
    "encoder_pet.encoder4.",
    "encoder_ct.encoder2.",
    "encoder_ct.encoder3.",
    "encoder_ct.encoder4.",
)

# Checkpoints trained with the gated PET-local refinement path use that path at
# inference, so they are not compatible with this network.
UNSUPPORTED_PREFIXES = ("pet_local_refine_proj.",)

# Some MONAI versions create cross-attention parameters inside every ViT
# TransformerBlock even though the ViT never enables cross-attention, so they
# are never used in the forward pass. Whether they exist depends on the
# installed MONAI version; they are ignored on both export and load.
_UNUSED_VIT_PARAM = re.compile(r"^encoder_(pet|ct)\.vit\.blocks\.\d+\.(norm_)?cross_attn\.")

_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


# ---------------------------------------------------------------------------
# State-dict cleaning (shared by loading and export)
# ---------------------------------------------------------------------------

def _strip_wrappers(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_state_dict(obj) -> Mapping[str, torch.Tensor]:
    """Return the tensor state dict from a training checkpoint or plain state dict."""
    if not isinstance(obj, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(obj).__name__}")
    for key in ("state_dict", "model_state_dict"):
        if key in obj and isinstance(obj[key], Mapping):
            return obj[key]
    return obj


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Strip DDP/compile prefixes and drop training-only parameters."""
    cleaned: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []
    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = _strip_wrappers(raw_key)
        if key.startswith(UNSUPPORTED_PREFIXES):
            raise ValueError(
                "This checkpoint uses the PET-local refinement path, which this release does not support."
            )
        if key.startswith(TRAINING_ONLY_PREFIXES) or _UNUSED_VIT_PARAM.match(key):
            dropped.append(key)
            continue
        cleaned[key] = value
    return cleaned, dropped


def inference_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """The model's state dict without parameters that are never used at inference."""
    return {k: v for k, v in model.state_dict().items() if not _UNUSED_VIT_PARAM.match(k)}


def load_into_model(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Strict load, except for the unused MONAI ViT cross-attention parameters."""
    usable = {k: v for k, v in state_dict.items() if not _UNUSED_VIT_PARAM.match(k)}
    missing, unexpected = model.load_state_dict(usable, strict=False)
    missing = [k for k in missing if not _UNUSED_VIT_PARAM.match(k)]
    if missing or unexpected:
        raise RuntimeError(
            f"Weights do not match the MM-PET-AI network. Missing: {missing[:10]} "
            f"({len(missing)} total); unexpected: {list(unexpected)[:10]} ({len(unexpected)} total)."
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _jsonable(value):
    return list(value) if isinstance(value, tuple) else value


def build_config(metadata: Optional[dict] = None) -> dict:
    from . import __version__

    return {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "mmpetai_version": __version__,
        "architecture": {k: _jsonable(v) for k, v in MODEL_KWARGS.items()},
        "preprocessing": {
            "target_spacing_mm": list(TARGET_SPACING),
            "ct_window_hu": list(CT_WINDOW_HU),
            "pet_units": "SUVbw",
            "pet_suv_max": PET_SUV_MAX,
        },
        "inference": {
            "roi_size": list(ROI_SIZE),
            "sw_overlap": SW_OVERLAP,
            "sw_mode": SW_MODE,
            "lesion_gate_threshold": LESION_GATE_THRESHOLD,
            "segmentation_threshold": SEGMENTATION_THRESHOLD,
        },
        "tracers": list(TRACERS),
        "metadata": dict(metadata or {}),
    }


def validate_config(config: Mapping) -> None:
    """Refuse weights whose architecture differs from this package's network."""
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"config.json is not an MM-PET-AI config (model_type={config.get('model_type')!r}).")
    if int(config.get("format_version", 0)) > FORMAT_VERSION:
        raise ValueError("These weights need a newer mmpetai version: pip install -U mmpetai")
    expected = {k: _jsonable(v) for k, v in MODEL_KWARGS.items()}
    found = config.get("architecture", {})
    mismatched = sorted(k for k in expected if found.get(k) != expected[k])
    if mismatched:
        raise ValueError(f"Weights were built for a different architecture (mismatched: {mismatched}).")


# ---------------------------------------------------------------------------
# Resolving and loading
# ---------------------------------------------------------------------------

def resolve_model(model: Optional[str] = None, revision: Optional[str] = None) -> Tuple[Path, Optional[Path]]:
    """Return local paths to (weights, config) for a Hub repo id or a local folder/file.

    Resolution order: ``model`` argument, ``$MMPETAI_MODEL``, ``DEFAULT_REPO_ID``.
    Hub downloads are cached by ``huggingface_hub`` and reused offline.
    """
    spec = str(model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_REPO_ID)
    path = Path(spec).expanduser()

    if path.is_dir():
        weights = path / WEIGHTS_FILENAME
        if not weights.is_file():
            raise FileNotFoundError(f"{path} does not contain {WEIGHTS_FILENAME}.")
        config = path / CONFIG_FILENAME
        return weights, (config if config.is_file() else None)
    if path.is_file():
        config = path.parent / CONFIG_FILENAME
        return path, (config if config.is_file() else None)

    if spec.startswith("<org>"):
        raise RuntimeError(
            "No default Hub repository is configured in this build. Pass --model <user>/<repo> "
            "or a local model folder."
        )
    if spec.count("/") != 1:
        raise FileNotFoundError(f"{spec!r} is neither an existing path nor a Hub repo id like 'user/repo'.")

    from huggingface_hub import hf_hub_download

    try:
        weights = Path(hf_hub_download(spec, WEIGHTS_FILENAME, revision=revision))
        config = Path(hf_hub_download(spec, CONFIG_FILENAME, revision=revision))
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch {spec!r} from the Hugging Face Hub ({type(exc).__name__}: {exc}). "
            f"If the repository is gated, accept its terms on the Hub and run `huggingface-cli login`. "
            f"Offline, pass a local model folder with --model."
        ) from exc
    return weights, config


def load_model_files(weights_path: Path, config_path: Optional[Path]) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Load and clean the state dict; validate config.json when present."""
    config: dict = {}
    if config_path is not None:
        config = json.loads(Path(config_path).read_text())
        validate_config(config)

    if Path(weights_path).suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
    else:
        try:
            obj = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {weights_path} safely. Convert training checkpoints with "
                f"tools/export_weights.py first."
            ) from exc
        state_dict = extract_state_dict(obj)

    cleaned, _ = clean_state_dict(state_dict)
    return cleaned, config


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_model_folder(state_dict: Mapping[str, torch.Tensor], folder: Path, metadata: Optional[dict] = None) -> Path:
    """Write model.safetensors and config.json into ``folder``."""
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(tensors, str(folder / WEIGHTS_FILENAME), metadata={"format": "pt", "model_type": MODEL_TYPE})
    (folder / CONFIG_FILENAME).write_text(json.dumps(build_config(metadata), indent=2) + "\n")
    return folder
