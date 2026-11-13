"""Deterministic preprocessing, identical to the evaluation pipeline used in the paper.

PET and CT are placed on the PET grid, reoriented to RAS, resampled to 3 mm,
cropped to the CT body foreground, intensity-normalized, and padded to at
least one 192^3 window. The crop/pad geometry is recorded so predictions can
be placed back onto the original PET grid exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    ResampleToMatchd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from .config import CT_WINDOW_HU, PET_SUV_MAX, ROI_SIZE, TARGET_SPACING

_KEYS = ["pet", "ct"]


class _ScaleByFactord(MapTransform):
    """Multiply images by the per-case factor in ``data[factor_key]`` (SUV conversion)."""

    def __init__(self, keys, factor_key: str = "suv_factor"):
        super().__init__(keys)
        self.factor_key = factor_key

    def __call__(self, data):
        d = dict(data)
        factor = float(d.get(self.factor_key, 1.0))
        if factor != 1.0:
            for key in self.key_iterator(d):
                d[key] = d[key] * factor
        return d


class _RecordShaped(MapTransform):
    """Store the current spatial shape of ``key`` under ``output_key``."""

    def __init__(self, key: str, output_key: str):
        super().__init__([key])
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        d[self.output_key] = tuple(int(v) for v in d[self.keys[0]].shape[-3:])
        return d


def build_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=_KEYS, ensure_channel_first=True, image_only=True, reader="NibabelReader"),
        ResampleToMatchd(keys=["ct"], key_dst="pet", mode="bilinear", padding_mode="border"),
        _ScaleByFactord(keys=["pet"]),
        Orientationd(keys=_KEYS, axcodes="RAS"),
        Spacingd(keys=_KEYS, pixdim=TARGET_SPACING, mode=("bilinear", "bilinear")),
        _RecordShaped("pet", "shape_before_cropping"),
        ScaleIntensityRanged(
            keys=["ct"], a_min=CT_WINDOW_HU[0], a_max=CT_WINDOW_HU[1], b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=_KEYS, source_key="ct", allow_smaller=True),
        ScaleIntensityRanged(keys=["pet"], a_min=0.0, a_max=PET_SUV_MAX, b_min=0.0, b_max=1.0, clip=True),
        _RecordShaped("pet", "shape_after_cropping_before_padding"),
        SpatialPadd(keys=_KEYS, spatial_size=ROI_SIZE, method="symmetric", mode="constant", constant_values=0),
        EnsureTyped(keys=_KEYS, dtype=torch.float32, track_meta=True),
    ])


@dataclass
class PreparedCase:
    image: torch.Tensor          # (1, 2, D, H, W) float32, channel 0 = PET, 1 = CT
    working_affine: np.ndarray   # 4x4 affine of the padded working grid
    ledger: Dict[str, Tuple[int, int, int]]


def _as_int_triplet(value) -> Tuple[int, int, int]:
    arr = np.asarray(value).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected three spatial values, got {arr.tolist()}")
    return tuple(int(v) for v in arr)


def prepare_case(pet_path: str, ct_path: str, suv_factor: float = 1.0, transforms: Compose = None) -> PreparedCase:
    transforms = transforms or build_transforms()
    d = transforms({"pet": str(pet_path), "ct": str(ct_path), "suv_factor": float(suv_factor)})

    if "foreground_start_coord" not in d or "foreground_end_coord" not in d:
        raise RuntimeError("Foreground crop coordinates were not recorded; check the installed MONAI version.")

    pet, ct = d["pet"], d["ct"]
    affine = pet.affine
    affine = affine.detach().cpu().numpy() if torch.is_tensor(affine) else np.asarray(affine)
    image = torch.cat([pet, ct], dim=0).unsqueeze(0)
    if hasattr(image, "as_tensor"):
        image = image.as_tensor()

    ledger = {
        "shape_before_cropping": _as_int_triplet(d["shape_before_cropping"]),
        "shape_after_cropping_before_padding": _as_int_triplet(d["shape_after_cropping_before_padding"]),
        "foreground_start_coord": _as_int_triplet(d["foreground_start_coord"]),
        "foreground_end_coord": _as_int_triplet(d["foreground_end_coord"]),
    }
    if min(ledger["shape_after_cropping_before_padding"]) <= 0:
        raise ValueError("CT foreground crop is empty; is the CT in Hounsfield units?")
    return PreparedCase(image=image.contiguous(), working_affine=np.asarray(affine, dtype=np.float64), ledger=ledger)
