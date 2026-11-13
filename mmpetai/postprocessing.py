"""Placing predictions back on the original PET grid, saving, and lesion summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

from .config import SEGMENTATION_THRESHOLD


def restore_from_ledger(
    array: np.ndarray,
    working_affine: np.ndarray,
    ledger: Mapping[str, Tuple[int, int, int]],
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undo symmetric padding and foreground cropping on the 3 mm working grid."""
    padded_shape = np.asarray(array.shape, dtype=np.int64)
    full_shape = np.asarray(ledger["shape_before_cropping"], dtype=np.int64)
    crop_shape = np.asarray(ledger["shape_after_cropping_before_padding"], dtype=np.int64)
    start = np.asarray(ledger["foreground_start_coord"], dtype=np.int64)
    end = np.asarray(ledger["foreground_end_coord"], dtype=np.int64)

    if np.any(padded_shape < crop_shape):
        raise ValueError(f"Prediction {tuple(padded_shape)} is smaller than the crop {tuple(crop_shape)}.")
    if np.any(start < 0) or np.any(end > full_shape) or not np.array_equal(end - start, crop_shape):
        raise ValueError("Inconsistent crop ledger.")

    pad_before = (padded_shape - crop_shape) // 2
    cropped = np.asarray(array)[tuple(slice(int(a), int(a + n)) for a, n in zip(pad_before, crop_shape))]
    restored = np.full(tuple(int(v) for v in full_shape), fill_value, dtype=cropped.dtype)
    restored[tuple(slice(int(a), int(b)) for a, b in zip(start, end))] = cropped

    affine = np.asarray(working_affine, dtype=np.float64).copy()
    linear = affine[:3, :3]
    affine[:3, 3] = affine[:3, 3] + linear @ pad_before.astype(np.float64) - linear @ start.astype(np.float64)
    return restored, affine


def probability_to_reference_grid(
    probability: np.ndarray,
    working_affine: np.ndarray,
    ledger: Mapping[str, Tuple[int, int, int]],
    reference: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    """Map a working-grid probability map onto the reference (original PET) grid."""
    prob, affine = restore_from_ledger(probability.astype(np.float32), working_affine, ledger, fill_value=0.0)
    target_shape = tuple(int(v) for v in reference.shape[:3])
    if prob.shape == target_shape and np.allclose(affine, reference.affine, rtol=1e-5, atol=1e-4):
        return prob
    placed = resample_from_to(
        nib.Nifti1Image(prob, affine), (target_shape, reference.affine),
        order=1, mode="constant", cval=0.0,
    )
    return np.clip(np.asarray(placed.dataobj, dtype=np.float32), 0.0, 1.0)


def binarize(probability: np.ndarray) -> np.ndarray:
    return (probability >= SEGMENTATION_THRESHOLD).astype(np.uint8)


def save_like(array: np.ndarray, reference: nib.spatialimages.SpatialImage, path: Path, dtype) -> None:
    """Save ``array`` with the reference image's geometry."""
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(array, dtype=dtype), reference.affine, header=header)
    img.header.set_slope_inter(1.0, 0.0)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None and int(qcode) > 0:
        img.set_qform(qform, int(qcode))
    if sform is not None and int(scode) > 0:
        img.set_sform(sform, int(scode))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def summarize_lesions(mask: np.ndarray, pet_suv: Optional[np.ndarray], voxel_spacing_mm) -> Dict[str, float]:
    """Lesion count (26-connected), total volume, and SUV statistics inside the mask."""
    mask = np.asarray(mask, dtype=bool)
    voxel_ml = float(np.prod(voxel_spacing_mm)) / 1000.0
    _, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    volume_ml = float(mask.sum()) * voxel_ml
    summary = {
        "lesion_count": int(n_components),
        "total_lesion_volume_ml": round(volume_ml, 3),
        "suv_max": None,
        "suv_mean": None,
        "total_lesion_uptake": None,
    }
    if pet_suv is not None and mask.any():
        values = np.asarray(pet_suv)[mask]
        suv_mean = float(values.mean())
        summary.update(
            suv_max=round(float(values.max()), 3),
            suv_mean=round(suv_mean, 3),
            total_lesion_uptake=round(suv_mean * volume_ml, 3),
        )
    return summary
