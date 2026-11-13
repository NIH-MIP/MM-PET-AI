"""High-level predictor: PET/CT NIfTI in, lesion mask and scan-level outputs out."""

from __future__ import annotations

import contextlib
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from . import __version__
from .config import (
    LESION_GATE_THRESHOLD,
    PET_SUV_SANITY_P999,
    ROI_SIZE,
    SW_MODE,
    SW_OVERLAP,
    TRACERS,
)
from .network import build_model
from .postprocessing import binarize, probability_to_reference_grid, save_like, summarize_lesions
from .preprocessing import build_transforms, prepare_case
from .weights import load_model_files, resolve_model

PathLike = Union[str, Path]

TRACER_NOTE = (
    "Auxiliary output. Tracer prediction was accurate on the internal test set but did not "
    "transfer to an external cohort (DEEP-PSMA); do not rely on it as a tracer label."
)


@dataclass
class PredictionResult:
    mask: np.ndarray                      # uint8, on the original PET grid
    reference: nib.spatialimages.SpatialImage
    lesion_presence_probability: float
    tracer_probabilities: Dict[str, float]
    lesion_summary: Dict[str, Optional[float]]
    suv_factor: float = 1.0
    probability: Optional[np.ndarray] = None
    runtime_s: float = 0.0
    inputs: Dict[str, str] = field(default_factory=dict)

    @property
    def predicted_tracer(self) -> str:
        return max(self.tracer_probabilities, key=self.tracer_probabilities.get)

    def save(self, mask_path: PathLike, probability_path: Optional[PathLike] = None) -> None:
        save_like(self.mask, self.reference, Path(mask_path), np.uint8)
        if probability_path is not None:
            if self.probability is None:
                raise ValueError("Probability map was not computed; call predict(..., return_probability=True).")
            save_like(self.probability, self.reference, Path(probability_path), np.float32)

    def to_dict(self) -> dict:
        return {
            "mmpetai_version": __version__,
            "inputs": self.inputs,
            "suv_factor": self.suv_factor,
            "lesion_presence_probability": round(self.lesion_presence_probability, 4),
            "lesions": self.lesion_summary,
            "tracer": {
                "predicted": self.predicted_tracer,
                "probabilities": {k: round(v, 4) for k, v in self.tracer_probabilities.items()},
                "note": TRACER_NOTE,
            },
            "runtime_s": round(self.runtime_s, 1),
        }


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_3d(path: PathLike, role: str) -> nib.spatialimages.SpatialImage:
    img = nib.load(str(path))
    if len(img.shape) == 4 and img.shape[3] == 1:
        img = img.slicer[:, :, :, 0]
    if len(img.shape) != 3:
        raise ValueError(f"{role} must be a 3D volume, got shape {img.shape}: {path}")
    return img


def check_inputs(pet_img, ct_img, suv_factor: float) -> np.ndarray:
    """Catch swapped channels and PET that is not in SUV. Returns PET in SUV."""
    pet = np.asarray(pet_img.get_fdata(dtype=np.float32)) * float(suv_factor)
    pet_min, pet_p999 = float(pet.min()), float(np.percentile(pet, 99.9))
    ct_min = float(np.asarray(ct_img.dataobj).min())
    if pet_min < -100:
        raise ValueError(f"PET has values down to {pet_min:.0f}; it looks like CT. Check the -pet / -ct order.")
    if pet_p999 > PET_SUV_SANITY_P999:
        raise ValueError(
            f"PET 99.9th percentile is {pet_p999:.0f}, which is not SUV. If the PET is in Bq/mL, pass "
            f"--suv-factor or --pet-dicom so it can be converted to body-weight SUV."
        )
    if ct_min > -100:
        raise ValueError(
            f"CT minimum is {ct_min:.0f} HU (no air); it may be PET or not in Hounsfield units. "
            f"Check the -pet / -ct order."
        )
    return pet


class MMPETAI:
    """Load the model once and segment any number of PET/CT pairs.

    ``model`` is a Hugging Face Hub repo id (downloaded and cached on first
    use) or a local folder containing ``model.safetensors`` and ``config.json``.
    If omitted, ``$MMPETAI_MODEL`` or the default Hub repository is used.

    Example
    -------
    >>> model = MMPETAI.from_pretrained()            # or MMPETAI.from_pretrained("user/MM-PET-AI")
    >>> result = model.predict("pet.nii.gz", "ct.nii.gz")
    >>> result.save("lesions.nii.gz")
    """

    def __init__(
        self,
        model: Optional[PathLike] = None,
        revision: Optional[str] = None,
        device: str = "auto",
        sw_batch_size: int = 1,
        overlap: float = SW_OVERLAP,
        use_amp: bool = True,
        lesion_gate: bool = True,
        stitch_on_cpu: bool = False,
        verbose: bool = True,
    ):
        self.device = _select_device(device)
        self.sw_batch_size = int(sw_batch_size)
        self.overlap = float(overlap)
        self.lesion_gate = bool(lesion_gate)
        self.stitch_device = torch.device("cpu") if stitch_on_cpu else self.device
        self.verbose = verbose

        self.amp_dtype = None
        if use_amp and self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                warnings.warn("GPU lacks bfloat16 support; running in float32.")
        if self.device.type == "cpu":
            warnings.warn("Running on CPU. Expect several minutes or more per scan.")

        self.weights_path, config_path = resolve_model(str(model) if model else None, revision=revision)
        state_dict, self.config = load_model_files(self.weights_path, config_path)
        self.model = build_model()
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self._transforms = build_transforms()

    @classmethod
    def from_pretrained(cls, model: Optional[PathLike] = None, **kwargs) -> "MMPETAI":
        """Hugging Face-style constructor; same arguments as ``MMPETAI(...)``."""
        return cls(model=model, **kwargs)

    def _roi_predictor(self, x: torch.Tensor):
        seg, lesion, tracer = self.model(x)
        seg, lesion, tracer = seg.float(), lesion.float(), tracer.float()
        if self.lesion_gate:
            keep = (torch.sigmoid(lesion) >= LESION_GATE_THRESHOLD).view(-1, 1, 1, 1, 1)
            seg = seg.masked_fill(~keep, -100.0)
        d, h, w = seg.shape[2:]
        lesion_map = lesion[:, :, None, None, None].repeat(1, 1, d, h, w)
        tracer_map = tracer[:, :, None, None, None].repeat(1, 1, d, h, w)
        return seg, lesion_map, tracer_map

    @torch.inference_mode()
    def predict(
        self,
        pet: PathLike,
        ct: PathLike,
        suv_factor: float = 1.0,
        return_probability: bool = False,
        input_checks: bool = True,
    ) -> PredictionResult:
        t0 = time.time()
        pet_img = _load_3d(pet, "PET")
        ct_img = _load_3d(ct, "CT")
        pet_suv = (
            check_inputs(pet_img, ct_img, suv_factor) if input_checks
            else np.asarray(pet_img.get_fdata(dtype=np.float32)) * float(suv_factor)
        )

        case = prepare_case(pet, ct, suv_factor=suv_factor, transforms=self._transforms)
        image = case.image.to(self.device)

        amp = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None else contextlib.nullcontext()
        )
        with amp:
            seg, lesion_map, tracer_map = sliding_window_inference(
                inputs=image,
                roi_size=ROI_SIZE,
                sw_batch_size=self.sw_batch_size,
                predictor=self._roi_predictor,
                overlap=self.overlap,
                mode=SW_MODE,
                sw_device=self.device,
                device=self.stitch_device,
                progress=self.verbose,
            )

        lesion_p = float(torch.sigmoid(lesion_map.float().mean(dim=(2, 3, 4))).reshape(-1)[0])
        tracer_p = torch.softmax(tracer_map.float().mean(dim=(2, 3, 4)), dim=1)[0].cpu().tolist()
        working_prob = torch.sigmoid(seg.float())[0, 0].cpu().numpy()
        del image, seg, lesion_map, tracer_map

        prob = probability_to_reference_grid(working_prob, case.working_affine, case.ledger, pet_img)
        mask = binarize(prob)
        summary = summarize_lesions(mask, pet_suv, pet_img.header.get_zooms()[:3])

        return PredictionResult(
            mask=mask,
            reference=pet_img,
            lesion_presence_probability=lesion_p,
            tracer_probabilities=dict(zip(TRACERS, tracer_p)),
            lesion_summary=summary,
            suv_factor=float(suv_factor),
            probability=prob if return_probability else None,
            runtime_s=time.time() - t0,
            inputs={"pet": str(pet), "ct": str(ct)},
        )
