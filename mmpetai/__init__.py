"""MM-PET-AI: whole-body lesion segmentation on FDG, PSMA, and DOTATATE PET/CT."""

__version__ = "1.0.0"

from .predictor import MMPETAI, PredictionResult  # noqa: E402

__all__ = ["MMPETAI", "PredictionResult", "__version__"]
