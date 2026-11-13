---
license: other            # TODO: set after institutional review
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - pet-ct
  - lesion-segmentation
  - fdg
  - psma
  - dotatate
  - 3d
---

# MM-PET-AI

One model for whole-body lesion segmentation on **¹⁸F-FDG**, **¹⁸F/⁶⁸Ga-PSMA**, and **⁶⁸Ga-DOTATATE** PET/CT.

The input is one PET/CT pair (NIfTI, PET in SUV, CT in HU). The output is a binary lesion mask on the PET grid, a scan-level lesion-presence probability, lesion statistics, and an auxiliary tracer prediction. No tracer label or organ mask is needed.

> **Research use only.** This model is not a medical device. Review every segmentation before use.

## Usage

Install the inference package. The weights in this repository download automatically the first time you run it and are cached afterwards.

```bash
pip install git+https://github.com/NIH-MIP/MM-PET-AI.git
```

Command line:

```bash
mmpetai -pet pet.nii.gz -ct ct.nii.gz -o lesions.nii.gz
mmpetai -i input_dir -o output_dir      # <case>_0000.nii.gz = PET, <case>_0001.nii.gz = CT
```

Python:

```python
from mmpetai import MMPETAI

model = MMPETAI.from_pretrained("zhanghaoyue/MM-PET-AI")
result = model.predict("pet.nii.gz", "ct.nii.gz")
result.save("lesions.nii.gz")
print(result.lesion_summary)
```

## Model

The model has asymmetric PET (ViT-Base) and CT (ViT-Small) encoders that exchange context bidirectionally at four depths. A PET-local convolutional branch keeps fine detail, and a shared UNETR decoder produces the segmentation. Scan-level heads predict lesion presence and tracer.

Training also used CT-derived organ pseudo-labels and component-aware small-lesion supervision. Only the PET/CT images are needed at inference.

Preprocessing: RAS orientation, 3 mm isotropic resampling, CT-foreground crop, CT window [-175, 250] HU, and PET window SUV [0, 15]. Inference: 192³ sliding windows, 33% overlap, Gaussian blending, lesion-presence gate 0.1, threshold 0.5.

## Training data

The model was trained on 2,814 public and institutional PET/CT scans across 14 protocols (1,832 training, 450 validation, and 532 test scans). The public sources are AutoPET FDG and PSMA and a multi-institution head and neck cancer FDG cohort. The institutional cohorts cover multiple myeloma, pediatric and young-adult solid tumors, metastatic prostate cancer, pheochromocytoma/paraganglioma, neuroendocrine tumors, olfactory neuroblastoma, and recurrent and high-risk prostate cancer, plus lesion-negative FDG scans. See the paper for details.

## Evaluation

See the paper for internal test, per-tracer, per-protocol, and external (DEEP-PSMA, QIN-Breast) results.

## Limitations

- **Tracer prediction is auxiliary.** It did not transfer to an external cohort (DEEP-PSMA), so do not use it as a tracer label.
- **DOTATATE evaluation was limited** (28 internal test scans, no external cohort).
- **Performance is lower in some settings:** small, low-uptake lesions and some prostate cancer, myeloma, and pediatric cohorts.
- **Annotation styles differ between source cohorts.**

## Citation

```bibtex
@article{zhang_mmpetai,
  title   = {MM-PET-AI: Unified Whole-Body Lesion Segmentation across FDG, DOTATATE, and PSMA PET/CT},
  author  = {Zhang, Haoyue and Vazquez-Quiroga, Alesia and Haque, Fahmida and Turkbey, Baris and Harmon, Stephanie},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {to appear}
}
```
