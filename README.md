# MM-PET-AI

Whole-body lesion segmentation on **FDG**, **PSMA**, and **DOTATATE** PET/CT with one model.

MM-PET-AI takes one PET/CT pair and returns a lesion mask on the PET grid. It does not need the tracer name, organ masks, or any manual input. It also reports a scan-level lesion-presence probability, basic lesion statistics (count, volume, SUVmax, SUVmean, total lesion uptake), and an auxiliary tracer prediction.

Model weights: **[huggingface.co/zhanghaoyue/MM-PET-AI](https://huggingface.co/zhanghaoyue/MM-PET-AI)**

> **Research use only.** MM-PET-AI is not a medical device and has not been validated for clinical decision-making. Review every segmentation before you use it.

## Installation

```bash
pip install git+https://github.com/NIH-MIP/MM-PET-AI.git
```

Install a [PyTorch build](https://pytorch.org/get-started/locally/) that matches your CUDA version first. Python 3.9 or newer is required. To read SUV factors from PET DICOM headers, also install the `dicom` extra: `pip install "mmpetai[dicom] @ git+https://github.com/NIH-MIP/MM-PET-AI.git"`.

You don't need to download the weights yourself. The first run fetches them from the Hugging Face Hub and caches them in `~/.cache/huggingface`. Later runs work offline.

## Command line

One scan:

```bash
mmpetai -pet pet.nii.gz -ct ct.nii.gz -o lesions.nii.gz
```

| Output | Content |
|---|---|
| `lesions.nii.gz` | Binary lesion mask (uint8) on the original PET grid |
| `lesions.json` | Lesion-presence probability, lesion count, volume, SUV statistics, tracer prediction |
| `lesions_prob.nii.gz` | Lesion probability map (only with `--save-probabilities`) |

A folder of scans, named `<case>_0000.nii.gz` (PET) and `<case>_0001.nii.gz` (CT):

```bash
mmpetai -i input_dir -o output_dir
```

This writes `<case>.nii.gz` and `<case>.json` for each case, plus `summary.csv`.

Useful options (see `mmpetai --help` for all of them):

| Option | Purpose |
|---|---|
| `--suv-factor F` / `--pet-dicom DIR` | Convert PET stored in Bq/mL to SUV |
| `-m, --model` | Use another Hub repo or a local model folder |
| `--revision v1.0` | Pin a specific model version for reproducibility |
| `-d cuda:1` | Choose the device |
| `--stitch-on-cpu` | Assemble very large volumes in system memory |

## Python

```python
from mmpetai import MMPETAI

model = MMPETAI.from_pretrained()                  # released model from the Hub
# model = MMPETAI.from_pretrained("path/to/local_model_folder")

result = model.predict("pet.nii.gz", "ct.nii.gz")  # reuse `model` for many scans
result.save("lesions.nii.gz")

print(result.lesion_summary)                       # count, volume, SUVmax, SUVmean, TLU
print(result.lesion_presence_probability)
mask = result.mask                                 # numpy uint8 array on the PET grid
```

## Input requirements

- **Format:** 3D NIfTI (`.nii` / `.nii.gz`). Convert DICOM with a tool such as [dcm2niix](https://github.com/rordenlab/dcm2niix).
- **PET units:** body-weight SUV. If your PET is in Bq/mL, use `--suv-factor` or `--pet-dicom`. The tool stops if the PET does not look like SUV.
- **CT units:** Hounsfield units. The CT may be on its own grid; it is resampled onto the PET grid.
- **Pairing:** PET and CT should come from the same examination.

## Offline and air-gapped use

On a machine with internet access, download the model folder:

```bash
huggingface-cli download zhanghaoyue/MM-PET-AI --local-dir mmpetai_model
```

Copy that folder across, then run with `--model mmpetai_model`, or set `MMPETAI_MODEL=/path/to/mmpetai_model`.

## Processing

The preprocessing matches the published evaluation. PET and CT are reoriented to RAS and resampled to 3 mm isotropic voxels. The volume is cropped to the CT body foreground, CT is windowed to [-175, 250] HU, and PET is windowed to SUV [0, 15].

Inference uses overlapping 192³ windows (33% overlap, Gaussian blending, bfloat16 on GPU). Windows with a lesion-presence probability below 0.1 are suppressed. The blended probability is mapped back to the original PET grid and thresholded at 0.5.

A CUDA GPU is strongly recommended. CPU-only inference works but is slow.

## Limitations

- **Tracer prediction is auxiliary.** It was accurate on the internal test set but did not transfer to an external cohort (DEEP-PSMA), so do not use it as a tracer label.
- **DOTATATE evaluation was limited** (28 internal test scans, no external cohort).
- **Performance varies by protocol and disease.** It is lower for small, low-uptake lesions and for some institutional prostate cancer and multiple myeloma cohorts. See the paper for details.

## Citation

```bibtex
@article{zhang_mmpetai,
  title   = {MM-PET-AI: Unified Whole-Body Lesion Segmentation across FDG, DOTATATE, and PSMA PET/CT},
  author  = {Zhang, Haoyue and Vazquez-Quiroga, Alesia and Haque, Fahmida and Turkbey, Baris and Harmon, Stephanie},
  journal = {arxiv},
  year    = {to appear}
}
```
## License

To be added.
