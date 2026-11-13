"""Command-line interface.

Single case:
    mmpetai -pet pet.nii.gz -ct ct.nii.gz -o lesions.nii.gz

Folder of cases (nnU-Net naming: <case>_0000.nii.gz = PET, <case>_0001.nii.gz = CT):
    mmpetai -i input_dir -o output_dir
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import __version__

_CASE_RE = re.compile(r"^(?P<case>.+)_(?P<ch>\d{4})\.nii(\.gz)?$")


def _strip_nii(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _sidecar(mask_path: Path, suffix: str) -> Path:
    return mask_path.with_name(_strip_nii(mask_path) + suffix)


def _find_cases(input_dir: Path, pet_channel: int, ct_channel: int) -> Tuple[Dict[str, Tuple[Path, Path]], List[str]]:
    channels: Dict[str, Dict[int, Path]] = {}
    for f in sorted(input_dir.iterdir()):
        m = _CASE_RE.match(f.name)
        if m:
            channels.setdefault(m.group("case"), {})[int(m.group("ch"))] = f
    cases, incomplete = {}, []
    for case, chans in channels.items():
        if pet_channel in chans and ct_channel in chans:
            cases[case] = (chans[pet_channel], chans[ct_channel])
        else:
            incomplete.append(case)
    return cases, incomplete


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmpetai",
        description="Whole-body lesion segmentation on FDG, PSMA, and DOTATATE PET/CT (MM-PET-AI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("-pet", "--pet", type=Path, help="PET NIfTI (SUV, or Bq/mL with --suv-factor / --pet-dicom)")
    io.add_argument("-ct", "--ct", type=Path, help="CT NIfTI in Hounsfield units")
    io.add_argument("-i", "--input-dir", type=Path,
                    help="folder of <case>_0000.nii.gz (PET) and <case>_0001.nii.gz (CT) files")
    io.add_argument("-o", "--output", type=Path, required=True,
                    help="output mask file (single case) or output folder (with -i)")
    io.add_argument("--pet-channel", type=int, default=0, help="channel index of PET files in -i mode")
    io.add_argument("--ct-channel", type=int, default=1, help="channel index of CT files in -i mode")
    io.add_argument("--save-probabilities", action="store_true", help="also save the lesion probability map")

    suv = p.add_argument_group("PET units (single case)")
    suv.add_argument("--suv-factor", type=float, default=None, help="multiply PET by this factor to get SUVbw")
    suv.add_argument("--pet-dicom", type=Path, default=None,
                     help="PET DICOM series folder; the SUV factor is read from its headers (needs pydicom)")

    run = p.add_argument_group("model / runtime")
    run.add_argument("-m", "--model", default=None,
                     help="Hugging Face repo id or local model folder (default: $MMPETAI_MODEL or the released model)")
    run.add_argument("--revision", default=None, help="Hub branch, tag, or commit to pin the model version")
    run.add_argument("-d", "--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    run.add_argument("--sw-batch-size", type=int, default=1, help="windows per forward pass (more = faster, more memory)")
    run.add_argument("--stitch-on-cpu", action="store_true", help="assemble the whole-volume output in CPU memory")
    run.add_argument("--no-amp", action="store_true", help="disable bfloat16 mixed precision")
    run.add_argument("--no-lesion-gate", action="store_true",
                     help="do not suppress windows with low lesion-presence probability (not the published setting)")
    run.add_argument("--skip-input-checks", action="store_true", help="skip PET-unit and channel-order checks")
    run.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    p.add_argument("--version", action="version", version=f"mmpetai {__version__}")
    return p


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def _write_outputs(result, mask_path: Path, save_probabilities: bool) -> dict:
    prob_path = _sidecar(mask_path, "_prob.nii.gz") if save_probabilities else None
    result.save(mask_path, prob_path)
    record = result.to_dict()
    record["outputs"] = {"mask": str(mask_path), **({"probability": str(prob_path)} if prob_path else {})}
    _sidecar(mask_path, ".json").write_text(json.dumps(record, indent=2))
    return record


def _resolve_suv_factor(args) -> float:
    if args.suv_factor is not None and args.pet_dicom is not None:
        raise SystemExit("Use either --suv-factor or --pet-dicom, not both.")
    if args.pet_dicom is not None:
        from .suv import suv_factor_from_dicom_dir
        return suv_factor_from_dicom_dir(str(args.pet_dicom))
    return 1.0 if args.suv_factor is None else float(args.suv_factor)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    single = args.pet is not None or args.ct is not None
    if single == (args.input_dir is not None):
        raise SystemExit("Give either -pet and -ct (single case) or -i (folder), not both.")
    if single and (args.pet is None or args.ct is None):
        raise SystemExit("Single-case mode needs both -pet and -ct.")
    if not single and (args.suv_factor is not None or args.pet_dicom is not None):
        raise SystemExit("--suv-factor / --pet-dicom apply to single-case mode; convert folder inputs to SUV first.")

    from .predictor import MMPETAI

    model = MMPETAI(
        model=args.model,
        revision=args.revision,
        device=args.device,
        sw_batch_size=args.sw_batch_size,
        use_amp=not args.no_amp,
        lesion_gate=not args.no_lesion_gate,
        stitch_on_cpu=args.stitch_on_cpu,
        verbose=not args.quiet,
    )
    _log(f"MM-PET-AI {__version__} | model: {model.weights_path} | device: {model.device}", args.quiet)

    predict_kwargs = dict(return_probability=args.save_probabilities, input_checks=not args.skip_input_checks)

    if single:
        mask_path = args.output
        if mask_path.suffix == "" or mask_path.is_dir():
            mask_path = mask_path / f"{_strip_nii(args.pet)}_lesions.nii.gz"
        result = model.predict(args.pet, args.ct, suv_factor=_resolve_suv_factor(args), **predict_kwargs)
        record = _write_outputs(result, mask_path, args.save_probabilities)
        lesions = record["lesions"]
        _log(
            f"Saved {mask_path} | lesions: {lesions['lesion_count']} | "
            f"volume: {lesions['total_lesion_volume_ml']} mL | {record['runtime_s']} s",
            args.quiet,
        )
        return 0

    cases, incomplete = _find_cases(args.input_dir, args.pet_channel, args.ct_channel)
    if incomplete:
        _log(f"Skipping {len(incomplete)} case(s) without both channels: {incomplete[:10]}", args.quiet)
    if not cases:
        raise SystemExit(f"No complete PET/CT pairs found in {args.input_dir}.")

    args.output.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for n, (case, (pet, ct)) in enumerate(sorted(cases.items()), start=1):
        _log(f"[{n}/{len(cases)}] {case}", args.quiet)
        try:
            result = model.predict(pet, ct, **predict_kwargs)
            record = _write_outputs(result, args.output / f"{case}.nii.gz", args.save_probabilities)
            rows.append({
                "case": case,
                "lesion_count": record["lesions"]["lesion_count"],
                "total_lesion_volume_ml": record["lesions"]["total_lesion_volume_ml"],
                "suv_max": record["lesions"]["suv_max"],
                "total_lesion_uptake": record["lesions"]["total_lesion_uptake"],
                "lesion_presence_probability": record["lesion_presence_probability"],
                "predicted_tracer": record["tracer"]["predicted"],
                "runtime_s": record["runtime_s"],
            })
        except Exception as exc:  # keep going; report at the end
            failures.append(case)
            _log(f"  FAILED: {type(exc).__name__}: {exc}", False)
            if not args.quiet:
                traceback.print_exc()

    if rows:
        with open(args.output / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _log(f"Done: {len(rows)} succeeded, {len(failures)} failed. Results in {args.output}", args.quiet)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
