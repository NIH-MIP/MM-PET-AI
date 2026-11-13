"""Body-weight SUV factor from PET DICOM headers (optional; requires pydicom).

PET NIfTIs converted directly from DICOM are often in Bq/mL. The model expects
SUVbw, so each such PET is multiplied by

    SUVbw factor = body weight [g] / injected dose decayed to scan start [Bq]
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

_DEFAULT_HALF_LIFE_S = {"18F": 6586.2, "68GA": 4062.6, "11C": 1223.4, "64CU": 45720.0, "89ZR": 282276.0}


def _parse_time(value) -> Optional[timedelta]:
    s = str(value or "").strip()
    if not s:
        return None
    whole, _, frac = s.partition(".")
    whole = whole.ljust(6, "0")[:6]
    try:
        hh, mm, ss = int(whole[:2]), int(whole[2:4]), int(whole[4:6])
    except ValueError:
        return None
    return timedelta(hours=hh, minutes=mm, seconds=ss, microseconds=int((frac or "0")[:6].ljust(6, "0")))


def _parse_datetime(date_value, time_value) -> Optional[datetime]:
    date_s = str(date_value or "").strip()[:8]
    t = _parse_time(time_value)
    if len(date_s) != 8 or t is None:
        return None
    try:
        return datetime.strptime(date_s, "%Y%m%d") + t
    except ValueError:
        return None


def _parse_dt(value) -> Optional[datetime]:
    s = str(value or "").strip()
    if len(s) < 8:
        return None
    rest = s[8:14] + (s[14:].split("+")[0].split("-")[0] if len(s) > 14 else "")
    return _parse_datetime(s[:8], rest)


def suv_factor_from_header(ds, earliest_acquisition: Optional[datetime] = None) -> Tuple[Optional[float], str]:
    """Return (factor, status). ``factor`` multiplies stored PET values to give SUVbw."""
    units = str(ds.get("Units", "")).strip().upper()
    decay = str(ds.get("DecayCorrection", "")).strip().upper()

    if units in {"GML", "G/ML"}:
        return 1.0, "already_suv"
    if units == "CNTS":
        philips = ds.get((0x7053, 0x1000))
        if philips is not None and float(philips.value) > 0:
            return float(philips.value), "philips_suv_scale_factor"
        return None, "unsupported_units:CNTS"
    if units != "BQML":
        return None, f"unsupported_units:{units or 'missing'}"

    try:
        weight_kg = float(ds.get("PatientWeight", None))
    except (TypeError, ValueError):
        weight_kg = float("nan")
    if not weight_kg > 0:
        return None, "missing_weight"

    seq = ds.get("RadiopharmaceuticalInformationSequence", None)
    if not seq:
        return None, "missing_radiopharmaceutical_info"
    rp = seq[0]
    try:
        dose_bq = float(rp.get("RadionuclideTotalDose", None))
    except (TypeError, ValueError):
        dose_bq = float("nan")
    if not dose_bq > 0:
        return None, "missing_injected_dose"

    if decay == "ADMIN":
        return weight_kg * 1000.0 / dose_bq, "ok"
    if decay != "START":
        return None, f"unsupported_decay_correction:{decay or 'missing'}"

    nuclide = ""
    code_seq = rp.get("RadionuclideCodeSequence", None)
    if code_seq:
        nuclide = str(code_seq[0].get("CodeMeaning", "")).replace("^", "").replace("-", "").upper()
    try:
        half_life = float(rp.get("RadionuclideHalfLife", None))
    except (TypeError, ValueError):
        half_life = _DEFAULT_HALF_LIFE_S.get(nuclide, float("nan"))
    if not half_life > 0:
        return None, "missing_half_life"

    scan_start = _parse_datetime(ds.get("SeriesDate"), ds.get("SeriesTime"))
    if earliest_acquisition is not None and (scan_start is None or earliest_acquisition < scan_start):
        scan_start = earliest_acquisition
    if scan_start is None:
        return None, "missing_scan_time"

    # Try the full injection datetime first, then the time of day on the scan
    # date (anonymized collections sometimes shift the injection date).
    candidates = []
    full = _parse_dt(rp.get("RadiopharmaceuticalStartDateTime", None))
    if full is not None:
        candidates.append(full)
    tod = _parse_time(rp.get("RadiopharmaceuticalStartTime", None))
    if tod is None and full is not None:
        tod = timedelta(hours=full.hour, minutes=full.minute, seconds=full.second)
    if tod is not None:
        on_scan_day = datetime(scan_start.year, scan_start.month, scan_start.day) + tod
        if on_scan_day > scan_start:
            on_scan_day -= timedelta(days=1)
        candidates.append(on_scan_day)
    if not candidates:
        return None, "missing_injection_time"

    max_delay_s = max(12 * 3600.0, 5 * half_life)
    for injection in candidates:
        delay_s = (scan_start - injection).total_seconds()
        if 0 <= delay_s <= max_delay_s:
            return weight_kg * 1000.0 / (dose_bq * math.pow(2.0, -delay_s / half_life)), "ok"
    return None, "implausible_uptake_delay"


def suv_factor_from_dicom_dir(dicom_dir: str) -> float:
    """SUV factor of the single PET series found under ``dicom_dir``."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("Reading DICOM requires pydicom: pip install 'mmpetai[dicom]'") from exc

    series = {}
    for dirpath, _, filenames in os.walk(dicom_dir):
        for name in sorted(filenames):
            try:
                ds = pydicom.dcmread(os.path.join(dirpath, name), stop_before_pixels=True)
            except Exception:
                continue
            if str(ds.get("Modality", "")).upper() != "PT":
                continue
            uid = str(ds.get("SeriesInstanceUID", dirpath))
            first, earliest = series.get(uid, (ds, None))
            acq = _parse_datetime(ds.get("AcquisitionDate"), ds.get("AcquisitionTime"))
            if acq is not None and (earliest is None or acq < earliest):
                earliest = acq
            series[uid] = (first, earliest)

    if not series:
        raise ValueError(f"No PET (Modality=PT) DICOM files found under {dicom_dir}")
    if len(series) > 1:
        raise ValueError(f"Found {len(series)} PET series under {dicom_dir}; point --pet-dicom at one series.")

    first, earliest = next(iter(series.values()))
    factor, status = suv_factor_from_header(first, earliest_acquisition=earliest)
    if factor is None:
        raise ValueError(f"Could not compute an SUV factor from {dicom_dir}: {status}")
    return factor
