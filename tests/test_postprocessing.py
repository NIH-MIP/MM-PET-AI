import numpy as np
import pytest

from mmpetai.postprocessing import binarize, restore_from_ledger, summarize_lesions


def _ledger(full, start, crop):
    end = tuple(s + c for s, c in zip(start, crop))
    return {
        "shape_before_cropping": full,
        "shape_after_cropping_before_padding": crop,
        "foreground_start_coord": start,
        "foreground_end_coord": end,
    }


def test_restore_places_crop_and_preserves_world_coordinates():
    full, start, crop, padded = (50, 60, 70), (5, 6, 7), (20, 30, 40), (24, 33, 40)
    pad_before = np.array([(p - c) // 2 for p, c in zip(padded, crop)])

    original = np.zeros(full, dtype=np.float32)
    original[5 + 3, 6 + 4, 7 + 5] = 1.0
    working = np.zeros(padded, dtype=np.float32)
    working[tuple(pad_before + np.array([3, 4, 5]))] = 1.0

    # Working-grid affine: 3 mm voxels, origin shifted by the crop and pad.
    base = np.diag([3.0, 3.0, 3.0, 1.0])
    base[:3, 3] = [-100.0, -120.0, -80.0]
    working_affine = base.copy()
    working_affine[:3, 3] = base[:3, 3] + 3.0 * (np.array(start) - pad_before)

    restored, affine = restore_from_ledger(working, working_affine, _ledger(full, start, crop))
    assert restored.shape == full
    np.testing.assert_array_equal(restored, original)
    np.testing.assert_allclose(affine, base)


def test_restore_rejects_inconsistent_ledger():
    bad = _ledger((10, 10, 10), (0, 0, 0), (5, 5, 5))
    bad["foreground_end_coord"] = (6, 5, 5)
    with pytest.raises(ValueError):
        restore_from_ledger(np.zeros((5, 5, 5)), np.eye(4), bad)


def test_binarize_and_summary():
    prob = np.zeros((10, 10, 10), dtype=np.float32)
    prob[1:3, 1:3, 1:3] = 0.9
    prob[7, 7, 7] = 0.6
    prob[5, 5, 5] = 0.4
    mask = binarize(prob)
    pet = np.full(prob.shape, 2.0, dtype=np.float32)
    pet[7, 7, 7] = 10.0

    summary = summarize_lesions(mask, pet, (2.0, 2.0, 2.0))
    assert summary["lesion_count"] == 2
    assert summary["total_lesion_volume_ml"] == pytest.approx(9 * 8 / 1000.0)
    assert summary["suv_max"] == pytest.approx(10.0)
