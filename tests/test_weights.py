import json

import pytest
import torch

from mmpetai.network import build_model
from mmpetai.weights import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    clean_state_dict,
    load_model_files,
    resolve_model,
    save_model_folder,
    validate_config,
)
import os


@pytest.fixture(scope="module")
def model():
    return build_model().eval()


def test_training_checkpoint_keys_load_strictly(model):
    training_sd = {f"module._orig_mod.{k}": v for k, v in model.state_dict().items()}
    training_sd.update({
        "ds1.weight": torch.zeros(1),
        "organ_head.weight": torch.zeros(1),
        "pet_local_aux_gate.query_proj.weight": torch.zeros(1),
        "pet_local_aux_head.weight": torch.zeros(1),
        "encoder_pet.encoder2.transp_conv_init.conv.weight": torch.zeros(1),
    })
    cleaned, dropped = clean_state_dict(training_sd)
    assert len(dropped) == 5
    model.load_state_dict(cleaned, strict=True)


def test_refinement_checkpoint_is_rejected(model):
    sd = dict(model.state_dict())
    sd["pet_local_refine_proj.weight"] = torch.zeros(1)
    with pytest.raises(ValueError):
        clean_state_dict(sd)


def test_model_folder_round_trip(model, tmp_path):
    folder = save_model_folder(model.state_dict(), tmp_path / "hf_model", {"note": "test"})
    assert (folder / WEIGHTS_FILENAME).is_file() and (folder / CONFIG_FILENAME).is_file()

    weights, config = resolve_model(str(folder))
    state_dict, cfg = load_model_files(weights, config)
    assert cfg["metadata"]["note"] == "test"
    build_model().load_state_dict(state_dict, strict=True)


def test_incompatible_config_is_rejected(model, tmp_path):
    folder = save_model_folder(model.state_dict(), tmp_path / "hf_model")
    cfg = json.loads((folder / CONFIG_FILENAME).read_text())
    cfg["architecture"]["hidden_size_ct"] = 768
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.skipif(not os.environ.get("MMPETAI_SLOW_TESTS"), reason="set MMPETAI_SLOW_TESTS=1 (192^3 forward pass)")
def test_forward_shapes(model):
    with torch.inference_mode():
        seg, lesion, tracer = model(torch.zeros(1, 2, 192, 192, 192))
    assert seg.shape == (1, 1, 192, 192, 192)
    assert lesion.shape == (1, 1)
    assert tracer.shape == (1, 3)
