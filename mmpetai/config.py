"""Fixed settings of the released MM-PET-AI model.

These values must match the configuration the weights were trained with.
Changing them changes the model's input distribution and invalidates results.
"""

# Sliding-window patch size; the ViT position embeddings are tied to it.
ROI_SIZE = (192, 192, 192)

# Working resolution (mm). PET and CT are resampled to this isotropic grid.
TARGET_SPACING = (3.0, 3.0, 3.0)

# CT window in HU, mapped linearly to [0, 1] with clipping.
CT_WINDOW_HU = (-175.0, 250.0)

# PET is expected in body-weight SUV; SUV in [0, PET_SUV_MAX] maps to [0, 1].
PET_SUV_MAX = 15.0

# Sliding-window inference settings used for all reported results.
SW_OVERLAP = 0.33
SW_MODE = "gaussian"

# Windows whose lesion-presence probability is below this threshold have
# their segmentation logits suppressed before blending.
LESION_GATE_THRESHOLD = 0.1

# Voxel threshold on the blended lesion probability.
SEGMENTATION_THRESHOLD = 0.5

# Order of the tracer-classification head outputs.
TRACERS = ("FDG", "DOTATATE", "PSMA")

# Sanity limit used to catch PET volumes that are not in SUV units.
PET_SUV_SANITY_P999 = 100.0

# Architecture hyperparameters of the released network.
MODEL_KWARGS = dict(
    img_size=ROI_SIZE,
    patch_size=(16, 16, 16),
    feature_size=16,
    hidden_size_pet=768,
    mlp_dim_pet=3072,
    num_heads_pet=12,
    hidden_size_ct=384,
    mlp_dim_ct=1536,
    num_heads_ct=6,
    pet_local_base_ch=16,
)
