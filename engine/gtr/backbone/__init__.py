"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Gated-token-recurrence ViT backbones. `csrc/` holds the optional hand-written CUDA
chunk_gla kernel (build with `bash engine/gtr/backbone/csrc/build.sh`).
"""

from .vit_adapter import ViTAdapter
from .vit_adapter_spatial_swiglu import ViTAdapterSpatialSwiGLU
