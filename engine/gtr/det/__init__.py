"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Horizontal-box detection head (DEIM / D-FINE lineage) and the instance-segmentation head
GTRSeg attaches to it. `hungarian_gpu_batch.py` is vendored from HA4DETR (Apache-2.0).
"""

from .decoder import GTRTransformer
from .criterion import GTRCriterion
from .matcher import HungarianMatcher
from .postprocessor import PostProcessor
