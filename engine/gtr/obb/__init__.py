"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Oriented-bounding-box head: the det decoder adapted to (cx, cy, w, h, a) rboxes with
angle distribution refinement, KLD / Chamfer costs and oriented contrastive denoising.
"""

from .decoder import OBBGTRTransformer
from .criterion import OBBGTRCriterion
from .matcher import OBBHungarianMatcher
from .postprocessor import OBBPostProcessor
