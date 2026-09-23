"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Monocular metric depth: DPT-style head on the 3-level GTR pyramid + scale-invariant log loss.
"""

from .gtrdepth import GTRDepth, DPTDepthHead
from .criterion import GTRDepthCriterion
