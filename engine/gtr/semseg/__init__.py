"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Semantic segmentation: dense FCN head on the encoder's stride-8 level, no queries.
"""

from .gtrsemseg import GTRSemSeg, SemSegHead, SemSegCriterion, SemSegPostProcessor
