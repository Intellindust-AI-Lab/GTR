"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Multi-person keypoint head ported from DETRPose: own model shell, deformable-attention
decoder, OKS-based criterion / matcher and keypoint postprocessor.
"""

from .gtrpose import GTRPose
from .decoder import GTRPoseTransformer
from .criterion import GTRPoseCriterion
from .matcher import GTRPoseHungarianMatcher
from .postprocessor import GTRPosePostProcessor
