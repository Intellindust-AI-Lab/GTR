"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# Importing a module registers its @register() classes with engine.core, so every class a
# YAML config names (`type:` or an injected `backbone:` / `decoder:` ...) must be reachable
# from here. Task packages only add their own decoder / criterion / matcher / postprocessor.

# Shared by every task: model shells, the GTR encoder and the two backbone flavours.
from .gtr import GTR, GTRSeg
from .hybrid_encoder import HybridEncoder, GTREncoder
from .backbone import ViTAdapter, ViTAdapterSpatialSwiGLU

# Horizontal boxes (+ instance masks through GTRSeg).
from .det import GTRTransformer, GTRCriterion, HungarianMatcher, PostProcessor

# Oriented bounding boxes: reuses the GTR model, swaps the four det pieces.
from .obb import OBBGTRTransformer, OBBGTRCriterion, OBBHungarianMatcher, OBBPostProcessor

# Keypoints: own model class + DETRPose-style decoder head.
from .pose import (GTRPose, GTRPoseTransformer, GTRPoseCriterion,
                   GTRPoseHungarianMatcher, GTRPosePostProcessor)

# Semantic segmentation: backbone + encoder + dense FCN head, no queries.
from .semseg import GTRSemSeg, SemSegHead, SemSegCriterion, SemSegPostProcessor

# Metric depth: DA-V2-style DPT head + SiLog loss on the GTR pyramid.
from .depth import GTRDepth, DPTDepthHead, GTRDepthCriterion
