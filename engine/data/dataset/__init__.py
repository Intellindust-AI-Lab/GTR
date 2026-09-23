"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_lmdb_dataset import CocoLmdbDetection
from .coco_pose_dataset import CocoPoseDetection
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .dota_dataset import DOTADetection, DOTA_CLASSES
from .dota_eval import DOTAEvaluator

# Semantic segmentation (Cityscapes 19-class trainId labels), mIoU evaluator.
from .cityscapes_semseg import CityscapesSemSeg
from .semseg_eval import SemSegEvaluator

# Monocular metric depth (NYU Depth V2 labeled .mat), DA-V2-style protocol.
from .nyu_depth_dataset import NYUDepthV2
from .depth_eval import DepthEvaluator, eval_depth
# Ultralytics-format depth LMDBs (mixed metric-depth pretraining).
from .ultra_depth_lmdb import UltraDepthLmdb
