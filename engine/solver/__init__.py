"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Dict

from ._solver import BaseSolver
from .det_solver import DetSolver
from .semseg_solver import SemSegSolver
from .depth_solver import DepthSolver

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'segmentation': DetSolver,
    'obb': DetSolver,
    'pose': DetSolver,
    'semantic_segmentation': SemSegSolver,
    'depth': DepthSolver,
}
