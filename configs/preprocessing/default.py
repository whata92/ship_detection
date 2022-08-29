import os
from yacs.config import CfgNode

_C = CfgNode()

_C.PREPROCESS = CfgNode()
_C.PREPROCESS.WIDTH = 512
_C.PREPROCESS.HEIGHT = 512
_C.PREPROCESS.OVERLAP = 0.2

_C.DATASET = CfgNode()
_C.DATASET.INFERENCE = CfgNode()
_C.DATASET.INFERENCE.PERCENTILE05 = -4.7426
_C.DATASET.INFERENCE.PERCENTILE95 = 0.3941

def get_cfg_defaults():
    return _C.clone()


def get_cfg_from_file(
    filepath: str,
) -> CfgNode:

    cfg = get_cfg_defaults()
    if os.path.exists(filepath):
        cfg.merge_from_file(filepath)

    cfg.freeze()
    return cfg
