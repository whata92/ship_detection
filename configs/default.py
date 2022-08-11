import os
from yacs.config import CfgNode

_C = CfgNode()

_C.DATASET = CfgNode()
_C.DATASET.TRAIN = CfgNode()
_C.DATASET.TRAIN.IMAGE = "/workspace/dataset/LS-SSDD-v1.0-OPEN/JPEGImages_sub"
_C.DATASET.TRAIN.ANNOTATION = "/workspace/dataset/train.json"
_C.DATASET.VAL = CfgNode()
_C.DATASET.VAL.IMAGE = "/workspace/dataset/LS-SSDD-v1.0-OPEN/JPEGImages_sub"
_C.DATASET.VAL.ANNOTATION = "/workspace/dataset/val.json"
_C.DATASET.INFERENCE = CfgNode()
_C.DATASET.INFERENCE.IMAGE_FILE = "/workspace/dataset/infer_all.txt"
_C.DATASET.NUM_CLASS = 2

_C.TRAIN = CfgNode()
_C.TRAIN.EPOCH = 20
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.OPTIMIZER = CfgNode()
_C.TRAIN.OPTIMIZER.LEARNING_RATE = 0.0001
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0005
_C.TRAIN.OPTIMIZER.LR_STEP_SIZE = 3
_C.TRAIN.OPTIMIZER.LR_GAMMA = 0.1


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
