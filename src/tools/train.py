import argparse
import sys
import os
import random
import numpy as np
import torch
import logging
from datetime import datetime

sys.path.append("/home/ubuntu/workspace/ship_detection")
ROOT = "/home/ubuntu/workspace/ship_detection"

from src.dataloader.dataloader import CocoDataset, get_dataloader
from src.utils.engine import evaluate
from src.model.faster_rcnn import get_model_instance_segmentation
from src.utils.logger import init_log
from configs.default import get_cfg_from_file


init_log("global", "info")
SEED = 42
CHECKPOINT_ROOT = os.path.join(
    ROOT,
    "checkpoints",
    datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


def parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="Path to the config file",
        type=str,
        default="configs/training/default.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("global")
    args = parser()
    logger.debug(f"config file: {args.cfg}")
    cfg = get_cfg_from_file(args.cfg)
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    train_dataset = CocoDataset(
        img_dir=cfg.DATASET.TRAIN.IMAGE,
        ann_file=cfg.DATASET.TRAIN.ANNOTATION,
    )

    val_dataset = CocoDataset(
        img_dir=cfg.DATASET.VAL.IMAGE,
        ann_file=cfg.DATASET.VAL.ANNOTATION
    )

    logger.info(f"Train length: {len(train_dataset)}, Val length: {len(val_dataset)}")

    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE
    )

    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE
    )

    device = (
        torch.device('cuda')
        if torch.cuda.is_available() else torch.device('cpu')
    )
    logger.debug(f"device: {device}")

    model = get_model_instance_segmentation(num_classes=cfg.DATASET.NUM_CLASS)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.TRAIN.OPTIMIZER.LEARNING_RATE,
        momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.TRAIN.OPTIMIZER.LR_STEP_SIZE,
        gamma=cfg.TRAIN.OPTIMIZER.LR_GAMMA
    )

    for epoch in range(cfg.TRAIN.EPOCH):
        for i, batch in enumerate(train_dataloader):
            model.train()
            images, targets = batch

            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()

            optimizer.step()

            if i == 1000:
                lr_scheduler.step()

            if (i + 1) % 10 == 0:
                logger.info(
                    f"epoch #{epoch + 1} Iteration #{i + 1} loss: {loss_value}"
                )

            if (i + 1) % 50 == 0:
                evaluate(model, val_dataloader, device=device)
                torch.save(
                    model.state_dict(),
                    os.path.join(CHECKPOINT_ROOT, f"epoch_{epoch + 1}_iter_{i + 1}.pth")
                    )
                logger.info(f"Saved weight: epoch_{epoch + 1}_iter_{i + 1}.pth")
