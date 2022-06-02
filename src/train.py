import torch
import os
import random
import logging
import numpy as np

from utils.engine import train_one_epoch, evaluate
from model.mask_rcnn import get_model_instance_segmentation
from dataloader.dataset import AirbusDS, CocoDataset, get_dataloader
from dataloader.augmentation import (
    get_train_transforms,
    get_valid_transforms
)

SEED = 42
logger = logging.getLogger(__name__)
# TODO: set logging function


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


if __name__ == "__main__":
    EPOCHS = 5
    clipping_value = 5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes=10)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # train_dataset = AirbusDS(
    #     dataset_file="/home/ubuntu/workspace/Airbus_ship/dataset/train.csv",
    #     dataset_root="/home/ubuntu/workspace/Airbus_ship/dataset/train",
    #     aug=True
    # )

    # val_dataset = AirbusDS(
    #     dataset_file="/home/ubuntu/workspace/Airbus_ship/dataset/val.csv",
    #     dataset_root="/home/ubuntu/workspace/Airbus_ship/dataset/val",
    #     aug=True
    # )

    train_aug = get_train_transforms()
    val_aug = get_valid_transforms()

    train_dataset = CocoDataset(
        # dataset_dir="/workspace/Airbus_ship/dataset",
        dataset_dir="/home/ubuntu/workspace/Airbus_ship/dataset",
        ann_file="train.json",
        mode="train",
        transforms=val_aug
    )

    val_dataset = CocoDataset(
        # dataset_dir="/workspace/Airbus_ship/dataset",
        dataset_dir="/home/ubuntu/workspace/Airbus_ship/dataset",
        ann_file="val.json",
        mode="val",
        transforms=val_aug
    )

    train_dataloader = get_dataloader(train_dataset)
    val_dataloader = get_dataloader(val_dataset, batch_size=1)

    for epoch in range(EPOCHS):
        # train_one_epoch(
        #     model,
        #     optimizer,
        #     train_dataloader,
        #     device,
        #     epoch,
        #     print_freq=100
        # )

        for i, batch in enumerate(train_dataloader):
            model.train()
            images, targets = batch

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()

            if i == 1000:
                lr_scheduler.step()

            if (i + 1) % 10 == 0:
                # TODO: Add evaluation algorithm
                print(f"epoch #{epoch + 1} Iteration #{i + 1} loss: {loss_value}")
                evaluate(model, val_dataloader, device=device)