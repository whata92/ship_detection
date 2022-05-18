import torch
import os
import random
import logging
import numpy as np

from model.mask_rcnn import get_model_instance_segmentation
from dataloader.dataset import AirbusDS, get_dataloader

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
    model = get_model_instance_segmentation(num_classes=2)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = AirbusDS(dataset_root="/workspace/dataset", aug=True)
    train_dataloader = get_dataloader(dataset)
    model.train()
    model.to(device)

    for epoch in range(EPOCHS):

        for i, batch in enumerate(train_dataloader):
            images, targets = batch

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()

            if (i + 1) % 10 == 0:
                # TODO: Add evaluation algorithm
                print(f"epoch #{epoch + 1} Iteration #{i + 1} loss: {loss_value}")