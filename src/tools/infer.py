import argparse
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from PIL import Image

import numpy as np
import torch

sys.path.append("/workspace")

from configs.default import get_cfg_from_file
from src.dataloader.dataloader import InferDataset, get_dataloader
from src.model.faster_rcnn import get_model_instance_segmentation
from src.utils.logger import init_log

init_log("global", "info")


def parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_root",
        dest="output_root",
        help="Path to the output path",
        type=str,
        default="output"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="Path to the config file",
        type=str,
        default="configs/inference/default.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("global")
    args = parser()
    logger.debug(f"config file: {args.cfg}")
    cfg = get_cfg_from_file(args.cfg)
    os.makedirs(args.output_root, exist_ok=True)

    device = (
        torch.device('cuda')
        if torch.cuda.is_available() else torch.device('cpu')
    )
    logger.debug(f"device: {device}")

    # Load weights for inference
    if device is torch.device('cpu'):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(args.checkpoint)
    weights = checkpoint

    # Create dataloader for inference
    with open(cfg["DATASET"]["INFERENCE"]["IMAGE_FILE"]) as fp:
        paths = fp.read()
        img_list = paths.split("\n")
    dataset = InferDataset(img_list)

    infer_dataloader = get_dataloader(
        dataset,
        infer_mode=True
    )

    result = {}

    # Create model
    with torch.no_grad():
        model = get_model_instance_segmentation(num_classes=cfg.DATASET.NUM_CLASS)
        model.load_state_dict(weights)
        model.eval()

        for i, batch in enumerate(infer_dataloader):
            img, img_path = batch
            img_path = img_path[0]
            img.to(device)

            output = model(img)[0]
            output["boxes"] = output["boxes"].to("cpu").numpy()
            output["labels"] = output["labels"].to("cpu").numpy()
            output["scores"] = output["scores"].to("cpu").numpy()

            result[img_path] = output

    # Create output csv

    # Create output image
    for key in result.keys():
        plt.figure()
        img_name = os.path.basename(key)
        img_numpy = np.array(Image.open(key))
        plt.imshow(img_numpy)

        ax = plt.gca()
        for box in result[key]["boxes"]:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            rect = patches.Rectangle(
                (center_x, center_y),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                fill=False
            )
            ax.add_patch(rect)
        output_path = os.path.join(
            args.output_root,
            img_name
        )
        plt.savefig(output_path)
