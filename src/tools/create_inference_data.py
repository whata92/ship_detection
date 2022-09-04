import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + "/../../"
    )
)

from configs.preprocessing.default import get_cfg_from_file
from src.utils.logger import init_log
from src.utils.raster_utils import (convert_img_to_np, crop_raster,
                                    generate_world_file)

init_log("global", "info")

def parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--img_path",
        dest="img_path",
        help="Path to the original satellite image",
        type=str,
        required=True
    )
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="Path to the config file",
        type=str,
        default="configs/preprocessing/default.yaml",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        help="Path to the output path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--crop_size",
        dest="crop_size",
        help="Subgrid size for cropping",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--overlap",
        dest="overlap",
        help="Overlap ratio of raster image [0, 1)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--discard_residual",
        dest="discard_residual",
        action='store_false',
        help="If checked, program discard residual portion of image"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("global")
    args = parser()
    cfg = get_cfg_from_file(args.cfg)
    logger.info(f"Image path: {args.img_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Crop image size: {args.crop_size}")
    logger.info(f"Overlap ratio: {args.overlap}")

    crop_raster(
        args.img_path,
        args.output_path,
        [args.crop_size, args.crop_size],
        [args.overlap, args.overlap],
        args.discard_residual
    )

    tif_imgs = glob.glob(os.path.join(args.output_path, "*.tif"))
    for tif_img in tif_imgs:
        out_filename = tif_img.replace(".tif", ".png")
        img_np = convert_img_to_np(tif_img, [0, 1, 2]).transpose(1, 2, 0)
        img_np_log = np.log(img_np + 1e-6)
        img_np_log = (
            (
                np.clip(
                    img_np_log,
                    cfg.DATASET.INFERENCE.PERCENTILE05,
                    cfg.DATASET.INFERENCE.PERCENTILE95
                ) - cfg.DATASET.INFERENCE.PERCENTILE05
            ) / (
                cfg.DATASET.INFERENCE.PERCENTILE95
                - cfg.DATASET.INFERENCE.PERCENTILE05
            )
        )
        img_np_log = (img_np_log * 255).astype(np.uint8)

        # Generate png file
        cv2.imwrite(out_filename, img_np_log)

        # Generate world file
        generate_world_file(tif_img, gen_prj=True)
