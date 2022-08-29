import argparse
import logging
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + "/../../"
    )
)

from src.utils.logger import init_log
from src.utils.raster_utils import crop_raster


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
    logger.debug(f"Image path: {args.img_path}")
    logger.debug(f"Output path: {args.output_path}")
    logger.debug(f"Crop image size: {args.crop_size}")
    logger.debug(f"Overlap ratio: {args.overlap}")

    crop_raster(
        args.img_path,
        args.output_path,
        [args.crop_size, args.crop_size],
        [args.overlap, args.overlap],
        args.discard_residual,
        save_png=True,
        color_band=[0, 1, 2]
    )
