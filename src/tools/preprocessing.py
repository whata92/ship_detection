import argparse
import sys
import logging

sys.path.append("/workspace")

from configs.default import get_cfg_from_file
from src.utils.logger import init_log
from src.utils.preprocessing_utils import crop_image

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
        "--annotation_path",
        dest="output_root",
        help="Path to the original annotation path",
        type=str,
        required=True
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

    # Get image and annotation path

    # Get the output folder

    # Get width, height and overlap of the output images

    # Calculate steps of cropping

    # load annotation file

    # Start for loop

    # Crop and save images by given coordinates

    # Crop and save annotations by given coordinates
