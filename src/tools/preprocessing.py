import os
import argparse
import sys
import logging
import glob

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + "/../../"
    )
)

from configs.default import get_cfg_from_file
from src.utils.logger import init_log
from src.utils.preprocessing_utils import (
    crop_image,
    crop_xml,
    item2object,
)

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
        dest="annotation_path",
        help="Path to the original annotation path",
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

    os.makedirs(args.output_path, exist_ok=True)

    # Get image and annotation path

    # Get the output folder

    # Get width, height and overlap of the output images
    width = cfg["PREPROCESS"]["WIDTH"]
    height = cfg["PREPROCESS"]["HEIGHT"]
    overlap = cfg["PREPROCESS"]["OVERLAP"]

    crop_image(
        img_file=args.img_path,
        size=[height, width],
        out_dir=args.output_path,
        overlap=0.2,
        residual_crop=False
    )

    crop_xml(
        xml_file=args.annotation_path,
        size=[height, width],
        out_dir=args.output_path,
        overlap=0.2,
        residual_crop=False
    )

    xml_list = glob.glob(
        os.path.join(args.output_path, "*.xml")
    )

    for xml in xml_list:
        item2object(xml)


    # Start for loop

    # Crop and save images by given coordinates

    # Crop and save annotations by given coordinates