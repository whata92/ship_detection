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
from src.utils.vector_utils import convert_polygon_to_point

init_log("global", "info")

def parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--geojson",
        dest="geojson",
        help="Path to the geojson file to convert",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out_geojson",
        dest="out_geojson",
        help="Path to the output geojson file",
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("global")
    args = parser()

    convert_polygon_to_point(args.geojson, args.out_geojson)
