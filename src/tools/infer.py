import argparse
import glob
import os
import sys

import geopandas as gpd
import pandas as pd
import rasterio as rio

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + "/../../"
    )
)

from mmdet.apis import inference_detector, init_detector

from src.utils.vector_utils import georeference_bboxes


def parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--img_dir",
        dest="img_dir",
        help="Path to the original satellite image",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to the output path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_cfg",
        dest="model_cfg",
        help="Path to the config file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint file",
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parser()
    os.makedirs(args.output_dir, exist_ok=True)

    png_list = glob.glob(os.path.join(args.img_dir, "*.png"))

    model = init_detector(args.model_cfg, args.checkpoint, device='cuda:0')

    for png in png_list:
        file_stem = os.path.splitext(os.path.basename(png))[0]
        date = file_stem.split("_")[0]
        out_file = os.path.join(args.output_dir, os.path.basename(png))
        result = inference_detector(model, png)
        model.show_result(png, result, out_file=out_file)

        with rio.open(png) as src:
            transform = src.transform

        with open(os.path.join(args.img_dir, file_stem + '.prj')) as fp:
            prj = fp.read()
            prj = rio.CRS.from_wkt(prj)

        gdf = georeference_bboxes(result[0], transform, prj, date)
        gdf.to_file(os.path.join(args.output_dir, file_stem + '.geojson'), driver="GeoJSON")

    geojsons = glob.glob(os.path.join(args.output_dir, '*.geojson'))
    for i, geojson in enumerate(geojsons):
        if i == 0:
            gdf = gpd.read_file(geojson)
        else:
            tmp = gpd.read_file(geojson)
            gdf = pd.concat([gdf, tmp])

    output_stem = args.img_dir.split('/')[-1]
    gdf.to_crs(epsg=4326, inplace=True)
    gdf.to_file(os.path.join(args.output_dir, output_stem + '.geojson'))
