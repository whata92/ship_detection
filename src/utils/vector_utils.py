from typing import Dict, List, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.features import rasterize
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
import json


def gdf_to_np(
    gdf_file: str,
    ref_img_path: str,
    target_shape: Union[Tuple[int], None],
    all_touched: bool = False,
) -> np.array:
    """Converts gdf to numpy array
    Args:
        gdf_file (str): Path to gdf file
        ref_img_path (str): Path to reference image
        target_shape (Union[Tuple[int], None]): Shape of output array.
                            If defined, should be in (width, height) format.
        all_touched (bool): Whether to rasterize all pixels touched by polygons
    Returns:
        np.array: Np.array corresponding to vector data alligned with ref_img_path
    """
    gdf = gpd.read_file(gdf_file)

    polygons = [(s, 1) for s in gdf["geometry"]]
    with rio.open(ref_img_path) as src:
        transform = src.transform
        size = (src.height, src.width)
    img_np = np.zeros(size)

    if polygons:
        rasterize(
            polygons,
            out_shape=size,
            out=img_np,
            transform=transform,
            fill=0,
            all_touched=all_touched,
        )

    if target_shape is not None:
        img_np = cv2.resize(img_np, dsize=target_shape)
    return img_np


def georeference_bboxes(
    bboxes: np.ndarray,
    transform: Affine,
    crs: str,
    date: str
) -> gpd.GeoDataFrame:
    """Georeferencing output bboxes from model output

    Args:
        bboxes (np.ndarray):
            bboxes to be transformed. It should be list of bboxes.
            [[x1, y1, x2, y2, conf], [...] ...]
        transform (Affine): Georeference affine transform matrix
        crs (str): CRS to be referenced
        date (str): Observation date of the satellite image. (YYYY-MM-DD)

    Returns:
        (gpd.GeoDataFrame): Georeferenced bbox
    """
    result = []
    date_list = [date] * len(bboxes)
    for bbox in bboxes:
        bbox = convert_bbox_to_polygon(bbox[:4])
        transform_for_func = np.array(transform)[[0, 1, 3, 4, 2, 5]]

        geo_bbox = affine_transform(bbox, transform_for_func)
        result.append(geo_bbox)
    d = {
        "geometry": result,
        "confidence": bboxes[:, 4],
        "date": date_list
    }
    output = gpd.GeoDataFrame(d)
    output.set_crs(epsg=crs.to_epsg(), inplace=True)
    return output


def convert_bbox_to_polygon(bbox: np.ndarray) -> Polygon:
    x1 = (bbox[0], bbox[1])
    x2 = (bbox[0], bbox[3])
    x3 = (bbox[2], bbox[3])
    x4 = (bbox[2], bbox[1])
    return Polygon([x1, x2, x3, x4])


def convert_geojson_to_geojsonld(
    geojson: str,
    output_file: str
) -> None:
    """Function to convert geojson to geojsonld (line-deliminated)

    Args:
        geojson (str): geojson to convert
        output_file (str): path to output geojsonld file
    """
    with open(geojson) as fp:
        geos = json.load(fp)

    features = geos["features"]
    features = [json.dumps(feature).replace(" ", "") for feature in features]

    with open(output_file, "w") as fp:
        fp.write('\n'.join(features))


def convert_polygon_to_point(
    geojson: str,
    output_file: str
) -> None:
    """Function to convert polygon geojson to point geojson

    Args:
        geojson (str): geojson to convert
        output_file (str): path to output geojson file
    """
    gdf = gpd.read_file(geojson)

    centers = gdf['geometry'].centroid
    gdf['geometry'] = centers

    gdf.to_file(output_file, driver='GeoJSON')
