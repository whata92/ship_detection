import os
from typing import List, Tuple, Union
import cv2

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import Polygon


def save_patch(
    src: rio.DatasetReader,
    img: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    save_path: str
) -> None:
    """Creates and saves a patch of the input image
        given its pixel coordinates and save_path
    Args:
        src (rio.DatasetReader): Rasterio dataset reader
                                 of the whole raster
        img (np.ndarray): Whole raster in np.ndarray format
        x_min (int): Minimum x coordinate of the patch
        x_max (int): Maximum x coordinate of the patch
        y_min (int): Minimum y coordinate of the patch
        y_max (int): Maximum y coordinate of the patch
        save_path (str): Path to save the patch
    """

    crop_size = (x_max - x_min, y_max - y_min)
    (west, north) = src.xy(x_min, y_min)
    (east, south) = src.xy(x_max, y_max)

    affine = rio.transform.from_bounds(
        west, south, east, north, crop_size[1], crop_size[0]
    )

    crop_img = img[:, x_min:x_max, y_min:y_max]

    (band, _, _) = img.shape

    with rio.open(
        save_path,
        "w",
        driver="GTiff",
        dtype=crop_img.dtype,
        height=crop_size[0],
        width=crop_size[1],
        count=band,
        crs=src.crs,
        transform=affine,
    ) as dst:

        dst.write(crop_img)


def crop_raster(
    input_img: str,
    save_dir: str,
    crop_size: List[int],
    overlap: List[int],
    residual_cropping: bool = False
):
    """Crop raster into subgrids
    Args:
        input_img (str): Path to raster file
        save_dir (str): Destination directory
        crop_size (List[int]): dimensions of the subgrid
        overlap (List[int]): fraction of a subgrid size in each axis that is
                             overlapped between subgrids
        residual_cropping (bool, optional): If True, last column and last row are
                                        aligned to the boarder, so that residuals
                                        are also in cropped subgrids.
                                        Defaults to False.
    """

    os.makedirs(save_dir, exist_ok=True)

    with rio.open(input_img) as src:

        img = np.array([src.read(i + 1) for i in range(src.count)])

        (_, x_size, y_size) = img.shape

        overlap_pxl = [int(crop_size[0] * overlap[0]), int(crop_size[1] * overlap[1])]

        if residual_cropping:
            last_val_x = x_size + 1 - overlap_pxl[0]
            last_val_y = y_size + 1 - overlap_pxl[1]
        else:
            last_val_x = x_size - crop_size[0] + 1
            last_val_y = y_size - crop_size[1] + 1

        for x_min in range(0, last_val_x, crop_size[0] - overlap_pxl[0]):
            for y_min in range(0, last_val_y, crop_size[1] - overlap_pxl[1]):
                x_min = min(x_min, x_size - crop_size[0])
                y_min = min(y_min, y_size - crop_size[1])
                x_max = x_min + crop_size[0]
                y_max = y_min + crop_size[1]

                crop_img_name = (
                    os.path.splitext(input_img)[0]
                    + f"_{x_min}_{y_min}_{x_max}_{y_max}.tif"
                )

                save_path = os.path.join(save_dir, os.path.split(crop_img_name)[-1])

                save_patch(src, img, x_min, x_max, y_min, y_max, save_path)


def convert_img_to_np(
    img: str,
    bands: Union[Tuple[int], None],
) -> np.array:
    """Convert img raster to numpy array. Raster can have any number of bands.
    Args:
        img (str): Path to WV .img file
        bands (Union[Tuple[int], None]): Tuple of bands to extract. If None,
                                         all bands are extracted.
    Returns:
        np.array: raster converted into np.array
    """
    with rio.open(img) as src:
        if bands is None:
            bands = range(src.count)
        bands = [src.read(band_idx + 1) for band_idx in bands]

    img_np = np.array(bands)

    return img_np


def create_geojson_of_img_area(img_path: str, out_geojson: str):
    """Given a raster file path, create a geojson file with a polygon of the
    image area.
    Args:
        img_path (str): Path to raster file
        out_geojson (str): Path to output geojson file
    """

    with rio.open(img_path) as src:
        transform = src.transform
        p1 = transform * (0, 0)
        p2 = transform * (0, src.height)
        p3 = transform * (src.width, src.height)
        p4 = transform * (src.width, 0)

    data = {"name": [], "geometry": []}
    data["geometry"] += [Polygon([p1, p2, p3, p4])]
    data["name"] += [img_path.split("/")[-1]]
    gdf = gpd.GeoDataFrame(data, crs=src.crs)
    gdf.to_file(out_geojson, driver="GeoJSON")
