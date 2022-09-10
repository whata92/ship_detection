from shapely.geometry import Polygon
import geopandas as gpd


def nms(gdf: gpd.GeoDataFrame, iou_thresh: float) -> gpd.GeoDataFrame:
    gdf.sort_values(by='confidence', inplace=True, ascending=False)
    geometries = list(gdf['geometry'])
    duplicate = [0] * len(gdf)
    for i in range(len(gdf)):
        shape1 = geometries[i]
        for j in range(i + 1, len(gdf)):
            shape2 = geometries[j]
            if iou(shape1, shape2) >= iou_thresh:
                duplicate[j] = 1
    gdf['duplicate'] = duplicate
    return gdf


def iou(polygon1: Polygon, polygon2: Polygon) -> float:
    union = polygon1.union(polygon2)
    intersection = polygon1.intersection(polygon2)

    return intersection.area / union.area
