import os
import sys
import geopandas as gpd
from shapely.geometry import Polygon

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + "/../../"
    )
)

from src.utils.general_utils import nms, iou


def test_iou():
    polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
    polygon2 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])

    assert iou(polygon1, polygon2) == 1/7


def test_nms():
    gdf = gpd.GeoDataFrame(
        {
            'geometry': [
                Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]),
                Polygon([(0, 0), (0, 1.9), (1.9, 1.9), (1.9, 0), (0, 0)]),
                Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
            ],
            'confidence': [1.0, 0.5, 0.3],
        }
    )

    nms(gdf, 0.8) == gpd.GeoDataFrame(
        {
            'geometry': [
                Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]),
                Polygon([(0, 0), (0, 1.9), (1.9, 1.9), (1.9, 0), (0, 0)]),
                Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
            ],
            'confidence': [1.0, 0.5, 0.3],
            'duplicate': [0, 1, 0]
        }
    )