import os
import sys
import shutil

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + "/../"
    )
)

from src.utils.preprocessing_utils import (
    crop_xml,
    check_bbox_inside
)

def test_crop_xml(default_path):
    xml_file = default_path["xml"]
    size = [512, 512]
    overlap = 0
    out_dir = "tests/tmp"
    os.makedirs(out_dir, exist_ok=True)

    assert xml_file == "tests/dummy_data/test.xml"
    try:
        crop_xml(xml_file, size, out_dir, overlap, residual_crop=False)
    finally:
        shutil.rmtree(out_dir)


def test_bbox_inside():
    img_area = [100, 100, 500, 500]
    bbox = [110, 120, 400, 410]

    assert check_bbox_inside(img_area, bbox)