from logging import root
import os
from tkinter import Wm
import xmltodict
import dicttoxml
from typing import List


def crop_xml(
    xml_file: str, 
    size: List[int], 
    out_dir: str,
    overlap: float = 0.2,
    residual_crop = False
) -> None:

    xml_stem = os.path.splitext(os.path.basename(xml_file))[0]
    with open(xml_file, 'r') as fp:
        content = xmltodict.parse(
            fp.read(),
            force_list=('object')
        )
    
    width = int(content["annotation"]["size"]["width"])
    height = int(content["annotation"]["size"]["height"])
    channel = int(content["annotation"]["size"]["depth"])

    stride_x = int(size[0] * (1 - overlap))
    stride_y = int(size[1] * (1 - overlap))

    step_x = width // stride_x
    step_y = height // stride_y

    step_x_list = [stride_x * i for i in range(step_x)]
    step_y_list = [stride_y * j for j in range(step_y)]

    if residual_crop:
        if step_x_list[-1] != width - size[0]:
            step_x_list.append(width - size[0])
        if step_y_list[-1] != height - size[1]:
            step_y_list.append(height - size[1])

    for w_min in step_x_list:
        for h_min in step_y_list:
            w_max = w_min + size[0]
            h_max = h_min + size[1]
            fname = f"{xml_stem}_{w_min}_{h_min}.xml"
            out_xml_path = os.path.join(out_dir, fname)
            img_area = [w_min, h_min, w_max, h_max]

            xml_dict = create_default_xml()
            xml_dict["annotation"]["size"]["width"] = size[0]
            xml_dict["annotation"]["size"]["height"] = size[1]
            xml_dict["annotation"]["size"]["depth"] = channel
            xml_dict["annotation"]["filename"] = fname
            xml_dict["annotation"]["path"] = out_xml_path

            for target in content["annotation"]["object"]:
                bbox = convert_bndbox_to_list(target["bndbox"])
                if check_bbox_inside(img_area, bbox):
                    xml_dict["annotation"]["object"].append(target)
            if len(xml_dict["annotation"]["object"]) > 0:
                with open(out_xml_path, 'w') as fp:
                    fp.write(
                        dicttoxml.dicttoxml(
                            xml_dict, 
                            attr_type=False, 
                            root=False
                        ).decode("utf-8")
                    )

def check_bbox_inside(img_area: List[int], bbox: List[int]) -> bool:
    if (
        img_area[0] <= bbox[0] and  # x_min
        img_area[1] <= bbox[1] and  # y_min
        img_area[2] >= bbox[2] and  # x_max
        img_area[3] >= bbox[3]      # y_max
    ):
        return True
    else:
        return False
    

def convert_bndbox_to_list(bndbox: dict) -> List[int]:
    x_min = int(bndbox["xmin"])
    y_min = int(bndbox["ymin"])
    x_max = int(bndbox["xmax"])
    y_max = int(bndbox["ymax"])
    return [x_min, y_min, x_max, y_max]


def create_default_xml() -> dict:
    return {
        "annotation": {
            "folder": "JPEGImages",
            "filename": "",
            "path": "",
            "source": {
                "database": "Unknown"
            },
            "size":{
                "width": 0,
                "height": 0,
                "depth": 0
            },
            "segmented": 0,
            "object": []
        }
    }