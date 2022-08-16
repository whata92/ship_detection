import os
import xmltodict
import dicttoxml
from PIL import Image
from typing import List
import numpy as np
import logging


logger = logging.getLogger("global.preprocessing_utils")

Image.MAX_IMAGE_PIXELS = 1000000000

def crop_image(
    img_file: str,
    size: List[int],
    out_dir: str,
    overlap: float = 0.2,
    residual_crop = False
) -> None:

    img_stem = os.path.splitext(os.path.basename(img_file))[0]

    stride_x = int(size[0] * (1 - overlap))
    stride_y = int(size[1] * (1 - overlap))

    img_np = np.array(Image.open(img_file))
    [height, width, _] = img_np.shape

    step_x = height // stride_x
    step_y = width // stride_y

    step_x_list = [stride_x * i for i in range(step_x)]
    step_y_list = [stride_y * j for j in range(step_y)]

    if residual_crop:
        if step_x_list[-1] != height - size[0]:
            step_x_list.append(height - size[0])
        if step_y_list[-1] != width - size[1]:
            step_y_list.append(width - size[1])

    for h_min in step_x_list:
        for w_min in step_y_list:
            h_max = h_min + size[0]
            w_max = w_min + size[1]
            cropped = img_np[h_min: h_max, w_min: w_max, :]

            fname = f"{img_stem}_{h_min}_{w_min}_{h_max}_{w_max}.jpg"
            out_img_path = os.path.join(out_dir, fname)

            img_pil = Image.fromarray(cropped)
            img_pil.save(out_img_path)
            logger.info(f"Processed: {out_img_path}")


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

    height = int(content["annotation"]["size"]["height"])
    width = int(content["annotation"]["size"]["width"])
    channel = int(content["annotation"]["size"]["depth"])

    stride_x = int(size[0] * (1 - overlap))
    stride_y = int(size[1] * (1 - overlap))

    step_x = height // stride_x
    step_y = width // stride_y

    step_x_list = [stride_x * i for i in range(step_x)]
    step_y_list = [stride_y * j for j in range(step_y)]

    if residual_crop:
        if step_x_list[-1] != height - size[0]:
            step_x_list.append(height - size[0])
        if step_y_list[-1] != width - size[1]:
            step_y_list.append(width - size[1])

    for h_min in step_x_list:
        for w_min in step_y_list:
            h_max = h_min + size[0]
            w_max = w_min + size[1]
            fname = f"{xml_stem}_{h_min}_{w_min}_{h_max}_{w_max}.xml"
            out_xml_path = os.path.join(out_dir, fname)
            img_area = [h_min, w_min, h_max, w_max]

            xml_dict = create_default_xml()
            xml_dict["annotation"]["size"]["height"] = size[0]
            xml_dict["annotation"]["size"]["width"] = size[1]
            xml_dict["annotation"]["size"]["depth"] = channel
            xml_dict["annotation"]["filename"] = fname
            xml_dict["annotation"]["path"] = out_xml_path

            for target in content["annotation"]["object"]:
                bbox = convert_bndbox_to_list(target["bndbox"])
                if check_bbox_inside(img_area, bbox):
                    target["name"] = "ship"
                    target["bndbox"]["xmin"] = int(target["bndbox"]["xmin"]) - w_min
                    target["bndbox"]["ymin"] = int(target["bndbox"]["ymin"]) - h_min
                    target["bndbox"]["xmax"] = int(target["bndbox"]["xmax"]) - w_min
                    target["bndbox"]["ymax"] = int(target["bndbox"]["ymax"]) - h_min
                    xml_dict["annotation"]["object"].append(target)
            if len(xml_dict["annotation"]["object"]) > 0:
                logger.info(f'{fname}: {len(xml_dict["annotation"]["object"])}')
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
    x_min = int(bndbox["ymin"])  # bndbox's y axis -> vertical
    y_min = int(bndbox["xmin"])  # bndbox's x axis -> horizontal
    x_max = int(bndbox["ymax"])
    y_max = int(bndbox["xmax"])
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


def item2object(xml_file: str):
    with open(xml_file, "r") as fp:
        original = fp.read()
    if "<item>" not in original:
        logger.warn(f"No item exists: {xml_file}")
        return
    original = original.replace("<object>", "")
    original = original.replace("</object>", "")
    original = original.replace("<item>", "<object>")
    original = original.replace("</item>", "</object>")
    os.remove(xml_file)
    with open(xml_file, "w") as fp:
        fp.write(original)
    logger.info(f"Processed: {xml_file}")
