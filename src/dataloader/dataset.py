import torch
import os
import PIL.Image as Image
import pandas as pd
import numpy as np
from skimage.measure import regionprops
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List

from .augmentation import (
    get_train_transforms,
    get_valid_transforms
)


class AirbusDS(Dataset):
    """
    A customized data loader.
    """
    def __init__(
        self, dataset_csv, dataset_root, aug=False, mode='train'
    ):
        """ Intialize the dataset
        """
        self.filenames = []
        self.root = dataset_root
        self.aug = aug
        self.mode = 'test'
        if mode in ['train', 'val']:
            self.mode = mode
            self.masks = pd.read_csv(dataset_csv).fillna(-1)
        if self.aug:
            self.transform = get_train_transforms()
        else:
            self.transform = get_valid_transforms()

        self.filenames = [
            os.path.join(self.root, fstem)
            for fstem in list(self.masks["ImageId"])
        ]
        self.len = len(self.filenames)

    # You must override __getitem__ and __len__
    def get_mask_boxes(self, ImageId):
        img_masks = self.masks.loc[
            self.masks['ImageId'] == ImageId, 'EncodedPixels'
        ].tolist()
        bboxes = []
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        if img_masks == [-1]:
            return all_masks, bboxes
        for mask in img_masks:
            target_mask = rle_decode(mask)
            props = regionprops(target_mask)
            for prop in props:
                bboxes.append(
                    [prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]]
                )
            all_masks += target_mask
        return all_masks, bboxes

    def calc_area(self, bboxes, mode="pascal_voc"):
        area = []
        if mode == "pascal_voc":
            for bbox in bboxes:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area.append(w * h)
        else:
            raise NotImplementedError()

        return area

    def __getitem__(self, idx):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[idx])
        ImageId = self.filenames[idx].split('/')[-1]
        if self.mode in ['train', 'val']:
            mask, bbox = self.get_mask_boxes(ImageId)
        if self.aug:
            if self.mode in ['train', 'val']:
                data = {
                    "image": np.array(image),
                    "mask": mask,
                    "bboxes": bbox,
                    "labels": np.array(["ship" for _ in bbox]),
                }
            else:
                data = {"image": np.array(image)}
            transformed = self.transform(**data)
            image = transformed["image"]
            if self.mode in ['train', 'val']:
                num_objs = len(bbox)
                target = {}
                target["boxes"] = torch.as_tensor(
                    transformed["bboxes"], dtype=torch.float32
                )
                target["masks"] = (
                    transformed["mask"][np.newaxis, : , :].to(torch.uint8)
                )
                target["labels"] = torch.ones((num_objs, ), dtype=torch.int64)
                target["area"] = torch.Tensor(self.calc_area(transformed["bboxes"]))
                target["iscrowd"] = torch.zeros((num_objs, ), dtype=torch.int64)
                target["image_id"] = torch.Tensor([idx])
                return image, target
            else:
                return image
        else:
            raise NotImplementedError()

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def rle_decode(mask_rle, shape=(768, 768)):
    '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns
            - numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def collate_fn(batch):
    images, labels = tuple(zip(*batch))
    labels = [{k: v for k, v in t.items()} for t in labels]
    return [images, labels]


def get_dataloader(
    dataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Callable[[List], List] = collate_fn
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader