import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO

from .augmentation import default_augmentation


def collate_fn(batch):
    images, labels = tuple(zip(*batch))
    labels = [{k: v for k, v in t.items()} for t in labels]
    return [images, labels]


def get_dataloader(
    dataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn=collate_fn
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader


class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.imgs_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = default_augmentation()

    def __getitem__(self, idx):
        '''
        Args:
            idx: index of sample to be fed
        return:
            dict containing:
            - PIL Image of shape (H, W)
            - target (dict) containing:
                - boxes:    FloatTensor[N, 4],
                    N being the n° of instances and it's bounding
                    boxe coordinates in [x0, y0, x1, y1] format,
                    ranging from 0 to W and 0 to H;
                - labels:   Int64Tensor[N], class label (0 is background);
                - image_id: Int64Tensor[1], unique id for each image;
                - area:     Tensor[N], area of bbox;
                - iscrowd:  UInt8Tensor[N], True or False;
                - masks:    UInt8Tensor[N, H, W], segmantation maps;
        '''
        img_id = self.img_ids[idx]
        img_obj = self.coco.loadImgs(img_id)[0]
        anns_obj = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        num_obj = len(anns_obj)

        img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name']))
        img = np.array(img)

        bboxes = [
            [
                ann['bbox'][0],
                ann['bbox'][1],
                ann['bbox'][0] + ann['bbox'][2],
                ann['bbox'][1] + ann['bbox'][3]
            ] for ann in anns_obj
        ]
        masks = [self.coco.annToMask(ann) for ann in anns_obj]
        areas = [ann['area'] for ann in anns_obj]

        target = {}
        if self.transforms is not None:
            masks = np.sum(np.array(masks), axis=0).astype(np.uint8)
            data = {
                "image": img,
                "mask": masks[:, :, np.newaxis],
                "bboxes": bboxes,
                "category_id": np.array(["ship" for _ in bboxes])
            }
            transformed = self.transforms(**data)
            img = torch.as_tensor(transformed["image"], dtype=torch.float32)
            target["boxes"] = transformed["bboxes"]
            target["masks"] = transformed["mask"]
        else:
            target["boxes"] = bboxes
            target["masks"] = np.array(masks)
        target["labels"] = torch.ones((num_obj, ), dtype=torch.int64)

        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["masks"] = torch.as_tensor(
            target["masks"].permute(1, 2, 0),
            dtype=torch.uint8
        )
        target["image_id"] = torch.tensor([img_id])
        target["area"] = torch.as_tensor(areas)
        target["iscrowd"] = torch.zeros(len(anns_obj), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.img_ids)
