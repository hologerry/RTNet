import torch
from imageio import imread
import numpy as np
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image
class SpatialDataset(Dataset):
    def __init__(self, datasets, H, W):
        self.datasets = datasets
        self.H = H
        self.W = W
        self.image_list = []
        self.label_list = []
        for dataset in self.datasets:
            path = os.path.join(dataset, "img_flip")
            file = sorted(os.listdir(path))
            for i in file:
                self.image_list.append(os.path.join(dataset, "img_flip", i))
                self.label_list.append(os.path.join(dataset, "label_flip", i[:-3] + "png"))
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, item):
        image = imread(self.image_list[item])
        label = imread(self.label_list[item])
        if len(label.shape) == 3:
             label = label[:, :, 0]
        image = img_normalize(image.astype(np.float32) / 255.)
        label = label.astype(np.float32)[:, :, np.newaxis] / 255.
        H, W, _ = image.shape
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.from_numpy(label.transpose((2, 0, 1)))
        return {'image': image, 'label': label, 'H': H, 'W': W}

class SpatialTestDataset(Dataset):
    def __init__(self, datasets, H, W):
        self.datasets = datasets
        self.H = H
        self.W = W
        self.image_list = []
        self.label_list = []
        for dataset in self.datasets:
            path = os.path.join(dataset, "img")
            file=os.listdir(path)
            for i in file:
                self.image_list.append(os.path.join(dataset, "img", i))
                self.label_list.append(os.path.join(dataset, "label", i[:-3]+"png"))

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, item):
        image = imread(self.image_list[item])
        label = imread(self.label_list[item])
        if len(label.shape) == 3:
             label = label[:, :, 0]
        image = img_normalize(image.astype(np.float32) / 255.)
        label = label.astype(np.float32)[:, :, np.newaxis] / 255.
        label_org = label.copy()
        H, W, _ = image.shape
        image = F.interpolate(torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0), (self.H, self.W), mode='bilinear', align_corners=True).squeeze(0)
        label = F.interpolate(torch.from_numpy(label.transpose((2, 0, 1))).unsqueeze(0), (self.H, self.W), mode='bilinear', align_corners=True).squeeze(0)
        return {'image': image, 'label': label, 'H': H, 'W': W, 'name': item, 'label_org': torch.from_numpy(label_org).permute(2,0,1)}

def my_collate_fn(batch):
    size = [256, 320, 384]
    H = size[np.random.randint(0, 3)]
    W = int(1.75 * H)
    img = []
    label = []
    for item in batch:
        img.append(F.interpolate(item['image'].unsqueeze(0), (H, W), mode='bilinear', align_corners=True))
        label.append(F.interpolate(item['label'].unsqueeze(0), (H, W), mode='bilinear', align_corners=True))
    return {'image': torch.cat(img, 0), 'label': torch.cat(label, 0)}