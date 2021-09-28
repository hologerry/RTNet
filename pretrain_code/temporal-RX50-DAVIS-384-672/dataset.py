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
class TemporalDataset(Dataset):
    def __init__(self, datasets, H, W):
        self.datasets = datasets
        self.H = H
        self.W = W
        self.fwflow_list = []
        self.bwflow_list = []
        self.label_list = []
        for dataset in self.datasets:
            path = os.path.join(dataset, "label_flip")
            file=sorted(os.listdir(path))
            for i in file:
                self.fwflow_list.append(os.path.join(dataset, "flow_img_flip", "fw_"+i))
                self.bwflow_list.append(os.path.join(dataset, "flow_img_flip", "bw_"+i))
                self.label_list.append(os.path.join(dataset, "label_flip", i))
    def __len__(self):
        return len(self.fwflow_list)
    def __getitem__(self, item):
        fw_flow = imread(self.fwflow_list[item])
        bw_flow = imread(self.bwflow_list[item])
        label = imread(self.label_list[item])
        if len(label.shape) == 3:
             label = label[:, :, 0]
        fw_flow = img_normalize(fw_flow.astype(np.float32) / 255.)
        bw_flow = img_normalize(bw_flow.astype(np.float32) / 255.)
        label = label.astype(np.float32)[:, :, np.newaxis] / 255.
        H, W, _ = bw_flow.shape
        fw_flow = torch.from_numpy(fw_flow.transpose((2, 0, 1)))
        bw_flow = torch.from_numpy(bw_flow.transpose((2, 0, 1)))
        label = torch.from_numpy(label.transpose((2, 0, 1)))
        return {'fw_flow': fw_flow, "bw_flow":bw_flow, 'label': label, 'H': H, 'W': W}

class TemporalTestDataset(Dataset):
    def __init__(self, datasets, H, W):
        self.datasets = datasets
        self.H = H
        self.W = W
        self.fwflow_list = []
        self.bwflow_list = []
        self.label_list = []
        for dataset in self.datasets:
            path = os.path.join(dataset, "label")
            file=os.listdir(path)
            for i in file:
                self.fwflow_list.append(os.path.join(dataset, "flow_img", "fw_"+i))
                self.bwflow_list.append(os.path.join(dataset, "flow_img", "bw_" + i))
                self.label_list.append(os.path.join(dataset, "label", i))

    def __len__(self):
        return len(self.fwflow_list)
    def __getitem__(self, item):
        fwflow = imread(self.fwflow_list[item])
        bwflow = imread(self.bwflow_list[item])
        label = imread(self.label_list[item])
        if len(label.shape) == 3:
             label = label[:, :, 0]
        fwflow = img_normalize(fwflow.astype(np.float32) / 255.)
        bwflow = img_normalize(bwflow.astype(np.float32) / 255.)
        label = label.astype(np.float32)[:, :, np.newaxis] / 255.
        label_org = label.copy()
        H, W, _ = fwflow.shape
        fwflow = F.interpolate(torch.from_numpy(fwflow.transpose((2, 0, 1))).unsqueeze(0), (self.H, self.W), mode='bilinear', align_corners=True).squeeze(0)
        bwflow = F.interpolate(torch.from_numpy(bwflow.transpose((2, 0, 1))).unsqueeze(0), (self.H, self.W),
                               mode='bilinear', align_corners=True).squeeze(0)
        label = F.interpolate(torch.from_numpy(label.transpose((2, 0, 1))).unsqueeze(0), (self.H, self.W), mode='bilinear', align_corners=True).squeeze(0)
        return {'fwflow': fwflow, 'bwflow': bwflow,
                'label': label, 'H': H, 'W': W, 'name': item, 'label_org': torch.from_numpy(label_org).permute(2,0,1)}

def my_collate_fn(batch):
    size = [256, 320, 384]
    H = size[np.random.randint(0, 3)]
    W = int(1.75 * H)
    fwflow = []
    bwflow = []
    label = []
    for item in batch:
        fwflow.append(F.interpolate(item['fw_flow'].unsqueeze(0), (H, W), mode='bilinear', align_corners=True))
        bwflow.append(F.interpolate(item['bw_flow'].unsqueeze(0), (H, W), mode='bilinear', align_corners=True))
        label.append(F.interpolate(item['label'].unsqueeze(0), (H, W), mode='bilinear', align_corners=True))
    return {'fw_flow': torch.cat(fwflow, 0), 'bw_flow': torch.cat(bwflow, 0), 'label': torch.cat(label, 0)}