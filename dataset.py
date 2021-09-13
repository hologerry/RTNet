import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import imread
from torch.utils.data import Dataset


def img_normalize(image):
    if len(image.shape) == 2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel, channel, channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
            / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image


class RTDataset(Dataset):
    def __init__(self, pathes, T, size):
        self.fwflow_list = []
        self.bwflow_list = []
        self.img_list = []
        self.label_list = []
        self.size = size
        self.T = T
        for path in pathes:
            file = sorted(os.listdir(os.path.join(path, "img_flip")))
            for i in file:
                self.img_list.append(os.path.join(path, "img_flip", i))
                self.label_list.append(os.path.join(path, "label_flip", i))
                self.fwflow_list.append(os.path.join(path, "flow_img_flip", "fw_"+i))
                self.bwflow_list.append(os.path.join(path, "flow_img_flip", "bw_"+i))
        self.dataset_len = len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        frame = [item]
        scope = 40
        other = np.random.randint(-scope, scope)
        while item + other >= self.dataset_len or item+other < 0 or other == 0:
            other = np.random.randint(-scope, scope)
        name1 = self.img_list[item]
        name2 = self.img_list[item+other]
        while name1.split('/')[-1].split("_")[0] != name2.split('/')[-1].split("_")[0] or name1.split('/')[-1].split("_")[-1] != name2.split('/')[-1].split("_")[-1]:
            other = np.random.randint(-scope, scope)
            while item + other >= self.dataset_len or item + other < 0 or other == 0:
                other = np.random.randint(-scope, scope)
            name2 = self.img_list[item + other]
        frame.append(item+other)
        videos, labels, fwflows, bwflows = [], [], [], []
        for i in frame:
            video = imread(self.img_list[i])
            fw = imread(self.fwflow_list[i])
            bw = imread(self.bwflow_list[i])
            label = imread(self.label_list[i])
            if len(label.shape) == 3:
                label = label[:, :, 0]
            label = label[:, :, np.newaxis]
            videos.append(img_normalize(video.astype(np.float32)/255.))
            labels.append(label.astype(np.float32)/255.)
            fwflows.append(img_normalize(fw.astype(np.float32)/255.))
            bwflows.append(img_normalize(bw.astype(np.float32) / 255.))
        video = torch.from_numpy(np.stack(videos, 0)).permute(0, 3, 1, 2)
        label = torch.from_numpy(np.stack(labels, 0)).permute(0, 3, 1, 2)
        fwflow = torch.from_numpy(np.stack(fwflows, 0)).permute(0, 3, 1, 2)
        bwflow = torch.from_numpy(np.stack(bwflows, 0)).permute(0, 3, 1, 2)
        if self.size is None:
            return {'video': video,
                    'label': label,
                    'fwflow': fwflow,
                    'bwflow': bwflow}
        else:
            return {'video': F.interpolate(video, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'label': F.interpolate(label, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'fwflow': F.interpolate(fwflow, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'bwflow': F.interpolate(bwflow, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True), }


def load_images(img_list, label_list, fwflow_list, bwflow_list, dataset_name, seq_path):
    images = glob.glob(os.path.join(seq_path, '*.jpg'))
    images = sorted(images)
    images.pop(0)  # pop first frame
    images.pop(-1)  # pop last frame
    labels = [img.replace('JPEGImages', 'Annotations').replace('.jpg', '.png') for img in images]
    fwflows = [img.replace(dataset_name, dataset_name+'_flow_fw').replace('.jpg', '.png') for img in images]
    bwflows = [img.replace(dataset_name, dataset_name+'_flow_bw').replace('.jpg', '.png') for img in images]
    img_list += images
    label_list += labels
    fwflow_list += fwflows
    bwflow_list += bwflows


class RTSegDataset(Dataset):
    def __init__(self, data_root, dataset_names, sub_datasets, size, scope=40, fw_only=False):
        self.fwflow_list = []
        self.bwflow_list = []
        self.img_list = []
        self.label_list = []
        self.size = size
        self.scope = scope
        self.fw_only = fw_only

        # ['PSEG', 'PSEG_v_flip', 'PSEG_h_flip', 'PSEG_hv_flip']
        self.dataset_names = dataset_names
        # ['blender_old', 'gen_mobilenet']
        self.sub_datasets = sub_datasets

        for dataset_name in self.dataset_names:
            for sub_dataset in self.sub_datasets:
                jpeg_path = os.path.join(data_root, dataset_name, sub_dataset, 'JPEGImages', '480p')

                if sub_dataset == 'blender_old':
                    sequences = os.listdir(jpeg_path)
                    for seq in sequences:
                        seq_path = os.path.join(jpeg_path, seq)
                        load_images(self.img_list, self.label_list, self.fwflow_list, self.bwflow_list, dataset_name, seq_path)

                elif sub_dataset == 'gen_mobilenet':
                    challenges = os.listdir(jpeg_path)
                    for cha in challenges:
                        sequences = os.listdir(os.path.join(jpeg_path, cha))
                        for seq in sequences:
                            seq_path = os.path.join(jpeg_path, cha, seq)
                            load_images(self.img_list, self.label_list, self.fwflow_list, self.bwflow_list, dataset_name, seq_path)


        self.dataset_len = len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        frame = [item]
        low = -self.scope
        high = self.scope
        other = np.random.randint(low, high)
        # prevent from index error
        while item + other >= self.dataset_len or item+other < 0 or other == 0:
            other = np.random.randint(low, high)
        name1 = self.img_list[item]
        name2 = self.img_list[item+other]
        # must from the same sequence
        while name1.split('/')[-2] != name2.split('/')[-2]:
            other = np.random.randint(low, high)
            while item + other >= self.dataset_len or item + other < 0 or other == 0:
                other = np.random.randint(low, high)
            name2 = self.img_list[item + other]

        frame.append(item+other)

        videos, labels, fwflows, bwflows = [], [], [], []
        for i in frame:
            video = imread(self.img_list[i])
            fw = imread(self.fwflow_list[i])
            bw = fw if self.fw_only else imread(self.bwflow_list[i])
            label = imread(self.label_list[i])

            assert video is not None, f"video is none {self.img_list[i]}"
            assert fw is not None, f"fw is none {self.fwflow_list[i]}"
            assert bw is not None, f"bw is none {self.bwflow_list[i]}"
            assert label is not None, f"label is none {self.label_list[i]}"

            label_sum = np.sum(label, axis=2)
            label_sum_mask = (label_sum > 0) * 255.0
            label = label_sum_mask[:, :, np.newaxis]
            videos.append(img_normalize(video.astype(np.float32) / 255.))
            labels.append(label.astype(np.float32) / 255.)
            fwflows.append(img_normalize(fw.astype(np.float32) / 255.))
            bwflows.append(img_normalize(bw.astype(np.float32) / 255.))

        video = torch.from_numpy(np.stack(videos, 0)).permute(0, 3, 1, 2)
        label = torch.from_numpy(np.stack(labels, 0)).permute(0, 3, 1, 2)
        fwflow = torch.from_numpy(np.stack(fwflows, 0)).permute(0, 3, 1, 2)
        bwflow = torch.from_numpy(np.stack(bwflows, 0)).permute(0, 3, 1, 2)

        if self.size is None:
            return {'video': video,
                    'label': label,
                    'fwflow': fwflow,
                    'bwflow': bwflow}
        else:
            return {'video': F.interpolate(video, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'label': F.interpolate(label, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'fwflow': F.interpolate(fwflow, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'bwflow': F.interpolate(bwflow, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True), }


class RTTestDataset(Dataset):
    def __init__(self, pathes, T, H, W):
        self.fwflow_list = []
        self.bwflow_list = []
        self.img_list = []
        self.label_list = []
        self.T = T
        self.H, self.W = H, W
        for path in pathes:
            file = sorted(os.listdir(os.path.join(path, "img")))
            for i in file:
                self.img_list.append(os.path.join(path, "img", i))
                self.label_list.append(os.path.join(path, "label", i[:-3] + "png"))
                self.fwflow_list.append(os.path.join(path, "flow_img", "fw_" + i[:-3] + "png"))
                self.bwflow_list.append(os.path.join(path, "flow_img", "bw_" + i[:-3] + "png"))
        self.dataset_len = len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        frame = [item]
        scope = 10
        other = np.random.randint(-scope, scope)
        while item + other >= self.dataset_len or item + other < 0 or other == 0:
            other = np.random.randint(-scope, scope)
        name1 = self.img_list[item]
        name2 = self.img_list[item + other]
        while name1.split('/')[-1].split("_")[0] != name2.split('/')[-1].split("_")[0]:
            other = np.random.randint(-scope, scope)
            while item + other >= self.dataset_len or item + other < 0 or other == 0:
                other = np.random.randint(-scope, scope)
            name2 = self.img_list[item + other]
        frame.append(item + other)
        videos, labels, fwflows, bwflows = [], [], [], []
        for i in frame:
            video = imread(self.img_list[i])
            fw = imread(self.fwflow_list[i])
            bw = imread(self.bwflow_list[i])
            label = imread(self.label_list[i])
            if len(label.shape) == 3:
                label = label[:, :, 0]
            label = label[:, :, np.newaxis]
            videos.append(img_normalize(video.astype(np.float32) / 255.))
            labels.append(label.astype(np.float32) / 255.)
            fwflows.append(img_normalize(fw.astype(np.float32) / 255.))
            bwflows.append(img_normalize(bw.astype(np.float32) / 255.))
            H, W = labels[0].shape[0], labels[0].shape[1]
        return {'video': F.interpolate(torch.from_numpy(np.stack(videos, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                'fwflow': F.interpolate(torch.from_numpy(np.stack(fwflows, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                'bwflow': F.interpolate(torch.from_numpy(np.stack(bwflows, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                "label_org": torch.from_numpy(np.stack([labels[0]], 0)).permute(0, 3, 1, 2),
                "H": H, "W": W, 'name': self.img_list[item].split("/")[-1]}


class RTTestNoLabelDataset(Dataset):
    def __init__(self, path, T, H, W, fw_only=False):
        self.fwflow_list = []
        self.bwflow_list = []
        self.img_list = []
        self.label_list = []
        self.T = T
        self.H, self.W = H, W
        self.fw_only = fw_only
        files = sorted(os.listdir(os.path.join(path, "img")))
        for filename in files:
            self.img_list.append(os.path.join(path, "img", filename))
            self.label_list.append(os.path.join(path, "img", filename))  # use the image as a placeholder
            self.fwflow_list.append(os.path.join(path, "fw_img", filename[:-3] + "png"))
            self.bwflow_list.append(os.path.join(path, "bw_img", filename[:-3] + "png"))
        self.dataset_len = len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        frame = [item]
        scope = 10

        other = np.random.randint(-scope, scope)
        while item + other >= self.dataset_len or item + other < 0 or other == 0:
            other = np.random.randint(-scope, scope)

        # other = -1 if item > 0 else 1

        frame.append(item + other)
        videos, labels, fwflows, bwflows = [], [], [], []

        for i in frame:
            video = imread(self.img_list[i])
            fw = imread(self.fwflow_list[i])
            bw = fw if self.fw_only else imread(self.bwflow_list[i])
            label = imread(self.label_list[i])
            if len(label.shape) == 3:
                label = label[:, :, 0]
            label = label[:, :, np.newaxis]
            videos.append(img_normalize(video.astype(np.float32) / 255.))
            labels.append(label.astype(np.float32) / 255.)
            fwflows.append(img_normalize(fw.astype(np.float32) / 255.))
            bwflows.append(img_normalize(bw.astype(np.float32) / 255.))
            H, W = labels[0].shape[0], labels[0].shape[1]

        return {'video': F.interpolate(torch.from_numpy(np.stack(videos, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                'fwflow': F.interpolate(torch.from_numpy(np.stack(fwflows, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                'bwflow': F.interpolate(torch.from_numpy(np.stack(bwflows, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                "label_org": torch.from_numpy(np.stack([labels[0]], 0)).permute(0, 3, 1, 2),
                "H": H, "W": W, 'name': self.img_list[item].split("/")[-1]}
