import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import imwrite
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RTTestNoLabelDataset
from model_RX50 import Interactive


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    setup_seed(1024)
    model_dir = "./saved_model/"
    results_dir = 'results/results_model_RX50_bwflow_only'
    os.makedirs(results_dir, exist_ok=True)
    batch_size_val = 1
    dataset = "../data/object_test"

    DAVIS_dataset = RTTestNoLabelDataset(dataset, 2, 384, int(384 * 1.75))
    DAVIS_dataloader = DataLoader(DAVIS_dataset, batch_size=1, shuffle=False, num_workers=4)

    net = Interactive().cuda()
    model_name = 'model_RX50.pth'
    ckpt = torch.load(model_dir + model_name)['state_dict']
    model_dict = net.state_dict()
    pretrained_dict = {k[7:]: v for k, v in ckpt.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()

    for data in tqdm(DAVIS_dataloader):
        img, fw_flow, bw_flow, label_org = data['video'].cuda(), data['fwflow'].cuda(), data['bwflow'].cuda(), data['label_org'].cuda()
        _, _, _, H, W = label_org.size()
        flow = torch.cat((fw_flow, bw_flow), 2)
        with torch.no_grad():
            out, _ = net(img, flow)
        out = F.interpolate(out[0], (H, W), mode='bilinear', align_corners=True)
        out = out[0, 0].cpu().numpy()
        out = (out - np.min(out) + 1e-12) / (np.max(out) - np.min(out) + 1e-12) * 255.
        out = out.astype(np.uint8)
        imwrite(os.path.join(results_dir, data['name'][0]), out)
