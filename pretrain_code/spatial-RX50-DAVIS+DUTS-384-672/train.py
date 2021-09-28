import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
from dataset import SpatialDataset, SpatialTestDataset, my_collate_fn
from model import Spatial
import os
import train_loss
import torch.nn.functional as F
import numpy as np
from S_measure import SMeasure
from imageio import imwrite

if __name__ == '__main__':
    lr = 1e-3

    def adjust_learning_rate(optimizer, decay_count, decay_rate=.9):
        optimizer.param_groups[0]['lr'] = max(lr * pow(decay_rate, decay_count), 1e-4)

    # ------- 2. set the directory of training dataset --------
    model_dir = "./ckpt/"
    if not os.path.isdir("ckpt"):
        os.mkdir("ckpt")
    epoch_num = 50
    batch_size_train = 8

    # Training Data
    datasets = ["../../DataSet//DAVIS/train", "../../DataSet/DUTS/train"]
    spatial_dataset = SpatialDataset(datasets=datasets, H=None, W=None)
    spatial_dataloader = DataLoader(spatial_dataset, batch_size=batch_size_train, shuffle=True, num_workers=batch_size_train*2, pin_memory=True, drop_last=True, collate_fn=my_collate_fn)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(spatial_dataset), len(spatial_dataloader)))

    # ------- 3. define model --------
    # define the net
    net = Spatial().cuda()
    print("---define optimizer...")
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    re_load = False
    # ------- 5. training process --------
    print("---start training...")
    optimizer.zero_grad()
    for epoch in range(1, epoch_num):
        iter_num = 0
        running_loss, running_loss0 = 0.0, 0.0
        if epoch>20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(1e-4, pow(0.9, epoch-20)*lr)
        net.train()
        ite_num_per=0
        for data in spatial_dataloader:
            optimizer.zero_grad()
            ite_num_per = ite_num_per + 1
            iter_num+=1
            img, label = data['image'].cuda(), data['label'].cuda()
            out = net(img)
            loss0, loss = train_loss.muti_bce_loss_fusion(out, label)
            running_loss += loss.item()  # total loss
            running_loss0 += loss0.item()
            loss.backward()
            optimizer.step()
            if ite_num_per%500==0:
                print(
                    "[epoch: {}/{}, iter: {}/{}, iter: {}] train loss: {:.5f}, loss0: {:.5f}".format(
                        epoch, epoch_num, ite_num_per, len(spatial_dataloader), iter_num,
                        running_loss / ite_num_per, running_loss0 / ite_num_per))
        torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                   model_dir + "epoch_{}_loss_{:.5f}_loss0_{:.5f}.pth".format(
                       epoch, running_loss / ite_num_per, running_loss0 / ite_num_per))