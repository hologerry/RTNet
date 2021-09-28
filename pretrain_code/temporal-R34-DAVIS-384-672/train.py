import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
from dataset import TemporalDataset, my_collate_fn, TemporalTestDataset
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
    epoch_num = 400
    batch_size_train = 12

    # Training Data
    datasets = ["../../DataSet/DAVIS/train"]
    spatial_dataset = TemporalDataset(datasets=datasets, H=None, W=None)
    spatial_dataloader = DataLoader(spatial_dataset, batch_size=batch_size_train, shuffle=True, num_workers=batch_size_train*2, pin_memory=True, drop_last=True, collate_fn=my_collate_fn)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(spatial_dataset), len(spatial_dataloader)))


    # ------- 3. define model --------
    # define the net
    net = Spatial().cuda()
    print("---define optimizer...")
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # ------- 5. training process --------
    optimizer.param_groups[0]['lr']=lr
    print("---start training...")
    decay_count=0
    iter_num = 0
    optimizer.zero_grad()
    for epoch in range(1, epoch_num):
        net.train()
        ite_num_per = 0
        running_loss = 0.0
        running_loss0 = 0.0
        for data in spatial_dataloader:
            iter_num+=1
            ite_num_per = ite_num_per + 1
            fwflow, bwflow, label = data['fw_flow'].cuda(), data['bw_flow'].cuda(),data['label'].cuda()
            # imwrite("fw.png", fwflow.cpu()[0].permute(1,2,0).numpy())
            # imwrite("bw.png", bwflow.cpu()[0].permute(1,2,0).numpy())
            # imwrite("label.png", label.cpu()[0, 0].numpy())
            # exit()
            out = net(torch.cat([fwflow, bwflow], 1))
            loss0, loss = train_loss.muti_bce_loss_fusion(out, label)
            running_loss += loss.item()  # total loss
            running_loss0 += loss0.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #if ite_num_per%500==0:
        print(
            "[epoch: {}/{}, iter: {}/{}, iter: {}] train loss: {:.5f}, loss0: {:.5f}".format(
                epoch, epoch_num, ite_num_per, len(spatial_dataloader), iter_num,
                running_loss / ite_num_per, running_loss0 / ite_num_per))
        if epoch % 4 == 0:
            torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                       model_dir + "epoch_{}_loss_{:.5f}_loss0_{:.5f}.pth".format(
                           epoch, running_loss / ite_num_per, running_loss0 / ite_num_per))
        if epoch >= 300:
            net.eval()
            decay_count+=1
            adjust_learning_rate(optimizer, decay_count)
            print(optimizer.param_groups[0]['lr'])