import argparse
import glob
import json
import os
import random
from shutil import copyfile

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# from Measurement import SMeasure

import train_loss
from dataset import RTSegDataset
from logger import setup_logger
from model_R34 import Interactive


def my_collate_fn(batch):
    size = [256, 320, 384]
    H = size[np.random.randint(0, 3)]
    W = int(1.75*H)
    img = []
    label = []
    fw_flow = []
    bw_flow = []
    for item in batch:
        img.append(F.interpolate(item['video'], (H, W), mode='bilinear', align_corners=True))
        label.append(F.interpolate(item['label'], (H, W), mode='bilinear', align_corners=True))
        bw_flow.append(F.interpolate(item['bwflow'], (H, W), mode='bilinear', align_corners=True))
        fw_flow.append(F.interpolate(item['fwflow'], (H, W), mode='bilinear', align_corners=True))
    return {'video': torch.stack(img, 0), 'label': torch.stack(label, 0),
            "bwflow": torch.stack(bw_flow, 0), "fwflow": torch.stack(fw_flow, 0)}


def adjust_learning_rate(optimizer, decay_count, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(1e-5, 5e-4 * pow(decay_rate, decay_count))
        logger.info(f"adjusting learning rate: {param_group['lr']}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # Training Data
    dataset = RTSegDataset(data_root=args.data_root, dataset_names=args.dataset_names, sub_datasets=args.sub_datasets, size=args.size, scope=args.scope, fw_only=args.fw_only)
    datasampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=args.local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=datasampler, num_workers=args.num_workers, collate_fn=my_collate_fn)
    logger.info(f"Training Set, DataSet Size:{len(dataset)}, DataLoader Size:{len(dataloader)}")

    # ------- Define model --------
    # define the net
    torch.cuda.set_device(args.local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Interactive(args.spatial_ckpt, args.temporal_ckpt))
    net = torch.nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank], find_unused_parameters=True)
    pretrained_net = []
    transformer = []
    lr = args.base_lr
    for name, param in net.named_parameters():
        if "Transformer" in name:
            transformer.append(param)
        elif "spatial_net" in name or "temporal_net" in name:
            param.requires_grad = False
            pretrained_net.append(param)
    param_group = [{"params": transformer, 'lr': lr},
                   {"params": pretrained_net, "lr": 0}]
    optimizer = optim.SGD(param_group, lr=lr, momentum=0.9, weight_decay=0.0005)

    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    tag = True
    for epoch in range(1, args.epoch_num+1):
        running_loss = 0.0
        running_spatial_loss = 0.0
        running_temporal_loss = 0.0
        ite_num_per = 0
        iter_num = 0
        datasampler.set_epoch(epoch)
        net.train()
        if tag and epoch > 4:
            lr = 5e-4
            for name, param in net.named_parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            tag = False
        if epoch > 15:
            adjust_learning_rate(optimizer, (epoch-20))
        for data_idx, data in enumerate(dataloader):
            ite_num_per = ite_num_per + 1
            iter_num = iter_num + 1
            img, fw_flow, bw_flow, label = data['video'].cuda(args.local_rank), \
                data['fwflow'].cuda(args.local_rank),\
                data['bwflow'].cuda(args.local_rank),\
                data['label'].cuda(args.local_rank)

            B, Seq, C, H, W = img.size()

            spatial_out, temporal_out = net(img, torch.cat((fw_flow, bw_flow), 2))
            spatial_loss0, spatial_loss = train_loss.muti_bce_loss_fusion(spatial_out, label.view(B * Seq, 1, H, W))
            temporal_loss0, temporal_loss = train_loss.muti_bce_loss_fusion(temporal_out, label.view(B * Seq, 1, H, W))

            loss = spatial_loss + temporal_loss
            running_loss += loss.item()
            running_spatial_loss += spatial_loss0.item()
            running_temporal_loss += temporal_loss0.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num % args.log_freq == 0:
                log_train_loss = running_loss / ite_num_per
                log_spatial_loss = running_spatial_loss / ite_num_per
                log_temporal_loss = running_temporal_loss / ite_num_per
                logger.info(f"[epoch: {epoch}/{args.epoch_num}, iter: {data_idx}/{len(dataloader)}, iter: {iter_num}]"
                            f" train loss: {log_train_loss:.5f}, spatial: {log_spatial_loss:.5f}, temporal:{log_temporal_loss:5f}")
                # tensorboard logger
                if summary_writer is not None:
                    step = (epoch - 1) * len(dataloader) + data_idx
                    summary_writer.add_scalar('lr', lr, step)
                    summary_writer.add_scalar('loss', log_train_loss, step)
                    summary_writer.add_scalar('spatial loss', log_spatial_loss, step)
                    summary_writer.add_scalar('temporal loss', log_temporal_loss, step)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            logger.info('==> Saving...')
            file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
            torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, file_name)
            copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='node rank for distributed training')
    parser.add_argument('--output_root', type=str, default='RTNet_output/')
    parser.add_argument('--exper_name', type=str, required=True, help='experiment name')
    parser.add_argument('--output_dir', type=str, help='output_root/exper_name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset_names', type=list, default=['PSEG', 'PSEG_v_flip', 'PSEG_h_flip', 'PSEG_hv_flip'])
    parser.add_argument('--sub_datasets', type=list, default=['blender_old', 'gen_mobilenet'])
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--scope', type=int, default=40)
    parser.add_argument('--fw_only', action='store_true')
    parser.add_argument('--spatial_ckpt', type=str, default='./RTNet/models/spatial_RX50.pth')
    parser.add_argument('--temporal_ckpt', type=str, default='./RTNet/models/temporal_RX50.pth')
    parser.add_argument('--log_freq', type=int, default=200)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--auto_resume', type=bool, default=True)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    cudnn.benchmark = True

    # setup logger
    args.output_dir = os.path.join(args.output_root, args.exper_name)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=args.output_dir, distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    if args.debug:
        set_seed(0)
        args.dataset_names = ['PSEG']
        args.sub_datasets = ['gen_mobilenet']

    main(args)
