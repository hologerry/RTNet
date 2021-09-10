#!/bin/bash

set -e
set -x

exper_name='pseg_scope40_rx50_gen_mobilenet'

master_addr=${MASTER_IP}
master_port=28652

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes ${OMPI_COMM_WORLD_SIZE} --node_rank ${OMPI_COMM_WORLD_RANK} --master_addr ${master_addr} --master_port ${master_port} \
    RTNet/train.py \
    --data_root ./data \
    --exper_name ${exper_name} \
    --batch_size 1 \
    --dataset_name_mode 0 \
    --sub_dataset_mode 1 \