#!/bin/bash

set -e
set -x

exper_name='debug'

python -m torch.distributed.launch --nproc_per_node=8 \
    RTNet/train.py \
    --data_dir ./data \
    --exper_name ${exper_name} \
    --batch_size 1 \
    --debug \
