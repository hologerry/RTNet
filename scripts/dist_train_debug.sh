#!/bin/bash

set -e
set -x

exper_name='debug'

python -m torch.distributed.launch --nproc_per_node=2 \
    RTNet/train.py \
    --data_root ./data \
    --exper_name ${exper_name} \
    --batch_size 1 \
    --debug \
