#!/bin/bash

export NVIDIA_SMI=$(nvidia-smi)

echo $NVIDIA_SMI

python robertaencdec.py \
    --dataDir "./../Data/ASAP-AES" \
    --numOfWorkers 0 \
    --num_epochs 40 \
    --batch_size 2 \
    --lr 0.05 \
    --mode train \
    --log_interval 10 \
    --prompt "1"


