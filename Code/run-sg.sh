#!/bin/bash

export NVIDIA_SMI=$(nvidia-smi)

echo $NVIDIA_SMI

:'
python skipgram.py \
    --dataDir "/home/achakravarty/Dissertation/Data/ASAP-AES" \
    --skipgram_file_path "/home/achakravarty/Dissertation/Data/Skipgram" \
    --numOfWorkers 0 \
    --embedding_dim 300 \
    --batch_size 8 \
    --num_epochs 40 \
    --log_interval 10 \
    --prompt '1'
'

python baseline-sg.py \
    --dataDir "/home/achakravarty/Dissertation/Data/ASAP-AES" \
    --skipgram_file_path "/home/achakravarty/Dissertation/Data/Skipgram" \
    --numOfWorkers 0 \
    --embedding_dim 300 \
    --num_epochs 40 \
    --batch_size 8 \
    --lr 0.001 \
    --mode train \
    --cnnfilters 100 \
    --cnn_window_size_small 2 \
    --cnn_window_size_medium 3 \
    --cnn_window_size_large 4 \
    --bgru_hidden_size 128 \
    --dropout 0.4 \
    --log_interval 10 \
    --prompt "1"

