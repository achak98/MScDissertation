#!/bin/bash

python main.py \
    --dataDir "/Users/ac/Desktop/unicode/Dissertation/Data/ASAP-AES" \
    --numOfWorkers 0 \
    --vocab_size 10000 \
    --embedding_dim 300 \
    --num_epochs 40 \
    --batch_size 128 \
    --lr 0.001 \
    --mode train \
    --cnnfilters 100 \
    --cnn_window_size_small 2 \
    --cnn_window_size_medium 3 \
    --cnn_window_size_large 4 \
    --bgru_hidden_size 128 \
    --dropout 0.4 \
    --log_interval 10
