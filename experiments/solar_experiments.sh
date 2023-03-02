#!/bin/bash


CUDA=0
TASK=solar
SIZE=64

python3 main.py --task $TASK \
    --epochs 100 \
    --device cuda:$CUDA \
    --dump_path /scratch/saad/$TASK/data-comparison/$SIZE/swav-climate+/t$RUN \
    --encoder swav-climate+ \
    --normalization data \
    --fine_tune_encoder False \
    --batch_size 16 \
    --data_size $SIZE