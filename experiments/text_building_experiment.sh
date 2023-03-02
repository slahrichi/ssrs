#!/bin/bash

CUDA=0
TASK=building
for SIZE in 64
do
    # SwAV-climate+
    for EPOCH in 10 20 30 40 50 60 70 80 90
    do
        for RUN in 1
        do
            python3 main.py --task $TASK \
                --epochs 100 \
                --device cuda:$CUDA \
                --dump_path /scratch/saad/no-fine-tune/$TASK/data-comparison/$SIZE/swav_climate+_ep$EPOCH/t$RUN \
                --encoder swav_climate+_ep$EPOCH \
                --normalization data \
                --fine_tune_encoder False \
                --batch_size 16 \
                --data_size $SIZE
        done
    done
done