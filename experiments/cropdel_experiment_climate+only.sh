#!/bin/bash

CUDA=0
TASK=crop_delineation
for SIZE in 64 128 256 512 1024
do
    # SwAV-climate+only-epochs
    for EPOCH in 100 125 50 175 200
    do
        for RUN in 1 2 3
        do
            python3 main.py --task $TASK \
                --epochs 100 \
                --device cuda:$CUDA \
                --dump_path /scratch/saad/no-fine-tune/$TASK/data-comparison/$SIZE/swav-climate+only-ep$EPOCH/t$RUN \
                --encoder swav-climate+only-ep$EPOCH \
                --normalization data \
                --fine_tune_encoder False \
                --batch_size 16 \
                --data_size $SIZE
        done
    done
done