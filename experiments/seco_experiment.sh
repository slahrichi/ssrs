#!/bin/bash

CUDA=7
TASK=crop_delineation
for SIZE in 64 128 256 512 1024
do
    # SeCo
    for RUN in 1 2 3 4 5
    do
    python3 main.py --task $TASK \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path /scratch/saad/no-fine-tune/$TASK/data-comparison/$SIZE/seco/t$RUN \
        --encoder seco \
        --normalization data \
        --fine_tune_encoder False \
        --batch_size 16 \
        --data_size $SIZE
    done

done