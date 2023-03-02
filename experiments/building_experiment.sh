#!/bin/bash

CUDA=1
TASK=building
for SIZE in 64 128 256 512 1024
do
    # Run each experiment 3 times
    for RUN in 1 2 3
    do
    python3 main.py --task $TASK \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path /scratch/saad/no-fine-tune/$TASK/data-comparison/$SIZE/supervised/t$RUN \
        --encoder imagenet \
        --normalization imagenet \
        --fine_tune_encoder False \
        --batch_size 16 \
        --data_size $SIZE
    done

    # SWAV-Imagenet
    for RUN in 1 2 3
    do
    python3 main.py --task $TASK \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path /scratch/saad/no-fine-tune/$TASK/data-comparison/$SIZE/swav-imagenet/t$RUN \
        --encoder swav \
        --normalization imagenet \
        --fine_tune_encoder False \
        --batch_size 16 \
        --data_size $SIZE
    done


    # SwAV-climate+
    for RUN in 1 2 3
    do
    python3 main.py --task $TASK \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path /scratch/saad/no-fine-tune/$TASK/data-comparison/$SIZE/swav-climate+/t$RUN \
        --encoder swav-climate+ \
        --normalization data \
        --fine_tune_encoder False \
        --batch_size 16 \
        --data_size $SIZE
    done

    #SwAV-climate+epochs
    for EPOCH in 0 10 20 30 40 50 60 70 80 90 99
    do
        for RUN in 1 2 3
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