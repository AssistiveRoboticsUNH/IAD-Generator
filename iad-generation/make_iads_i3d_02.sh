#!/bin/bash

SPLIT=2
SIZE=25

MODEL="~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_0"$SPLIT"_"$SIZE
LIST_DIR="~/datasets/UCF-101/listFiles"
OUT_DIR="ucf_0"$SPLIT"/ucf_"$SIZE
python iad_generator.py $MODEL $LIST_DIR/trainlist0$SPLIT_$SIZE.list --prefix $OUT_DIR/ucf_$SIZE


