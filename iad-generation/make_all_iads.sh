#!/bin/bash

SPLIT=1
SIZE=100

python iad_generator.py ~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_0$SPLIT_$SIZE ~/datasets/UCF-101/listFiles/trainlist0$SPLIT_$SIZE.list --prefix ucf_0$SPLIT/ucf_$SIZE/ucf_$SIZE_train
python iad_generator.py ~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_0$SPLIT_$SIZE ~/datasets/UCF-101/listFiles/testlist0$SPLIT_$SIZE.list --prefix ucf_0$SPLIT/ucf_$SIZE/ucf_$SIZE_test --min_max_file ucf_0$SPLIT/ucf_$SIZE/min_maxes.npz
	


