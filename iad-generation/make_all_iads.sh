#!/bin/bash

for SPLIT in '01' '02' '03'
do 
	for SIZE in 100 75 50 25
	do
		echo "python iad_generator.py ~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_$SPLIT_$SIZE ~/datasets/UCF-101/listFiles/trainlist$SPLIT_$SIZE.list --prefix ucf_$SPLIT/ucf_$SIZE/ucf_$SIZE_train"
		echo "python iad_generator.py ~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_$SPLIT_$SIZE ~/datasets/UCF-101/listFiles/trainlist$SPLIT_$SIZE.list --prefix ucf_$SPLIT/ucf_$SIZE/ucf_$SIZE_test --min_max_file ucf_$SPLIT/ucf_$SIZE/min_maxes.npz"
	done
done