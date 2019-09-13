#!/bin/bash

for SPLIT in 1 
do
	for SIZE in 50 25 #100 75 
	do

		MODEL="~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_0"$SPLIT"_"$SIZE
		LIST_DIR="~/datasets/UCF-101/listFiles"
		OUT_DIR="ucf_0"$SPLIT"/ucf_"$SIZE

		CMD_TRAIN="python iad_generator.py "$MODEL" "$LIST_DIR"/trainlist0"$SPLIT"_"$SIZE".list --dst_directory "$OUT_DIR" --prefix ucf_"$SIZE"_train" 
		echo $CMD_TRAIN
		eval $CMD_TRAIN

		CMD_TEST="python iad_generator.py "$MODEL" "$LIST_DIR"/testlist0"$SPLIT".list --dst_directory "$OUT_DIR" --prefix ucf_"$SIZE"_test --min_max_file "$OUT_DIR"/min_maxes.npz"
		echo $CMD_TEST
		eval $CMD_TEST
	done
done
