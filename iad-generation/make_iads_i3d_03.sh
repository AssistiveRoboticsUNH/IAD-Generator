#!/bin/bash

SPLIT=3

for SIZE in 100 75 50 25
do

	MODEL="~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_0"$SPLIT"_"$SIZE
	LIST_DIR="~/datasets/UCF-101/listFiles"
	OUT_DIR="ucf_0"$SPLIT"/ucf_"$SIZE

	echo "python iad_generator.py "$MODEL" "$LIST_DIR"/trainlist0"$SPLIT"_"$SIZE".list --prefix "$OUT_DIR"/ucf_"$SIZE"_train"
	echo "python iad_generator.py "$MODEL" "$LIST_DIR"/testlist0"$SPLIT".list --prefix "$OUT_DIR"/ucf_"$SIZE"_test --min_max_file "$OUT_DIR"/min_maxes.npz"
	
done

