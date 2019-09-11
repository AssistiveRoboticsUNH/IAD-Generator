#!/bin/bash

SPLIT=1
SIZE=100

MODEL="~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_0"$SPLIT"_"$SIZE
LIST_DIR="~/datasets/UCF-101/listFiles"
OUT_DIR="ucf_0"$SPLIT"/ucf_"$SIZE

echo "python iad_generator.py "$MODEL" "$LIST_DIR"/trainlist0"$SPLIT"_"$SIZE".list --prefix "$OUTDIR"/ucf_"$SIZE"_train"
echo "python iad_generator.py "$MODEL" "$LIST_DIR"/testlist0"$SPLIT".list --prefix "$OUTDIR"/ucf_"$SIZE"_test --min_max_file "$OUTDIR"/min_maxes.npz"
	


