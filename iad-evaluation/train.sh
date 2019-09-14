#!/bin/bash

for SIZE in 25 50
do
	IAD_DIR="../iad-generation/ucf_01/ucf_"$SIZE
	LIST_DIR="~/datasets/UCF-101/listFiles/"

	CMD="python ensemble.py model/ucf_"$SIZE" 101 "$IAD_DIR" --gpu 0--test "$LIST_DIR"testlist01.list --train "$LIST_DIR"trainlist01_"$SIZE".list"
	echo $CMD
	eval $CMD
done