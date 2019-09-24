#!/bin/bash

for SIZE in 100 #75 #50 #25 #50
do
	IAD_DIR="~/datasets/ICRA_2020/hmdb/hmdb_iads/hmdb_"$SIZE #"../iad-generation/ucf_01/ucf_"$SIZE
	LIST_DIR="~/datasets/HMDB-51/listFiles/"

	CMD="python ensemble_window_group.py model/hmdb_"$SIZE" 51 "$IAD_DIR" --gpu 0 --test "$LIST_DIR"testlist01.list"
	echo $CMD
	eval $CMD
done
