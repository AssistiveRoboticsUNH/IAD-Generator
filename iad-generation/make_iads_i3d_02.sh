#!/bin/bash

SPLIT=2
SIZE=25

python iad_generator.py "$MODEL" "$LIST_DIR"/trainlist0"$SPLIT"_"$SIZE".list --prefix "$OUT_DIR"/ucf_"$SIZE"_train
python iad_generator.py "$MODEL" "$LIST_DIR"/testlist0"$SPLIT".list --prefix "$OUT_DIR"/ucf_"$SIZE"_test --min_max_file "$OUT_DIR"/min_maxes.npz
	



