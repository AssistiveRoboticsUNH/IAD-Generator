#!/bin/bash

SPLIT=2
SIZE=25

python iad_generator.py $MODEL $LIST_DIR/trainlist0$SPLIT_$SIZE.list --prefix $OUT_DIR/ucf_$SIZE_train


