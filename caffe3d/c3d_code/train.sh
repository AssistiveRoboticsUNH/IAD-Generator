#!/usr/bin/env sh
set -e

DATASET=bm

./build/tools/caffe \
  train \
  --solver=examples/c3d_$DATASET\/exec/c3d_solver.prototxt \
  $@ \
  2>&1 | tee examples/c3d_$DATASET\/exec/c3d_train.log
