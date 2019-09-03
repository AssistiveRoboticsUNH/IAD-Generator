#!/usr/bin/env sh
set -e

DATASET=bm

./build/tools/caffe \
  test \
  -model examples/c3d_$DATASET\/exec/c3d_test.prototxt \
  -weights examples/c3d_$DATASET\/exec/weights_iter_640.caffemodel\
  -gpu 0\
  -iterations 306

