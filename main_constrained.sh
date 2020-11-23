# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/cle_macho/mini_imagenet/train_split/unlabeled_use"
ARCH="alexnet"
LR=0.05
WD=-5
K=500
WORKERS=4
EXP="exp_constrained"
PYTHON="python"
BATCH=32
DIR_LABELED="/home/cle_macho/mini_imagenet/train_split/labeled_use"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main_constrained_dc.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --batch ${BATCH} --sobel --verbose --workers ${WORKERS} --epochs 500 \
  --reassign 3 --checkpoints 150000

CUDA_VISIBLE_DEVICES=0 ${PYTHON} compute_deepcluster_features_constrained_dc.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --batch ${BATCH} --sobel --verbose --workers ${WORKERS}
