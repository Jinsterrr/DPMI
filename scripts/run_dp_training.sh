#!/bin/bash

# Batch run DP mechanism training (DPMI)

set -e

METHODS=(dp_sgd adp_clip adp_alloc loss tanhact)
NET=resnet
DATASET=cifar10
EPOCHS=60
BATCHSIZE=256
LR=0.01
EPS=2.0

for method in "${METHODS[@]}"; do
  echo "[DPMI] Training with $method..."
  cd ../dp_mechanism/DP_training/DPMLBench/algorithms/$method
  python main.py --net $NET --dataset $DATASET --private -p --eps $EPS --epoch $EPOCHS --batchsize $BATCHSIZE --lr $LR
  cd -
done 