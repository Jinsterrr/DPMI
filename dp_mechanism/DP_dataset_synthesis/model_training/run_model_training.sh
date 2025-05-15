#!/bin/bash

# Common parameters
SAVE_DIR="./logs"
TEST_PATH="./dataset_cifar10"
EPOCHS=200
BATCH_SIZE=128

# Run DP-MEPF experiment
python generic_train.py \
  --data_source dpmepf \
  --train_datapath "/path/to/dpmepf_data.npz" \
  --test_datapath $TEST_PATH \
  --model wrn \
  --depth 40 \
  --width 4 \
  --exp_name "dpmepf_experiment" \
  --save_dir $SAVE_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --gpu 0

# Run DPSDA experiment
python generic_train.py \
  --data_source dpsda \
  --train_datapath "/path/to/dpsda_data.npz" \
  --test_datapath $TEST_PATH \
  --model wrn \
  --depth 28 \
  --width 10 \
  --exp_name "dpsda_experiment" \
  --save_dir $SAVE_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --gpu 0

# Run PrivImage experiment
python generic_train.py \
  --data_source privimage \
  --train_datapath "/path/to/privimage_data/" \
  --test_datapath $TEST_PATH \
  --model resnet20 \
  --exp_name "privimage_experiment" \
  --save_dir $SAVE_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --gpu 0