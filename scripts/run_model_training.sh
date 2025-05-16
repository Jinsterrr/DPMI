#!/bin/bash

# Batch run model training on synthetic datasets (DPMI)

set -e

cd ../dp_mechanism/DP_dataset_synthesis/model_training

# Example for DP-MEPF synthesized data
python generic_train.py --data_source dpmepf --train_datapath <path_to_dpmepf_data> --exp_name dpmefp_exp

# Example for DPSDA synthesized data
python generic_train.py --data_source dpsda --train_datapath <path_to_dpsda_data> --exp_name dpsda_exp

# Example for Privimage synthesized data
python generic_train.py --data_source privimage --train_datapath <path_to_privimage_data> --exp_name privimage_exp

cd - 