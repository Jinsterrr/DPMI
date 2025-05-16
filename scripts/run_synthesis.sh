#!/bin/bash

# Batch run DP dataset synthesis (DPMI)

set -e

# DP-MEPF
cd ../dp_mechanism/DP_dataset_synthesis/dataset_synthesis/DP-MEPF/code
python dp_mepf.py --dataset cifar10 --gen_output tanh --matched_moments mean --dp_tgt_eps 1. --seed 1
cd -

# DPSDA (example, adjust as needed)
cd ../dp_mechanism/DP_dataset_synthesis/dataset_synthesis/DPSDA
# python <your_dpsda_script.py> --dataset cifar10 --eps 1.0 --seed 1
cd -

# Privimage (example, adjust as needed)
cd ../dp_mechanism/DP_dataset_synthesis/dataset_synthesis/Privimage
# python <your_privimage_script.py> --dataset cifar10 --eps 1.0 --seed 1
cd - 