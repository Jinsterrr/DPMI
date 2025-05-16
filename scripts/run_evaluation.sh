#!/bin/bash

# Batch run explanation and evaluation (DPMI)

set -e

cd ../explanation_evaluation

# Example for DP-MEPF trained model
echo "[DPMI] Evaluating DP-MEPF model..."
python evaluate_metrics.py --model_path <path_to_dpmefp_model> --model_name dpmefp_model

# Example for DPSDA trained model
echo "[DPMI] Evaluating DPSDA model..."
python evaluate_metrics.py --model_path <path_to_dpsda_model> --model_name dpsda_model

# Example for Privimage trained model
echo "[DPMI] Evaluating Privimage model..."
python evaluate_metrics.py --model_path <path_to_privimage_model> --model_name privimage_model

cd - 