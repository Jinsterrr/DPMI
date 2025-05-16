#!/bin/bash

# DPMI setup script
echo "[DPMI] Setting up Python virtual environment..."
python3 -m venv dpmi_env
source dpmi_env/bin/activate
echo "[DPMI] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
echo "[DPMI] Setup complete. Activate with: source dpmi_env/bin/activate" 