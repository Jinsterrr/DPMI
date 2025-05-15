# DPMI

## Overview
This project is a collection of differential privacy mechanisms, evaluation tools, and datasets for privacy-preserving machine learning research. It includes modules for dataset synthesis, model training, and explanation evaluation, supporting a variety of privacy-preserving algorithms and benchmarks.

## Installation
1. Clone the repository
2. Install dependencies for each submodule as needed (see their respective `requirements.txt` files).

## Usage
- Refer to the `README.md` files in each submodule for detailed usage instructions.
- Example scripts and notebooks are provided in the respective folders.

## Directory Structure
- `dp_mechanism/` - Differential privacy mechanisms, including dataset synthesis and model training.
- `explanation_evaluation/` - Tools and scripts for evaluating explanation methods.


## Contributing
Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Third-Party Libraries
This project includes or depends on the following third-party libraries, each with their own licenses:

- **DP-MEPF**  
  Path: `dp_mechanism/DP_dataset_synthesis/dataset_synthesis/DP-MEPF`  
  License: MIT  
  Copyright: Park Lab ML

- **DPSDA**  
  Path: `dp_mechanism/DP_dataset_synthesis/dataset_synthesis/DPSDA`  
  License: MIT  
  Copyright: Microsoft Corporation

- **DP-ImaGen**  
  Path: `dp_mechanism/DP_dataset_synthesis/dataset_synthesis/Privimage/DP-ImaGen`  
  License: MIT  
  Copyright: SunnierLee

- **DP-ImaGen/src/PRIVIMAGE+D/model**  
  Path: `dp_mechanism/DP_dataset_synthesis/dataset_synthesis/Privimage/DP-ImaGen/src/PRIVIMAGE+D/model`  
  License: Apache 2.0

- **adaclipoptimizer.py (Meta Platforms, Inc.)**  
  Path: `dp_mechanism/DP_training/DPMLBench/algorithms/adp_clip/adaclipoptimizer.py`  
  License: Apache 2.0  
  Copyright: Meta Platforms, Inc. and affiliates

Please refer to the respective LICENSE files in each submodule for more information. 