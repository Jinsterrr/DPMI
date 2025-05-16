# DPMI: Differential Privacy Model and Interpretation Benchmark

This repository is the official implementation of the DPMI framework, supporting end-to-end benchmarking for differential privacy mechanisms, dataset synthesis, model training, and interpretability evaluation.

## System Overview

DPMI provides a holistic pipeline to evaluate how differential privacy (DP) mechanisms affect model explainability. The framework covers two types of DP noise injection (DP training and DP dataset synthesis) and a broad set of explainability methods and metrics. The system is modular and extensible for benchmarking new privacy or explainability techniques.

<p align="center">
  <img src="pipeline.pdf" alt="DPMI System Pipeline" width="600"/>
</p>

**Figure:** Architectural overview of DPMI for assessing the impact of DP on model explainability. The system comprises three main modules: Method Pool, DP Mechanism, and Explanation Evaluation.

## Key Features
- **Comprehensive DP Mechanisms:** Supports both in-training DP (e.g., DPSGD, Adpclip, Adpalloc, loss, tanhact) and dataset synthesis (e.g., DP-MEPF, DPSDA, Privimage).
- **Rich Explainability Metrics:** Evaluates faithfulness (ROAD), robustness (RIS), and sparseness (Gini Index) across multiple explanation methods (gradient-based, activation-based, class activation-based, perturbation-based, Shapley value-based).
- **Reproducible Benchmark:** Includes scripts for all stages: DP training, data synthesis, model training, and evaluation.
- **Open-source and extensible:** Easy to add new DP or explainability methods.

## Requirements

To install requirements:

```bash
bash setup.sh
# or
pip install -r requirements.txt
```

## Dataset Preparation

DPMI supports standard image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Download and preprocess datasets as required by each DP method. For DP dataset synthesis, ensure the generated data is saved in a format compatible with `generic_train.py`.

## DP Mechanism Training

To train models with DP mechanisms (e.g., DPSGD, Adpclip, Adpalloc, loss, tanhact), use the scripts in `dp_mechanism/DP_training/DPMLBench/algorithms/`.

Example (DP-SGD):

```bash
cd dp_mechanism/DP_training/DPMLBench/algorithms/dp_sgd
python main.py --net resnet --dataset cifar10 --private -p --eps 2.0 --epoch 60 --batchsize 256 --lr 0.01
```

Other methods (Adpclip, Adpalloc, loss, tanhact) are available in their respective subdirectories.

## DP Dataset Synthesis

To synthesize datasets using three representative methods:

- **DP-MEPF**:

```bash
cd dp_mechanism/DP_dataset_synthesis/dataset_synthesis/DP-MEPF/code
python dp_mepf.py --dataset cifar10 --gen_output tanh --matched_moments mean --dp_tgt_eps 1. --seed 1
```

- **DPSDA** and **Privimage**: See their respective README files for usage and options.

## Model Training on Synthetic Data

Train a model using the synthesized dataset with the generic training script:

```bash
cd dp_mechanism/DP_dataset_synthesis/model_training
python generic_train.py --data_source dpmepf --train_datapath <path_to_synthesized_data> --exp_name <experiment_name>
```

## Explanation and Evaluation

Evaluate trained models using interpretability metrics:

```bash
cd explanation_evaluation
python evaluate_metrics.py --model_path <path_to_model> --model_name <model_name>
```

## Differential Privacy Methods

DPMI supports two main categories of DP mechanisms:
- **DP Training:** Noise is injected during model training (e.g., DPSGD, Adpclip, Adpalloc, loss, tanhact).
- **DP Dataset Synthesis:** Synthetic data is generated with DP guarantees (e.g., DP-MEPF, DPSDA, Privimage), and then used for downstream model training.

See the paper and code for detailed hyperparameters and method descriptions.

## Explainability Metrics

DPMI evaluates explanations using three main metrics:
- **Faithfulness (ROAD):** How well the explanation reflects the model's true decision process.
- **Robustness (RIS):** Stability of explanations under small input perturbations.
- **Sparseness (Gini Index):** How concise and focused the explanation is.

Multiple explanation methods are supported, including gradient-based, activation-based, class activation-based, perturbation-based, and Shapley value-based approaches.

## Main Results

DPMI provides a comprehensive comparison of DP methods across explainability metrics. Below are detailed results on CIFAR-10 (ResNet20, $\epsilon=4$), averaged over five runs. For clarity, only mean values are shown (standard deviations omitted).

### Faithfulness Scores
| Method      | InputXGrad | FovEx | FeatureAblation | Occlusion | GradientShap | KernelShap | Activation | GradXActivation | GradCam | GradCamPlusScore |
|-------------|------------|-------|-----------------|-----------|--------------|------------|------------|-----------------|---------|------------------|
| Normal      | 0.479      | 0.433 | 0.521           | 0.499     | 0.509        | 0.143      | 0.477      | 0.488           | 0.521   | 0.401            |
| DP-SGD      | 0.257      | 0.394 | 0.308           | 0.398     | 0.277        | 0.149      | 0.424      | 0.395           | 0.390   | 0.330            |
| AdpAlloc    | 0.358      | 0.340 | 0.332           | 0.326     | 0.323        | 0.109      | 0.323      | 0.288           | 0.289   | 0.328            |
| AdpClip     | 0.333      | 0.414 | 0.388           | 0.450     | 0.347        | 0.102      | 0.378      | 0.295           | 0.302   | 0.349            |
| FocalLoss   | 0.330      | 0.407 | 0.361           | 0.388     | 0.328        | 0.118      | 0.397      | 0.365           | 0.379   | 0.328            |
| TanhAct     | 0.315      | 0.285 | 0.381           | 0.326     | 0.334        | 0.123      | 0.205      | 0.318           | 0.320   | 0.244            |
| DPSDA       | 0.362      | 0.390 | 0.421           | 0.530     | 0.345        | 0.133      | 0.392      | 0.366           | 0.362   | 0.280            |
| DP-MEPF     | 0.316      | 0.428 | 0.261           | 0.377     | 0.328        | 0.192      | 0.342      | 0.337           | 0.296   | 0.355            |
| PrivImage   | 0.643      | 0.527 | 0.629           | 0.537     | 0.632        | 0.612      | 0.575      | 0.494           | 0.504   | 0.444            |

### Robustness Scores
| Method      | InputXGrad | FovEx | FeatureAblation | Occlusion | GradientShap | KernelShap | Activation | GradXActivation | GradCam | GradCamPlusScore |
|-------------|------------|-------|-----------------|-----------|--------------|------------|------------|-----------------|---------|------------------|
| Normal      | -3.44      | -6.69 | -2.82           | -2.49     | -5.61        | -5.06      | 5.21       | 2.05            | 2.00    | 2.16             |
| DP-SGD      | -3.54      | -6.74 | -2.97           | -1.99     | -5.84        | -4.76      | 5.60       | 2.55            | 2.55    | 1.31             |
| AdpAlloc    | -3.45      | -5.48 | -2.84           | -2.37     | -6.04        | -4.76      | 5.75       | 2.17            | 2.20    | 0.31             |
| AdpClip     | -3.37      | -3.55 | -2.84           | -2.32     | -5.77        | -4.70      | 5.81       | 2.02            | 2.08    | 2.16             |
| FocalLoss   | -3.29      | -6.17 | -2.91           | -2.39     | -5.72        | -4.85      | 5.52       | 2.42            | 2.38    | 2.08             |
| TanhAct     | -3.01      | -7.38 | -2.98           | -2.55     | -5.92        | -4.70      | 0.27       | 2.17            | 2.13    | -4.74            |
| DPSDA       | -3.28      | -4.10 | -2.50           | -1.83     | -5.90        | -4.88      | 5.89       | 2.51            | 2.49    | -0.31            |
| DP-MEPF     | -3.44      | -6.51 | -2.99           | -1.78     | -5.49        | -5.17      | 5.50       | 2.65            | 2.77    | -1.71            |
| PrivImage   | -3.57      | -9.27 | -3.30           | -2.72     | -5.75        | -4.46      | 4.52       | 1.28            | 1.23    | -2.52            |

### Sparseness Scores
| Method      | InputXGrad | FovEx | FeatureAblation | Occlusion | GradientShap | KernelShap | Activation | GradXActivation | GradCam | GradCamPlusScore |
|-------------|------------|-------|-----------------|-----------|--------------|------------|------------|-----------------|---------|------------------|
| Normal      | 0.554      | 0.698 | 0.547           | 0.478     | 0.553        | 0.589      | 0.239      | 0.459           | 0.470   | 0.470            |
| DP-SGD      | 0.543      | 0.718 | 0.536           | 0.464     | 0.545        | 0.539      | 0.173      | 0.385           | 0.374   | 0.270            |
| AdpAlloc    | 0.544      | 0.707 | 0.543           | 0.486     | 0.539        | 0.538      | 0.199      | 0.415           | 0.420   | 0.355            |
| AdpClip     | 0.548      | 0.706 | 0.541           | 0.468     | 0.547        | 0.551      | 0.204      | 0.416           | 0.419   | 0.264            |
| FocalLoss   | 0.536      | 0.704 | 0.533           | 0.459     | 0.531        | 0.545      | 0.197      | 0.393           | 0.385   | 0.279            |
| TanhAct     | 0.519      | 0.722 | 0.525           | 0.438     | 0.519        | 0.643      | 0.409      | 0.294           | 0.286   | 0.444            |
| DPSDA       | 0.598      | 0.724 | 0.587           | 0.504     | 0.607        | 0.532      | 0.163      | 0.429           | 0.431   | 0.323            |
| DP-MEPF     | 0.540      | 0.733 | 0.550           | 0.434     | 0.539        | 0.605      | 0.155      | 0.356           | 0.367   | 0.256            |
| PrivImage   | 0.610      | 0.699 | 0.559           | 0.484     | 0.605        | 0.557      | 0.149      | 0.388           | 0.401   | 0.395            |

**Metric explanations:**
- *Faithfulness*: Higher is better; measures how well explanations reflect the model's true decision process.
- *Robustness*: Higher (less negative) is better; measures stability of explanations under input perturbations.
- *Sparseness*: Higher is better; measures how concise and focused the explanation is.

- **PrivImage** achieves the best faithfulness and sparseness among DP methods across most explanation types.
- **DP dataset synthesis** methods generally preserve explanation fidelity better than DP training methods.
- **Gradient clipping** in DP-SGD impacts explainability more than noise addition.
- **Wider networks** perform better with most DP data synthesis, while simpler models maintain more faithful explanations in DP training.

## Pre-trained Models

You can download pretrained models here:

- [DPMI Example Model](https://drive.google.com/xxxxxx) trained on CIFAR10 with specific parameters.

## Scripts

Batch scripts are provided in the `scripts/` directory for each stage:
- `run_dp_training.sh`: Batch run all DP training methods
- `run_synthesis.sh`: Batch run all dataset synthesis methods
- `run_model_training.sh`: Batch train models on all synthetic datasets
- `run_evaluation.sh`: Batch evaluate all trained models

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## Citation

If you use this codebase, please cite the relevant papers and this repository. 