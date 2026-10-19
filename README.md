# Sidewinder: Adversarial Network Traffic Generation & Defense Evaluation Framework

**Sidewinder** is a unified research benchmark designed to evaluate modern Machine Learning and Deep Learning Network Intrusion Detection Systems (NIDS) against adversarial traffic attacks in both **Closed-World** and **Open-World** settings. 

The framework combines traffic sequence modeling (BERT/Transformer-based mimicry) with Reinforcement Learning agents (Soft Actor-Critic with recurrent/sequence backbones like GRU, LSTM, and Mamba) and classic mutation algorithms to generate adversarial network perturbations and analyze target defense robustness.

---

## Repository Structure

```text
Sidewinder/
├── advGenerate/                   # Adversarial Generation & RL Attack Framework
│   ├── config/                    # Configuration files
│   │   └── sac.yaml               # SAC algorithm hyperparameter configurations
│   ├── environments/              # Gym environments wrapping target NIDS defenses
│   │   ├── cnn_environment_eval.py
│   │   ├── dl_environment_eval.py
│   │   ├── environment.py         # Base environment abstract class
│   │   ├── pandora_fast.py
│   │   ├── pandora_seq_environment_eval.py
│   │   ├── prob_environment_eval.py
│   │   ├── transformer_environment_eval.py
│   │   ├── vib_ensemble.py
│   │   └── vib_environment_eval.py
│   ├── amoeba_eval.py             # Amoeba attack baseline evaluator
│   ├── buffer.py                  # Experience replay buffer implementations
│   ├── compile_dataset.py         # Dataset aggregation and preparation
│   ├── config_parser.py           # Utility for parsing YAML configurations
│   ├── mamba_pytorch.py           # PyTorch implementation of Mamba sequence block
│   ├── mutate_and_inject.py       # Traffic mutation and payload injection module
│   ├── netmasqurade_eval.py       # NetMasquerade attack evaluator
│   ├── random_mutation.py         # Random mutation attack generator
│   ├── rl_dataset.py              # PyTorch Dataset objects for RL trajectories
│   ├── rl_types.py                # Dataclasses and typing definitions for RL
│   ├── rl_utils.py                # Utility helper functions for RL modules
│   ├── sac.py                     # Standard Soft Actor-Critic (SAC) implementation
│   ├── sac_gru.py                 # SAC with GRU sequence memory
│   ├── sac_lstm.py                # SAC with LSTM sequence memory
│   ├── sac_mamba.py               # SAC with Mamba sequence state model
│   ├── sanity_test.py             # Environment verification & test suite
│   ├── test_mamba.py              # Mamba architecture unit tests
│   ├── train.py                   # Main training script for RL agents
│   ├── train_cnn_ids.py           # Trainer for CNN-based IDS classifier
│   ├── train_dl_ids.py            # Trainer for Deep Learning IDS classifier
│   ├── train_ml_ids.py            # Trainer for classical ML IDS classifiers
│   ├── train_pandora_seq.py       # Trainer for Pandora Sequence NIDS
│   ├── train_prob_classifier.py   # Trainer for probabilistic defense models
│   ├── train_transformer_ids.py   # Trainer for Transformer-based IDS
│   └── train_vib_ensemble.py      # Trainer for Variational Information Bottleneck (VIB) IDS
│
├── trafficMimic/                  # Generative Traffic Sequence Mimicry Models
│   ├── config/
│   │   └── bert.yaml              # BERT architecture & pretraining hyper-parameters
│   ├── dataset/
│   │   ├── build_real_vocab.py    # Vocabulary builder from packet payloads/tokens
│   │   ├── dataset.py             # Sequence dataset loaders
│   │   └── vocab.py               # Vocabulary mapping and tokenizer logic
│   ├── model/                     # Custom Transformer & BERT Neural Models
│   │   ├── architecture/          # BERT, Encoder, Decoder, Transformer modules
│   │   ├── attention/             # Single-head and Multi-head attention mechanisms
│   │   ├── embedding/             # Positional, Token, Segment, and Transformer embeddings
│   │   └── utils/                 # FeedForward, GELU, LayerNorm, and Sublayer connection modules
│   ├── bert_train.py              # Training entry point for BERT traffic mimicry model
│   └── utils.py                   # Sequence modeling utility functions
│
├── ablation_runner.py             # Execution manager for ablation experiments
├── check_vocab_size.py            # Utility script to inspect build vocabulary sizes
├── eval_metrics.py                # Comprehensive accuracy, F1, and evasion metric computations
├── shared_types.py                # Shared data types, constants, and data models
├── visualize_boundary.py          # Decision boundary visualization and mapping tool
├── build_dataset.sh               # Pipeline step 1: Dataset extraction and pre-processing
├── build_cnn_models.sh            # Pipeline step 2a: Train CNN target defense models
├── build_dl_models.sh             # Pipeline step 2b: Train DL target defense models
├── build_ml_models.sh             # Pipeline step 2c: Train classical ML baseline models
├── build_pandora_model.sh         # Pipeline step 2d: Train Pandora target defense models
├── build_transformer_models.sh    # Pipeline step 2e: Train Transformer defense models
├── build_vib_models.sh            # Pipeline step 2f: Train VIB Ensemble defense models
├── run_ablation.sh                # Execution script for hyperparameter ablation studies
├── run_amoeba_pipeline.sh         # Execution script for Amoeba baseline attack
├── run_mutate_and_inject.sh       # Script for packet mutation and injection execution
├── run_netmasqurade.sh            # Execution script for NetMasquerade attack
├── run_netmasqurade_extended_budget.sh # NetMasquerade evaluation under high mutation budgets
├── run_random_mutation.sh         # Execution script for Random Mutation baseline attack
├── run_rl_pipeline.sh             # Execution script for training RL generation agents (SAC)
└── run_visualize.sh               # Execution script to render decision boundary plots
```


## Key Components & Workflow

### 1. Traffic Mimicry (`trafficMimic/`)
Trains sequence-based language representations (BERT/Transformer architectures) on benign network packet payloads and sequences. These models capture high-dimensional network protocol syntax to generate realistic obfuscated network traffic.

### 2. Target Defense Classifiers (`advGenerate/train_*.py`)
Constructs and trains target defense architectures in **Closed-World** (2-Class: Benign vs. Malicious) and **Open-World** (3-Class: Benign, Malicious, UNK) operational settings:
* **CNN / Deep Learning (DL)**
* **Pandora Sequence Classifier**
* **Transformer-based IDS**
* **Variational Information Bottleneck (VIB) Ensemble**
* **Kitsune / Autoencoder-like Defenses**

### 3. Adversarial Generation Agents (`advGenerate/`)
Generates evasion traffic using various attack models against target Gym environments (`advGenerate/environments/`):
* **Soft Actor-Critic (SAC)** with sequence memory backbones (`sac.py`, `sac_gru.py`, `sac_lstm.py`, `sac_mamba.py`).
* **Baselines**: NetMasquerade (`netmasqurade_eval.py`), Amoeba (`amoeba_eval.py`), and Random Mutations (`random_mutation.py`).

---

## Execution Guide

Make all shell scripts executable before running the pipeline:

```bash
chmod +x *.sh
```

## Step 1: Build Datasets & Vocabularies

Extract features and build network token vocabularies:

``` bash
./build_dataset.sh
python trafficMimic/dataset/build_real_vocab.py
```

## Step 2: Train Generative Traffic Mimicry Model

Pretrain the BERT traffic representation model:

```bash
python trafficMimic/bert_train.py --config trafficMimic/config/bert.yaml
```

## Step 3: Train Target Defense Models

Train baseline NIDS classifiers across target architectures:

```bas
./build_cnn_models.sh
./build_dl_models.sh
./build_ml_models.sh
./build_pandora_model.sh
./build_transformer_models.sh
./build_vib_models.sh
```

## Step 4: Run Adversarial Generation & Attacks

Train RL agents or run baseline attack suites against target defenses:

### Reinforcement Learning Pipeline (SAC / SAC-Mamba / SAC-LSTM):

```bash
./run_rl_pipeline.sh
```

### NetMasquerade Attack:

```bash
./run_netmasqurade.sh
```

### Amoeba Attack Pipeline:

```bash
./run_amoeba_pipeline.sh
```

### Random Mutation Baseline:

```bash
./run_random_mutation.sh
```

## Step 5: Evaluate Performance & Run Ablation Studies

Compute accuracy, F1-scores, evasion success rates, and run feature ablation:

```bash
python eval_metrics.py
./run_ablation.sh
```

## Step 6: Visualize Decision Boundaries

Render classification decision boundaries and feature spaces:

```bash
./run_visualize.sh
```

## Prerequisites & Requirements

* **OS:** Linux / WSL2 / MSYS2 (Bash execution environment required for `.sh` scripts)
* **Python:** 3.8+ (3.12+ recommended)
* **PyTorch:** 2.0+ (CUDA support strongly recommended)

### Installation & Dependencies

Choose one of the following methods to install the project dependencies:

#### Option 1: Using `uv` (Recommended)
Using [`uv`](https://github.com/astral-sh/uv) automatically manages PyTorch CUDA wheels and build-time requirements (`mamba-ssm`, `causal-conv1d`):

```bash
# 1. Install uv (if not already installed)
curl -sSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 2. Sync environment and install all dependencies
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate
```

## Option 2: Using Standard pip (requirements.txt)

Note: Because mamba-ssm and causal-conv1d compile CUDA/C++ extensions during installation, you must pre-install torch and numpy before running requirements.txt

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Pre-install PyTorch & build prerequisites
pip install torch numpy setuptools wheel --index-url [https://download.pytorch.org/whl/cu132](https://download.pytorch.org/whl/cu132)

# 3. Install remaining dependencies
pip install -r requirements.txt
```