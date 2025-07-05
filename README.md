# CCPS: Calibrating LLM Confidence by Probing Perturbed Representation Stability

This repository contains the official implementation for the paper: **"CCPS: Calibrating LLM Confidence by Probing Perturbed Representation Stability"**.

Our CCPS method offers a novel approach to estimate the confidence of Large Language Models (LLMs) by analyzing the stability of their internal representations when subjected to targeted adversarial perturbations. This repository provides the code to reproduce our experiments, including feature extraction, model training, and evaluation.

<strong>Paper:</strong> <a href="https://arxiv.org/abs/2505.21772" target="_blank" rel="noopener noreferrer">https://arxiv.org/abs/2505.21772</a>

## Table of Contents
1. [Overview](#overview)
2. [Data](#data)
3. [Setup](#setup)
4. [Workflow and Usage](#workflow-and-usage)
    * [Step 0: Answer Generation & Assessment](#step-0-answer-generation--assessment)
    * [Step 1: Hidden State & Perturbation Data Extraction](#step-1-hidden-state--perturbation-data-extraction)
    * [Step 2: Feature Calculation](#step-2-feature-calculation)
    * [Step 3: Contrastive Model Training](#step-3-contrastive-model-training)
    * [Step 4: Classifier Model Training](#step-4-classifier-model-training)
    * [Step 5: Evaluation](#step-5-evaluation)
5. [License](#license)
6. [Citation](#citation)

## Overview

CCPS operates on frozen base LLMs. The core idea is to:
1.  Apply targeted adversarial perturbations to the final hidden states that generate an LLM's answer tokens.
2.  Extract a rich set of features reflecting the model's response to these perturbations.
3.  Train a lightweight classifier on these features to predict the correctness of the LLM's answer, which serves as its confidence score.

This repository provides the scripts to replicate these steps.

## Data

All datasets used for training, validation, and evaluation in this work are publicly available on Hugging Face Datasets:

<strong>🤗 Dataset Hub Link:</strong> <a href="https://huggingface.co/datasets/ledengary/CCPS" target="_blank" rel="noopener noreferrer">https://huggingface.co/datasets/ledengary/CCPS</a>

The dataset includes:
* **Training/Validation Sets:**
    * `CT-CHOICE`: Multiple-choice (MC) format.
    * `CT-OE`: Open-ended (OE) format.
* **Test Sets:**
    * `MMLU-CHOICE`: MMLU benchmark in MC format.
    * `MMLU-PRO-CHOICE`: MMLU-Pro benchmark in MC format.
    * `MMLU-OE`: MMLU benchmark in OE format.

Each dataset is further organized by the base LLM used to generate the responses:
* `Meta-Llama-3.1-8B-Instruct`
* `Qwen2.5-14B-Instruct`
* `Mistral-Small-24B-Instruct-2501`
* `Qwen2.5-32B-Instruct`

Please refer to the [Hugging Face dataset card](https://huggingface.co/datasets/ledengary/CCPS) for detailed structure and instructions on how to load specific subsets.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ledengary/CCPS.git
    cd CCPS
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv ccps_env
    source ccps_env/bin/activate  # On Windows: ccps_env\Scripts\activate
    ```
3.  **Install dependencies:**

    This project was developed using Python 3.12.9. All required packages are listed in the `requirements.txt` file. To install them:

    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** We recommend installing `vllm` and `unsloth` from source for optimal performance and compatibility. See:
    > - https://github.com/vllm-project/vllm
    > - https://github.com/unslothai/unsloth

4.  **Data:** Download the required data splits from [https://huggingface.co/datasets/ledengary/CCPS](https://huggingface.co/datasets/ledengary/CCPS) and place them in a local `data/` directory (or adjust paths in scripts accordingly). For example, `data/CT-CHOICE/Meta-Llama-3.1-8B-Instruct/train.jsonl`.

## Workflow and Usage

The CCPS pipeline involves several sequential steps. Ensure you have downloaded the necessary base LLMs (e.g., via `unsloth` or HuggingFace Hub) and the datasets.

### Step 0: Answer Generation & Assessment
The `preprocessing/` directory contains utility scripts that were used for the initial generation of LLM responses and their subsequent assessment (labeling), which formed the basis of the datasets now hosted on Hugging Face. Users of this repository primarily focused on reproducing CCPS results with the provided datasets might not need to run these preprocessing scripts directly, but they offer insight into the data creation pipeline.

* **`answer_with_vllm.py`**: This script is used for efficiently generating answers from the base LLMs using the vLLM library. It supports different configurations for MC and OE answer generation.

    **Example Command (MC):**
    ```bash
    python preprocessing/answer_with_vllm.py \
        --visible_cudas "0,1" \
        --data_location "../../data/CT-CHOICE/" \
        --output_dir "../../data/CT-CHOICE/" \
        --output_subdir "OOTB" \
        --llm_id "unsloth/Qwen2.5-32B-Instruct" \
        --llm_dir "unsloth/Qwen2.5-32B-Instruct" \
        --dtype "bfloat16" \
        --temp 0 \
        --gpu_memory 0.9 \
        --tensor_parallel 2 \
        --max_seq_len 1 \
        --chat_template "qwen"
    ```
    **Example Command (OE):**
    ```bash
    python preprocessing/answer_with_vllm.py \
        --visible_cudas "0,1" \
        --data_location "../../data/CT-OE/" \
        --output_dir "../../data/CT-OE/" \
        --output_subdir "OOTB" \
        --llm_id "unsloth/Qwen2.5-32B-Instruct" \
        --llm_dir "unsloth/Qwen2.5-32B-Instruct" \
        --dtype "bfloat16" \
        --temp 0 \
        --gpu_memory 0.9 \
        --tensor_parallel 2 \
        --max_seq_len 30 \
        --chat_template "qwen"
    ```

* **`answer_assessment.py`**: This script handles the labeling of the LLM-generated responses as correct or incorrect. For MC, this involves string matching. For OE responses, it queries `gpt-4o-mini` via its API.

    **Example Commands:**
    ```bash
    # For MC using substring matching
    python preprocessing/answer_assessment.py --data_dir "../../data/CT-CHOICE/answered/" --output_dir "../../data/CT-CHOICE/labeled/" --grading_method "substring"

    # For OE using GPT-based grading
    python preprocessing/answer_assessment.py --data_dir "../../data/CT-OE/answered/" --output_dir "../../data/CT-OE/labeled/" --grading_method "gpt"

### Step 1: Hidden State & Perturbation Data Extraction
This step extracts the original final hidden states, original logits, Jacobian vectors, and perturbed hidden states/logits for each token in the LLM-generated answers.

**Script:** `models/origpert_hidden_state_logit_extraction.py`

**Example Command:**
```bash
python models/origpert_hidden_state_logit_extraction.py \
    --gpu_ids 0 \
    --llm_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --test_dataset_name "MMLU-CHOICE" \ # Or "CT-CHOICE-WRG", "MMLU-PRO-CHOICE", "CT-OE-WRG", "MMLU-OE".
    --pei_radius 20.0 \
    --pei_steps 5 \
    --output_dir "../representations/OrigPert" \ # Adjust path as needed
    --use_unsloth # If using Unsloth for model loading
```
The `--test_dataset_name` argument specifies the dataset to process (e.g., MMLU-CHOICE). The script expects corresponding data subdirectories as outlined in the Data section. This script generates .npz files for states/logits and metadata.jsonl in the `--output_dir`.

### Step 2: Feature Calculation
This step takes the raw data from Step 1 (original/perturbed hidden states and logits) and calculates the $D_f=75$ features for each token.

**Script:** `models/origpert_feature_extraction.py`

**Example Command:**
```bash
python models/origpert_feature_extraction.py \
    --gpu_ids 0 \
    --input_dir "../representations/OrigPert" \ # Should match output_dir from Step 1
    --test_dataset_name "CT-CHOICE-WRG" \ # Or MMLU-CHOICE, MMLU-PRO-CHOICE, etc.
    --llm_id "unsloth/Qwen2.5-32B-Instruct" \ # Corresponding LLM for the input data
    --output_dir "../features" \ # Adjust path as needed
    --eps_search_low 0.01 \
    --eps_search_high 20.0 \
    --use_unsloth # If using Unsloth for model loading (if model is needed in this script)
```
The `--input_dir` should point to the directory where outputs from Step 1 for the specified `--test_dataset_name` and `--llm_id` are stored. This script generates .pkl and .csv files containing the features, organized by dataset and LLM, within the `--output_dir`.

### Step 3: Contrastive Model Training
This step trains the contrastive encoder ($E_{MC}$ for MC or $E_{OE}$ for OE) using the features extracted in Step 2. The goal is to learn discriminative embeddings for the features.

**Script for MC:** `models/ccps_contrastive_train.py`

**Example Command:**
```bash
python models/ccps_contrastive_train.py \
    --visible_cudas 0 \
    --feature_dir "../features" \
    --train_dataset_name "CT-CHOICE-WRG" \
    --test_dataset_name "MMLU-CHOICE" \ # Used for naming output model directory structure
    --llm_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --val_ratio 0.1 \
    --hidden_dims "64,32,16" \
    --embed_dim 8 \
    --activation "elu" \
    --dropout 0.05 \
    --loss_type "contrastive" \
    --margin 1.0 \
    --batch_size 64 \
    --train_steps 5000 \
    --eval_steps 500 \
    --log_steps 25 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --seed 23 \
    --num_workers 1 \
    --output_dir "../trained_models/CCPS/contrastive_ov" \ # Adjust path as needed
```

**Script for OE:** `models/ccps_oe_contrastive_train.py`

**Example Command:**
```bash
python models/ccps_oe_contrastive_train.py \
    --visible_cudas 0 \
    --feature_dir "../features" \
    --train_dataset_name "CT-OE-WRG" \
    --test_dataset_name "MMLU-OE" \ # Used for naming output model directory structure
    --llm_id "unsloth/Qwen2.5-32B-Instruct" \
    --val_ratio 0.1 \
    --hidden_dims "64,32,16" \
    --kernel_sizes "3,3,3" \
    --embed_dim 16 \
    --activation "relu" \
    --dropout 0.05 \
    --loss_type "contrastive" \
    --margin 1.0 \
    --batch_size 32 \
    --train_steps 5000 \
    --eval_steps 500 \
    --log_steps 25 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --seed 23 \
    --num_workers 4 \
    --max_seq_length 30 \
    --output_dir "../trained_models/CCPS/contrastive_ov" \ # Adjust path as needed
```
These scripts save the trained contrastive model (.pt), the feature scaler (.pkl), and the training configuration (config.json) in the specified `--output_dir`, organized by dataset and LLM.

### Step 4: Classifier Model Training
This step trains the final classifier head ($C$) on top of the contrastive encoder ($E_{MC}$ or $E_{OE}$) trained in Step 3. This stage fine-tunes the encoder and trains the classifier head jointly by default.

**Script for MC:** `models/ccps_classifier_train.py`

**Example Command:**
```bash
python models/ccps_classifier_train.py \
    --visible_cudas "0" \
    --feature_dir "../features" \
    --train_dataset_name "CT-CHOICE-WRG" \
    --test_dataset_name "MMLU-PRO-CHOICE" \ # Used for naming output model directory structure
    --llm_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --val_ratio 0.1 \
    --contrastive_model_path "../trained_models/CCPS/contrastive_ov" \ # Path from Step 3
    --classifier_hidden_dims "48,24,12" \
    --activation "elu" \
    --batch_size 32 \
    --train_steps 5000 \
    --eval_steps 500 \
    --log_steps 25 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --seed 23 \
    --num_workers 1 \
    --output_dir "../trained_models/CCPS/classifier_ov" \ # Adjust path as needed
```

**Script for OE:** `models/ccps_oe_classifier_train.py`

**Example Command:**
```bash
python models/ccps_oe_classifier_train.py \
    --visible_cudas "0" \
    --feature_dir "../features" \
    --train_dataset_name "CT-OE-WRG" \
    --test_dataset_name "MMLU-OE" \ # Used for naming output model directory structure
    --llm_id "unsloth/Qwen2.5-32B-Instruct" \
    --val_ratio 0.1 \
    --max_seq_length 30 \
    --contrastive_model_path "../trained_models/CCPS/contrastive_ov" \ # Path from Step 3
    --classifier_hidden_dims "32" \
    --activation "relu" \
    --batch_size 32 \
    --train_steps 5000 \
    --eval_steps 500 \
    --log_steps 25 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --seed 23 \
    --num_workers 1 \
    --output_dir "../trained_models/CCPS/classifier_ov" \ # Adjust path as needed
```
These scripts save the trained final classifier model (classifier_model.pt or classifier_oe_model.pt) and its training configuration (config.json) in the specified `--output_dir`, organized by dataset and LLM.

### Step 5: Evaluation
This step uses the trained CCPS models (contrastive encoder + classifier) from Step 3 and Step 4 to predict confidence scores on the test sets and calculate the final evaluation metrics.

**Script for MC:** `models/ccps_uncertainty.py`

**Example Command:**
```bash
python evaluation/ccps_uncertainty.py \
    --visible_cudas "0" \
    --test_data_dir "../data/MMLU-PRO-CHOICE/tasks" \ # Path to the root of MMLU-style test task folders
    --data_subdir "OOTB-F" \ # Subdirectory structure within each task folder
    --test_dataset_name "MMLU-PRO-CHOICE" \
    --pretrained_dataset_name "CT-CHOICE-WRG" \ # Name of the training dataset used for the loaded models
    --llm_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --feature_dir "../features" \ # Directory to load test set features
    --contrastive_model_path "../trained_models/CCPS/contrastive_ov" \
    --classifier_model_path "../trained_models/CCPS/classifier_ov" \
    --results_dir "../results/CCPS-ov_uncertainty" \ # Adjust path as needed
    --batch_size 32
```

**Script for OE:** `models/ccps_oe_uncertainty.py`

**Example Command:**
```bash
python evaluation/ccps_oe_uncertainty.py \
    --visible_cudas "0" \
    --test_data_dir "../data/MMLU-OE/tasks" \ # Path to the root of MMLU-style test task folders
    --data_subdir "OOTB-F" \ # Subdirectory structure within each task folder
    --test_dataset_name "MMLU-OE" \
    --pretrained_dataset_name "CT-OE-WRG" \ # Name of the training dataset used for the loaded models
    --llm_id "unsloth/Qwen2.5-32B-Instruct" \
    --feature_dir "../features" \ # Directory to load test set features
    --contrastive_model_path "../trained_models/CCPS/contrastive_ov" \
    --classifier_model_path "../trained_models/CCPS/classifier_ov" \
    --results_dir "../results/CCPS-ov_uncertainty" \ # Adjust path as needed
    --max_seq_length 30 \
    --batch_size 32
```

These scripts output performance metrics (e.g., ECE, Brier, ACC, AUCPR, AUROC), saved as JSON files within a subdirectory structure under the specified `--results_dir`.

## License
This project is licensed under the **MIT License** - see the LICENSE file for details

## Citation
If you use CCPS or this codebase in your research, please cite our paper:

```bibtex
@misc{ccps,
      title={Calibrating LLM Confidence by Probing Perturbed Representation Stability}, 
      author={Reza Khanmohammadi and Erfan Miahi and Mehrsa Mardikoraem and Simerjot Kaur and Ivan Brugere and Charese H. Smiley and Kundan Thind and Mohammad M. Ghassemi},
      year={2025},
      eprint={2505.21772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21772}, 
}
