import sys
import argparse
import os
import json
import copy
import pickle
import numpy as np
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Perform uncertainty estimation using pretrained CCPS OE model on test data")
parser.add_argument('--visible_cudas', type=str, required=True, help='Visible CUDA devices')
parser.add_argument('--test_data_dir', type=str, required=True, help='Test data directory (root of tasks)')
parser.add_argument('--data_subdir', type=str, required=True, help='Subdirectory under each task where assessed outputs are saved')
parser.add_argument('--test_dataset_name', type=str, required=True, help='Test dataset name')
parser.add_argument('--pretrained_dataset_name', type=str, required=True, help='Train dataset name (used for the contrastive model)')
parser.add_argument('--tasks', type=str, required=False, nargs='*', help='List of Tasks to process', default=None)
parser.add_argument('--llm_id', type=str, required=True, help='LLM ID')
parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing feature files')
parser.add_argument('--contrastive_model_path', type=str, required=True, help='Path to the trained contrastive OE model directory')
parser.add_argument('--classifier_model_path', type=str, required=True, help='Path to the trained classifier OE model directory')
parser.add_argument('--results_dir', type=str, required=True, help='Directory to save uncertainty estimation results')
parser.add_argument('--seed', type=int, required=False, help="Seed for reproducibility", default=23)
parser.add_argument('--batch_size', type=int, required=False, help="Batch size for evaluation", default=32)
parser.add_argument('--dtype', type=str, required=False, help="Data type to use", default="bfloat16")
parser.add_argument('--device', type=str, required=False, help="Device to use", default="cuda")
parser.add_argument('--max_seq_length', type=int, required=False, help="Maximum sequence length for tokens", default=30)
parser.add_argument('--isoreg_path', type=str, default=None, help='Path to the Isotonic Regression model directory')
parser.add_argument('--eps_search_high', type=float, default=20, help='maximum eps searched for')
args = parser.parse_args()

print('=' * 50)
print('args:', args)
print('=' * 50)

# Set the GPU number before further imports
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_cudas)
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Import from the utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (
    set_visible_cudas, 
    seed_everything,
    get_dtype,
    check_all_task_records_are_uncertainty_estimated,
    compute_task_metrics,
    save_task_data_and_metrics,
    check_all_task_records_are_answered,
    check_all_task_records_are_assessed,
)
sys.path = original_sys_path

# Set seed and visible CUDA devices
seed_everything(args.seed)
set_visible_cudas(args.visible_cudas)

original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from ccps import (
    ConvEmbeddingNet,
    ClassifierWithEmbedding
)
sys.path = original_sys_path

def remove_eot_token_ids(data, llm_id):
    """Remove end-of-text tokens from the data"""
    def return_eot_token_ids(llm_id):
        if "llama" in llm_id.lower():
            return [128009]
        elif "mistral" in llm_id.lower():
            return [2]
        elif "qwen" in llm_id.lower():
            return [151645, 198]
        else:
            raise ValueError(f"Unknown LLM ID: {llm_id}")
    
    eot_token_ids = return_eot_token_ids(llm_id)
    data = data[~data['token_id'].isin(eot_token_ids)]
    return data

def load_answered_task_data(task, data_subdir, llm_id, test_data_dir):
    """
    Loads the previously saved answered and assessed task data.
    Assumes the file is saved as 'answered_data.pkl' under:
      {test_data_dir}/{task}/answered/{data_subdir}/{llm_id_replaced}/
    """
    llm_id_dir = llm_id.replace('/', '-')
    data_path = os.path.join(test_data_dir, task, "answered", data_subdir, llm_id_dir, "task_data_answered.pkl")
    print("Loading assessed task data from:", data_path)
    with open(data_path, "rb") as f:
        task_data = pickle.load(f)
    print('Loaded', len(task_data), 'records')
    unique_hash_ids = list(set([item['hash_id'] for item in task_data]))
    print(f"Source - Count of unique hash_ids: {len(unique_hash_ids)}")
    return task_data

def load_model_config(model_path):
    """Load model configuration from a JSON file"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_contrastive_model(contrastive_model_path, device, dtype):
    """Load the pretrained contrastive model for OE tasks"""
    print("Loading contrastive model from:", contrastive_model_path)
    config = load_model_config(contrastive_model_path)
    
    # Get model architecture details
    model_config = config.get("model", {})
    input_dim = model_config.get("input_dim")
    embed_dim = model_config.get("embed_dim", config.get("embed_dim", 4))
    
    # Get hidden_dims and kernel_sizes
    hidden_dims_str = config.get("hidden_dims", "64,64")
    kernel_sizes_str = config.get("kernel_sizes", "3,3")
    
    hidden_dims = [int(dim) for dim in hidden_dims_str.split(',')]
    kernel_sizes = [int(k) for k in kernel_sizes_str.split(',')]
    
    activation = config.get("activation", "relu")
    dropout = config.get("dropout", 0.2)
    
    # Create model with same architecture
    model = ConvEmbeddingNet(
        input_dim=input_dim, 
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        kernel_sizes=kernel_sizes,
        activation=activation,
        dropout=dropout
    )
    
    # Load model weights
    model_path = os.path.join(contrastive_model_path, "contrastive_oe_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device, dtype=dtype)
    model.eval()
    
    print(f"Contrastive model loaded successfully.")
    
    # Load scaler
    scaler_path = os.path.join(contrastive_model_path, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load feature columns
    columns_path = os.path.join(contrastive_model_path, "feature_columns.json")
    with open(columns_path, "r") as f:
        feature_columns = json.load(f)
    
    return model, scaler, feature_columns, embed_dim, input_dim

def load_classifier_model(classifier_model_path, contrastive_model, embed_dim, device, dtype):
    """Load the pretrained classifier model for OE tasks"""
    print("Loading classifier model from:", classifier_model_path)
    config = load_model_config(classifier_model_path)
    
    # Parse classifier hidden dimensions if provided
    classifier_hidden_dims = None
    if config.get("classifier_hidden_dims"):
        classifier_hidden_dims = [int(dim) for dim in config["classifier_hidden_dims"].split(',')]
    
    # Create classifier model
    model = ClassifierWithEmbedding(
        embedding_model=contrastive_model,
        embed_dim=embed_dim,
        hidden_dims=classifier_hidden_dims,
        num_classes=2,
        activation=config.get("activation", "relu"),
        dropout=config.get("dropout", 0.2),
        use_dropout=config.get("use_dropout", True)
    )
    
    # Load model weights
    model_path = os.path.join(classifier_model_path, "classifier_oe_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device, dtype=dtype)
    model.eval()
    
    print(f"Classifier model loaded successfully.")
    return model

def load_test_features(task, feature_dir, llm_id):
    """Load the test features for a given task"""
    task_path = os.path.join(feature_dir, task, "llm_output_features.pkl")
    print("Loading test features from:", task_path)
    
    if os.path.exists(task_path):
        with open(task_path, "rb") as f:
            data = pickle.load(f)
        data = remove_eot_token_ids(data, llm_id)
        print(f"Loaded test features with shape {data.shape}")
        unique_hash_ids = list(set(data['hash_id'].values))
        print(f"Feature - Count of unique hash_ids: {len(unique_hash_ids)}")
        return data
    else:
        print(f"File path {task_path} does not exist.")
        return None

def prepare_sequential_data(test_features, exclude_cols, scaler, feature_columns, max_seq_length, eps_search_high):
    """Process and prepare sequential data from test features"""
    print("Processing sequential data...")
    
    # Extract identifiers for each sample
    sample_identifiers = test_features[['hash_id', 'sample_idx_in_task', 'task_name']].drop_duplicates()
    print(f"Found {len(sample_identifiers)} unique samples")
    
    # Create a list to store processed samples
    processed_samples = []
    
    # Process each sample
    for idx, identifier in tqdm(sample_identifiers.iterrows(), total=len(sample_identifiers), desc="Processing samples"):
        sample_data = test_features[
            (test_features['hash_id'] == identifier['hash_id']) & 
            (test_features['sample_idx_in_task'] == identifier['sample_idx_in_task']) & 
            (test_features['task_name'] == identifier['task_name'])
        ]
        
        # Sort by token index to ensure correct sequence order
        sample_data = sample_data.sort_values('token_idx_in_response')
        
        # Extract features (drop exclude columns)
        feat_cols = [col for col in feature_columns if col in test_features.columns]
        if len(feat_cols) != len(feature_columns):
            missing_cols = [col for col in feature_columns if col not in test_features.columns]
            print(f"Warning: Missing {len(missing_cols)} feature columns in test data. Examples: {missing_cols[:5]}")
        
        features = sample_data[feat_cols].copy()
        
        # Replace inf values with eps_search_high for epsilon_to_flip_token
        if 'epsilon_to_flip_token' in features.columns:
            features['epsilon_to_flip_token'] = features['epsilon_to_flip_token'].replace([np.inf], eps_search_high)
        
        # Replace all inf/-inf with NaN
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        # Skip sample if no valid tokens
        if len(features) == 0:
            continue
        
        # Save processed sample
        processed_samples.append({
            'hash_id': identifier['hash_id'],
            'sample_idx_in_task': identifier['sample_idx_in_task'],
            'task_name': identifier['task_name'],
            'features': features.values,  # Numpy array of features
            'seq_length': len(features),  # Actual sequence length before padding
        })
    
    # Apply scaling to all samples
    for sample in processed_samples:
        sample['features_scaled'] = scaler.transform(sample['features'])
    
    return processed_samples

def evaluate_test_sequences(classifier_model, test_sequences, batch_size, device, dtype, max_seq_length):
    """Evaluate test sequences with the classifier model"""
    model = classifier_model
    model.eval()
    
    all_logits = []
    all_probs = []
    all_hash_ids = []
    
    # Process data in batches
    for i in range(0, len(test_sequences), batch_size):
        batch_end = min(i + batch_size, len(test_sequences))
        batch_samples = test_sequences[i:batch_end]
        
        # Prepare batch data
        batch_features = [s['features_scaled'] for s in batch_samples]
        batch_seq_lens = [min(s['seq_length'], max_seq_length) for s in batch_samples]
        batch_hash_ids = [s['hash_id'] for s in batch_samples]
        
        # Get feature dimension
        feature_dim = batch_features[0].shape[1] if len(batch_features) > 0 else 0
        
        # Pad sequences
        batch_padded = np.zeros((len(batch_features), max_seq_length, feature_dim))
        for j, (features, seq_len) in enumerate(zip(batch_features, batch_seq_lens)):
            batch_padded[j, :seq_len, :] = features[:seq_len]
        
        # Convert to tensors
        batch_tensor = torch.tensor(batch_padded, dtype=dtype).to(device)
        seq_lens_tensor = torch.tensor(batch_seq_lens, dtype=torch.long).to(device)
        
        # Evaluate
        with torch.no_grad():
            logits, _ = model(batch_tensor, seq_lens_tensor)
            probs = F.softmax(logits, dim=1)
            
            all_logits.append(logits.cpu().float().numpy())
            all_probs.append(probs.cpu().float().numpy())
            all_hash_ids.extend(batch_hash_ids)
    
    # Concatenate results
    all_logits = np.vstack(all_logits) if all_logits else np.array([])
    all_probs = np.vstack(all_probs) if all_probs else np.array([])
    
    return all_logits, all_probs, all_hash_ids

def match_predictions_to_task_data(task_data, all_hash_ids, all_logits, all_probs):
    """Match predictions to task data based on hash_id"""
    print(f"Matching predictions to task data for {len(all_hash_ids)} records")
    task_data_with_predictions = copy.deepcopy(task_data)
    
    # Create a map from hash_id to predictions for fast lookup
    hash_id_to_preds = {}
    for i, hash_id in enumerate(all_hash_ids):
        hash_id_to_preds[hash_id] = {
            'logits': all_logits[i].tolist(),
            'p_false': float(all_probs[i, 0]),
            'p_true': float(all_probs[i, 1])
        }
    
    # Add predictions to task data
    matched_count = 0
    for item in task_data_with_predictions:
        hash_id = item['hash_id']
        if hash_id in hash_id_to_preds:
            preds = hash_id_to_preds[hash_id]
            item['logits'] = preds['logits']
            item['p_false'] = preds['p_false']
            item['p_true'] = preds['p_true']
            matched_count += 1
    
    print(f"Matched {matched_count} records out of {len(task_data_with_predictions)}")
    
    # Check if all records were matched
    if matched_count < len(task_data_with_predictions):
        print(f"Warning: {len(task_data_with_predictions) - matched_count} records were not matched with predictions")
    
    return task_data_with_predictions

def setup_path():
    results_dir = os.path.join(args.results_dir, args.data_subdir, args.test_dataset_name, args.llm_id.replace('/', '-'))
    contrastive_model_path = os.path.join(args.contrastive_model_path, f"{args.pretrained_dataset_name}_{args.test_dataset_name}", args.llm_id.replace('/', '-'))
    classifier_model_path = os.path.join(args.classifier_model_path, f"{args.pretrained_dataset_name}_{args.test_dataset_name}", args.llm_id.replace('/', '-'))
    feature_dir = os.path.join(args.feature_dir, args.test_dataset_name, args.llm_id.replace('/', '-'))
    return results_dir, contrastive_model_path, classifier_model_path, feature_dir

def compute_overall_metrics(task_metrics_list):
    """Compute average metrics across all tasks"""
    if not task_metrics_list:
        return {}
    
    # Initialize with the first task's metrics structure
    overall_metrics = {}
    for metric_name, metric_value in task_metrics_list[0].items():
        if isinstance(metric_value, dict):
            overall_metrics[metric_name] = {k: [] for k in metric_value.keys()}
        else:
            overall_metrics[metric_name] = []
    
    # Collect metrics from all tasks
    for task_metrics in task_metrics_list:
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    if v is not None:  # Skip None values
                        overall_metrics[metric_name][k].append(v)
            else:
                if metric_value is not None:  # Skip None values
                    overall_metrics[metric_name].append(metric_value)
    
    # Compute averages
    average_metrics = {}
    for metric_name, metric_values in overall_metrics.items():
        if isinstance(metric_values, dict):
            average_metrics[metric_name] = {}
            for k, v in metric_values.items():
                if v:  # Check if the list is not empty
                    average_metrics[metric_name][k] = sum(v) / len(v)
                else:
                    average_metrics[metric_name][k] = None
        else:
            if metric_values:  # Check if the list is not empty
                average_metrics[metric_name] = sum(metric_values) / len(metric_values)
            else:
                average_metrics[metric_name] = None
    
    return average_metrics

def main():
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    results_dir, contrastive_model_path, classifier_model_path, feature_dir = setup_path()
    
    # Load models
    contrastive_model, scaler, feature_columns, embed_dim, input_dim = load_contrastive_model(
        contrastive_model_path, device, dtype
    )
    
    classifier_model = load_classifier_model(
        classifier_model_path, contrastive_model, embed_dim, device, dtype
    )
    
    # Define columns to exclude for feature processing
    exclude_cols = [
        'hash_id', 'task_name', 'sample_idx_in_task', 'token_idx_in_response',
        'token_str', 'token_id', 'query_label_sample', 'answer_type',
        'pei_curve_token', 'wrong_answer_idx',
    ]
    
    # If no specific tasks are provided, assume all task directories in test_data_dir
    tasks = args.tasks
    if tasks is None:
        tasks = [d for d in os.listdir(args.test_data_dir) if os.path.isdir(os.path.join(args.test_data_dir, d))]
    
    # List to store metrics from all tasks
    all_task_metrics = []
    
    for task_idx, task in enumerate(tasks):
        print('=' * 20, f"Processing uncertainty estimation for task: {task} ({task_idx+1}/{len(tasks)})", '=' * 20)
        task_results_dir = os.path.join(results_dir, task)
        os.makedirs(task_results_dir, exist_ok=True)
        final_task_data_file = os.path.join(task_results_dir, "task_data_ue.pkl")
        final_task_metrics_file = os.path.join(task_results_dir, "task_metrics.json")
        
        print('Final task data file:', final_task_data_file)
        print('Final task metrics file:', final_task_metrics_file)
        
        # Load task data
        task_data = load_answered_task_data(task, args.data_subdir, args.llm_id, args.test_data_dir)
        check_all_task_records_are_answered(task_data)
        check_all_task_records_are_assessed(task_data)
        print('All task records have been answered and assessed')
        
        # Load test features
        test_features = load_test_features(task, feature_dir, args.llm_id)
        
        # Process and prepare sequential data
        test_sequences = prepare_sequential_data(
            test_features, exclude_cols, scaler, feature_columns, 
            args.max_seq_length, args.eps_search_high
        )
        print(f"Prepared {len(test_sequences)} test sequences")
        
        # Evaluate test sequences
        all_logits, all_probs, all_hash_ids = evaluate_test_sequences(
            classifier_model, test_sequences, args.batch_size, device, dtype, args.max_seq_length
        )
        
        # Match predictions to task data
        task_data_ue = match_predictions_to_task_data(task_data, all_hash_ids, all_logits, all_probs)
        
        # Check that all records have uncertainty estimation
        check_all_task_records_are_uncertainty_estimated(task_data_ue)
        
        # Compute metrics
        task_metrics = compute_task_metrics(task_data_ue)
        print('Task metrics:', task_metrics)
        
        # Store metrics for overall calculation
        all_task_metrics.append(task_metrics)
        
        # Save results
        save_task_data_and_metrics(task_results_dir, task_data_ue, task_metrics)
        
        print('=' * 20, f"Done with task {task_idx + 1}/{len(tasks)}: {task}", '=' * 20)
    
    # Compute and save overall performance metrics
    if all_task_metrics:
        overall_metrics = compute_overall_metrics(all_task_metrics)
        print('=' * 20, "Overall Performance Metrics", '=' * 20)
        print(json.dumps(overall_metrics, indent=2))
        
        # Save overall metrics to a JSON file next to the task directories
        overall_metrics_file = os.path.join(results_dir, "overall_performance.json")
        with open(overall_metrics_file, 'w') as f:
            json.dump(overall_metrics, f, indent=2)
        print(f"Overall performance metrics saved to: {overall_metrics_file}")
    else:
        print("No task metrics were collected. Overall performance metrics not calculated.")

if __name__ == "__main__":
    main()

## Large llm_ids:
# unsloth/Meta-Llama-3.1-8B-Instruct
# unsloth/Qwen2.5-14B-Instruct
# unsloth/Mistral-Small-24B-Instruct-2501
# unsloth/Qwen2.5-32B-Instruct

# Example usage:
# python ccps_oe_uncertainty.py \
#   --visible_cudas "0" \
#   --test_data_dir "../data/MMLU-OE/tasks" \
#   --data_subdir "OOTB-F" \
#   --test_dataset_name "MMLU-OE" \
#   --pretrained_dataset_name "CT-OE-WRG" \
#   --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#   --feature_dir "../features" \
#   --contrastive_model_path "../trained_models/CCPS/contrastive_ov" \
#   --classifier_model_path "../trained_models/CCPS/classifier_ov" \
#   --results_dir "../results/CCPS-ov_uncertainty" \
#   --max_seq_length 30 \
#   --batch_size 32 \
#   --eps_search_high 20

# 2>&1 | tee ccps_oe_uncertainty.log