import gc
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_extraction.log")
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Extract Features from Saved Representations')

# Input configuration
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs to use')
parser.add_argument('--input_dir', type=str, required=True,
                    help='Directory where representations are saved')
parser.add_argument('--test_dataset_name', type=str, required=True,
                    help='Test dataset name (e.g., CT-CHOICE-WRG)')
parser.add_argument('--llm_id', type=str, required=True,
                    help='LLM model identifier')
parser.add_argument('--use_unsloth', action='store_true', 
                    help='Use Unsloth for model loading')
parser.add_argument('--dtype', type=str, default='float16', 
                    choices=['bfloat16', 'float16', 'float32'], 
                    help='Model dtype')

# Output configuration
parser.add_argument('--output_dir', type=str, default='../features',
                    help='Directory for saving extracted features')

# Feature calculation configuration
parser.add_argument('--eps_search_low', type=float, default=0.0,
                    help='Epsilon search lower bound')
parser.add_argument('--eps_search_high', type=float, default=20.0,
                    help='Epsilon search upper bound')
parser.add_argument('--bisection_iterations', type=int, default=10,
                    help='Number of bisection iterations')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

def load_representations(task_dir, answer_type):
    """Load saved representations for a specific answer type"""
    answer_dir = os.path.join(task_dir, answer_type)
    
    if not os.path.exists(answer_dir):
        logger.warning(f"Directory not found: {answer_dir}")
        return None
    
    try:
        # Load arrays
        original_hidden_states = np.load(os.path.join(answer_dir, 'original_hidden_states.npz'))['data']
        original_logits = np.load(os.path.join(answer_dir, 'original_logits.npz'))['data']
        jacobian_vectors = np.load(os.path.join(answer_dir, 'jacobian_vectors.npz'))['data']
        perturbed_hidden_states = np.load(os.path.join(answer_dir, 'perturbed_hidden_states.npz'))['data']
        
        # Load metadata
        metadata = []
        with open(os.path.join(answer_dir, 'metadata.jsonl'), 'r') as f:
            for line in f:
                metadata.append(json.loads(line))
        
        return {
            'original_hidden_states': original_hidden_states,
            'original_logits': original_logits,
            'jacobian_vectors': jacobian_vectors,
            'perturbed_hidden_states': perturbed_hidden_states,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Error loading representations from {answer_dir}: {e}")
        return None

def generate_perturbed_logits_batch(perturbed_hidden_states, model):
    """Generate logits for all perturbed hidden states in one batch"""
    with torch.no_grad():
        # Convert numpy array to tensor and move to device
        if isinstance(perturbed_hidden_states, np.ndarray):
            perturbed_hidden_states = torch.from_numpy(perturbed_hidden_states).to(model.device)
        
        # Batch process through lm_head
        # Shape: [num_perturbations, hidden_size]
        if perturbed_hidden_states.dim() == 2:
            perturbed_hidden_states = perturbed_hidden_states.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Process in batch
        perturbed_hidden_states.to(model.device)
        logits = model.lm_head(perturbed_hidden_states).squeeze(1)  # [batch, vocab_size]
    
    return logits

def extract_comprehensive_features(original_hidden_state, original_logits, jacobian_vector,
                                 perturbed_hidden_states, actual_token_id, 
                                 perturbation_metadata, model):
    """Extract all comprehensive features for a token"""
    features = {}
    
    # Convert to torch tensors
    H_0 = torch.from_numpy(original_hidden_state).to(model.device)
    L_0 = torch.from_numpy(original_logits).to(model.device)
    J_T = torch.from_numpy(jacobian_vector).to(model.device)
    
    # Generate all perturbed logits at once
    perturbed_logits_tensor = generate_perturbed_logits_batch(perturbed_hidden_states, model)
    
    # I. Features from Original State
    probs_0 = torch.softmax(L_0, dim=-1).squeeze()
    log_probs_0 = torch.log_softmax(L_0, dim=-1).squeeze()
    
    # Get top-2 indices
    top2_values, top2_indices = torch.topk(L_0.squeeze(), 2)
    argmax_0 = top2_indices[0].item()
    second_best_0 = top2_indices[1].item() if len(top2_indices) > 1 else argmax_0
    
    features['original_log_prob_actual'] = log_probs_0[actual_token_id].item()
    features['original_prob_actual'] = probs_0[actual_token_id].item()
    features['original_logit_actual'] = L_0[0, actual_token_id].item()
    features['original_prob_argmax'] = probs_0[argmax_0].item()
    features['original_logit_argmax'] = L_0[0, argmax_0].item()
    
    # Entropy
    features['original_entropy'] = -torch.sum(probs_0 * torch.log(probs_0 + 1e-6)).item()
    
    # Margins
    features['original_margin_logit_top1_top2'] = L_0[0, argmax_0].item() - L_0[0, second_best_0].item()
    features['original_margin_prob_top1_top2'] = probs_0[argmax_0].item() - probs_0[second_best_0].item()
    
    # Norms
    features['original_norm_logits_L2'] = torch.norm(L_0).item()
    features['original_std_logits'] = torch.std(L_0).item()
    features['original_norm_hidden_state_L2'] = torch.norm(H_0).item()
    
    # Boolean features
    features['is_actual_token_original_argmax'] = int(actual_token_id == argmax_0)
    
    # II. Overall Perturbation Metrics
    features['jacobian_norm_token'] = torch.norm(J_T).item()
    
    # Calculate epsilon-to-flip using the pre-generated logits
    original_argmax = torch.argmax(L_0).item()
    epsilon_to_flip = float('inf')
    for i, metadata in enumerate(perturbation_metadata):
        if torch.argmax(perturbed_logits_tensor[i]).item() != original_argmax:
            epsilon_to_flip = metadata['perturbation_radius']
            break
    features['epsilon_to_flip_token'] = epsilon_to_flip
    
    # Calculate PEI using pre-generated logits
    log_probs_original = torch.log_softmax(L_0, dim=-1)
    log_p_original = log_probs_original[0, actual_token_id].item()
    f_values = [0.0]
    
    perturbed_log_probs = torch.log_softmax(perturbed_logits_tensor, dim=-1)
    for i in range(len(perturbed_hidden_states)):
        log_p_perturbed = perturbed_log_probs[i, actual_token_id].item()
        f_k = log_p_original - log_p_perturbed
        f_values.append(max(0.0, f_k))
    
    pei_steps = len(perturbed_hidden_states)
    pei_value = 0.0
    for k in range(pei_steps):
        pei_value += (f_values[k] + f_values[k+1]) / 2.0
    pei_value = pei_value / pei_steps if pei_steps > 0 else 0.0
    
    features['pei_value_token'] = pei_value
    features['pei_curve_token'] = f_values
    
    # III. Features from Perturbed States
    perturbed_features = {
        'perturbed_log_prob_actual': [],
        'perturbed_prob_actual': [],
        'perturbed_logit_actual': [],
        'delta_log_prob_actual_from_original': [],
        'perturbed_prob_argmax': [],
        'perturbed_logit_argmax': [],
        'did_argmax_change_from_original': [],
        'perturbed_entropy': [],
        'perturbed_margin_logit_top1_top2': [],
        'perturbed_norm_logits_L2': [],
        'kl_div_perturbed_from_original': [],
        'js_div_perturbed_from_original': [],
        'cosine_sim_logits_perturbed_to_original': [],
        'cosine_sim_hidden_perturbed_to_original': [],
        'l2_dist_hidden_perturbed_from_original': []
    }
    
    # Process all perturbations using pre-generated logits
    perturbed_probs = torch.softmax(perturbed_logits_tensor, dim=-1)
    
    for i in range(len(perturbed_hidden_states)):
        H_p = torch.from_numpy(perturbed_hidden_states[i]).to(model.device)
        L_p = perturbed_logits_tensor[i].to(model.device)
        probs_p = perturbed_probs[i].to(model.device)
        log_probs_p = perturbed_log_probs[i].to(model.device)
        
        # Basic metrics
        perturbed_features['perturbed_log_prob_actual'].append(log_probs_p[actual_token_id].item())
        perturbed_features['perturbed_prob_actual'].append(probs_p[actual_token_id].item())
        perturbed_features['perturbed_logit_actual'].append(L_p[actual_token_id].item())
        
        # Delta from original
        delta_log_prob = log_probs_0[actual_token_id].item() - log_probs_p[actual_token_id].item()
        perturbed_features['delta_log_prob_actual_from_original'].append(delta_log_prob)
        
        # Argmax features
        argmax_p = torch.argmax(L_p).item()
        perturbed_features['perturbed_prob_argmax'].append(probs_p[argmax_p].item())
        perturbed_features['perturbed_logit_argmax'].append(L_p[argmax_p].item())
        perturbed_features['did_argmax_change_from_original'].append(int(argmax_p != argmax_0))
        
        # Entropy
        entropy_p = -torch.sum(probs_p * torch.log(probs_p + 1e-6)).item()
        perturbed_features['perturbed_entropy'].append(entropy_p)
        
        # Margin
        top2_p = torch.topk(L_p, 2).indices
        if len(top2_p) >= 2:
            margin = L_p[top2_p[0]].item() - L_p[top2_p[1]].item()
            perturbed_features['perturbed_margin_logit_top1_top2'].append(margin)
        else:
            perturbed_features['perturbed_margin_logit_top1_top2'].append(0.0)
        
        # Norms
        perturbed_features['perturbed_norm_logits_L2'].append(torch.norm(L_p).item())
        
        # Divergences
        kl_div = torch.nn.functional.kl_div(log_probs_p, probs_0, reduction='sum').item()
        perturbed_features['kl_div_perturbed_from_original'].append(kl_div)
        
        # JS divergence
        m_probs = 0.5 * (probs_0 + probs_p)
        js_div = 0.5 * torch.nn.functional.kl_div(log_probs_0, m_probs, reduction='sum') + \
                 0.5 * torch.nn.functional.kl_div(log_probs_p, m_probs, reduction='sum')
        perturbed_features['js_div_perturbed_from_original'].append(js_div.item())
        
        # Similarities
        cos_sim_logits = torch.nn.functional.cosine_similarity(L_0.squeeze(), L_p.squeeze(), dim=0).item()
        perturbed_features['cosine_sim_logits_perturbed_to_original'].append(cos_sim_logits)
        
        cos_sim_hidden = torch.nn.functional.cosine_similarity(H_0.squeeze(), H_p.squeeze(), dim=0).item()
        perturbed_features['cosine_sim_hidden_perturbed_to_original'].append(cos_sim_hidden)
        
        # L2 distance
        l2_dist = torch.norm(H_p - H_0).item()
        perturbed_features['l2_dist_hidden_perturbed_from_original'].append(l2_dist)
    
    # Add summary statistics
    for metric_name, values in perturbed_features.items():
        if values:
            features[f'{metric_name}_min'] = min(values)
            features[f'{metric_name}_max'] = max(values)
            features[f'{metric_name}_mean'] = np.mean(values)
            features[f'{metric_name}_std'] = np.std(values) if len(values) > 1 else 0.0
    
    return features

def process_answer_type(representations, answer_type, model, args):
    """Process all tokens for a specific answer type"""
    if representations is None:
        return None
    
    all_features = []
    
    for sample_meta in tqdm(representations['metadata'], desc=f"Processing {answer_type}"):
        sample_idx = sample_meta['sample_idx']
        hash_id = sample_meta['hash_id']
        query_label = sample_meta['query_label']
        task_name = sample_meta['task_name']
        
        for token_meta in sample_meta['tokens']:
            # Extract data for this token
            token_idx = token_meta['token_idx']
            token_id = token_meta['token_id']
            token_str = token_meta['token_str']
            
            # Get indices
            orig_hidden_idx = token_meta['original_hidden_state_idx']
            orig_logits_idx = token_meta['original_logits_idx']
            jacobian_idx = token_meta['jacobian_idx']
            perturb_start = token_meta['perturbed_start_idx']
            perturb_end = token_meta['perturbed_end_idx']
            
            # Get data
            original_hidden = representations['original_hidden_states'][orig_hidden_idx]
            original_logits = representations['original_logits'][orig_logits_idx]
            jacobian = representations['jacobian_vectors'][jacobian_idx]
            
            # Get perturbed hidden states
            perturbed_hidden_states = representations['perturbed_hidden_states'][perturb_start:perturb_end+1]
            
            # Get perturbation metadata
            perturbation_metadata = token_meta['perturbation_metadata']
            
            # Calculate features
            features = extract_comprehensive_features(
                original_hidden, original_logits, jacobian,
                perturbed_hidden_states, token_id,
                perturbation_metadata, model
            )
            
            # Add metadata
            features.update({
                'hash_id': hash_id,
                'task_name': task_name,
                'sample_idx_in_task': sample_idx,
                'token_idx_in_response': token_idx,
                'token_str': token_str,
                'token_id': token_id,
                'query_label_sample': query_label,
                'answer_type': answer_type
            })
            
            # Add wrong answer index if present
            if 'wrong_answer_idx' in sample_meta:
                features['wrong_answer_idx'] = sample_meta['wrong_answer_idx']
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)

def save_features(df, output_dir, filename):
    """Save features to CSV and pickle"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved features to {csv_path}")
    
    # Save as pickle
    pkl_path = os.path.join(output_dir, f"{filename}.pkl")
    df.to_pickle(pkl_path)
    logger.info(f"Saved features to {pkl_path}")

def load_model_and_tokenizer(llm_id, use_unsloth, dtype_str):
    logger.info(f"Loading model and tokenizer: {llm_id}")
    
    if dtype_str == "bfloat16": dtype = torch.bfloat16
    elif dtype_str == "float16": dtype = torch.float16
    else: dtype = torch.float32

    if use_unsloth:
        logger.info("Using unsloth for model loading.")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_id, max_seq_length=5000, dtype=dtype, load_in_4bit=False
        )
    else:
        logger.info("Using transformers for model loading.")
        tokenizer = AutoTokenizer.from_pretrained(llm_id)
        model = AutoModelForCausalLM.from_pretrained(
            llm_id, torch_dtype=dtype, quantization_config=None,
            trust_remote_code=True, use_cache=False, device_map='auto'
        )
    model.eval()
    return model, tokenizer

def main():
    # Construct input directory path
    input_dir = os.path.join(
        args.input_dir,
        args.test_dataset_name,
        args.llm_id.replace('/', '-'),
    )
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.llm_id,
        args.use_unsloth,
        args.dtype
    )
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir,
        args.test_dataset_name,
        args.llm_id.replace('/', '-')
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "feature_extraction_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Process each task
    task_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for task_name in task_dirs:
        logger.info(f"\nProcessing task: {task_name}")
        task_input_dir = os.path.join(input_dir, task_name)
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Process each answer type
        for answer_type in ['llm_output', 'target', 'wrong_answer']:
            logger.info(f"Processing {answer_type} for task {task_name}")
            
            # Load representations
            representations = load_representations(task_input_dir, answer_type)
            
            if representations is not None:
                # Calculate features
                features_df = process_answer_type(representations, answer_type, model, args)
                
                if features_df is not None and not features_df.empty:
                    # Save features
                    save_features(features_df, task_output_dir, f"{answer_type}_features")
                    logger.info(f"Processed {len(features_df)} tokens for {answer_type}")
            else:
                logger.warning(f"No representations found for {answer_type} in {task_name}")
    
    logger.info(f"\nFeature extraction complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()


## New LLMs:
# "unsloth/Meta-Llama-3.1-8B-Instruct"
# "unsloth/Qwen2.5-14B-Instruct"
# "unsloth/Mistral-Small-24B-Instruct-2501"
# "unsloth/Qwen2.5-32B-Instruct"

# python origpert_feature_extraction.py \
#     --gpu_ids "3" \
#     --input_dir "../representations/OrigPert" \
#     --test_dataset_name "CT-CHOICE-WRG" \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --output_dir "../features" \
#     --eps_search_low 0.01 \
#     --eps_search_high 20 \
#     --bisection_iterations 10 \
#     --use_unsloth

# 2>&1 | tee origpert_feature_ct_choice.log

# python origpert_feature_extraction.py \
#     --gpu_ids "6" \
#     --input_dir "../representations/OrigPert" \
#     --test_dataset_name "CT-OE-WRG" \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --output_dir "../features" \
#     --eps_search_low 0.01 \
#     --eps_search_high 20 \
#     --bisection_iterations 10 \
#     --use_unsloth

# python origpert_feature_extraction.py \
#     --gpu_ids "4" \
#     --input_dir "../representations/OrigPert" \
#     --test_dataset_name "MMLU-CHOICE" \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --output_dir "../features" \
#     --eps_search_low 0.01 \
#     --eps_search_high 20 \
#     --bisection_iterations 10 \
#     --use_unsloth

# python origpert_feature_extraction.py \
#     --gpu_ids "5" \
#     --input_dir "../representations/OrigPert" \
#     --test_dataset_name "MMLU-PRO-CHOICE" \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --output_dir "../features" \
#     --eps_search_low 0.01 \
#     --eps_search_high 20 \
#     --bisection_iterations 10 \
#     --use_unsloth

# python origpert_feature_extraction.py \
#     --gpu_ids "7" \
#     --input_dir "../representations/OrigPert" \
#     --test_dataset_name "MMLU-OE" \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --output_dir "../features" \
#     --eps_search_low 0.01 \
#     --eps_search_high 20 \
#     --bisection_iterations 10 \
#     --use_unsloth