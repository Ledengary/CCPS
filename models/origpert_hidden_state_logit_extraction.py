import gc
import os
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extract_hidden_states.log")
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Extract Hidden States, Logits, and Jacobians')

# Model configuration
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs to use')
parser.add_argument('--llm_id', type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct', 
                    help='LLM model identifier')
parser.add_argument('--test_dataset_name', type=str, default='CT-CHOICE-WRG', 
                    help='Test dataset name')
parser.add_argument('--dtype', type=str, default='bfloat16', 
                    choices=['bfloat16', 'float16', 'float32'], 
                    help='Model dtype')
parser.add_argument('--use_unsloth', action='store_true', 
                    help='Use Unsloth for model loading')

# Analysis configuration
parser.add_argument('--max_samples_per_task', type=int, default=None, 
                    help='Maximum samples per task, None for all')

# PEI configuration
parser.add_argument('--pei_radius', type=float, default=10.0, 
                    help='Maximum perturbation radius for PEI calculation')
parser.add_argument('--pei_steps', type=int, default=5, 
                    help='Number of steps for PEI integration')

# Output configuration
parser.add_argument('--output_dir', type=str, default='hidden_states_results', 
                    help='Directory for saving results')

args = parser.parse_args()

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

def extract_choice_options(item, test_dataset_name):
    import re
    if "choice" in test_dataset_name.lower():
        context = item['context']
        parts = context.split("Choices:\n")
        if not parts or len(parts) < 2:
            item['wrong_answers'] = []
            item['all_choices'] = []
            return item
        options_text = parts[-1]
        choices = re.findall(r'\(\s*([a-zA-Z])\s*\):', options_text)
        wrong_targets_str = item.get('wrong_targets', '')
        wrong_letters = []
        if wrong_targets_str:
            wrong_indices = wrong_targets_str.split("<<OPJ>>")
            for idx_str in wrong_indices:
                if idx_str.isdigit():
                    idx = int(idx_str)
                    if 0 <= idx < len(choices):
                        wrong_letters.append(choices[idx])
        item['wrong_answers'] = wrong_letters
        item['all_choices'] = choices
    else:
        wrong_target_texts = item.get('wrong_target_texts', '')
        wrong_answers = []
        if wrong_target_texts:
            wrong_answers = [txt.strip() for txt in wrong_target_texts.split("<<OPJ>>") if txt.strip()]
        item['wrong_answers'] = wrong_answers
        item['all_choices'] = item.get('all_choices', [])
    return item


def ct_system_prompt_choice():
    return "You are an expert who responds with concise, correct answers. For multiple-choice questions, respond only with the letter of the correct option (e.g., a, b, c, d, ...). Do not include any explanation or additional text."

def ct_system_prompt_oe():
    return "You are an expert who responds with concise, correct answers. Directly state the answer without phrases like 'the correct answer is'"

def load_task_data_ct(test_dataset_name, llm_id):
    logger.info(f"Loading CT data from {test_dataset_name}")
    task_data = {}
    data_dir = f"../data/{test_dataset_name}/"
    
    task_address = os.path.join(data_dir, "labeled", llm_id.replace('/', '-'), "OOTB-F", "train_ans_lbl.jsonl")
    if not os.path.exists(task_address):
        logger.warning(f"Warning: Data file not found at {task_address}")
        generic_task_address = os.path.join(data_dir, "train_ans_lbl.jsonl")
        if os.path.exists(generic_task_address):
            task_address = generic_task_address
            logger.info(f"Found data at {task_address}")
        else:
            logger.error(f"Could not find data at general path either: {generic_task_address}")
            return {}

    all_data = []
    try:
        with jsonlines.open(task_address) as reader:
            for obj in reader:
                updated_obj = extract_choice_options(obj, test_dataset_name)
                all_data.append(updated_obj)
    except FileNotFoundError:
        logger.error(f"ERROR: File not found at {task_address}. Please check the path.")
        return {}

    unique_dataset_names = set(obj['dataset_name'] for obj in all_data if 'dataset_name' in obj)
    task_data = {dataset_name: [] for dataset_name in unique_dataset_names}
    for obj in all_data:
        if 'dataset_name' in obj:
            task_data[obj['dataset_name']].append(obj)
    
    for k, v in task_data.items():
        logger.info(f"Loaded {len(v)} {k} samples.")
    
    return task_data

def load_task_data_mmlu(test_dataset_name, llm_id):
    print(f"Loading MMLU task data from {test_dataset_name}")
    task_data = {}
    data_dir = f"../data/{test_dataset_name}/tasks/"
    for task in os.listdir(data_dir):
        task_address = data_dir + task + f"/answered/OOTB-F/{llm_id.replace('/', '-')}/task_data_answered.pkl"
        if os.path.exists(task_address):
            with open(task_address, "rb") as f:
                task_data[task] = pickle.load(f)
            print(f"Loaded {len(task_data[task])} {task} samples.")
    print(f"Total tasks loaded: {len(task_data)}")

    all_data = []
    for task_name, task_samples in task_data.items():
        for sample in task_samples:
            if "choice" in test_dataset_name.lower():
                system_prompt = ct_system_prompt_choice()
            else:
                system_prompt = ct_system_prompt_oe()
            updated_sample = extract_choice_options(sample, test_dataset_name)
            prompt = updated_sample["prompt"]
            context = updated_sample["context"]
            target_prompt = updated_sample["target_prompt"]
            user_prompt = f"{prompt}{context}{target_prompt}"
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            updated_sample["conversation"] = conversation
            updated_sample["dataset_name"] = task_name
            updated_sample["query_label"] = updated_sample['assessment']
            # raise an error if wrong_answers are not present
            if "wrong_answers" not in updated_sample:
                raise ValueError(f"Missing 'wrong_answers' in sample: {updated_sample}")
            all_data.append(updated_sample)
    task_data = {task_name: [] for task_name in set(sample['dataset_name'] for sample in all_data)}
    for sample in all_data:
        if 'dataset_name' in sample:
            task_data[sample['dataset_name']].append(sample)
    for k, v in task_data.items():
        print(f"Loaded {len(v)} {k} samples.")
    print(f"Total tasks loaded: {len(task_data)}: {list(task_data.keys())}")
    return task_data

def convert_to_chat_template(conversation, tokenizer, add_generation_prompt):
    try:
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        # Fallback formatting
        text = ""
        for item in conversation:
            text += f"{item['role']}: {item['content']}\n"
        if add_generation_prompt and conversation and conversation[-1]['role'] == 'user':
            text += "assistant:"
        return text

def get_jacobian_for_token(model, input_ids, token_id):
    """Calculate Jacobian vector for a specific token"""
    model.zero_grad()
    
    # Forward pass without gradients
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_state = outputs.hidden_states[-1][:, -1, :].clone().detach()
    
    # Make slice require grad for lm_head backprop
    hidden_state_grad = hidden_state.requires_grad_(True)
    
    # Pass through lm_head
    logits = model.lm_head(hidden_state_grad.unsqueeze(1))
    logits_for_loss = logits.squeeze(1)
    
    # Calculate loss and backward
    log_probs = torch.log_softmax(logits_for_loss, dim=-1)
    loss = -log_probs[0, token_id]
    
    loss.backward()
    
    if hidden_state_grad.grad is not None:
        jacobian_vector = hidden_state_grad.grad.clone().detach()
    else:
        jacobian_vector = torch.zeros_like(hidden_state_grad)
    
    model.zero_grad()
    return jacobian_vector

def extract_hidden_states_and_logits(sample, model, tokenizer, answer_text, answer_type, 
                                   pei_radius, pei_steps, task_name):
    """
    Extract original and perturbed hidden states and logits for a sample
    """
    conversation = sample['conversation'].copy()
    
    # Create conversation with the answer appended
    conversation_with_response = conversation.copy()
    if answer_text:
        conversation_with_response.append({'role': 'assistant', 'content': answer_text})
    
    prompt_only_str = convert_to_chat_template(conversation, tokenizer, add_generation_prompt=True)
    full_prompt_str = convert_to_chat_template(conversation_with_response, tokenizer, add_generation_prompt=False)

    input_ids = tokenizer.encode(prompt_only_str, return_tensors="pt").to(model.device)
    full_ids = tokenizer.encode(full_prompt_str, return_tensors="pt").to(model.device)

    response_start_idx = input_ids.shape[1]
    if full_ids.shape[1] < response_start_idx:
        response_ids = []
    else:
        response_ids = full_ids[0, response_start_idx:].tolist()
    
    if not response_ids:
        logger.warning(f"No response IDs found for {answer_type}. Skipping.")
        return None
    
    # Store results for all tokens
    original_hidden_states_list = []
    original_logits_list = []
    jacobian_vectors_list = []
    perturbed_hidden_states_list = []
    perturbed_logits_list = []
    metadata_list = []
    
    current_input = input_ids
    
    # Process each token in the response
    for token_idx, token_id in enumerate(response_ids):
        # Get original hidden state and logits
        with torch.no_grad():
            outputs = model(current_input, output_hidden_states=True, return_dict=True)
            hidden_state = outputs.hidden_states[-1][:, -1, :].clone()
            logits = model.lm_head(hidden_state.unsqueeze(1))[:, -1, :]
        
        # Store original states
        original_hidden_states_list.append(hidden_state.cpu().float().numpy())
        original_logits_list.append(logits.cpu().float().numpy())
        
        # Calculate Jacobian for perturbation
        jacobian_vector = get_jacobian_for_token(model, current_input, token_id)
        jacobian_norm = torch.norm(jacobian_vector)
        
        # Store Jacobian vector
        jacobian_vectors_list.append(jacobian_vector.cpu().float().numpy())
        
        # Initialize lists for this token's perturbations
        token_perturbed_hidden_states = []
        token_perturbed_logits = []
        perturbed_metadata = []
        
        # Generate perturbations
        if jacobian_norm.item() > 1e-9:
            jacobian_direction = jacobian_vector / jacobian_norm
            delta_r = pei_radius / pei_steps
            
            for k in range(1, pei_steps + 1):
                r_k = k * delta_r
                
                # Perturb the hidden state
                perturbed_hidden = hidden_state + r_k * jacobian_direction
                
                # Get logits for perturbed state
                with torch.no_grad():
                    perturbed_logits = model.lm_head(perturbed_hidden.unsqueeze(1))[:, -1, :]
                
                token_perturbed_hidden_states.append(perturbed_hidden.cpu().float().numpy())
                token_perturbed_logits.append(perturbed_logits.cpu().float().numpy())
                
                perturbed_metadata.append({
                    'perturbation_step': k,
                    'perturbation_radius': r_k,
                    'jacobian_norm': jacobian_norm.item()
                })
        else:
            # For zero Jacobian, just repeat the original state
            for k in range(1, pei_steps + 1):
                token_perturbed_hidden_states.append(hidden_state.cpu().float().numpy())
                token_perturbed_logits.append(logits.cpu().float().numpy())
                
                perturbed_metadata.append({
                    'perturbation_step': k,
                    'perturbation_radius': 0.0,
                    'jacobian_norm': 0.0
                })
        
        perturbed_hidden_states_list.extend(token_perturbed_hidden_states)
        perturbed_logits_list.extend(token_perturbed_logits)
        
        # Store metadata for this token
        token_metadata = {
            'token_idx': token_idx,
            'token_id': token_id,
            'token_str': tokenizer.convert_ids_to_tokens(token_id),
            'original_hidden_state_idx': len(original_hidden_states_list) - 1,
            'original_logits_idx': len(original_logits_list) - 1,
            'jacobian_idx': len(jacobian_vectors_list) - 1,  # Add Jacobian index
            'perturbed_start_idx': len(perturbed_hidden_states_list) - pei_steps,
            'perturbed_end_idx': len(perturbed_hidden_states_list) - 1,
            'perturbation_metadata': perturbed_metadata,
            'task_name': task_name  # Add task name
        }
        metadata_list.append(token_metadata)
        
        # Update input for next token
        if token_idx < len(response_ids) - 1:
            next_token_tensor = torch.tensor([[token_id]], device=model.device)
            current_input = torch.cat([current_input, next_token_tensor], dim=1)
    
    return {
        'original_hidden_states': np.array(original_hidden_states_list),
        'original_logits': np.array(original_logits_list),
        'jacobian_vectors': np.array(jacobian_vectors_list),  # Add Jacobian vectors
        'perturbed_hidden_states': np.array(perturbed_hidden_states_list),
        'perturbed_logits': np.array(perturbed_logits_list),
        'metadata': metadata_list
    }

def save_task_data(task_output_dir, answer_type, all_samples_data, task_name):
    """Save the extracted data for all samples in a task"""
    answer_dir = os.path.join(task_output_dir, answer_type)
    os.makedirs(answer_dir, exist_ok=True)
    
    # Combine data from all samples
    all_original_hidden_states = []
    all_original_logits = []
    all_jacobian_vectors = []
    all_perturbed_hidden_states = []
    all_perturbed_logits = []
    combined_metadata = []
    
    original_offset = 0
    jacobian_offset = 0
    perturbed_offset = 0
    
    for sample_idx, sample_data in enumerate(all_samples_data):
        if sample_data is None:
            continue
            
        # Add to combined arrays
        all_original_hidden_states.append(sample_data['original_hidden_states'])
        all_original_logits.append(sample_data['original_logits'])
        all_jacobian_vectors.append(sample_data['jacobian_vectors'])
        all_perturbed_hidden_states.append(sample_data['perturbed_hidden_states'])
        all_perturbed_logits.append(sample_data['perturbed_logits'])
        
        # Update metadata with global indices and add task information
        sample_metadata = {
            'sample_idx': sample_idx,
            'hash_id': sample_data['hash_id'],
            'query_label': sample_data.get('query_label', None),
            'task_name': task_name,  # Add task name at sample level
            'dataset_name': sample_data.get('dataset_name', task_name),  # Add dataset name
            'answer_type': answer_type,  # Add answer type
            'wrong_answer_idx': sample_data.get('wrong_answer_idx', None),  # Include if present
            'tokens': []
        }
        
        for token_meta in sample_data['metadata']:
            token_meta_global = token_meta.copy()
            token_meta_global['original_hidden_state_idx'] += original_offset
            token_meta_global['original_logits_idx'] += original_offset
            token_meta_global['jacobian_idx'] += jacobian_offset
            token_meta_global['perturbed_start_idx'] += perturbed_offset
            token_meta_global['perturbed_end_idx'] += perturbed_offset
            sample_metadata['tokens'].append(token_meta_global)
        
        original_offset += len(sample_data['metadata'])
        jacobian_offset += len(sample_data['jacobian_vectors'])
        perturbed_offset += len(sample_data['perturbed_hidden_states'])
        
        combined_metadata.append(sample_metadata)
    
    # Concatenate all arrays
    if all_original_hidden_states:
        original_hidden_states = np.vstack(all_original_hidden_states)
        original_logits = np.vstack(all_original_logits)
        jacobian_vectors = np.vstack(all_jacobian_vectors)
        perturbed_hidden_states = np.vstack(all_perturbed_hidden_states)
        perturbed_logits = np.vstack(all_perturbed_logits)
        
        # Save arrays
        np.savez_compressed(
            os.path.join(answer_dir, 'original_hidden_states.npz'),
            data=original_hidden_states.astype(np.float16)
        )
        np.savez_compressed(
            os.path.join(answer_dir, 'original_logits.npz'),
            data=original_logits.astype(np.float16)
        )
        np.savez_compressed(
            os.path.join(answer_dir, 'jacobian_vectors.npz'),
            data=jacobian_vectors.astype(np.float16)
        )
        np.savez_compressed(
            os.path.join(answer_dir, 'perturbed_hidden_states.npz'),
            data=perturbed_hidden_states.astype(np.float16)
        )
        # np.savez_compressed(
        #     os.path.join(answer_dir, 'perturbed_logits.npz'),
        #     data=perturbed_logits.astype(np.float16)
        # )
        
        # Save metadata
        with open(os.path.join(answer_dir, 'metadata.jsonl'), 'w') as f:
            for sample_meta in combined_metadata:
                f.write(json.dumps(sample_meta) + '\n')
        
        logger.info(f"Saved {len(combined_metadata)} samples for {answer_type}")
        logger.info(f"Original states shape: {original_hidden_states.shape}")
        logger.info(f"Jacobian vectors shape: {jacobian_vectors.shape}")
        logger.info(f"Perturbed states shape: {perturbed_hidden_states.shape}")

def check_files_exist(task_output_dir, answer_type):
    """Check if all necessary files exist for a specific answer type in a task"""
    answer_dir = os.path.join(task_output_dir, answer_type)
    
    if not os.path.exists(answer_dir):
        return False
    
    required_files = [
        'original_hidden_states.npz',
        'original_logits.npz',
        'jacobian_vectors.npz',
        'perturbed_hidden_states.npz',
        'metadata.jsonl'
    ]
    
    for file_name in required_files:
        file_path = os.path.join(answer_dir, file_name)
        if not os.path.exists(file_path):
            return False
    
    # Check if metadata file is valid and not empty
    metadata_path = os.path.join(answer_dir, 'metadata.jsonl')
    try:
        with open(metadata_path, 'r') as f:
            # Check if there's at least one valid line
            first_line = f.readline().strip()
            if not first_line:
                return False
            # Try to parse it as JSON
            json.loads(first_line)
    except (IOError, json.JSONDecodeError):
        return False
    
    return True

def main():
    # Create timestamp for unique output directory
    root_output_dir = os.path.join(args.output_dir, args.test_dataset_name, args.llm_id.replace('/', '-'))
    os.makedirs(root_output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(root_output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    logger.info(f"CUDA_VISIBLE_DEVICES set to {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.llm_id,
        args.use_unsloth,
        args.dtype
    )
    
    # Load task data
    if "mmlu" in args.test_dataset_name.lower():
        task_data = load_task_data_mmlu(args.test_dataset_name, args.llm_id)
    else:
        task_data = load_task_data_ct(args.test_dataset_name, args.llm_id)
    
    logger.info(f"Processing tasks: {list(task_data.keys())}")
    
    # Process each task
    for t, (task_name, samples_in_task) in enumerate(task_data.items()):
        logger.info(f"\nProcessing Task {t + 1}/{len(task_data)}: {task_name}")
        
        task_output_dir = os.path.join(root_output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Limit samples if specified
        if args.max_samples_per_task is not None:
            samples_to_process = samples_in_task[:args.max_samples_per_task]
        else:
            samples_to_process = samples_in_task
        
        # Process each answer type
        # for answer_type in ['llm_output', 'target', 'wrong_answer']:
        for answer_type in ['llm_output']:
            # Check if files already exist for this answer type
            if check_files_exist(task_output_dir, answer_type):
                logger.info(f"Files already exist for {answer_type} in task {task_name}. Skipping.")
                continue
        
            logger.info(f"\nProcessing {answer_type} for task {task_name}")
            all_samples_data = []
            
            for sample_idx, sample_data in enumerate(tqdm(samples_to_process, 
                                                        desc=f"{answer_type} - {task_name}")):
                if answer_type == 'llm_output' and 'llm_output' in sample_data:
                    answer_text = sample_data.get('llm_output', "")
                    query_label = sample_data.get('query_label', 0)
                    
                    result = extract_hidden_states_and_logits(
                        sample_data, model, tokenizer, answer_text, answer_type,
                        args.pei_radius, args.pei_steps, task_name
                    )
                    
                    if result is not None:
                        result['hash_id'] = sample_data.get('hash_id', 'unknown')
                        if result['hash_id'] == 'unknown':
                            result['hash_id'] = sample_data.get('id', 'unknown')
                        result['query_label'] = query_label
                        result['dataset_name'] = sample_data.get('dataset_name', task_name)
                        all_samples_data.append(result)
                
                elif answer_type == 'target' and 'target' in sample_data:
                    answer_text = sample_data.get('target', "")
                    
                    result = extract_hidden_states_and_logits(
                        sample_data, model, tokenizer, answer_text, answer_type,
                        args.pei_radius, args.pei_steps, task_name
                    )
                    
                    if result is not None:
                        result['hash_id'] = sample_data.get('hash_id', f'unknown_{sample_idx}')
                        result['query_label'] = 1  # Target is always correct
                        result['dataset_name'] = sample_data.get('dataset_name', task_name)
                        all_samples_data.append(result)
                
                elif answer_type == 'wrong_answer' and 'wrong_answers' in sample_data:
                    wrong_answers = sample_data.get('wrong_answers', [])
                    
                    for wrong_idx, wrong_text in enumerate(wrong_answers):
                        if not wrong_text:
                            continue
                        
                        result = extract_hidden_states_and_logits(
                            sample_data, model, tokenizer, wrong_text, answer_type,
                            args.pei_radius, args.pei_steps, task_name
                        )
                        
                        if result is not None:
                            result['hash_id'] = sample_data.get('hash_id', f'unknown_{sample_idx}')
                            result['query_label'] = 0  # Wrong answer is always incorrect
                            result['wrong_answer_idx'] = wrong_idx
                            result['dataset_name'] = sample_data.get('dataset_name', task_name)
                            all_samples_data.append(result)
                
                # Clear GPU cache periodically
                if sample_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Save all data for this answer type
            if all_samples_data:
                save_task_data(task_output_dir, answer_type, all_samples_data, task_name)
    
    logger.info(f"\nExtraction complete. Results saved to {root_output_dir}")
    return root_output_dir

if __name__ == "__main__":
    output_path = main()
    print(f"\nExtraction completed successfully. Results saved to: {output_path}")
    

## New LLMs:
# "unsloth/Meta-Llama-3.1-8B-Instruct"
# "unsloth/Qwen2.5-14B-Instruct"
# "unsloth/Mistral-Small-24B-Instruct-2501"
# "unsloth/Qwen2.5-32B-Instruct"

# python origpert_hidden_state_logit_extraction.py \
#     --gpu_ids 0 \
#     --llm_id "unsloth/Qwen2.5-14B-Instruct" \
#     --test_dataset_name "CT-CHOICE-WRG" \
#     --pei_radius 20.0 \
#     --pei_steps 5 \
#     --output_dir "../representations/OrigPert" \
#     --use_unsloth

# 2>&1 | tee origpert_ct_choice.log

# python origpert_hidden_state_logit_extraction.py \
#     --gpu_ids 6 \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --test_dataset_name "CT-OE-WRG" \
#     --pei_radius 20.0 \
#     --pei_steps 5 \
#     --output_dir "../representations/OrigPert" \
#     --use_unsloth

# python origpert_hidden_state_logit_extraction.py \
#     --gpu_ids 7 \
#     --llm_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
#     --test_dataset_name "MMLU-CHOICE" \
#     --pei_radius 20.0 \
#     --pei_steps 5 \
#     --output_dir "../representations/OrigPert" \
#     --use_unsloth

# python origpert_hidden_state_logit_extraction.py \
#     --gpu_ids 7 \
#     --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#     --test_dataset_name "MMLU-OE" \
#     --pei_radius 20.0 \
#     --pei_steps 5 \
#     --output_dir "../representations/OrigPert" \
#     --use_unsloth