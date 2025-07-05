import argparse
import os
import sys
from tqdm import tqdm
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Answer questions using a specified LLM with vLLM.")
parser.add_argument('--data_location', type=str, required=True, help='Input data directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--output_subdir', type=str, required=True, help='Output subdirectory')
parser.add_argument('--llm_id', type=str, required=True, help='LLM ID')
parser.add_argument('--llm_dir', type=str, required=False, help='LLM directory')
parser.add_argument('--visible_cudas', type=str, required=True, help='Visible CUDA devices')
parser.add_argument('--dtype', type=str, required=False, default=None, help='Model precision')
parser.add_argument('--temp', type=float, required=True, help='Temperature for LLM')
parser.add_argument('--gpu_memory', type=float, required=True, help='GPU memory allocation for VLLM')
parser.add_argument('--tensor_parallel', type=int, required=True, help='Tensor parallel for VLLM')
parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility')
parser.add_argument('--max_seq_len', type=int, required=True, help='Maximum sequence length for LLM and tokenizer')
parser.add_argument('--chat_template', type=str, default="openai", help='Chat template for VLLM')
parser.add_argument('--tokenizer_mode', type=str, default=None, help='Tokenizer mode for VLLM')
parser.add_argument('--config_format', type=str, default=None, help='Config format for VLLM')
parser.add_argument('--quantization', type=str, default=None, help='Quantization for VLLM')
parser.add_argument('--load_format', type=str, default=None, help='Load format for VLLM')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

# Import from utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (
    set_visible_cudas, 
    load_jsonl, 
    save_jsonl, 
    seed_everything, 
    ct_system_prompt,
    ct_system_prompt_choice, 
    initialize_tokenizer
)
sys.path = original_sys_path
seed_everything(args.seed)
set_visible_cudas(args.visible_cudas)


def format_prompt(llm_id, system_prompt, user_prompt, tokenizer, apply_template=False):
    if "gemma" in llm_id:
        conversation = [
            {"role": "user", "content": user_prompt},
        ]
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    if apply_template:
        return tokenizer.apply_chat_template(conversation, tokenize=False)
    else:
        return conversation


def process_file(llm, input_file_path, output_file_path, tokenizer, args):
    """Process a single JSONL file, generate output using LLM, and save to output file."""
    logging.info(f'Starting to process file: {input_file_path}')
    # Load records from the input JSONL file
    records = load_jsonl(input_file_path)
    # records = records[16000:]

    conversations = []
    for rec in records:
        if args.max_seq_len == 1:
            system_prompt = ct_system_prompt_choice()
        else:
            system_prompt = ct_system_prompt()
        prompt = rec.get("prompt", "")
        context = rec.get("context", "")
        target_prompt = rec.get("target_prompt", "")
        user_prompt = f"{prompt}{context}{target_prompt}"
        conversation = format_prompt(args.llm_id, system_prompt, user_prompt, tokenizer, apply_template=False)
        conversations.append(conversation)

    logging.info(f'Number of conversations prepared: {len(conversations)}')
    output_texts = llm.batch_chat_query(
        conversations, 
        temperature=args.temp, 
        max_tokens=args.max_seq_len, 
        use_tqdm=True,
        chat_template_content_format=args.chat_template
    )
    logging.info('Batch chat query completed')

    # Update existing records with LLM output
    for rec, output_text, conversation in zip(records, output_texts, conversations):
        rec["conversation"] = conversation
        rec["llm_output"] = output_text

    save_jsonl(records, output_file_path)
    logging.info(f'Processed {input_file_path} and saved to {output_file_path}')


def process_all_files(llm, tokenizer, args):
    """Process all JSONL files in the data_location."""
    # Create output directory under output_dir with llm_id sanitized (replace '/' with '-')
    sanitized_llm_id = args.llm_id.replace('/', '-')
    output_base_dir = os.path.join(args.output_dir, 'answered', sanitized_llm_id, args.output_subdir)
    os.makedirs(output_base_dir, exist_ok=True)

    # Dynamically find all JSONL files in the data_location
    jsonl_files = glob.glob(os.path.join(args.data_location, '*.jsonl'))

    for input_file_path in jsonl_files:
        file_name = os.path.basename(input_file_path)
        output_file_name = file_name.replace('.jsonl', '_ans.jsonl')
        output_file_path = os.path.join(output_base_dir, output_file_name)

        # Skip processing if the output file already exists
        if os.path.exists(output_file_path):
            logging.info(f'Skipping {input_file_path} as output already exists.')
            continue

        process_file(llm, input_file_path, output_file_path, tokenizer, args)

# Argument parsing
if __name__ == "__main__":
    # set VLLM_WORKER_MULTIPROC_METHOD to spawn
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    print('VLLM_WORKER_MULTIPROC_METHOD:', os.getenv("VLLM_WORKER_MULTIPROC_METHOD"))

    original_sys_path = sys.path.copy()
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))
    if utils_path not in sys.path:
        sys.path.append(utils_path)
    from talk2llm import Talk2LLM
    sys.path = original_sys_path
    tokenizer = initialize_tokenizer(args.llm_dir, max_seq_length=args.max_seq_len)

    llm = Talk2LLM(
        model_id=args.llm_dir, 
        dtype=args.dtype, 
        gpu_memory_utilization=args.gpu_memory, 
        tensor_parallel_size=args.tensor_parallel, 
        enforce_eager=False,
        tokenizer_mode=args.tokenizer_mode,
        config_format=args.config_format,
        quantization=args.quantization,
        load_format=args.load_format,
    )
    process_all_files(llm, tokenizer, args)


# ====================================================================
## Large llm_ids:
# unsloth/Meta-Llama-3.1-8B-Instruct (8.03B parameters)
# unsloth/Qwen2.5-14B-Instruct (14.8B parameters)
# unsloth/Mistral-Small-24B-Instruct-2501 (23.6B parameters)
# unsloth/Qwen2.5-32B-Instruct (32.8B parameters)
# unsloth/Meta-Llama-3.1-70B-Instruct (70.6B parameters)

# CHOICE
# python answer_with_vllm.py \
#   --visible_cudas "3,5" \
#   --data_location "../../data/CT-CHOICE-WRG/" \
#   --output_dir "../../data/CT-CHOICE-WRG/" \
#   --output_subdir "OOTB" \
#   --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#   --llm_dir "unsloth/Qwen2.5-32B-Instruct" \
#   --dtype "bfloat16" \
#   --temp 0 \
#   --gpu_memory 0.9 \
#   --tensor_parallel 2 \
#   --max_seq_len 1 \
#   --chat_template "qwen"

# OE
# python answer_with_vllm.py \
#   --visible_cudas "0,1,2,3" \
#   --data_location "../../data/CT-OE-WRG/" \
#   --output_dir "../../data/CT-OE-WRG/" \
#   --output_subdir "OOTB" \
#   --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#   --llm_dir "unsloth/Qwen2.5-32B-Instruct" \
#   --dtype "bfloat16" \
#   --temp 0 \
#   --gpu_memory 0.9 \
#   --tensor_parallel 4 \
#   --max_seq_len 30 \
#   --chat_template "qwen" 