import argparse
import os
import sys
import json
import openai
from tqdm import tqdm
import multiprocessing
from functools import partial
import glob
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description="Assess LLM answers using GPT-4o")
parser.add_argument('--grading_method', type=str, required=True, help="The grading method to use (gpt or substring)")
parser.add_argument("--data_dir", type=str, required=True, help="Base data directory (e.g., '../../data/CT-CHOICE/answered/')")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files")
parser.add_argument("--prefix", type=str, required=False, help="Prefix of the LLM ID to process (e.g., 'unsloth')")
args = parser.parse_args()
print('=' * 50)
print('args:', args)
print('=' * 50)


# Import from utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (
    grade_with_gpt,
    grade_with_substring
)
sys.path = original_sys_path

def process_sample_with_gpt(sample):
    """
    Process a single sample by adding a 'query_label' key determined by GPT-4o.

    Expected keys in sample: "question" and "llm_answer" (or similar).
    Adjust key names if your JSON structure differs.
    
    Args:
        sample (dict): A single JSON record.

    Returns:
        dict: The sample with an additional key "query_label" (int 0 or 1).
    """
    # Adjust the following key names as needed.
    question = sample.get("context", "")
    gt_answer = sample.get("target", "")
    llm_answer = sample.get("llm_output", "")
    
    # Query GPT-4o to get the label.
    label_str = grade_with_gpt(question, gt_answer, llm_answer)
    sample["query_label"] = int(label_str)
    return sample

def process_file_with_gpt(file_path, output_file):
    """
    Process a single file: read JSON/JSONL data, run GPT-4o queries in parallel,
    add a "query_label" to each sample, and then write out a new file.
    
    Args:
        file_path (str): Path to the input file.
        output_file (str): Path to write the output file.
    """
    samples = []
    # Check extension to decide how to read the file.
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Process samples in parallel using available CPUs.
    num_processes = multiprocessing.cpu_count()
    processed_samples = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i, sample in enumerate(tqdm(
            pool.imap(process_sample_with_gpt, samples),
            total=len(samples),
            desc=f"Processing {os.path.basename(file_path)}"
        )):
            processed_samples.append(sample)
            # Add a delay after every 1000 samples
            if (i + 1) % 5000 == 0:
                print("Pausing for 10 seconds to avoid API overload...")
                time.sleep(10)
    
    # Retry labeling for samples with a query_label of -1
    retry_samples = [sample for sample in processed_samples if sample['query_label'] == -1]
    if retry_samples:
        print(f"Retrying labeling for {len(retry_samples)} samples with label -1.")
        for sample in retry_samples:
            # Retry querying GPT-4o
            label_str = grade_with_gpt(sample.get("question", ""), sample.get("target", ""), sample.get("llm_output", ""))
            sample["query_label"] = int(label_str)

    # Write output as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Processed {len(processed_samples)} samples from {os.path.basename(file_path)} and saved to {output_file}")


def process_sample_with_substring(sample):
    """
    Process a single sample by adding a 'query_label' key determined by GPT-4o.

    Expected keys in sample: "question" and "llm_answer" (or similar).
    Adjust key names if your JSON structure differs.
    
    Args:
        sample (dict): A single JSON record.

    Returns:
        dict: The sample with an additional key "query_label" (int 0 or 1).
    """
    # Adjust the following key names as needed.
    question = sample.get("context", "")
    gt_answer = sample.get("target", "")
    llm_answer = sample.get("llm_output", "")
    
    # Use substring check to get the label.
    label_str = grade_with_substring(question, gt_answer, llm_answer)
    sample["query_label"] = int(label_str)
    return sample

def process_file_with_substring(file_path, output_file):
    """
    Process a single file: read JSON/JSONL data, run substring check,
    add a "query_label" to each sample, and then write out a new file.
    
    Args:
        file_path (str): Path to the input file.
        output_file (str): Path to write the output file.
    """
    samples = []
    # Check extension to decide how to read the file.
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    processed_samples = [process_sample_with_substring(sample) for sample in samples]

    # Write output as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Processed {len(processed_samples)} samples from {os.path.basename(file_path)} and saved to {output_file}")


def main():
    # Iterate over all subdirectories in the input directory.
    for llm_id in os.listdir(args.data_dir):
        if args.prefix:
            if not llm_id.startswith(args.prefix):
                continue
        input_dir = os.path.join(args.data_dir, llm_id)
        for subdir in os.listdir(input_dir):
            input_dir = os.path.join(args.data_dir, llm_id, subdir)
            print(f"Processing directory: {input_dir}")
            # Create output directory: output_dir/llm_id
            output_llm_dir = os.path.join(args.output_dir, llm_id, subdir)
            os.makedirs(output_llm_dir, exist_ok=True)

            # List all JSON and JSONL files in the input directory.
            file_patterns = [os.path.join(input_dir, "*.jsonl"), os.path.join(input_dir, "*.json")]
            input_files = []
            for pattern in file_patterns:
                input_files.extend(glob.glob(pattern))

            if not input_files:
                print(f"No .json or .jsonl files found in {input_dir}")
                continue

            # Process each file.
            for file_path in input_files:
                base = os.path.basename(file_path)
                # Create output filename: e.g., train_ans.jsonl -> train_ans_lbl.jsonl
                name, ext = os.path.splitext(base)
                output_file = os.path.join(output_llm_dir, f"{name}_lbl.jsonl")
                
                print(f"Processing file: {file_path}")
                print(f"Output file: {output_file}")
                
                # Skip processing if the output file already exists
                if os.path.exists(output_file):
                    print(f"Skipping {base} as it is already processed.")
                    continue

                print(f"Processing file: {base}")
                if args.grading_method == "gpt":
                    process_file_with_gpt(file_path, output_file)
                elif args.grading_method == "substring":
                    process_file_with_substring(file_path, output_file)
                else:
                    raise ValueError(f"Invalid grading method: {args.grading_method}")
                print('=' * 80)

if __name__ == "__main__":
    main()

## Example usage:
# python answer_assessment.py --data_dir "../../data/CT-CHOICE-WRG/answered/"  --output_dir "../../data/CT-CHOICE-WRG/labeled/" --grading_method "substring"
# python answer_assessment.py --data_dir "../../data/CT-OE-WRG/answered/"  --output_dir "../../data/CT-OE-WRG/labeled/" --grading_method "gpt"
# python answer_assessment.py --prefix "unsloth" --data_dir "../../data/CT-CHOICE/answered/"  --output_dir "../../data/CT-CHOICE/labeled/"