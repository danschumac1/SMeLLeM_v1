import json
import os
import random
import argparse

'''
### path must end with _full.jsonl !!! ###
==================================================================
python ./src/data_managment/sampler.py \
    --input_path=./data/MCQ_1_TS_full.jsonl

==================================================================
'''

def parse_args():
    """Argument parsing function"""
    parser = argparse.ArgumentParser(description="TKGinfer Generation Script")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--num_rows', type=int, default=300, help='Number of rows to sample')
    return parser.parse_args()

def sample_jsonl(input_path, num_samples):
    """
    Randomly sample `num_samples` lines from a JSONL file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if num_samples > len(lines):
        raise ValueError(f"Requested {num_samples} samples, but file only has {len(lines)} lines.")

    random.seed(42)
    sampled_lines = random.sample(lines, num_samples)
    return [json.loads(line) for line in sampled_lines]

def write_jsonl(data, input_path):
    """
    Writes a list of JSON objects to a JSONL file with 'full' replaced by 'sample' in the filename.
    """
    dir_path = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    
    if "full" not in file_name:
        raise ValueError("Expected 'full' in the input filename.")
    
    new_file_name = file_name.replace("_full", "")
    output_path = os.path.join(dir_path, new_file_name)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    args = parse_args()
    sample = sample_jsonl(args.input_path, args.num_rows)
    write_jsonl(sample, args.input_path)

if __name__ == '__main__':
    main()
