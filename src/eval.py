'''
@author: Dan Schumacher
@date: 2025-05-05

TO RUN: 
python src/eval.py --input_path data/generations/mcq/1/vision.jsonl --output_path data/results/results.tsv

'''

import argparse
import json
import os
from pathlib import Path
from typing import List
import csv


def extract_info_from_path(input_path: str):
    """Extract dataset, subset, and prompt_type from the input path."""
    parts = Path(input_path).parts
    # Example path: data/generations/mcq/1/vision.jsonl
    try:
        dataset = parts[2]
        subset = parts[3]
        prompt_type = Path(parts[4]).stem
    except IndexError:
        raise ValueError("Path structure unexpected. Expected: data/generations/<dataset>/<subset>/<prompt_type>.jsonl")
    return dataset, subset, prompt_type

def read_jsonl(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def evaluate_mcq(data: List[dict]):
    """Calculate accuracy based on answer_letter in pred text."""
    total = 0
    correct = 0
    for row in data:
        total += 1
        pred = row.get("pred", "").lower()
        # print(pred)
        answer = ' '.join(row.get("answer", "").lower().split()[:5])
        # print(answer)
        if answer  in pred:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total, correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_path", help="Optional path to write detailed results")
    args = parser.parse_args()

    data = read_jsonl(args.input_path)
    accuracy, total, correct = evaluate_mcq(data)
    dataset, subset, prompt_type = extract_info_from_path(args.input_path)

    results = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "dataset": dataset,
        "subset": subset,
        "prompt_type": prompt_type,
        "input_file": args.input_path,
    }

    print(json.dumps(results, indent=2))

    if args.output_path:
        # check if the path exists, if not create it
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # if the file exists, does it have content?
        if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
            # append to the file
            with open(args.output_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=results.keys(), delimiter="\t")
                writer.writerow(results)
        else:
            # create the file and write the header
            with open(args.output_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=results.keys(), delimiter="\t")
                writer.writeheader()
                writer.writerow(results)


if __name__ == "__main__":
    main()
