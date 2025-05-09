'''
2025-05-01
Author: Dan Schumacher

See ./bin/prompting.sh

or run directly like so 

python ./src/chatTS_prompting_v1.py \
    --input_path "./data/chatts/llm_qa_1000_sp.jsonl" \
    --prompt_path "./resources/prompts/llm_qa.yaml" \
    --output_path "./test.jsonl"
'''

import json
import sys
from tqdm import tqdm
from utils.file_io import  load_tsdata_list
from utils.prompting.prompter import OpenAIPrompter
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--input_path', type=str, help='Path to the input JSONL file',),
    parser.add_argument('--prompt_path', type=str, help="What yaml to use for prompting")
    parser.add_argument('--output_path', type=str, help="Where do you want to save output?")
    parser.add_argument('--context', type=bool, default=False, help="Context for the prompt")
    return parser.parse_args()

def main():
    args = parse_args()
    # Extract dataset and subset from the input path: ./data/<dataset>/<subset>.jsonl
    input_parts = args.input_path.strip().split('/')
    dataset = input_parts[2]
    subset = os.path.splitext(input_parts[3])[0]  # remove .jsonl
    # Extract prompt file name (without .yaml) from the prompt path: ./resources/prompts/<dataset>/<subset>/<prompt_file>.yaml
    prompt_file = os.path.splitext(os.path.basename(args.prompt_path))[0]
    print(f"Running prompting for:")
    print(f"\tDataset     : {dataset}")
    print(f"\tSubset      : {subset}")
    print(f"\tPrompt file : {prompt_file}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        f.write('')

    llm_qa_data = load_tsdata_list(args.input_path)
    # llm_qa_data = llm_qa_data[:3] # XXX uncomment for testing
    # we used the 0th and 44th examples in our few shot examples -> remove them
    llm_qa_data = [item for i, item in enumerate(llm_qa_data) if i not in (0, 44)]
    llm_qa_data = llm_qa_data       #[:3] # REMOVE

    if args.context:
        prompt_headers= {
            "background_info": "Here is some background information:\n".upper(),
            "additional_info": "Here are some useful clues:\n".upper(),
            "time_series": "Here is the time series you will analyze:\n".upper(),
            "question": "Here is the question you need to answer:\n".upper(),
            "options_text": "Here are the options you can chose from:\n".upper(),
        },
    else:
        prompt_headers= {
            "time_series": "Here is the time series you will analyze:\n",
            "question": "Here is the question you need to answer:\n",
            "options_text": "Here are the options you can chose from:\n",
        }

    prompter = OpenAIPrompter(
        prompt_path=args.prompt_path,
        prompt_headers=prompt_headers,
        llm_model= "gpt-4o-mini",
        temperature=0.1,
    )

    for i, row in enumerate(tqdm(llm_qa_data, desc="Generating responses", file=sys.stdout)):
        if args.context:
            input_texts = {
                "background_info": row.description,
                "additional_info": row.characteristics,
                "time_series": row.series,
                "question": row.question,
                "options_text": row.options_text,
            }
        else:
            input_texts = {
                "time_series": row.series,
                "question": row.question,
                "options_text": row.options_text,
            }
        response_json = prompter.get_completion(input_texts)
        abcd_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        ouput = {
            "uuid": row.uuid,
            "pred": response_json[0],
            "answer_letter": abcd_map[row.answer_index],
            "answer": row.answer,
            "question": row.question,
        }

        with open(args.output_path, 'a') as f:
            f.write(json.dumps(ouput) + "\n")

if __name__ == "__main__":
    main()
