'''
2025-05-01
Author: Dan Schumacher
How to run:
   python ./src/vision_prompting_v0.py \
    --input_path ./data/chatts/llm_qa.jsonl \
    --output_path ./test.jsonl
'''

import argparse
import json
import base64
import sys
from openai import OpenAI
import dotenv
import os

from tqdm import tqdm
from utils.vis import plot_time_series
from utils.file_io import load_tsdata_list
from utils.prompting.prompter import OpenAIPrompter

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--input_path', type=str, help='Path to the input JSONL file',),
    parser.add_argument('--prompt_path', type=str, help="What yaml to use for prompting")
    parser.add_argument('--output_path', type=str, help="Where do you want to save output?")
    return parser.parse_args()
    
def main():
    args = parse_args()
    # Extract dataset and subset from the input path: ./data/<dataset>/<subset>.jsonl
    input_parts = args.input_path.strip().split('/')
    dataset = input_parts[2]
    subset = os.path.splitext(input_parts[3])[0]  # remove .jsonl
    print(f"Running vision prompting for:")
    print(f"\tDataset     : {dataset}")
    print(f"\tSubset      : {subset}")

    # clear the output file
    with open(args.output_path, 'w') as f:
        f.write("")

    data = load_tsdata_list(args.input_path)
    # data = data[:3] # XXX uncomment for testing
    print(f"Loaded {len(data)} rows from {args.input_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load environment variables from .env file
    dotenv.load_dotenv("./resources/.env")
    client = OpenAI()
    prompter = OpenAIPrompter(
        prompt_path=args.prompt_path,
        prompt_headers= {
            "background_info": "Here is some background information\n:".upper(),
            "additional_info": "Here are some useful clues:\n".upper(),
            "question": "Here is the question you need to answer:\n".upper(),
            "options_text": "Here are the options you can chose from:\n".upper(),
        },
        llm_model= "gpt-4o-mini",
        temperature=0.1,
    )

    # Path to your image
    image_path = "figures/0_tiny.png"

    # Getting the Base64 string
    first_row = True
    for idx, row in enumerate(tqdm(data, total=len(data), desc="Processing rows", file=sys.stdout)):
        # tqdm.write(f"Processing idx={idx}")
        plot_time_series(row, 0) # this saves over 0_tiny.png
        input_texts = {
            "background_info": row.description,
            "additional_info": row.characteristics,
            "question": row.question,
            "options_text": row.options_text
        }
        prompter.add_image(input_texts, image_path)

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