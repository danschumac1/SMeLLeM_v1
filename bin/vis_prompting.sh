#!/bin/bash
# TO RUN : nohup ./bin/vis_prompting.sh > ./logs/vis_prompting.log 2>&1 &
#               ./bin/vis_prompting.sh
# 718033
# CHANGE THESE
datasets=(
    "mcq"
    # "chatts"
    # "tarr"
)
subsets=(
    "1"
)
# context=true
context=false

for dataset in "${datasets[@]}"; do
    for subset in "${subsets[@]}"; do
        input_path="./data/$dataset/${subset}.jsonl"

        output_path="./data/generations/$dataset/$subset/vision_$context.jsonl"

            python ./src/vision_prompting_v0.py \
                --input_path "$input_path" \
                --prompt_path "./resources/prompts/mcq/1/vision.yaml" \
                --output_path "$output_path"
    done
done
