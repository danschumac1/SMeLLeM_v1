#!/bin/bash
# TO RUN : nohup ./bin/prompting.sh > ./logs/prompting.log 2>&1 &
#               ./bin/prompting.sh
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
prompt_files=( # don't put .yaml
    # "generic_fs"
    "generic_zs"
    # "reasoning_zs"
    # "reasoning_fs"
)
# context=true
context=false

for dataset in "${datasets[@]}"; do
    for subset in "${subsets[@]}"; do
        input_path="./data/$dataset/${subset}.jsonl"

        for prompt_file in "${prompt_files[@]}"; do
            prompt_path="./resources/prompts/$dataset/$subset/${prompt_file}.yaml"
            output_path="./data/generations/$dataset/$subset/${prompt_file}_$context.jsonl"

            python ./src/prompting_v1.py \
                --input_path "$input_path" \
                --context "$context" \
                --prompt_path "$prompt_path" \
                --output_path "$output_path"
        done
    done
done
