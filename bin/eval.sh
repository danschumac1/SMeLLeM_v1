#!/bin/bash
# TO RUN : ./bin/eval.sh

# === CONFIGURE BELOW ===
datasets=(
    "mcq"
    # "chatts"
    # "tarr"
)
subsets=(
    "1"
)
prompt_files=( # without .jsonl
    "generic_zs"
    "vision"
    # "generic_fs"
    # "reasoning_zs"
    # "reasoning_fs"
)
clear=0

# === Optionally clear results file ===
if [ "$clear" == "1" ]; then
    > ./data/results/results.tsv
fi

# === LOOP THROUGH ALL COMBINATIONS ===
for dataset in "${datasets[@]}"; do
    for subset in "${subsets[@]}"; do
        for prompt_file in "${prompt_files[@]}"; do
            input_path="./data/generations/$dataset/$subset/${prompt_file}.jsonl"
            output_path="./data/results/results.tsv"
            mkdir -p "$(dirname "$output_path")"

            python ./src/eval.py \
                --input_path "$input_path" \
                --output_path "$output_path" 
        done
    done
done
