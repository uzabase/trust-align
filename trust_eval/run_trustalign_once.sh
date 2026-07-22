#!/bin/bash
set -e

cd /home/uzabase/work/trust-align/trust_eval

run_eval() {
    local model=$1
    local model_tag=$2
    local dataset=$3
    local temp=${4:-0.1}
    local outdir="results/${model_tag}/${dataset}"

    echo ""
    echo "======================================================"
    echo " model=${model} dataset=${dataset} temp=${temp}"
    echo " output: ${outdir}"
    echo "======================================================"

    python evaluate.py \
        --data-type "${dataset}" \
        --concurrency 1 \
        --model "${model}" \
        --temperature "${temp}" \
        --output-dir "${outdir}"
}

echo "######## trustalign_qwen2.5_7b (temp=0.1) ########"
for dataset in qampari asqa expertqa; do
    run_eval "declare-lab/trustalign_qwen2.5_7b" "trustalign-qwen2.5-7b" "${dataset}"
done

echo "######## trustalign_qwen2.5_3b (temp=0.1) ########"
for dataset in qampari asqa expertqa; do
    run_eval "declare-lab/trustalign_qwen2.5_3b" "trustalign-qwen2.5-3b" "${dataset}"
done

echo ""
echo "======================================================"
echo " ALL DONE at $(date)"
echo "======================================================"
