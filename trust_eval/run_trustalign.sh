#!/bin/bash
set -e

cd /home/uzabase/work/trust-align/trust_eval

run_eval() {
    local model=$1
    local model_tag=$2
    local dataset=$3
    local run=$4
    local temp=${5:-0.1}
    local outdir="results/${model_tag}/${dataset}/run_${run}"

    echo ""
    echo "======================================================"
    echo " model=${model} dataset=${dataset} run=${run} temp=${temp}"
    echo " output: ${outdir}"
    echo "======================================================"

    python evaluate.py \
        --data-type "${dataset}" \
        --concurrency 1 \
        --model "${model}" \
        --temperature "${temp}" \
        --output-dir "${outdir}"
}

# ── Phase 5: declare-lab/trustalign_qwen2.5_7b × 5 runs × 3 datasets ─────────
echo "######## Phase 5: trustalign_qwen2.5_7b (temp=0.1) ########"
for dataset in qampari asqa expertqa; do
    for run in 1 2 3 4 5; do
        run_eval "declare-lab/trustalign_qwen2.5_7b" "trustalign-qwen2.5-7b" "${dataset}" "${run}"
    done
done

# ── Phase 6: declare-lab/trustalign_qwen2.5_3b × 5 runs × 3 datasets ─────────
echo "######## Phase 6: trustalign_qwen2.5_3b (temp=0.1) ########"
for dataset in qampari asqa expertqa; do
    for run in 1 2 3 4 5; do
        run_eval "declare-lab/trustalign_qwen2.5_3b" "trustalign-qwen2.5-3b" "${dataset}" "${run}"
    done
done

echo ""
echo "======================================================"
echo " ALL TRUSTALIGN RUNS DONE at $(date)"
echo "======================================================"
