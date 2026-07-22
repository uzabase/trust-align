#!/bin/bash
set -e

cd /home/uzabase/work/trust-align/trust_eval

run_eval() {
    local model=$1
    local model_tag=$2
    local dataset=$3
    local run=$4
    local temp=${5:-0.5}
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

# ── Phase 1: Qwen2.5-7B-Instruct × 5 runs × 3 datasets ──────────────────────
echo "######## Phase 1: Qwen2.5-7B-Instruct ########"
for dataset in qampari asqa expertqa; do
    for run in 1 2 3 4 5; do
        run_eval "Qwen/Qwen2.5-7B-Instruct" "qwen-2-5-7b" "${dataset}" "${run}"
    done
done

# ── Phase 2: Qwen2.5-3B-Instruct × 残り4 runs × 3 datasets ──────────────────
echo "######## Phase 2: Qwen2.5-3B-Instruct (run 2-5) ########"
for dataset in qampari asqa expertqa; do
    for run in 2 3 4 5; do
        run_eval "Qwen/Qwen2.5-3B-Instruct" "qwen-2-5-3b" "${dataset}" "${run}"
    done
done

# ── Phase 3: Qwen2.5-1.5B-Instruct × 5 runs × 3 datasets ────────────────────
echo "######## Phase 3: Qwen2.5-1.5B-Instruct ########"
for dataset in qampari asqa expertqa; do
    for run in 1 2 3 4 5; do
        run_eval "Qwen/Qwen2.5-1.5B-Instruct" "qwen-2-5-1-5b" "${dataset}" "${run}"
    done
done

# ── Phase 4: Qwen2.5-0.5B-Instruct × 5 runs × 3 datasets ────────────────────
echo "######## Phase 4: Qwen2.5-0.5B-Instruct ########"
for dataset in qampari asqa expertqa; do
    for run in 1 2 3 4 5; do
        run_eval "Qwen/Qwen2.5-0.5B-Instruct" "qwen-2-5-0-5b" "${dataset}" "${run}"
    done
done

echo ""
echo "======================================================"
echo " ALL DONE at $(date)"
echo "======================================================"
