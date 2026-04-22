#!/bin/bash

# AST Training Script for Cattle Eructation Detection
# Cow-Independent 5-fold Cross-Validation
#
# Usage:
#   ./ast_training.sh <data_dir> <exp_root> [pretrained_model_path]
#
# Example:
#   ./ast_training.sh \
#       /path/to/json_files_cow_indep \
#       /path/to/experiments/ast \
#       /path/to/audioset_10_10_0.4593.pth
#
# JSON layout (5 folds, fixed test set):
#   <data_dir>/train_fold{0..4}.json
#   <data_dir>/val_fold{0..4}.json
#   <data_dir>/test.json
#   <data_dir>/label_index.csv

set -e

# ── Argument parsing ─────────────────────────────────────────────────────────
AST_DATASET=${1:?Usage: $0 <data_dir> <exp_root> [pretrained_model_path]}
EXP_ROOT=${2:?Usage: $0 <data_dir> <exp_root> [pretrained_model_path]}
PRETRAINED_MODEL=${3:-""}

label_index="${AST_DATASET}/label_index.csv"

# ── Training Hyperparameters ─────────────────────────────────────────────────
batch_size=12
n_epochs=100
warmup=5
patience=15
lr=1e-5
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
freqm=48
timem=48
mixup=0

# ── Audio settings ───────────────────────────────────────────────────────────
fstride=10
tstride=10

# Imagenet pretrained backbone (always True for AST fine-tuning)
imagenet_pretrain=True

# Verify data directory
if [ ! -f "${label_index}" ]; then
    echo "ERROR: label_index.csv not found at: ${label_index}"
    echo "Please ensure the data directory contains the required files."
    exit 1
fi

# ── Filter / Resample sweep ──────────────────────────────────────────────────
# Set to (True) or (False) to control preprocessing.
# The paper evaluates all 4 combinations: filter × resample
filter_options=(True False)
resample_options=(True False)

echo "=================================================="
echo "AST Training for Cattle Eructation Detection"
echo "Dataset      : ${AST_DATASET}"
echo "Experiments  : ${EXP_ROOT}"
echo "Folds        : 0..4 (cow-independent 5-fold CV)"
echo "=================================================="

for filter_val in "${filter_options[@]}"; do
    for resample_val in "${resample_options[@]}"; do

        # Determine sample rate
        # Original audio is 8 kHz; resample=True → 16 kHz (AST default)
        if [ "${resample_val}" == "False" ]; then
            sample_rate=8000
        else
            sample_rate=16000
        fi

        # Determine pretrained model usage
        if [ -n "${PRETRAINED_MODEL}" ] && [ -f "${PRETRAINED_MODEL}" ]; then
            audioset_pretrain=True
            pretrained_arg="--audioset-pretrain True --pretrained-model-path ${PRETRAINED_MODEL}"
        else
            audioset_pretrain=False
            pretrained_arg="--audioset-pretrain False"
        fi

        exp_tag="ast_filter_${filter_val}_resample_${resample_val}"
        echo ""
        echo "Configuration: filter=${filter_val}, resample=${resample_val}, sr=${sample_rate}"
        echo "Experiment tag: ${exp_tag}"

        # ── Loop through 5 cow-independent folds ─────────────────────────
        for i in $(seq 0 4); do
            exp_dir="${EXP_ROOT}/${exp_tag}/fold_${i}"
            mkdir -p "${exp_dir}"

            echo "  Running Fold ${i}..."

            TRAIN_JSON="${AST_DATASET}/train_fold${i}.json"
            VAL_JSON="${AST_DATASET}/val_fold${i}.json"
            TEST_JSON="${AST_DATASET}/test.json"

            python run.py \
                --data-train "${TRAIN_JSON}" \
                --data-val "${VAL_JSON}" \
                --data-eval "${TEST_JSON}" \
                --label-csv "${label_index}" \
                --exp-dir "${exp_dir}" \
                --lr ${lr} \
                --n-epochs ${n_epochs} \
                --batch-size ${batch_size} \
                --warmup-epochs ${warmup} \
                --patience ${patience} \
                --lrscheduler-start ${lrscheduler_start} \
                --lrscheduler-decay ${lrscheduler_decay} \
                --lrscheduler-step ${lrscheduler_step} \
                --fstride ${fstride} \
                --tstride ${tstride} \
                --freqm ${freqm} \
                --timem ${timem} \
                --mixup ${mixup} \
                --imagenet-pretrain ${imagenet_pretrain} \
                ${pretrained_arg} \
                --filter ${filter_val} \
                --resample ${resample_val} \
                --sample-rate ${sample_rate} \
                --eval-test True

            echo "  Fold ${i} Completed."
        done
    done
done

echo ""
echo "All AST experiments completed."
