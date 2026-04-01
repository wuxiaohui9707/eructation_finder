#!/bin/bash

# PANNs (Cnn14) Training Script for Cattle Eructation Detection
# Cow-Independent 5-fold Cross-Validation
#
# Usage:
#   ./train_new.sh <data_dir> <exp_root> [pretrained_model_path]
#
# Example:
#   ./train_new.sh \
#       /path/to/json_files_cow_indep \
#       /path/to/experiments/panns \
#       /path/to/Cnn14_mAP=0.431.pth
#
# JSON layout (5 folds, fixed test set):
#   <data_dir>/train_fold{0..4}.json
#   <data_dir>/val_fold{0..4}.json
#   <data_dir>/test.json
#   <data_dir>/label_index.csv

set -e

# ── Argument parsing ─────────────────────────────────────────────────────────
DATASET=${1:?Usage: $0 <data_dir> <exp_root> [pretrained_model_path]}
EXP_ROOT=${2:?Usage: $0 <data_dir> <exp_root> [pretrained_model_path]}
PRETRAINED_MODEL=${3:-""}

label_index="${DATASET}/label_index.csv"

# ── Training Hyperparameters ─────────────────────────────────────────────────
batch_size=16
n_epochs=100
warmup=5
patience=15

# ── PANNs-specific audio settings ────────────────────────────────────────────
# Official Cnn14 always uses mel_bins=64
PANNS_MEL_BINS=64
PANNS_TARGET_LENGTH=500

if [ -n "${PRETRAINED_MODEL}" ] && [ -f "${PRETRAINED_MODEL}" ]; then
    PANNS_PRETRAINED_PATH="${PRETRAINED_MODEL}"
    echo "Using pretrained weights: ${PANNS_PRETRAINED_PATH}"
else
    PANNS_PRETRAINED_PATH=""
    echo "Training from scratch (no pretrained weights)"
fi

# Verify data directory
if [ ! -f "${label_index}" ]; then
    echo "ERROR: label_index.csv not found at: ${label_index}"
    exit 1
fi

# ── Filter / Resample sweep ──────────────────────────────────────────────────
# The paper evaluates combinations of filter × resample
filter_options=(False)
resample_options=(False)

echo "=================================================="
echo "PANNs Training for Cattle Eructation Detection"
echo "Dataset      : ${DATASET}"
echo "Experiments  : ${EXP_ROOT}"
echo "Folds        : 0..4 (cow-independent 5-fold CV)"
echo "=================================================="

for filter_val in "${filter_options[@]}"; do
    for resample_val in "${resample_options[@]}"; do

        # Determine sample rate
        # Original audio is 8 kHz
        # resample=False → 8 kHz (original)
        # resample=True  → 32 kHz (official Cnn14 pretrained spec)
        if [ "${resample_val}" == "False" ]; then
            sample_rate=8000
        else
            sample_rate=32000
        fi

        exp_tag="panns_filter_${filter_val}_resample_${resample_val}"
        echo ""
        echo "Configuration: filter=${filter_val}, resample=${resample_val}, sr=${sample_rate}"

        for i in $(seq 0 4); do
            exp_dir="${EXP_ROOT}/${exp_tag}/fold_${i}"
            mkdir -p "${exp_dir}"

            echo "  Running Fold ${i}..."

            TRAIN_JSON="${DATASET}/train_fold${i}.json"
            VAL_JSON="${DATASET}/val_fold${i}.json"
            TEST_JSON="${DATASET}/test.json"

            python run.py \
                --model panns \
                --data-train "${TRAIN_JSON}" \
                --data-val   "${VAL_JSON}" \
                --data-eval  "${TEST_JSON}" \
                --label-csv  "${label_index}" \
                --exp-dir    "${exp_dir}" \
                --sample-rate "${sample_rate}" \
                --mel-bins "${PANNS_MEL_BINS}" \
                --target-length "${PANNS_TARGET_LENGTH}" \
                --batch-size "${batch_size}" \
                --n-epochs "${n_epochs}" \
                --warmup-epochs "${warmup}" \
                --patience "${patience}" \
                --filter "${filter_val}" \
                --resample "${resample_val}" \
                --panns-pretrained-path "${PANNS_PRETRAINED_PATH}" \
                --eval-test True

            echo "  Fold ${i} Completed."
        done
    done
done

echo ""
echo "All PANNs experiments completed."
