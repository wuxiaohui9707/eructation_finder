#!/bin/bash

# Feature Engineering-Based Classifiers Training Script
# Cattle Eructation Detection — Cow-Independent 5-fold CV
#
# Uses pyAudioAnalysis for feature extraction + sklearn classifiers
# (SVM, SVM-RBF, Random Forest, Gradient Boosting, Extra Trees)
#
# Usage:
#   ./train_new.sh <data_dir> <exp_root>
#
# Example:
#   ./train_new.sh \
#       /path/to/json_files_cow_indep \
#       /path/to/experiments/pyaudioanalysis
#
# JSON layout (5 folds, fixed test set):
#   <data_dir>/train_fold{0..4}.json
#   <data_dir>/val_fold{0..4}.json
#   <data_dir>/test.json
#   <data_dir>/label_index.csv

set -e

# ── Argument parsing ─────────────────────────────────────────────────────────
DATASET=${1:?Usage: $0 <data_dir> <exp_root>}
EXP_ROOT=${2:?Usage: $0 <data_dir> <exp_root>}

label_index="${DATASET}/label_index.csv"

# Verify data directory
if [ ! -f "${label_index}" ]; then
    echo "ERROR: label_index.csv not found at: ${label_index}"
    exit 1
fi

# ── Filter / Resample sweep ──────────────────────────────────────────────────
filter_options=(False)
resample_options=(False)

echo "=================================================="
echo "Feature Engineering Classifiers Training"
echo "Dataset      : ${DATASET}"
echo "Experiments  : ${EXP_ROOT}"
echo "Folds        : 0..4 (cow-independent 5-fold CV)"
echo "=================================================="

for filter_val in "${filter_options[@]}"; do
    for resample_val in "${resample_options[@]}"; do

        # Determine sample rate
        if [ "${resample_val}" == "False" ]; then
            sample_rate=8000
        else
            sample_rate=16000
        fi

        exp_tag="pyaudio_filter_${filter_val}_resample_${resample_val}"
        echo ""
        echo "Configuration: filter=${filter_val}, resample=${resample_val}, sr=${sample_rate}"

        for i in $(seq 0 4); do
            exp_dir="${EXP_ROOT}/${exp_tag}/fold_${i}"
            mkdir -p "${exp_dir}"

            echo "  Running Fold ${i}..."

            TRAIN_JSON="${DATASET}/train_fold${i}.json"
            VAL_JSON="${DATASET}/val_fold${i}.json"
            TEST_JSON="${DATASET}/test.json"

            python run_pyaudio.py \
                --data-train "${TRAIN_JSON}" \
                --data-val   "${VAL_JSON}" \
                --data-eval  "${TEST_JSON}" \
                --label-csv  "${label_index}" \
                --exp-dir    "${exp_dir}" \
                --classifier all \
                --sample-rate "${sample_rate}" \
                --filter "${filter_val}" \
                --resample "${resample_val}"

            echo "  Fold ${i} Completed."
        done
    done
done

echo ""
echo "All feature engineering experiments completed."
