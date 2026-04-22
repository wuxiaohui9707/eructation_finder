# Cattle Eructation (Burp) Detection System

This repository contains the official codebase for the quantitative acoustic analysis and detection of cattle rumination and eructation (burps). The project has been refactored into a rigorous, process-centric pipeline to ensure usability, reproducibility, and high modularity. 

It supports state-of-the-art Deep Learning models like **Audio Spectrogram Transformer (AST)** and **PANNs (Cnn14)**, as well as Traditional Machine Learning ensembles (via `pyAudioAnalysis`).

---

## 📂 Project Architecture

The codebase is organized into a chronological 5-step machine learning lifecycle, supported by a shared `core` package.

```text
.
├── core/                           # Shared foundational logic (Models, Data Modules, Utils)
├── data_preparation/               # Step 1: Data extraction and JSON metadata preparation
├── model_training/                 # Step 2: PyTorch Lightning model training scripts
├── threshold_finetune/             # Step 3: Optimal threshold search and evaluation
├── inference_unseen_audio/         # Step 4: Sliding-window inference on continuous raw audio
└── visualization/                  # Step 5: Feature visualization, t-SNE, and attention heatmaps
```

---

## 🚀 Quick Start Workflow

Follow this step-by-step workflow to reproduce the experiments or train your own models.

### Step 1: Data Preparation
Convert your raw audio directories and label annotations into training-ready JSON formats.
```bash
python data_preparation/dataset_make.py \
    --wav_path /path/to/raw/audio \
    --label_json /path/to/labels.json \
    --output_dir /path/to/output_json_folder
```

### Step 2: Model Training
Train a chosen architecture using isolated event data. AST and PANNs perfectly share the unified Kaldi-based Mel-Spectrogram extraction infrastructure (`core/data_module.py`). 

**Train AST:**
```bash
python model_training/train_AST.py \
    --data_dir /path/to/output_json_folder \
    --exp_dir experiments/ast_train \
    --freq_division_mode uniform \
    --batch_size 16 \
    --max_epochs 100
```

**Train PANNs:**
```bash
python model_training/train_PANNs.py \
    --data_dir /path/to/output_json_folder \
    --exp_dir experiments/panns_train \
    --freq_division_mode uniform \
    --mel_bins 64 \
    --batch_size 16
```

### Step 3: Threshold Finetuning
Calculate maximum F1-scores and Recalls by sweeping optimal thresholds on validation datasets.
```bash
python threshold_finetune/get_best_threshold_AST.py \
    --ensemble_dir "experiments/ast_train/cv_fold_*" \
    --json_dir /path/to/output_json_folder \
    --label_csv /path/to/label_index.csv \
    --output_csv results_ast_threshold.csv
```

### Step 4: Inference on Unseen Audio (Sliding Window)
Employ a continuous sliding-window inference to detect burp events on long, continuous recordings.
```bash
python inference_unseen_audio/inference_AST.py \
    --ensemble_dir "experiments/ast_train/cv_fold_*" \
    --audio_input /path/to/continuous/cattle_audio.wav \
    --label_csv /path/to/label_index.csv \
    --exp_dir inference_outputs \
    --window_size 5 --step_size 0.5 \
    --nms True
```

### Step 5: Visualization
Generate insightful graphics comparing standard uniform mel-filterbanks with custom 1kHz-split filterbanks, or plot t-SNE embeddings.
```bash
# Visualize AST structural patch heatmaps
python visualization/ast_visualization.py \
    --audio /path/to/sample.wav \
    --n_mels 128 \
    --patch_size 16 \
    --output ast_vis_

# Plot t-SNE feature embeddings
python visualization/plot_tsne.py \
    --features_file /path/to/features.npz \
    --output_dir tsne_outputs
```

---

## 🛠 Prerequisites

Ensure you have a Python `3.8+` environment configured. Install dependencies including:

* `torch` , `torchaudio`, `torchvision` (Compatible with your CUDA version)
* `pytorch-lightning`
* `scikit-learn`, `pandas`, `numpy`, `scipy`
* `librosa`, `matplotlib`
* `pyAudioAnalysis` (Needs to be physically present via Git submodules or in the root directory for Feature Engineering models).

---
*Note: This architecture allows any new deep learning acoustic model to plug directly into the `core/models` layer without rewriting dataset loading or evaluation pipelines.*
