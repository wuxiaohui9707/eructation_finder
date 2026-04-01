import argparse
import glob
import json
import os
import csv
import numpy as np
import torch
import torchaudio
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
import sys
import os

# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.models.ast_model import ASTModelVis
from core.data_module import resample_audio

def load_label_mapping(label_csv):
    """Load bidirectional label mappings from CSV"""
    df = pd.read_csv(label_csv)
    return {
        'index_to_mid': {int(row['index']): row['mid'] for _, row in df.iterrows()},
        'mid_to_index': {row['mid']: int(row['index']) for _, row in df.iterrows()}
    }

def load_ensemble_models(ensemble_dir_pattern):
    """Load ensemble models from checkpoint directories"""
    model_paths = glob.glob(os.path.join(ensemble_dir_pattern, "checkpoints", "best_model.ckpt"))
    models = []
    for ckpt_path in model_paths:
        model = ASTModelVis()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(torch.device('cuda')).eval()
        models.append(model)
    return models

def make_features(waveform, sr, mel_bins=128, target_length=500,
                  norm_mean=4.2677393, norm_std=4.5689974):
    """Generate Mel-spectrogram features (matches AudiosetDataset._wav2fbank exactly)"""
    # Subtract waveform mean (same as _wav2fbank line 271 in ast_data_module.py)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
    )
    n_frames = fbank.shape[0]
    if n_frames < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - n_frames))
    else:
        fbank = fbank[:target_length, :]
    # norm_mean is POSITIVE (e.g. 4.2677393); subtract it, do NOT negate
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

def process_dataset(json_files, models, label_mappings, resample=True):
    mid_to_index = label_mappings['mid_to_index']
    index_to_mid = label_mappings['index_to_mid']
    pos_class_mid = index_to_mid[0]
    
    records = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f).get('data', [])
        
        for item in data:
            try:
                waveform, sr = resample_audio(item['wav'], target_sample_rate=16000, resample=resample)
                
                mel = make_features(waveform, sr)
                mel = mel.unsqueeze(0).to('cuda', dtype=torch.float32)
                
                probs = []
                for model in models:
                    with torch.no_grad():
                        logits = model(mel)
                    
                        # Handle dimension variations
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(0)
                            
                        # Process different output types
                        if logits.shape[1] == 1:
                            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                        elif logits.shape[1] == 2:
                            prob = torch.softmax(logits, dim=1)[:, 0].squeeze().cpu().numpy()
                        else:
                            raise ValueError(f"Invalid output dimension: {logits.shape[1]}")
                        
                        probs.append(prob)
                
                avg_prob = np.mean(probs)
                records.append({
                    'target': mid_to_index[item['labels']],
                    pos_class_mid: avg_prob
                })
            
            except Exception as e:
                print(f"Error processing {item['wav']}: {str(e)}")
                continue
    return records

def find_best_threshold(records, pos_class_mid, output_csv):
    """Find optimal threshold by two criteria and save sweep to CSV.

    Returns
    -------
    best_f1_thr   : threshold maximising F1
    best_f1       : F1 at best_f1_thr
    best_rec_thr  : threshold with highest recall; tie-broken by highest precision
    best_rec_pre  : precision at best_rec_thr
    best_rec_val  : recall value at best_rec_thr
    results       : list of (threshold, f1) tuples
    """
    thresholds = np.arange(0, 1.01, 0.01)
    results = []

    best_f1_thr   = 0.0
    best_f1       = 0.0
    best_rec_thr  = 0.0
    best_rec_val  = 0.0
    best_rec_pre  = 0.0   # precision at peak recall (tie-breaker)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Threshold', 'F1_Score', 'Recall', 'Precision'])

        for threshold in thresholds:
            y_true = [x['target'] for x in records]
            y_pred = [0 if x[pos_class_mid] >= threshold else 1 for x in records]
            f1        = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            recall    = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)

            writer.writerow([f"{threshold:.2f}", f"{f1:.4f}",
                             f"{recall:.4f}", f"{precision:.4f}"])

            # Criterion 1: max F1
            if f1 > best_f1:
                best_f1     = f1
                best_f1_thr = threshold

            # Criterion 2: max recall; tie-break by max precision
            if (recall > best_rec_val) or \
               (recall == best_rec_val and precision > best_rec_pre):
                best_rec_val = recall
                best_rec_pre = precision
                best_rec_thr = threshold

            results.append((threshold, f1))

    return best_f1_thr, best_f1, best_rec_thr, best_rec_pre, best_rec_val, results

def main():
    parser = argparse.ArgumentParser(description="Find best classification threshold for AST ensemble")

    parser.add_argument("--ensemble_dir", type=str, required=True,
                        help="Glob pattern for fold dirs (each must have checkpoints/best_model*.ckpt)")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory containing val_*.json and test_*.json files")
    parser.add_argument("--label_csv", type=str, required=True,
                        help="CSV with label mappings (columns: index, mid)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV for threshold sweep results")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds (must match ast_dataset_make_new.py)")
    parser.add_argument("--resample", type=bool, default=True,
                        help="Resample audio to 16kHz")
    parser.add_argument("--mode", type=str, default="merge", choices=["indep", "merge"],
                        help="Mode of operation: 'indep' for independent fold processing, 'merge' for merged ensemble processing.")
    args = parser.parse_args()

    label_mappings = load_label_mapping(args.label_csv)
    pos_class_mid  = label_mappings['index_to_mid'][0]

    # ------------------------------------------------------------------ #
    # Shared test set (fixed across all folds)
    # ------------------------------------------------------------------ #
    test_json = os.path.join(args.json_dir, "test.json")
    if not os.path.isfile(test_json):
        raise FileNotFoundError(f"test.json not found: {test_json}")

    if args.mode == "indep":
        # Collect per-fold model paths so we can load fold-specific models
        fold_model_dirs = sorted(glob.glob(args.ensemble_dir))
        if not fold_model_dirs:
            raise FileNotFoundError(f"No fold directories found for pattern: {args.ensemble_dir}")
        print(f"Found {len(fold_model_dirs)} fold model directories.")

        fold_thresholds = []
        fold_val_f1s    = []
        fold_results    = []   # (threshold, test_f1, test_recall, test_precision)

        for fold_i in range(args.n_folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_i}")
            print(f"{'='*60}")

            # ---- Load THIS fold's model only (for threshold search on its val set) ----
            # Using only the matched fold model avoids data leakage: the val set for
            # fold i was never seen by fold i's model during training.
            fold_dir = fold_model_dirs[fold_i] if fold_i < len(fold_model_dirs) else None
            if fold_dir is None:
                print(f"  WARNING: no model directory for fold {fold_i}, skipping.")
                continue
            fold_models = load_ensemble_models(fold_dir)
            if not fold_models:
                print(f"  WARNING: no checkpoint found in {fold_dir}, skipping.")
                continue

            # ---- Val set for fold i ----
            val_json = os.path.join(args.json_dir, f"val_fold{fold_i}.json")
            if not os.path.isfile(val_json):
                print(f"  WARNING: {val_json} not found, skipping fold {fold_i}.")
                continue

            fold_output_csv = args.output_csv.replace(".csv", f"_fold{fold_i}.csv")

            print(f"  Val JSON : {val_json}")
            val_records  = process_dataset([val_json], fold_models, label_mappings,
                                           args.resample)
            pd.DataFrame(val_records).to_csv(
                args.output_csv.replace(".csv", f"_val_records_fold{fold_i}.csv"), index=False)

            best_thr, val_f1, rec_thr, rec_pre, rec_val, _ = find_best_threshold(
                val_records, pos_class_mid, fold_output_csv)
            print(f"  [F1]    Best threshold={best_thr:.2f}  val F1={val_f1:.4f}")
            print(f"  [Recall] Best threshold={rec_thr:.2f}  "
                  f"recall={rec_val:.4f}  precision@peak_recall={rec_pre:.4f}")
            fold_thresholds.append(best_thr)
            fold_val_f1s.append(val_f1)

            # ---- Evaluate on the shared test set using THIS fold's model ----
            print(f"  Test JSON: {test_json}")
            test_records = process_dataset([test_json], fold_models, label_mappings,
                                           args.resample)
            y_true = [x['target'] for x in test_records]
            y_pred = [0 if x[pos_class_mid] >= best_thr else 1 for x in test_records]
            t_f1   = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            t_rec  = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            t_pre  = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
            print(f"  [F1 thr]    Test F1={t_f1:.4f}  Recall={t_rec:.4f}  Precision={t_pre:.4f}")

            # ---- Evaluate with recall-optimised threshold ----
            y_pred_rec = [0 if x[pos_class_mid] >= rec_thr else 1 for x in test_records]
            t_f1_rec  = f1_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
            t_rec_rec = recall_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
            t_pre_rec = precision_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
            print(f"  [Recall thr] Test F1={t_f1_rec:.4f}  Recall={t_rec_rec:.4f}  Precision={t_pre_rec:.4f}")

            fold_results.append((best_thr, t_f1, t_rec, t_pre,
                                 rec_thr,  t_f1_rec, t_rec_rec, t_pre_rec))

        # ------------------------------------------------------------------ #
        # Summary across all folds
        # ------------------------------------------------------------------ #
        print(f"\n{'='*60}")
        print("Summary (mean ± std across folds)")
        print(f"{'='*60}")
        thrs  = [r[0] for r in fold_results]
        f1s   = [r[1] for r in fold_results]
        recs  = [r[2] for r in fold_results]
        pres  = [r[3] for r in fold_results]
        rec_thrs  = [r[4] for r in fold_results]
        f1s_rec   = [r[5] for r in fold_results]
        recs_rec  = [r[6] for r in fold_results]
        pres_rec  = [r[7] for r in fold_results]
        print("  --- Best-F1 threshold ---")
        print(f"  Threshold  : {np.mean(thrs):.3f} ± {np.std(thrs):.3f}")
        print(f"  Test F1    : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  Test Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        print(f"  Test Prec  : {np.mean(pres):.4f} ± {np.std(pres):.4f}")
        print("  --- Best-recall (max-precision) threshold ---")
        print(f"  Threshold  : {np.mean(rec_thrs):.3f} ± {np.std(rec_thrs):.3f}")
        print(f"  Test F1    : {np.mean(f1s_rec):.4f} ± {np.std(f1s_rec):.4f}")
        print(f"  Test Recall: {np.mean(recs_rec):.4f} ± {np.std(recs_rec):.4f}")
        print(f"  Test Prec  : {np.mean(pres_rec):.4f} ± {np.std(pres_rec):.4f}")

        # Save summary
        summary_path = args.output_csv.replace(".csv", "_summary.csv")
        pd.DataFrame(fold_results,
                     columns=['f1_threshold', 'test_f1', 'test_recall', 'test_precision',
                              'rec_threshold', 'test_f1_rec', 'test_recall_rec', 'test_precision_rec']
                     ).to_csv(summary_path, index=False)
        print(f"\nPer-fold summary saved to: {summary_path}")

    elif args.mode == "merge":
        print(f"\n{'='*60}")
        print("Merged Ensemble Mode")
        print(f"{'='*60}")

        # Load all models from all folds into a single ensemble
        all_model_dirs = sorted(glob.glob(args.ensemble_dir))
        if not all_model_dirs:
            raise FileNotFoundError(f"No fold directories found for pattern: {args.ensemble_dir}")
        
        merged_models = []
        for fold_dir in all_model_dirs:
            models_in_fold = load_ensemble_models(fold_dir)
            merged_models.extend(models_in_fold)
        
        if not merged_models:
            raise RuntimeError("No models loaded for merged ensemble.")
        print(f"Loaded {len(merged_models)} models for the merged ensemble.")

        # ---- Val set for threshold search (all val sets combined) ----
        val_json_files = [os.path.join(args.json_dir, f"val_fold{i}.json") for i in range(args.n_folds)]
        val_json_files = [f for f in val_json_files if os.path.isfile(f)]
        if not val_json_files:
            raise FileNotFoundError(f"No validation JSON files found in {args.json_dir}")
        
        print(f"  Val JSONs: {val_json_files}")
        val_records = process_dataset(val_json_files, merged_models, label_mappings,
                                      args.resample)
        
        merged_output_csv = args.output_csv.replace(".csv", "_merged_val.csv")
        best_thr, val_f1, rec_thr, rec_pre, rec_val, _ = find_best_threshold(
            val_records, pos_class_mid, merged_output_csv)
        print(f"  [F1]    Best threshold={best_thr:.2f}  val F1={val_f1:.4f}")
        print(f"  [Recall] Best threshold={rec_thr:.2f}  "
              f"recall={rec_val:.4f}  precision@peak_recall={rec_pre:.4f}")

        # ---- Evaluate on the shared test set using the merged ensemble ----
        print(f"  Test JSON: {test_json}")
        test_records = process_dataset([test_json], merged_models, label_mappings,
                                       args.resample)
        y_true = [x['target'] for x in test_records]

        # Calculate performance across ALL thresholds on the test set
        merged_test_sweep_csv = args.output_csv.replace(".csv", "_test_sweep.csv")
        find_best_threshold(test_records, pos_class_mid, merged_test_sweep_csv)
        print(f"  Test set sweep saved to: {merged_test_sweep_csv}")

        # Evaluate with F1-optimised threshold from combined validation set
        y_pred_f1 = [0 if x[pos_class_mid] >= best_thr else 1 for x in test_records]
        t_f1_f1   = f1_score(y_true, y_pred_f1, pos_label=0, zero_division=0)
        t_rec_f1  = recall_score(y_true, y_pred_f1, pos_label=0, zero_division=0)
        t_pre_f1  = precision_score(y_true, y_pred_f1, pos_label=0, zero_division=0)
        print(f"  [F1 thr]    Test F1={t_f1_f1:.4f}  Recall={t_rec_f1:.4f}  Precision={t_pre_f1:.4f}")

        # Evaluate with recall-optimised threshold from combined validation set
        y_pred_rec = [0 if x[pos_class_mid] >= rec_thr else 1 for x in test_records]
        t_f1_rec  = f1_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
        t_rec_rec = recall_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
        t_pre_rec = precision_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
        print(f"  [Recall thr] Test F1={t_f1_rec:.4f}  Recall={t_rec_rec:.4f}  Precision={t_pre_rec:.4f}")

        # Save summary for merged mode
        summary_path = args.output_csv.replace(".csv", "_merged_summary.csv")
        merged_summary = pd.DataFrame([{
            'f1_threshold': best_thr, 'test_f1': t_f1_f1, 'test_recall': t_rec_f1, 'test_precision': t_pre_f1,
            'rec_threshold': rec_thr, 'test_f1_rec': t_f1_rec, 'test_recall_rec': t_rec_rec, 'test_precision_rec': t_pre_rec
        }])
        merged_summary.to_csv(summary_path, index=False)
        print(f"\nMerged ensemble summary saved to: {summary_path}")


if __name__ == "__main__":
    main()