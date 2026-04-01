import argparse
import glob
import json
import os
import csv
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

# ────────────────────────────────────────────────────────────────────────────
# Import unified pipeline helpers.
# ────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

sys.path.append(os.path.join(SCRIPT_DIR, '..'))

# Make pyAudioAnalysis importable
_PAA_DIR = os.path.join(SCRIPT_DIR, '..', 'pyAudioAnalysis')
if os.path.isdir(_PAA_DIR):
    sys.path.insert(0, os.path.abspath(_PAA_DIR))

from model_training.train_pyAudioAnalysis import extract_file_features

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

def load_label_mapping(label_csv):
    """Load bidirectional label mappings from CSV (index <-> mid)."""
    df = pd.read_csv(label_csv)
    return {
        'index_to_mid': {int(row['index']): row['mid'] for _, row in df.iterrows()},
        'mid_to_index': {row['mid']: int(row['index']) for _, row in df.iterrows()},
    }

# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def load_ensemble(ensemble_dir_pattern, classifier):
    """
    Find all fold directories matching ensemble_dir_pattern and load the
    sklearn model + scaler for the given classifier.
    """
    fold_dirs = sorted(glob.glob(ensemble_dir_pattern))
    if not fold_dirs:
        raise FileNotFoundError(
            f"No directories found matching: {ensemble_dir_pattern}"
        )

    models, scalers = [], []
    for fold_dir in fold_dirs:
        model_path  = os.path.join(fold_dir, classifier, "model.pkl")
        scaler_path = os.path.join(fold_dir, classifier, "scaler.pkl")
        if not os.path.exists(model_path):
            print(f"  [skip] model not found: {model_path}")
            continue
        if not os.path.exists(scaler_path):
            print(f"  [skip] scaler not found: {scaler_path}")
            continue
        with open(model_path,  "rb") as f:
            models.append(pickle.load(f))
        with open(scaler_path, "rb") as f:
            scalers.append(pickle.load(f))
        print(f"  Loaded: {fold_dir}  [{classifier}]")

    if not models:
        raise FileNotFoundError(
            f"No {classifier} model.pkl / scaler.pkl found in any fold under "
            f"{ensemble_dir_pattern}"
        )

    print(f"Loaded {len(models)} ensemble model(s) for classifier='{classifier}'.")
    return models, scalers

# ────────────────────────────────────────────────────────────────────────────
# Dataset processing
# ────────────────────────────────────────────────────────────────────────────

def process_dataset(json_files, models, scalers, label_mappings,
                    mid_window=1.0, mid_step=1.0, short_window=0.05, short_step=0.05,
                    resample=False, target_sr=16000, apply_filter=False, cutoff_freq=1024):
    """
    Run ensemble inference over all items in the given JSON files, treating each audio file as one event.
    """
    mid_to_index = label_mappings['mid_to_index']
    index_to_mid = label_mappings['index_to_mid']
    pos_class_mid = index_to_mid[0]           # positive class (index 0)

    records = []
    
    # We iterate over all valid files
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f).get('data', [])

        for item in data:
            wav_path = item['wav']
            try:
                # Extract features using pyAudioAnalysis logic from run_pyaudio
                fv = extract_file_features(wav_path, mid_window, mid_step, short_window, short_step,
                                           resample, target_sr, apply_filter, cutoff_freq)
                if fv is None:
                    continue
                
                # Ensemble inference
                probs = []
                for model, scaler in zip(models, scalers):
                    x = scaler.transform(fv.reshape(1, -1))
                    
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(x)[0, 0] # Index 0 represents label 0
                    else:
                        d = model.decision_function(x)[0]
                        # Handling multi-class vs single-class outputs
                        if isinstance(d, np.ndarray) and d.size > 1:
                            d = d[0]
                        p = 1.0 / (1.0 + np.exp(-d))
                        
                    probs.append(p)
                
                avg_prob = float(np.mean(probs))
                records.append({
                    'target': mid_to_index[item['labels']],
                    pos_class_mid: avg_prob,
                })

            except Exception as e:
                print(f"[warn] Skipped {wav_path}: {e}")
                continue

    return records


# ────────────────────────────────────────────────────────────────────────────
# Threshold search
# ────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find best classification threshold for pyAudioAnalysis ensemble")

    parser.add_argument("--ensemble_dir", type=str, required=True,
                        help="Glob pattern for fold dirs (each must have <classifier>/model.pkl)")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory containing val_*.json and test_*.json files")
    parser.add_argument("--label_csv", type=str, required=True,
                        help="CSV with label mappings (columns: index, mid)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV for threshold sweep results")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--mode", type=str, default=mode, choices=["indep", "merge"],
                        help="Mode of operation: 'indep' for independent fold processing, 'merge' for merged ensemble processing.")
                        
    parser.add_argument('--classifier', type=str, default='gradientboosting',
                        choices=['svm', 'svm_rbf', 'randomforest', 'gradientboosting', 'extratrees'],
                        help='Which trained classifier to use for inference')

    # Audio preprocessing options (must match training config)
    parser.add_argument("--resample", type=str2bool, default=False,
                        help="Resample audio to --target_sr")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate (used when --resample is set)")
    parser.add_argument("--filter", type=str2bool, default=False,
                        help="Apply low-pass filter")
    parser.add_argument("--cutoff_freq", type=int, default=1024,
                        help="Cutoff frequency for low-pass filter")

    # pyAudioAnalysis feature config (must match training config)
    parser.add_argument('--mid_window',   type=float, default=1.0,
                        help='Mid-term window length (seconds)')
    parser.add_argument('--mid_step',     type=float, default=1.0,
                        help='Mid-term step (seconds)')
    parser.add_argument('--short_window', type=float, default=0.05,
                        help='Short-term window length (seconds)')
    parser.add_argument('--short_step',   type=float, default=0.05,
                        help='Short-term step (seconds)')

    args = parser.parse_args()

    # ── Load label mappings ─────────────────────────────────────────────────
    label_mappings = load_label_mapping(args.label_csv)
    pos_class_mid  = label_mappings['index_to_mid'][0]

    # ------------------------------------------------------------------ #
    # Shared test set (fixed across all folds)
    # ------------------------------------------------------------------ #
    test_json = os.path.join(args.json_dir, "test.json")
    if not os.path.isfile(test_json):
        raise FileNotFoundError(f"test.json not found: {test_json}")

    if args.mode == "indep":
        fold_dirs = sorted(glob.glob(args.ensemble_dir))
        if not fold_dirs:
            raise FileNotFoundError(f"No fold directories found for pattern: {args.ensemble_dir}")
        print(f"Found {len(fold_dirs)} fold model directories.")

        fold_thresholds = []
        fold_val_f1s    = []
        fold_results    = []   # (threshold, test_f1, test_recall, test_precision)

        for fold_i in range(args.n_folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_i}")
            print(f"{'='*60}")

            fold_dir = fold_dirs[fold_i] if fold_i < len(fold_dirs) else None
            if fold_dir is None:
                print(f"  WARNING: no model directory for fold {fold_i}, skipping.")
                continue
            
            try:
                fold_models, fold_scalers = load_ensemble(fold_dir, args.classifier)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}, skipping.")
                continue

            val_json = os.path.join(args.json_dir, f"val_fold{fold_i}.json")
            if not os.path.isfile(val_json):
                print(f"  WARNING: {val_json} not found, skipping fold {fold_i}.")
                continue

            fold_output_csv = args.output_csv.replace(".csv", f"_fold{fold_i}.csv")

            infer_kwargs = dict(
                models=fold_models,
                scalers=fold_scalers,
                label_mappings=label_mappings,
                mid_window=args.mid_window,
                mid_step=args.mid_step,
                short_window=args.short_window,
                short_step=args.short_step,
                resample=args.resample,
                target_sr=args.target_sr,
                apply_filter=args.filter,
                cutoff_freq=args.cutoff_freq,
            )

            print(f"  Val JSON : {val_json}")
            val_records  = process_dataset([val_json], **infer_kwargs)
            pd.DataFrame(val_records).to_csv(
                args.output_csv.replace(".csv", f"_val_records_fold{fold_i}.csv"), index=False)

            best_thr, val_f1, rec_thr, rec_pre, rec_val, _ = find_best_threshold(
                val_records, pos_class_mid, fold_output_csv)
            print(f"  [F1]    Best threshold={best_thr:.2f}  val F1={val_f1:.4f}")
            print(f"  [Recall] Best threshold={rec_thr:.2f}  "
                  f"recall={rec_val:.4f}  precision@peak_recall={rec_pre:.4f}")
            fold_thresholds.append(best_thr)
            fold_val_f1s.append(val_f1)

            print(f"  Test JSON: {test_json}")
            test_records = process_dataset([test_json], **infer_kwargs)
            y_true = [x['target'] for x in test_records]
            
            y_pred = [0 if x[pos_class_mid] >= best_thr else 1 for x in test_records]
            t_f1   = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            t_rec  = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            t_pre  = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
            print(f"  [F1 thr]    Test F1={t_f1:.4f}  Recall={t_rec:.4f}  Precision={t_pre:.4f}")

            y_pred_rec = [0 if x[pos_class_mid] >= rec_thr else 1 for x in test_records]
            t_f1_rec  = f1_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
            t_rec_rec = recall_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
            t_pre_rec = precision_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
            print(f"  [Recall thr] Test F1={t_f1_rec:.4f}  Recall={t_rec_rec:.4f}  Precision={t_pre_rec:.4f}")

            fold_results.append((best_thr, t_f1, t_rec, t_pre,
                                 rec_thr,  t_f1_rec, t_rec_rec, t_pre_rec))

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

        try:
            merged_models, merged_scalers = load_ensemble(args.ensemble_dir, args.classifier)
        except Exception as e:
            raise RuntimeError(f"Failed to load merged ensemble models: {e}")

        val_json_files = [os.path.join(args.json_dir, f"val_fold{i}.json") for i in range(args.n_folds)]
        val_json_files = [f for f in val_json_files if os.path.isfile(f)]
        if not val_json_files:
            raise FileNotFoundError(f"No validation JSON files found in {args.json_dir}")
        
        infer_kwargs = dict(
            models=merged_models,
            scalers=merged_scalers,
            label_mappings=label_mappings,
            mid_window=args.mid_window,
            mid_step=args.mid_step,
            short_window=args.short_window,
            short_step=args.short_step,
            resample=args.resample,
            target_sr=args.target_sr,
            apply_filter=args.filter,
            cutoff_freq=args.cutoff_freq,
        )

        print(f"  Val JSONs: {val_json_files}")
        val_records = process_dataset(val_json_files, **infer_kwargs)
        
        merged_output_csv = args.output_csv.replace(".csv", "_merged_val.csv")
        best_thr, val_f1, rec_thr, rec_pre, rec_val, _ = find_best_threshold(
            val_records, pos_class_mid, merged_output_csv)
        print(f"  [F1]    Best threshold={best_thr:.2f}  val F1={val_f1:.4f}")
        print(f"  [Recall] Best threshold={rec_thr:.2f}  "
              f"recall={rec_val:.4f}  precision@peak_recall={rec_pre:.4f}")

        print(f"  Test JSON: {test_json}")
        test_records = process_dataset([test_json], **infer_kwargs)
        y_true = [x['target'] for x in test_records]

        # Calculate performance across ALL thresholds on the test set
        merged_test_sweep_csv = args.output_csv.replace(".csv", "_test_sweep.csv")
        find_best_threshold(test_records, pos_class_mid, merged_test_sweep_csv)
        print(f"  Test set sweep saved to: {merged_test_sweep_csv}")

        y_pred_f1 = [0 if x[pos_class_mid] >= best_thr else 1 for x in test_records]
        t_f1_f1   = f1_score(y_true, y_pred_f1, pos_label=0, zero_division=0)
        t_rec_f1  = recall_score(y_true, y_pred_f1, pos_label=0, zero_division=0)
        t_pre_f1  = precision_score(y_true, y_pred_f1, pos_label=0, zero_division=0)
        print(f"  [F1 thr]    Test F1={t_f1_f1:.4f}  Recall={t_rec_f1:.4f}  Precision={t_pre_f1:.4f}")

        y_pred_rec = [0 if x[pos_class_mid] >= rec_thr else 1 for x in test_records]
        t_f1_rec  = f1_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
        t_rec_rec = recall_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
        t_pre_rec = precision_score(y_true, y_pred_rec, pos_label=0, zero_division=0)
        print(f"  [Recall thr] Test F1={t_f1_rec:.4f}  Recall={t_rec_rec:.4f}  Precision={t_pre_rec:.4f}")

        summary_path = args.output_csv.replace(".csv", "_merged_summary.csv")
        merged_summary = pd.DataFrame([{
            'f1_threshold': best_thr, 'test_f1': t_f1_f1, 'test_recall': t_rec_f1, 'test_precision': t_pre_f1,
            'rec_threshold': rec_thr, 'test_f1_rec': t_f1_rec, 'test_recall_rec': t_rec_rec, 'test_precision_rec': t_pre_rec
        }])
        merged_summary.to_csv(summary_path, index=False)
        print(f"\nMerged ensemble summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
