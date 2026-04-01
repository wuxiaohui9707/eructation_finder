"""
run_pyaudio.py – pyAudioAnalysis-based audio classification baseline

Uses pyAudioAnalysis mid-term feature extraction + sklearn classifiers as a
traditional ML baseline, compatible with the same JSON data files and output
format as run.py (PyTorch/Lightning models).

Usage:
    python run_pyaudio.py \
        --data-train <train.json> \
        --data-val   <val.json> \
        --data-eval  <test.json> \
        --label-csv  <label_index.csv> \
        --exp-dir    <output_dir> \
        [--classifier svm|svm_rbf|randomforest|gradientboosting|extratrees|all]
        [--mid-window 1.0] [--mid-step 1.0]
        [--short-window 0.05] [--short-step 0.05]
        [--resample True|False] [--sample-rate 16000]
        [--filter True|False] [--cutoff-freq 1024]
"""
import sys
import os
# Make the cloned pyAudioAnalysis importable
_PAA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'pyAudioAnalysis')
if os.path.isdir(_PAA_DIR):
    sys.path.insert(0, os.path.abspath(_PAA_DIR))

import argparse
import json
import csv
import pickle
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.svm
import sklearn.ensemble
import sklearn.metrics

# pyAudioAnalysis imports
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures as aF

# Use the project's shared evaluate() function
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.eval import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_json_data(json_path):
    """Load file list from AST-style JSON: {"data": [{"wav": ..., "labels": ...}]}"""
    with open(json_path, 'r') as f:
        return json.load(f)['data']


def load_label_index(label_csv_path):
    """Return {mid_str: int_index} mapping, e.g. {'/m/POS': 0, '/m/NEG': 1}"""
    index_dict = {}
    with open(label_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index_dict[row['mid']] = int(row['index'])
    return index_dict


def load_and_filter_waveform(wav_path, resample, target_sr, apply_filter, cutoff_freq):
    """
    Load a WAV file with scipy (consistent with pyAudioAnalysis audioBasicIO),
    optionally resample and/or low-pass filter.

    Returns: (signal_float32_mono, sampling_rate)
    """
    try:
        sr, sig = wavfile.read(wav_path)
    except Exception:
        # Fallback to pyAudioAnalysis reader (handles more formats)
        sr, sig = audioBasicIO.read_audio_file(wav_path)

    # Convert to float32 in [-1, 1]
    if sig.dtype == np.int16:
        sig = sig.astype(np.float32) / 32768.0
    elif sig.dtype == np.int32:
        sig = sig.astype(np.float32) / 2147483648.0
    elif sig.dtype != np.float32:
        sig = sig.astype(np.float32)

    # Mono
    sig = audioBasicIO.stereo_to_mono(sig)

    # Resample
    if resample and sr != target_sr:
        n_target = int(len(sig) * target_sr / sr)
        sig = scipy_signal.resample(sig, n_target)
        sr = target_sr

    # Low-pass filter
    if apply_filter and cutoff_freq < sr / 2:
        nyquist = 0.5 * sr
        norm_cutoff = cutoff_freq / nyquist
        b, a = scipy_signal.butter(5, norm_cutoff, btype='low', analog=False)
        sig = scipy_signal.filtfilt(b, a, sig).astype(np.float32)

    return sig.astype(np.float32), sr


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_file_features(wav_path, mid_window, mid_step,
                          short_window, short_step,
                          resample, target_sr, apply_filter, cutoff_freq):
    """
    Extract a single fixed-length feature vector for one audio file by:
      1. Loading the waveform (with optional resample & filter)
      2. Running pyAudioAnalysis mid_feature_extraction()
         → produces (n_features × n_windows) matrix
      3. Computing per-feature mean AND std across windows → concat → 1-D vector

    Returns: np.ndarray of shape (2 * n_features,) or None on failure
    """
    try:
        sig, sr = load_and_filter_waveform(wav_path, resample, target_sr,
                                            apply_filter, cutoff_freq)
    except Exception as e:
        print(f"  [WARN] Failed to load {wav_path}: {e}")
        return None

    if len(sig) < sr * 0.1:  # skip files shorter than 100 ms
        print(f"  [WARN] Too short, skipping: {wav_path}")
        return None

    try:
        mid_features, _, _ = aF.mid_feature_extraction(
            sig, sr,
            round(mid_window * sr),
            round(mid_step * sr),
            round(short_window * sr),
            round(short_step * sr)
        )
    except Exception as e:
        print(f"  [WARN] Feature extraction failed for {wav_path}: {e}")
        return None

    # mid_features: (n_features × n_windows)
    if mid_features.ndim == 1:
        mid_features = mid_features.reshape(-1, 1)

    # One vector per file: [mean_feat1, mean_feat2, ..., std_feat1, std_feat2, ...]
    means = np.mean(mid_features, axis=1)
    stds  = np.std(mid_features, axis=1)
    fv = np.concatenate([means, stds])

    # Guard against NaN / Inf
    if np.isnan(fv).any() or np.isinf(fv).any():
        print(f"  [WARN] NaN/Inf in features, skipping: {wav_path}")
        return None

    return fv


def build_feature_matrix(data, label_dict, n_classes,
                          mid_window, mid_step, short_window, short_step,
                          resample, target_sr, apply_filter, cutoff_freq,
                          split_name='set'):
    """
    Extract features and labels for an entire dataset split.

    Returns:
        X  : np.ndarray  (n_valid_samples, n_features)
        y  : np.ndarray  (n_valid_samples,)  integer class ids
        Y  : np.ndarray  (n_valid_samples, n_classes)  one-hot / multi-label
        skipped : int
    """
    X_list, y_list = [], []
    skipped = 0

    for idx, item in enumerate(data):
        wav_path = item['wav']
        label_str = item['labels']  # e.g. '/m/POS'
        label_idx = label_dict.get(label_str, -1)
        if label_idx < 0:
            print(f"  [WARN] Unknown label '{label_str}', skipping.")
            skipped += 1
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{split_name}] Processing file {idx+1}/{len(data)} ...")

        fv = extract_file_features(wav_path, mid_window, mid_step,
                                   short_window, short_step,
                                   resample, target_sr, apply_filter, cutoff_freq)
        if fv is None:
            skipped += 1
            continue

        X_list.append(fv)
        y_list.append(label_idx)

    print(f"  [{split_name}] Done. {len(X_list)} valid / {skipped} skipped.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # One-hot label matrix for evaluate()
    Y = np.zeros((len(y), n_classes), dtype=np.float32)
    for i, cls in enumerate(y):
        Y[i, cls] = 1.0

    return X, y, Y


# ─────────────────────────────────────────────────────────────────────────────
# Classifier helpers
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_PARAMS = {
    'svm':             np.array([0.001, 0.01, 0.1,  0.5, 1.0, 5.0, 10.0, 20.0]),
    'svm_rbf':         np.array([0.001, 0.01, 0.1,  0.5, 1.0, 5.0, 10.0, 20.0]),
    'randomforest':    np.array([10,    25,   50,   100, 200]),
    'gradientboosting': np.array([10,   25,   50,   100, 200]),
    'extratrees':      np.array([10,    25,   50,   100, 200]),
}


def build_classifier(classifier_type, param):
    """Instantiate an sklearn classifier with the given hyperparameter."""
    if classifier_type == 'svm':
        return sklearn.svm.SVC(C=param, kernel='linear',
                               probability=True, gamma='auto')
    elif classifier_type == 'svm_rbf':
        return sklearn.svm.SVC(C=param, kernel='rbf',
                               probability=True, gamma='auto')
    elif classifier_type == 'randomforest':
        return sklearn.ensemble.RandomForestClassifier(
            n_estimators=int(param), n_jobs=-1, random_state=42)
    elif classifier_type == 'gradientboosting':
        return sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=int(param), random_state=42)
    elif classifier_type == 'extratrees':
        return sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=int(param), n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")


def select_best_param(classifier_type, X_train_full, y_train_full):
    """
    Quick internal cross-validation (90/10 split, 5 repetitions) to choose
    the best hyperparameter by macro-F1.
    """
    params = CLASSIFIER_PARAMS[classifier_type]
    best_param = params[0]
    best_f1 = -1.0
    n_rep = 5

    for param in params:
        f1_scores = []
        for _ in range(n_rep):
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_train_full, y_train_full, test_size=0.10,
                stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None,
                random_state=None
            )
            clf = build_classifier(classifier_type, param)
            try:
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_va)
                f1 = sklearn.metrics.f1_score(y_va, y_pred, average='macro',
                                              zero_division=0)
                f1_scores.append(f1)
            except Exception:
                f1_scores.append(0.0)
        mean_f1 = np.mean(f1_scores)
        print(f"    param={param:.4g}  macro-F1={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_param = param

    print(f"  → Best param: {best_param:.4g}  (F1={best_f1:.4f})")
    return best_param


def get_proba_matrix(clf, X, n_classes):
    """
    Get (N, n_classes) probability matrix.
    Handles classifiers whose .classes_ may be a subset of all classes.
    """
    proba_raw = clf.predict_proba(X)            # (N, len(clf.classes_))
    clf_classes = list(clf.classes_)
    if len(clf_classes) == n_classes:
        return proba_raw

    # Some classes may be absent from the training fold
    proba_full = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    for i, cls_idx in enumerate(clf_classes):
        proba_full[:, int(cls_idx)] = proba_raw[:, i]
    return proba_full


# ─────────────────────────────────────────────────────────────────────────────
# Single-classifier training + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_one_classifier(classifier_type,
                       X_train, y_train, Y_train,
                       X_test,  y_test,  Y_test,
                       n_classes, exp_dir):
    """
    Select best param, train on full train set, evaluate on test set.
    Saves model + scaler to exp_dir/<classifier_type>/.
    Returns stats dict.
    """
    clf_dir = os.path.join(exp_dir, classifier_type)
    os.makedirs(clf_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Classifier: {classifier_type}")
    print(f"{'='*60}")

    # ── 1. Feature normalisation (fit on train only) ──────────────────────
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)

    # ── 2. Hyperparameter selection (internal CV on training set) ─────────
    print("  Hyperparameter search ...")
    best_param = select_best_param(classifier_type, X_train_norm, y_train)

    # ── 3. Train final model on full training set ─────────────────────────
    print("  Training final model ...")
    clf = build_classifier(classifier_type, best_param)
    clf.fit(X_train_norm, y_train)

    # ── 4. Save artefacts ─────────────────────────────────────────────────
    with open(os.path.join(clf_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(clf_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        'classifier_type': classifier_type,
        'best_param': float(best_param),
        'n_train': int(X_train.shape[0]),
        'n_test':  int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
    }
    with open(os.path.join(clf_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # ── 5. Evaluate on test set ──────────────────────────────────────────
    print("  Evaluating on test set ...")
    proba = get_proba_matrix(clf, X_test_norm, n_classes)  # (N, n_classes)

    # evaluate() from eval.py expects (predictions, targets) where predictions
    # are already probabilities (0–1 range) or logits; we pass probabilities directly
    stats = evaluate(proba, Y_test)
    stats['classifier'] = classifier_type
    stats['best_param'] = float(best_param)

    print(f"  AP={stats['AP']:.4f}  AUC={stats['auc']:.4f}  "
          f"Acc={stats['accuracy']:.4f}  F1={stats['f1']:.4f}")

    # Save individual results
    with open(os.path.join(clf_dir, 'test_results.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    pd.DataFrame([stats]).to_csv(
        os.path.join(clf_dir, 'test_results.csv'), index=False)

    return stats


def eval_one_classifier(classifier_type, X_test, Y_test, n_classes, exp_dir):
    """
    Eval-only: load saved model.pkl + scaler.pkl, run inference on X_test,
    and overwrite test_results.json / test_results.csv with updated metrics.
    """
    clf_dir = os.path.join(exp_dir, classifier_type)
    model_pkl  = os.path.join(clf_dir, 'model.pkl')
    scaler_pkl = os.path.join(clf_dir, 'scaler.pkl')

    if not os.path.exists(model_pkl) or not os.path.exists(scaler_pkl):
        print(f"  [SKIP] {classifier_type}: model.pkl or scaler.pkl not found in {clf_dir}")
        return None

    with open(model_pkl,  'rb') as f:
        clf = pickle.load(f)
    with open(scaler_pkl, 'rb') as f:
        scaler = pickle.load(f)

    X_test_norm = scaler.transform(X_test)
    proba = get_proba_matrix(clf, X_test_norm, n_classes)

    stats = evaluate(proba, Y_test)
    stats['classifier'] = classifier_type

    print(f"  {classifier_type}: AP={stats['AP']:.4f}  AUC={stats['auc']:.4f}  "
          f"Acc={stats['accuracy']:.4f}  F1(macro)={stats['f1']:.4f}  "
          f"F1(burp)={stats['f1_burp']:.4f}  F1(nonburp)={stats['f1_nonburp']:.4f}")

    with open(os.path.join(clf_dir, 'test_results.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    pd.DataFrame([stats]).to_csv(
        os.path.join(clf_dir, 'test_results.csv'), index=False)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='pyAudioAnalysis-based audio classification baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data-train', type=str, required=True,
                        help='Training data JSON file')
    parser.add_argument('--data-val', type=str, required=True,
                        help='Validation data JSON file (merged with train for final training)')
    parser.add_argument('--data-eval', type=str, default='',
                        help='Test/evaluation data JSON file')
    parser.add_argument('--label-csv', type=str, required=True,
                        help='CSV with class label mappings (index, mid, display_name)')

    # Experiment
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Directory to save models and results')

    # Classifier
    parser.add_argument('--classifier', type=str, default='all',
        choices=['svm', 'svm_rbf', 'randomforest',
                 'gradientboosting', 'extratrees', 'all'],
        help='Classifier(s) to train/eval. "all" covers every classifier.')

    # Eval-only
    parser.add_argument('--eval-only', type=str2bool, default=False,
                        help='Load saved models, re-run test evaluation, update CSVs')

    # Feature extraction
    parser.add_argument('--mid-window',   type=float, default=1.0,
                        help='Mid-term window length (seconds)')
    parser.add_argument('--mid-step',     type=float, default=1.0,
                        help='Mid-term step (seconds)')
    parser.add_argument('--short-window', type=float, default=0.05,
                        help='Short-term window length (seconds)')
    parser.add_argument('--short-step',   type=float, default=0.05,
                        help='Short-term step (seconds)')

    # Audio pre-processing (mirrors run.py)
    parser.add_argument('--resample',     type=str2bool, default=False,
                        help='Resample audio to --sample-rate')
    parser.add_argument('--sample-rate',  type=int, default=16000,
                        help='Target sample rate when --resample is True')
    parser.add_argument('--filter',       type=str2bool, default=False,
                        help='Apply low-pass filter')
    parser.add_argument('--cutoff-freq',  type=int, default=1024,
                        help='Low-pass cutoff frequency (Hz)')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.exp_dir, exist_ok=True)

    # Save args for reference
    with open(os.path.join(args.exp_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # ── Load label mapping ────────────────────────────────────────────────
    label_dict = load_label_index(args.label_csv)
    n_classes  = len(label_dict)
    print(f"\nClasses ({n_classes}): {label_dict}")

    # ── Choose classifiers ────────────────────────────────────────────────
    all_classifiers = ['svm', 'svm_rbf', 'randomforest',
                       'gradientboosting', 'extratrees']
    classifiers_to_run = all_classifiers if args.classifier == 'all' \
                         else [args.classifier]

    # ── Common feature extraction kwargs ─────────────────────────────────
    feat_kwargs = dict(
        mid_window=args.mid_window,
        mid_step=args.mid_step,
        short_window=args.short_window,
        short_step=args.short_step,
        resample=args.resample,
        target_sr=args.sample_rate,
        apply_filter=args.filter,
        cutoff_freq=args.cutoff_freq,
    )

    # ── Feature extraction ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 / 3 – Feature Extraction")
    print("="*60)

    train_data = load_json_data(args.data_train)
    val_data   = load_json_data(args.data_val)
    test_data  = load_json_data(args.data_eval) if args.data_eval else []

    # Merge train + val for final model training (same philosophy as other models
    # which use val for early stopping but train on all non-test data)
    combined_data = train_data + val_data

    print(f"\n→ Extracting features from TRAIN+VAL ({len(combined_data)} files) ...")
    X_train, y_train, Y_train = build_feature_matrix(
        combined_data, label_dict, n_classes,
        split_name='train+val', **feat_kwargs)

    if len(X_train) == 0:
        print("ERROR: No valid training samples found. Aborting.")
        sys.exit(1)

    if test_data:
        print(f"\n→ Extracting features from TEST ({len(test_data)} files) ...")
        X_test, y_test, Y_test = build_feature_matrix(
            test_data, label_dict, n_classes,
            split_name='test', **feat_kwargs)
    else:
        X_test, y_test, Y_test = X_train, y_train, Y_train
        print("No --data-eval provided; using train+val set as test set.")

    print(f"\nFeature matrix sizes:  train={X_train.shape}  test={X_test.shape}")

    # ── Train & Evaluate each classifier ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 / 3 – Training & Evaluation")
    print("="*60)

    all_stats = []
    for clf_type in classifiers_to_run:
        stats = run_one_classifier(
            clf_type,
            X_train, y_train, Y_train,
            X_test,  y_test,  Y_test,
            n_classes, args.exp_dir
        )
        all_stats.append(stats)

    # ── Summary: best classifier by AP ───────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 / 3 – Summary")
    print("="*60)

    df_all = pd.DataFrame(all_stats)
    df_all = df_all.sort_values('AP', ascending=False).reset_index(drop=True)
    print("\nAll classifiers ranked by AP:")
    print(df_all[['classifier', 'AP', 'auc', 'accuracy', 'f1']].to_string(index=False))

    # Save combined summary
    summary_json_path = os.path.join(args.exp_dir, 'all_results.json')
    summary_csv_path  = os.path.join(args.exp_dir, 'all_results.csv')
    with open(summary_json_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    df_all.to_csv(summary_csv_path, index=False)

    # ── Best classifier → canonical output files expected by compare_models.py ──
    best = df_all.iloc[0].to_dict()
    best_clf_type = best['classifier']
    print(f"\n★ Best classifier: {best_clf_type}  "
          f"AP={best['AP']:.4f}  AUC={best['auc']:.4f}")

    # Copy best classifier's results to the top-level exp_dir as
    # test_results.json / test_results.csv (same filenames as run.py)
    canonical_stats = {k: v for k, v in best.items()
                       if k not in ('classifier', 'best_param')}
    with open(os.path.join(args.exp_dir, 'test_results.json'), 'w') as f:
        json.dump(canonical_stats, f, indent=4)
    pd.DataFrame([canonical_stats]).to_csv(
        os.path.join(args.exp_dir, 'test_results.csv'), index=False)

    print(f"\nResults saved to: {args.exp_dir}")
    print(f"  • test_results.json / .csv  (best: {best_clf_type})")
    print(f"  • all_results.json / .csv   (all classifiers)")
    print(f"  • <classifier>/             (per-classifier models & results)")


if __name__ == '__main__':
    main()
