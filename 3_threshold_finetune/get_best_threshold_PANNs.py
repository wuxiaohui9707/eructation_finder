import argparse
import glob
import json
import os
import csv
import sys
import pickle
import numpy as np
import torch
import torchaudio
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

# ────────────────────────────────────────────────────────────────────────────
# Import unified pipeline helpers.
# The script is expected to be run from the 'unified/' directory,
# or that directory must be on PYTHONPATH.
# ────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from core.data_module import resample_audio, low_pass_filter
from core.models.models import get_model_class
from model_training.train_PANNs import PANNsLightningModule


# ────────────────────────────────────────────────────────────────────────────
# Label helpers
# ────────────────────────────────────────────────────────────────────────────

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

def load_ensemble_models(ensemble_dir_pattern, classes_num, device):
    """
    Load all PANNs fold checkpoints matching *ensemble_dir_pattern*.

    Checkpoints are saved by PyTorch Lightning as UnifiedLightningModule.
    We unwrap the inner Cnn14 model and return a list of eval-mode models.
    """
    ckpt_paths = glob.glob(
        os.path.join(ensemble_dir_pattern, "checkpoints", "best_model*.ckpt")
    )
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoint files found matching: "
            f"{os.path.join(ensemble_dir_pattern, 'checkpoints', 'best_model*.ckpt')}"
        )

    # Keep only the newest checkpoint file
    latest_ckpt = max(ckpt_paths, key=os.path.getmtime)
    ckpt_paths = [latest_ckpt]

    models = []
    model_mel_bins = None # To store the detected mel_bins from the first model
    for ckpt_path in sorted(ckpt_paths):
        print(f"[PANNs] Loading checkpoint: {ckpt_path}")

        # Load the args stored alongside the checkpoint (args.pkl lives next to checkpoints/)
        # ── Step 1: peek at raw state_dict to get ground-truth shapes ──────
        raw_ckpt   = torch.load(ckpt_path, map_location="cpu")
        state_dict = raw_ckpt.get("state_dict", {})

        # Infer mel_bins from bn0 (always present in Cnn14)
        if "model.bn0.weight" in state_dict:
            actual_mel_bins = state_dict["model.bn0.weight"].shape[0]
        else:
            actual_mel_bins = classes_num  # shouldn't happen, but safe fallback

        # Infer classes_num from the final linear layer
        if "model.fc_audioset.weight" in state_dict:
            actual_classes = state_dict["model.fc_audioset.weight"].shape[0]
        else:
            actual_classes = classes_num

        print(f"  detected  mel_bins={actual_mel_bins}  classes_num={actual_classes}")

        # ── Step 2: build args with ground-truth values ─────────────────────
        fold_dir = os.path.dirname(os.path.dirname(ckpt_path))  # fold_* dir
        args_pkl  = os.path.join(fold_dir, "args.pkl")

        if os.path.exists(args_pkl):
            with open(args_pkl, "rb") as f:
                saved_args = pickle.load(f)
            # Patch with checkpoint-inferred values to guarantee consistency
            saved_args.mel_bins  = actual_mel_bins
            saved_args.label_dim = actual_classes
            # Skip loading pretrained weights — they're already baked into the ckpt
            saved_args.panns_pretrained_path = None
        else:
            import argparse as _ap
            saved_args = _ap.Namespace(
                model="panns",
                label_dim=actual_classes,
                mel_bins=actual_mel_bins,
                panns_pretrained_path=None,
                lr=1e-3,
                lrscheduler_step=1,
                lrscheduler_decay=0.5,
                warmup_epochs=0,
                lrscheduler_start=0,
            )

        # ── Step 3: load the full Lightning checkpoint ──────────────────────
        model_class = get_model_class(saved_args.model)
        pl_module = PANNsLightningModule.load_from_checkpoint(
            ckpt_path,
            model_class=model_class,
            args=saved_args,
            map_location="cpu",
        )

        inner_model = pl_module.model.to(device).eval()
        models.append(inner_model)

    print(f"[PANNs] Loaded {len(models)} model(s).")
    return models


# ────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ────────────────────────────────────────────────────────────────────────────

def make_features(waveform, sr, mel_bins=64, target_length=1000,
                  norm_mean=-4.2677393, norm_std=4.5689974):
    """
    Generate a normalised log-mel spectrogram matching the training pipeline.

    Returns a tensor of shape [1, Time, mel_bins] ready for PANNs
    (UnifiedLightningModule.forward adds the channel dim, but we add it here
    so we can skip wrapping around the inner Cnn14 directly).
    """
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10.0,   # 10 ms
        frame_length=25.0,  # 25 ms
    )  # [Time, mel_bins]

    n_frames = fbank.shape[0]
    if n_frames < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - n_frames))
    else:
        fbank = fbank[:target_length, :]

    # Normalise (same as training)
    fbank = (fbank - norm_mean) / (norm_std * 2)

    # Add batch and channel dimensions: [1, 1, Time, mel_bins]
    return fbank.unsqueeze(0).unsqueeze(0)


# ────────────────────────────────────────────────────────────────────────────
# Dataset processing
# ────────────────────────────────────────────────────────────────────────────

def process_dataset(json_files, models, label_mappings,
                    resample=False, apply_filter=False,
                    target_sample_rate=16000, mel_bins=64, target_length=1000,
                    norm_mean=-4.2677393, norm_std=4.5689974,
                    device=torch.device('cpu')):
    """
    Run ensemble inference over all items in the given JSON files.

    Returns a list of dicts  {'target': int,  <pos_class_mid>: float_prob}
    """
    mid_to_index = label_mappings['mid_to_index']
    index_to_mid = label_mappings['index_to_mid']
    pos_class_mid = index_to_mid[0]           # positive class (index 0)

    records = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f).get('data', [])

        for item in data:
            try:
                # 1. Load & resample
                waveform, sr = resample_audio(
                    item['wav'],
                    target_sample_rate=target_sample_rate,
                    resample=resample,
                )

                # 2. Low-pass filter
                if apply_filter:
                    waveform_np = waveform.numpy()
                    if waveform_np.ndim > 1:
                        waveform_np = waveform_np.squeeze()
                    waveform_np = low_pass_filter(waveform_np, sr).copy()
                    waveform = torch.from_numpy(waveform_np)
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)

                # 3. Mean-centre (same as training)
                waveform = waveform - waveform.mean()

                # 4. Make log-mel spectrogram [1, 1, Time, mel_bins]
                mel = make_features(
                    waveform, sr,
                    mel_bins=mel_bins,
                    target_length=target_length,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                ).to(device, dtype=torch.float32)

                # 5. Ensemble inference
                probs = []
                for model in models:
                    with torch.no_grad():
                        logits = model(mel)   # [1, classes_num]

                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)

                    # CRITICAL BUG FIX: The models were trained using nn.BCEWithLogitsLoss()
                    # Therefore we must strictly use torch.sigmoid to get the true probability
                    if logits.shape[1] == 1:
                        prob = torch.sigmoid(logits.squeeze()).cpu().item()
                    elif logits.shape[1] >= 2:
                        prob = torch.sigmoid(logits)[:, 0].squeeze().cpu().item()
                    else:
                        raise ValueError(f"Unexpected output dim: {logits.shape[1]}")

                    probs.append(prob)

                avg_prob = float(np.mean(probs))
                records.append({
                    'target': mid_to_index[item['labels']],
                    pos_class_mid: avg_prob,
                })

            except Exception as e:
                print(f"[warn] Skipped {item.get('wav', '?')}: {e}")
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
    parser = argparse.ArgumentParser(description="Find best classification threshold for PANNs ensemble")

    parser.add_argument("--ensemble_dir", type=str, required=True,
                        help="Glob pattern for fold dirs (each must have checkpoints/best_model*.ckpt)")
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

    # Audio preprocessing options (must match training config)
    parser.add_argument("--resample", type=str2bool, default=True,
                        help="Resample audio to --sample_rate")
    parser.add_argument("--sample_rate", type=int, default=32000,
                        help="Target sample rate (used when --resample is set)")
    parser.add_argument("--filter", type=str2bool, default=False,
                        help="Apply low-pass filter")

    # PANNs feature config (must match training config)
    parser.add_argument("--mel_bins", type=int, default=64,
                        help="Number of mel bins (64 for official pretrained Cnn14)")
    parser.add_argument("--target_length", type=int, default=500,
                        help="Target spectrogram time frames")
    parser.add_argument("--norm_mean", type=float, default=-4.2677393,
                        help="Spectrogram normalisation mean")
    parser.add_argument("--norm_std", type=float, default=4.5689974,
                        help="Spectrogram normalisation std")
    parser.add_argument("--classes_num", type=int, default=2,
                        help="Number of output classes (used as fallback if args.pkl missing)")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

            fold_dir = fold_model_dirs[fold_i] if fold_i < len(fold_model_dirs) else None
            if fold_dir is None:
                print(f"  WARNING: no model directory for fold {fold_i}, skipping.")
                continue
            fold_models = load_ensemble_models(fold_dir, classes_num=args.classes_num, device=device)
            if not fold_models:
                print(f"  WARNING: no checkpoint found in {fold_dir}, skipping.")
                continue

            val_json = os.path.join(args.json_dir, f"val_fold{fold_i}.json")
            if not os.path.isfile(val_json):
                print(f"  WARNING: {val_json} not found, skipping fold {fold_i}.")
                continue

            fold_output_csv = args.output_csv.replace(".csv", f"_fold{fold_i}.csv")

            infer_kwargs = dict(
                models=fold_models,
                label_mappings=label_mappings,
                resample=args.resample,
                apply_filter=args.filter,
                target_sample_rate=args.sample_rate,
                mel_bins=args.mel_bins,
                target_length=args.target_length,
                norm_mean=args.norm_mean,
                norm_std=args.norm_std,
                device=device,
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

        all_model_dirs = sorted(glob.glob(args.ensemble_dir))
        if not all_model_dirs:
            raise FileNotFoundError(f"No fold directories found for pattern: {args.ensemble_dir}")
        
        merged_models = []
        for fold_dir in all_model_dirs:
            models_in_fold = load_ensemble_models(fold_dir, classes_num=args.classes_num, device=device)
            merged_models.extend(models_in_fold)
        
        if not merged_models:
            raise RuntimeError("No models loaded for merged ensemble.")
        print(f"Loaded {len(merged_models)} models for the merged ensemble.")

        val_json_files = [os.path.join(args.json_dir, f"val_fold{i}.json") for i in range(args.n_folds)]
        val_json_files = [f for f in val_json_files if os.path.isfile(f)]
        if not val_json_files:
            raise FileNotFoundError(f"No validation JSON files found in {args.json_dir}")
        
        infer_kwargs = dict(
            models=merged_models,
            label_mappings=label_mappings,
            resample=args.resample,
            apply_filter=args.filter,
            target_sample_rate=args.sample_rate,
            mel_bins=args.mel_bins,
            target_length=args.target_length,
            norm_mean=args.norm_mean,
            norm_std=args.norm_std,
            device=device,
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