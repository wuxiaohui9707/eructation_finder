"""
inference_pyaudio.py
--------------------
Sliding-window ensemble inference for pyAudioAnalysis traditional-ML models
(SVM, SVM-RBF, Random Forest, Gradient Boosting, Extra Trees).

Each fold's trained sklearn classifier (.pkl) and feature scaler (.pkl) are
loaded from the experiment directories, predictions are averaged across folds,
and the result is post-processed (smoothing → thresholding → weighted NMS).

Usage example:
  python inference_pyaudio.py \\
      --classifier svm_rbf \\
      --ensemble_dir "/path/to/experiments/fold_*" \\
      --audio_input  "/path/to/audio_dir" \\
      --label_csv    "/path/to/label_index.csv" \\
      --exp_dir      "/path/to/output_dir"
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── pyAudioAnalysis imports ───────────────────────────────────────────────────
_PYAUDIO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pyAudioAnalysis"
)
if _PYAUDIO_ROOT not in sys.path:
    sys.path.insert(0, _PYAUDIO_ROOT)

try:
    from pyAudioAnalysis import MidTermFeatures as aF
    from pyAudioAnalysis import audioBasicIO
except ImportError as e:
    raise ImportError(
        f"Cannot import pyAudioAnalysis from {_PYAUDIO_ROOT}. "
        "Make sure the library is cloned/installed.\nOriginal error: {e}"
    )

# ── Optional scipy for filtering ─────────────────────────────────────────────
try:
    from scipy.signal import resample_poly, butter, sosfiltfilt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ── torchaudio (primary audio loader – handles DVI_ADPCM and other formats) ───
try:
    import torch
    import torchaudio
    TORCHAUDIO_OK = True
except ImportError:
    TORCHAUDIO_OK = False

start_time = time.time()


# ---------------------------------------------------------------------------
# Feature extraction helpers (must match run_pyaudio.py training settings)
# ---------------------------------------------------------------------------

# Configuration defaults (these will be overridden by CLI args during extraction)
MID_WINDOW  = 1.0    # seconds
MID_STEP    = 1.0    # seconds
SHORT_WIN   = 0.05   # seconds
SHORT_STEP  = 0.05   # seconds

def load_audio(audio_path, resample, target_sr, apply_filter, cutoff_freq=1024):
    """
    Load a WAV file as a mono float64 numpy array.

    Uses torchaudio as primary backend (handles DVI_ADPCM and other
    compressed formats that scipy.io.wavfile cannot parse).
    Falls back to pyAudioAnalysis/audioBasicIO if torchaudio is unavailable.

    Args:
        audio_path  : path to WAV file
        resample    : bool – resample to target_sr
        target_sr   : target sample rate
        apply_filter: bool – apply low-pass Butterworth filter
        cutoff_freq : cutoff frequency in Hz for the low-pass filter

    Returns:
        (signal, sample_rate) where signal is 1-D float64 numpy array
    """
    if TORCHAUDIO_OK:
        waveform, sr = torchaudio.load(audio_path)        # (C, T)
        waveform = waveform.mean(dim=0)                    # mono
        signal = waveform.numpy().astype(np.float64)
    else:
        sr, signal = audioBasicIO.read_audio_file(audio_path)
        signal = audioBasicIO.stereo_to_mono(signal).astype(np.float64)

    # Resample
    if resample and sr != target_sr:
        if TORCHAUDIO_OK:
            t = torch.from_numpy(signal).float().unsqueeze(0)
            t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=target_sr)
            signal = t.squeeze(0).numpy().astype(np.float64)
        elif SCIPY_OK:
            from math import gcd
            g = gcd(target_sr, sr)
            signal = resample_poly(signal, target_sr // g, sr // g)
        sr = target_sr

    # Low-pass filter
    if apply_filter and SCIPY_OK:
        nyq = sr / 2.0
        if cutoff_freq < nyq:
            sos = butter(5, cutoff_freq / nyq, btype='low', output='sos')
            signal = sosfiltfilt(sos, signal)

    return signal.astype(np.float64), sr



def extract_features(signal, sr, mid_window=1.0, mid_step=1.0, short_window=0.05, short_step=0.05):
    """
    Extract a single fixed-length feature vector for the given signal.

    Uses mid-term feature extraction (mean + std over short-term windows)

    Returns:
        1-D numpy array of shape (2 * n_short_features,)
    """
    n_mid   = int(mid_window  * sr)
    n_step  = int(mid_step    * sr)
    n_short = int(short_window   * sr)
    n_sstep = int(short_step  * sr)

    try:
        mid_feats, _, _ = aF.mid_feature_extraction(
            signal, sr, n_mid, n_step, n_short, n_sstep
        )
        if mid_feats.ndim == 1:
            mid_feats = mid_feats.reshape(-1, 1)

        # mid_feats: (n_features, n_windows)
        feat = np.concatenate([
            np.mean(mid_feats, axis=1),
            np.std(mid_feats, axis=1)
        ])
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")

    # Guard against NaN / Inf (happens with absolute silence)
    if np.isnan(feat).any() or np.isinf(feat).any():
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat.astype(np.float64)


# ---------------------------------------------------------------------------
# Ensemble loading
# ---------------------------------------------------------------------------

def load_ensemble(ensemble_dir_pattern, classifier):
    """
    Find all fold directories matching ensemble_dir_pattern and load the
    sklearn model + scaler for the given classifier.

    Returns:
        (models, scalers, n_loaded)
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


# ---------------------------------------------------------------------------
# Per-window prediction (ensemble)
# ---------------------------------------------------------------------------

def predict_window(signal_window, sr, models, scalers, 
                   mid_window=1.0, mid_step=1.0, short_window=0.05, short_step=0.05):
    """
    Extract features from signal_window and return the averaged positive-class
    probability across all ensemble models.

    Returns:
        float – mean predicted probability of the positive class (index 0)
    """
    feat = extract_features(signal_window, sr, mid_window, mid_step, short_window, short_step)
    probs = []
    for model, scaler in zip(models, scalers):
        x = scaler.transform(feat.reshape(1, -1))
        # predict_proba returns shape (1, n_classes); col 0 = positive class
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0, 0]
        else:
            # decision_function fallback (e.g. linear SVM without proba)
            d = model.decision_function(x)[0]
            if isinstance(d, np.ndarray) and d.size > 1:
                d = d[0]
            p = 1.0 / (1.0 + np.exp(-d))
        probs.append(p)
    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Post-processing (verbatim copy from inference_unseen_audio.py)
# ---------------------------------------------------------------------------

def weighted_nms(events):
    merged = []
    for event in events:
        if not merged:
            merged.append(event)
        else:
            last = merged[-1]
            overlap = min(last['end'], event['end']) - max(last['start'], event['start'])
            if overlap > 0:
                total_weight = last['prob'] + event['prob']
                new_start = (last['start'] * last['prob'] + event['start'] * event['prob']) / total_weight
                new_end   = (last['end']   * last['prob'] + event['end']   * event['prob']) / total_weight
                merged[-1] = {
                    'start': new_start,
                    'end':   new_end,
                    'prob':  max(last['prob'], event['prob'])
                }
            else:
                merged.append(event)
    return merged


def process_csv(csv_path, output_dir, threshold=0.5, smooth_window=3, label='burp'):
    df = pd.read_csv(csv_path)
    predictions = df[[  'start_time', 'end_time', label]].values

    total_duration = int(np.ceil(predictions[-1, 1]))
    time_prob = np.zeros(total_duration)
    weight    = np.zeros(total_duration)

    for start, end, prob in predictions:
        s, e = int(np.floor(start)), int(np.ceil(end))
        for t in range(s, e):
            if t < total_duration:
                time_prob[t] += prob
                weight[t]    += 1

    time_prob = np.divide(time_prob, weight,
                          out=np.zeros_like(time_prob), where=weight != 0)
    smoothed  = np.convolve(time_prob,
                            np.ones(smooth_window) / smooth_window,
                            mode='same')

    events, current = [], None
    for t in range(len(smoothed)):
        if smoothed[t] >= threshold:
            if current is None:
                current = {'start': t, 'end': t, 'prob': smoothed[t]}
            else:
                current['end']  = t
                current['prob'] = max(current['prob'], smoothed[t])
        else:
            if current is not None:
                events.append(current)
                current = None
    if current is not None:
        events.append(current)

    merged = weighted_nms(events)
    final_events = [
        {'start': e['start'], 'end': e['end'] + 1, 'prob': round(e['prob'], 4)}
        for e in merged
    ]

    # Visualise
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    plt.figure(figsize=(15, 5))
    plt.plot(smoothed, label='Smoothed Probability')
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    labeled = False
    for ev in final_events:
        plt.axvspan(ev['start'], ev['end'], alpha=0.3, color='green',
                    label='Burp Event' if not labeled else '')
        labeled = True
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.title('Burp Event Detection — pyAudioAnalysis')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

    return final_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sliding-window ensemble inference using pyAudioAnalysis sklearn models."
    )

    # ── Default paths ─────────────────────────────────────────────────────────
    _base = ""
    _ens  = ""
    _out  = ""

    parser.add_argument('--classifier',   type=str, default='gradientboosting',
                        choices=['svm', 'svm_rbf', 'randomforest', 'gradientboosting', 'extratrees'],
                        help='Which trained classifier to use for inference')
    parser.add_argument('--ensemble_dir', type=str, required=True,
                        help='Glob pattern for fold directories (e.g. .../fold_*)')
    parser.add_argument('--audio_input',  type=str, required=True,
                        help='Path to a .WAV file OR a folder of .WAV files')
    parser.add_argument('--recursive',    action='store_true', default=False,
                        help='Search sub-folders recursively for .WAV files')
    parser.add_argument('--label_csv',    type=str, required=True,
                        help='label_index.csv (index → display_name)')
    parser.add_argument('--exp_dir',      type=str, required=True,
                        help='Output directory for predictions and plots')

    # ── Sliding window ────────────────────────────────────────────────────────
    parser.add_argument('--window_size',  type=float, default=5.0,
                        help='Sliding window size in seconds')
    parser.add_argument('--step_size',    type=float, default=0.5,
                        help='Step between windows in seconds')

    # ── Audio preprocessing (MUST match training!) ────────────────────────────
    parser.add_argument('--resample',     type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Resample audio to --target_sr (True/False)')
    parser.add_argument('--target_sr',    type=int, default=16000,
                        help='Target sample rate when resampling')
    parser.add_argument('--filter',       type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Apply 1024 Hz low-pass filter (True/False)')
    parser.add_argument('--cutoff_freq',  type=int, default=1024,
                        help='Low-pass filter cutoff frequency in Hz')
                        
    # ── pyAudioAnalysis feature config ────────────────────────────────────────
    parser.add_argument('--mid_window',   type=float, default=1.0, help='Mid-term window (s)')
    parser.add_argument('--mid_step',     type=float, default=1.0, help='Mid-term step (s)')
    parser.add_argument('--short_window', type=float, default=0.05, help='Short-term window (s)')
    parser.add_argument('--short_step',   type=float, default=0.05, help='Short-term step (s)')

    # ── Post-processing ───────────────────────────────────────────────────────
    parser.add_argument('--threshold',    type=float, default=0.5,
                        help='Probability threshold for event detection')
    parser.add_argument('--smooth',       type=int,   default=3,
                        help='Smoothing window size (seconds)')

    args = parser.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)

    # ── Label mapping ─────────────────────────────────────────────────────────
    label_dict = {}
    with open(args.label_csv) as f:
        for row in csv.DictReader(f):
            idx  = int(row['index'])
            name = row.get('display_name', row.get('mid', str(idx)))
            label_dict[idx] = name
    target_label = label_dict.get(0, 'burp')
    print(f"Target label: '{target_label}' (index 0)")

    # ── Collect audio files ───────────────────────────────────────────────────
    audio_input = args.audio_input
    if os.path.isfile(audio_input):
        audio_files = [audio_input]
    elif os.path.isdir(audio_input):
        pat = os.path.join(audio_input, '**', '*.[Ww][Aa][Vv]') if args.recursive \
              else os.path.join(audio_input, '*.[Ww][Aa][Vv]')
        audio_files = sorted(glob.glob(pat, recursive=args.recursive))
        if not audio_files:
            raise FileNotFoundError(f"No .WAV files found in: {audio_input}")
        print(f"Found {len(audio_files)} audio file(s) in: {audio_input}")
    else:
        raise FileNotFoundError(f"--audio_input not found: {audio_input}")

    # ── Load ensemble ─────────────────────────────────────────────────────────
    models, scalers = load_ensemble(args.ensemble_dir, args.classifier)

    # ── Process each file ─────────────────────────────────────────────────────
    for file_idx, audio_file in enumerate(audio_files, start=1):
        t0 = time.time()
        print(f"\n[{file_idx}/{len(audio_files)}] Processing: {audio_file}")

        signal, sr = load_audio(
            audio_file,
            resample=args.resample,
            target_sr=args.target_sr,
            apply_filter=args.filter,
            cutoff_freq=args.cutoff_freq
        )

        win_samples  = int(args.window_size * sr)
        step_samples = int(args.step_size   * sr)
        total_samples = len(signal)

        results = []
        for start in range(0, total_samples - win_samples + 1, step_samples):
            end    = start + win_samples
            window = signal[start:end]
            try:
                prob = predict_window(window, sr, models, scalers,
                                      mid_window=args.mid_window,
                                      mid_step=args.mid_step,
                                      short_window=args.short_window,
                                      short_step=args.short_step)
            except Exception as e:
                print(f"  [warn] window {start}–{end}: {e}")
                continue

            results.append({
                'start_time': round(start / sr, 2),
                'end_time':   round(end   / sr, 2),
                target_label: round(prob, 4),
            })

        if not results:
            print("  [warn] No windows produced results – skipping file.")
            continue

        # ── Write per-window CSV ──────────────────────────────────────────────
        audio_name = os.path.splitext(os.path.basename(audio_file))[0]
        csv_path   = os.path.join(args.exp_dir, f"{audio_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['start_time', 'end_time', target_label])
            writer.writeheader()
            writer.writerows(results)
        print(f"  Per-window CSV : {csv_path}  ({len(results)} windows)")

        # ── Post-process → NMS → visualise ───────────────────────────────────
        final_events = process_csv(
            csv_path, args.exp_dir,
            threshold=args.threshold,
            smooth_window=args.smooth,
            label=target_label
        )
        summary_path = os.path.join(
            args.exp_dir,
            f"Summary_inference_pyaudio_{args.classifier}_{audio_name}.csv"
        )
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Start', 'End', 'Probability'])
            writer.writeheader()
            for ev in final_events:
                writer.writerow({
                    'Start': int(ev['start']),
                    'End':   int(ev['end']),
                    'Probability': float(ev['prob'])
                })
        print(f"  Event summary  : {summary_path}  ({len(final_events)} event(s))")
        print(f"  File time      : {time.time() - t0:.1f}s")

    print(f"\n=== All done — {len(audio_files)} file(s) in "
          f"{time.time() - start_time:.1f}s total ===")


if __name__ == '__main__':
    main()
