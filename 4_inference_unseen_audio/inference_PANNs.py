"""
inference_unseen_audio.py
--------------------------
Sliding-window ensemble inference for PANNs (Cnn14) model.

Loads trained PANNs checkpoints from multiple folds, runs sliding-window
inference on continuous audio recordings, and post-processes predictions
with smoothing + thresholding + weighted NMS.

Usage example:
  python inference_unseen_audio.py \
    --model panns \
    --num_classes 2 \
    --ensemble_dir "/path/to/panns_experiment/fold_*" \
    --audio_input "/path/to/audio.wav" \
    --label_csv "/path/to/label_index.csv" \
    --exp_dir "output_dir" \
    --resample True \
    --filter True
"""

import argparse
import os
import csv
import glob
import numpy as np
import torch
import torchaudio
import time
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Local imports — all from the unified pipeline
from core.data_module import resample_audio, low_pass_filter
from core.models.models import get_model_class

start_time = time.time()


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def make_features(waveform, sr, mel_bins=64, target_length=500,
                  norm_mean=-4.2677393, norm_std=4.5689974):
    """
    Compute a log-mel spectrogram from a waveform tensor and normalise it.

    Args:
        waveform      : torch.Tensor [1, num_samples]
        sr            : sample rate
        mel_bins      : number of mel filter banks (must match training config)
        target_length : fixed time-frame length (pad or trim)
        norm_mean     : dataset normalisation mean (from training)
        norm_std      : dataset normalisation std  (from training)

    Returns:
        fbank : torch.Tensor [target_length, mel_bins]  (float32)
    """
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10.0,   # 10 ms hop
        frame_length=25.0,  # 25 ms window
    )

    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p), mode='constant')
    elif p < 0:
        fbank = fbank[:target_length, :]

    # Normalise the same way as during training
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank


# ---------------------------------------------------------------------------
# Sliding-window audio processor
# ---------------------------------------------------------------------------

def process_audio_file(audio_file, window_size, step_size,
                       resample, apply_filter,
                       mel_bins=64, target_length=500,
                       target_sr=16000,
                       norm_mean=-4.2677393, norm_std=4.5689974):
    """
    Generator: slide a window over the audio file and yield mel-spectrograms.

    Yields:
        (input_tensor, start_sample, end_sample)
        input_tensor : torch.Tensor [1, 1, target_length, mel_bins]  on CUDA
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load & optionally resample
    if resample:
        waveform, sr = resample_audio(audio_file, target_sample_rate=target_sr, resample=True)
    else:
        waveform, sr = torchaudio.load(audio_file)

    # Optionally apply low-pass filter
    if apply_filter:
        waveform_np = waveform.numpy()
        waveform_np = low_pass_filter(waveform_np, sr).copy()
        waveform = torch.from_numpy(waveform_np)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    window_samples = int(window_size * sr)
    step_samples   = int(step_size   * sr)

    for start in range(0, waveform.shape[1] - window_samples + 1, step_samples):
        end = start + window_samples
        window_waveform = waveform[:, start:end]

        fbank = make_features(window_waveform, sr,
                              mel_bins=mel_bins,
                              target_length=target_length,
                              norm_mean=norm_mean,
                              norm_std=norm_std)

        # Shape: [target_length, mel_bins] → [1, 1, target_length, mel_bins]
        input_tensor = fbank.unsqueeze(0).unsqueeze(0).float().to(device)

        yield input_tensor, start, end, sr


# ---------------------------------------------------------------------------
# Post-processing helpers (unchanged from original)
# ---------------------------------------------------------------------------

def weighted_nms(events):
    """
    Merge overlapping events by weighting start/end by probability.
    All input events must have a 'prob' key.
    """
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


def process_csv(csv_path, output_dir, threshold=0.5, smooth_window=3):
    """
    Build a time-probability curve from per-window predictions,
    apply smoothing + thresholding, run NMS, and save a visualisation.
    """
    df = pd.read_csv(csv_path)
    predictions = df[['start_time', 'end_time', 'burp']].values

    total_duration = int(np.ceil(predictions[-1, 1]))
    time_prob = np.zeros(total_duration)
    weight    = np.zeros(total_duration)

    for start, end, prob in predictions:
        start_sec = int(np.floor(start))
        end_sec   = int(np.ceil(end))
        for t in range(start_sec, end_sec):
            if t < total_duration:
                time_prob[t] += prob
                weight[t]    += 1

    time_prob = np.divide(time_prob, weight,
                          out=np.zeros_like(time_prob), where=weight != 0)
    smoothed  = np.convolve(time_prob,
                            np.ones(smooth_window) / smooth_window,
                            mode='same')

    events = []
    current_event = None
    for t in range(len(smoothed)):
        if smoothed[t] >= threshold:
            if current_event is None:
                # Use 'prob' key consistently so weighted_nms can always access it
                current_event = {'start': t, 'end': t, 'prob': smoothed[t]}
            else:
                current_event['end']  = t
                current_event['prob'] = max(current_event['prob'], smoothed[t])
        else:
            if current_event is not None:
                events.append(current_event)
                current_event = None
    if current_event is not None:
        events.append(current_event)

    merged = weighted_nms(events)
    final_events = [
        {'start': e['start'],
         'end':   e['end'] + 1,
         'prob':  round(e['prob'], 4)}
        for e in merged
    ]

    # Visualise
    filename = os.path.basename(csv_path).split('.')[0]
    plt.figure(figsize=(15, 5))
    plt.plot(smoothed, label='Smoothed Probability')
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    already_labeled = False
    for event in final_events:
        label_str = 'Burp Event' if not already_labeled else ''
        plt.axvspan(event['start'], event['end'], alpha=0.3, color='green', label=label_str)
        already_labeled = True
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.title('Burp Event Detection Timeline')
    plt.legend()
    out_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Visualisation saved to {out_path}")

    return final_events


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ensemble_models(ensemble_dir_pattern, model_name, num_classes,
                         pretrained=False, in_channels=1):
    """
    1. Find all best_model.ckpt files matching ensemble_dir_pattern.
    2. For each checkpoint, instantiate the correct model class via get_model_class().
    3. Load the state_dict (handles the 'model.' prefix from PyTorch Lightning).
    4. Return a list of models on CUDA/CPU in eval mode.

    Args:
        ensemble_dir_pattern : glob pattern ending with the fold directory,
                               e.g. ".../resnet_filter_True/mccv_fold_*"
        model_name           : one of the supported names in models/__init__.py
                               e.g. 'resnet', 'mobilenet', 'mobilenet_fa_enhanced', 'panns'
        num_classes          : number of output classes (must match training)
        pretrained           : whether the architecture uses pretrained backbone
                               (usually False when loading a trained checkpoint)
        in_channels          : 1 or 3  (for CNN models)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Lightning auto-versions checkpoints as best_model-v1.ckpt, best_model-v2.ckpt, etc.
    # For each fold directory, pick the NEWEST best_model*.ckpt by modification time.
    fold_dirs = sorted(glob.glob(ensemble_dir_pattern))
    ckpt_paths = []
    for fold_dir in fold_dirs:
        candidates = glob.glob(os.path.join(fold_dir, "checkpoints", "best_model*.ckpt"))
        if candidates:
            newest = max(candidates, key=os.path.getmtime)
            ckpt_paths.append(newest)

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found matching: "
            f"{os.path.join(ensemble_dir_pattern, 'checkpoints', 'best_model*.ckpt')}"
        )

    model_class = get_model_class(model_name)

    ensemble = []
    for ckpt_path in ckpt_paths:
        print(f"Loading model: {ckpt_path}")

        # Load checkpoint first so we can inspect the state_dict
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if 'state_dict' in checkpoint:
            # PyTorch Lightning wraps the model as self.model → strip "model." prefix
            raw_sd = checkpoint['state_dict']
            new_sd = {k[len('model.'):] if k.startswith('model.') else k: v
                      for k, v in raw_sd.items()}
        else:
            new_sd = checkpoint

        # Instantiate model using the correct architecture args
        model_lower = model_name.lower()
        if model_lower == 'panns':
            # Auto-detect mel_bins from checkpoint (handles both 64 and 128 variants)
            ckpt_mel_bins = new_sd['bn0.weight'].shape[0]
            print(f"  [PANNs] detected mel_bins={ckpt_mel_bins} from checkpoint")
            model = model_class(classes_num=num_classes, mel_bins=ckpt_mel_bins)
        else:
            model = model_class(
                num_classes=num_classes,
                pretrained=pretrained,
                in_channels=in_channels,
            )

        model.load_state_dict(new_sd)
        model.to(device)
        model.eval()
        ensemble.append(model)


    return ensemble, device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sliding-window ensemble inference for unified CNN audio models."
    )

    # Model selection
    parser.add_argument('--model', type=str, default='panns',
                        help='Model architecture (must match training config)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (must match training config)')
    parser.add_argument('--in_channels', type=int, default=1, choices=[1, 3],
                        help='Input channels for CNN models (1 = mono mel-spec)')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Set True only if loading untrained backbone weights')

    # Data / paths
    parser.add_argument('--ensemble_dir', type=str, required=True,
                        help='Glob pattern for fold directories containing checkpoints/best_model.ckpt')
    parser.add_argument('--audio_input', type=str, required=True,
                        help='Path to a single .WAV file OR a folder containing .WAV/.wav files')
    parser.add_argument('--recursive', action='store_true', default=False,
                        help='If --audio_input is a folder, search sub-folders recursively')
    parser.add_argument('--label_csv', type=str, required=True,
                        help='CSV with label index mapping (columns: index, display_name or mid)')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Output directory for predictions and plots')

    # Sliding window
    parser.add_argument('--window_size', type=float, default=5.0,
                        help='Window size in seconds')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size in seconds between windows')

    # Audio preprocessing (must match training settings!)
    parser.add_argument('--resample', type=bool, default=True,
                        help='Resample audio to --target_sr before inference')
    parser.add_argument('--target_sr', type=int, default=32000,
                        help='Target sample rate when resampling')
    parser.add_argument('--filter', type=bool, default=True,
                        help='Apply low-pass filter before inference')
    parser.add_argument('--mel_bins', type=int, default=64,
                        help='Number of mel filter banks (must match training)')
    parser.add_argument('--target_length', type=int, default=500,
                        help='Fixed number of time frames (must match training)')
    parser.add_argument('--norm_mean', type=float, default=-4.2677393,
                        help='Spectrogram normalisation mean (from training dataset stats)')
    parser.add_argument('--norm_std', type=float, default=4.5689974,
                        help='Spectrogram normalisation std  (from training dataset stats)')

    # Post-processing
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Probability threshold for event detection')
    parser.add_argument('--smooth', type=int, default=3,
                        help='Smoothing window size (seconds)')

    args = parser.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load label mapping
    # ------------------------------------------------------------------
    label_dict = {}
    with open(args.label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx  = int(row['index'])
            name = row.get('display_name', row.get('mid', str(idx)))
            label_dict[idx] = name

    target_label = label_dict.get(0, 'burp')
    print(f"Target label: '{target_label}' (index 0)")

    # ------------------------------------------------------------------
    # 2. Collect audio files
    # ------------------------------------------------------------------
    audio_input = args.audio_input
    if os.path.isfile(audio_input):
        audio_files = [audio_input]
    elif os.path.isdir(audio_input):
        pattern = os.path.join(audio_input, '**', '*.[Ww][Aa][Vv]') if args.recursive \
                  else os.path.join(audio_input, '*.[Ww][Aa][Vv]')
        audio_files = sorted(glob.glob(pattern, recursive=args.recursive))
        if not audio_files:
            raise FileNotFoundError(
                f"No .WAV/.wav files found in: {audio_input}"
            )
        print(f"Found {len(audio_files)} audio file(s) in: {audio_input}")
    else:
        raise FileNotFoundError(f"--audio_input path not found: {audio_input}")

    # ------------------------------------------------------------------
    # 3. Load ensemble (once, shared across all files)
    # ------------------------------------------------------------------
    ensemble_models, device = load_ensemble_models(
        ensemble_dir_pattern=args.ensemble_dir,
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
    )
    print(f"Loaded {len(ensemble_models)} model(s) for ensembling.")

    # ------------------------------------------------------------------
    # 4. Process each file
    # ------------------------------------------------------------------
    for file_idx, audio_file in enumerate(audio_files, start=1):
        file_start = time.time()
        print(f"\n[{file_idx}/{len(audio_files)}] Processing: {audio_file}")

        results    = []
        sr_for_time = None

        for input_tensor, start, end, sr in process_audio_file(
                audio_file,
                args.window_size,
                args.step_size,
                args.resample,
                args.filter,
                mel_bins=args.mel_bins,
                target_length=args.target_length,
                target_sr=args.target_sr,
                norm_mean=args.norm_mean,
                norm_std=args.norm_std):

            if sr_for_time is None:
                sr_for_time = sr

            with torch.no_grad():
                probs_list = []
                for model in ensemble_models:
                    logits = model(input_tensor)
                    probs  = torch.sigmoid(logits)
                    probs_list.append(probs.squeeze(0).cpu().numpy())

            ensemble_probs = np.mean(probs_list, axis=0)
            target_prob    = float(ensemble_probs[0])

            results.append({
                'start_time': round(start / sr_for_time, 2),
                'end_time':   round(end   / sr_for_time, 2),
                target_label: round(target_prob, 4),
            })

        # ---- Write per-window CSV ----
        audio_name      = os.path.basename(audio_file).rsplit('.', 1)[0]
        csv_output_path = os.path.join(
            args.exp_dir, f'{audio_name}.csv'
        )
        fieldnames = ['start_time', 'end_time', target_label]
        with open(csv_output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Per-window CSV : {csv_output_path}  ({len(results)} windows)")

        # ---- Post-processing: smooth → threshold → NMS → visualise ----
        final_events = process_csv(
            csv_output_path,
            output_dir=args.exp_dir,
            threshold=args.threshold,
            smooth_window=args.smooth,
        )
        summary_path = os.path.join(
            args.exp_dir, f'Summary_inference_{args.model}_{audio_name}.csv'
        )
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Start', 'End', 'Probability'])
            writer.writeheader()
            for ev in final_events:
                writer.writerow({
                    'Start':       int(ev['start']),
                    'End':         int(ev['end']),
                    'Probability': float(ev['prob']),
                })
        print(f"  Event summary  : {summary_path}  ({len(final_events)} event(s))")
        print(f"  File time      : {time.time() - file_start:.1f}s")

    print(f"\n=== All done — {len(audio_files)} file(s) processed in "
          f"{time.time() - start_time:.1f}s total ===")


if __name__ == '__main__':
    main()
