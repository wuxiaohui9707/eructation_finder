import argparse
import os
import csv
import glob
import numpy as np
import torch
import torchaudio
from scipy import signal
import time
import matplotlib.pyplot as plt
import pandas as pd

import sys
# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Local imports
from core.models.ast_model import ASTModelVis
from core.data_module import resample_audio

start_time = time.time()

def make_features(waveform, sr, mel_bins, target_length=500):
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
    )
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    # Normalize as in your original AST code
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

def process_audio_file(audio_file, window_size, step_size, resample, apply_filter):
    """
    Sliding-window generator that yields (mel_spectrogram_data, start_sample, end_sample).
    """
    waveform, sr = torchaudio.load(audio_file)

    if resample:
        waveform, sr = resample_audio(audio_file, target_sample_rate=16000, resample=True)

    window_samples = int(window_size * sr)
    step_samples = int(step_size * sr)

    # Generate windows
    for start in range(0, waveform.shape[1] - window_samples + 1, step_samples):
        end = start + window_samples
        window_waveform = waveform[:, start:end]

        mel_spectrogram = make_features(window_waveform, sr, mel_bins=128)
        # Expand dims to (1, time, freq) for the model
        mel_spectrogram_data = mel_spectrogram.expand(1, 500, 128).float()
        mel_spectrogram_data = mel_spectrogram_data.to(torch.device('cuda'), dtype=torch.float32)

        yield mel_spectrogram_data, start, end

def weighted_nms(events):
    """
    An example of merging overlapping events by weighting their start/end
    based on probability. Adjust logic as needed.
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
                new_start = (last['start']*last['prob'] + event['start']*event['prob']) / total_weight
                new_end = (last['end']*last['prob'] + event['end']*event['prob']) / total_weight
                merged[-1] = {
                    'start': new_start,
                    'end': new_end,
                    'prob': max(last['prob'], event['prob'])
                }
            else:
                merged.append(event)
    return merged

def process_csv(csv_path, output_dir, prob_col='burp', threshold=0.5, smooth_window=3):
    """
    Take the per-window predictions, build a time-probability curve,
    apply smoothing and thresholding, and then do non-maximum suppression (NMS).
    """
    df = pd.read_csv(csv_path)
    # the column with probabilities
    predictions = df[['start_time', 'end_time', prob_col]].values

    # Calculate total duration (rounded up)
    total_duration = int(np.ceil(predictions[-1, 1]))

    # Build time-probability curve
    time_prob = np.zeros(total_duration)
    weight = np.zeros(total_duration)

    for start, end, prob in predictions:
        start_sec = int(np.floor(start))
        end_sec = int(np.ceil(end))
        for t in range(start_sec, end_sec):
            if t < total_duration:
                time_prob[t] += prob
                weight[t] += 1

    # Avoid divide-by-zero
    time_prob = np.divide(time_prob, weight, out=np.zeros_like(time_prob), where=weight != 0)

    # Simple smoothing
    smoothed = np.convolve(time_prob, np.ones(smooth_window) / smooth_window, mode='same')

    # Threshold and collect events
    events = []
    current_event = None
    for t in range(len(smoothed)):
        if smoothed[t] >= threshold:
            if current_event is None:
                current_event = {'start': t, 'end': t, 'max_prob': smoothed[t]}
            else:
                current_event['end'] = t
                current_event['max_prob'] = max(current_event['max_prob'], smoothed[t])
        else:
            if current_event is not None:
                events.append(current_event)
                current_event = None

    # Handle last event
    if current_event is not None:
        events.append(current_event)

    # Merge events if needed
    merged = weighted_nms(events)

    final_events = []
    for event in merged:
        final_events.append({
            'start': event['start'],
            'end': event['end'] + 1,
            'prob': round(event.get('max_prob', event['prob']), 4)
        })

    # Visualization
    filename = os.path.basename(csv_path).split('.')[0]
    plt.figure(figsize=(15, 5))
    plt.plot(smoothed, label='Smoothed Probability')
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')

    # Highlight events
    already_labeled = False
    for event in final_events:
        # only label the patch once
        label_str = 'Burp Event' if not already_labeled else ""
        plt.axvspan(event['start'], event['end'], alpha=0.3, color='green', label=label_str)
        already_labeled = True

    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.title('Burp Event Detection Timeline')
    plt.legend()
    out_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Visualization saved to {out_path}")

    return final_events

def load_ensemble_models(ensemble_dir_pattern, label_csv):
    """
    1. Find all best_model.ckpt files matching ensemble_dir_pattern
       (e.g. .../ast_filter_True_resample_True/cv_fold_*/checkpoints/best_model.ckpt).
    2. For each checkpoint, load an ASTModelVis() instance.
    3. Return a list of these models on CUDA in eval mode.
    """
    # Load label dict once here if needed
    # (Though in your code you do it in main, so adapt as you wish.)
    model_paths = sorted(glob.glob(os.path.join(ensemble_dir_pattern, "checkpoints", "best_model.ckpt")))
    ensemble_models = []
    for ckpt_path in model_paths:
        print(f"Loading model: {ckpt_path}")
        model = ASTModelVis()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # If 'state_dict' is a nested dict, adapt the key loading
        if 'state_dict' in checkpoint:
            # Some checkpoints have a prefix 'model.'
            model.load_state_dict({
                k.replace('model.', ''): v 
                for k,v in checkpoint['state_dict'].items()
            })
        else:
            model.load_state_dict(checkpoint)

        model.to('cuda')
        model.eval()
        ensemble_models.append(model)
    return ensemble_models
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ensemble_dir', type=str, required=True,
                        help='Glob pattern for directories of best_model.ckpt (e.g. .../cv_fold_*/checkpoints/best_model.ckpt)')
    parser.add_argument('--audio_input', type=str, required=True,
                        help='Path to a single .WAV/.wav file OR a folder containing .WAV/.wav files')
    parser.add_argument('--recursive', action='store_true', default=False,
                        help='If --audio_input is a folder, search sub-folders recursively')
    parser.add_argument('--label_csv', type=str, required=True,
                        help='Path to the CSV file containing label names')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--window_size', type=float, default=5,
                        help='Window size in seconds for sliding over audio')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size in seconds between consecutive windows')
    parser.add_argument('--resample', type=bool, default=True,
                        help='Whether to resample the audio to 16kHz')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold (default: 0.5)')
    parser.add_argument('--smooth', type=int, default=3,
                        help='Smoothing window size (default: 3)')
    parser.add_argument('--ensemble', type=str, default='True', choices=['True', 'False'],
                        help='True to average models, False to output each model separately')
    parser.add_argument('--nms', type=str, default='False', choices=['True', 'False'],
                        help='True to apply weighted NMS and generate plots, False to keep raw output')

    args = parser.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)

    is_ensemble = (args.ensemble == 'True')
    is_nms      = (args.nms == 'True')

    # ------------------------------------------------------------------
    # 1. Load label mappings
    # ------------------------------------------------------------------
    label_dict = {}
    with open(args.label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_dict[int(row['index'])] = row['display_name']
    burp_label = label_dict.get(0, 'burp')

    # ------------------------------------------------------------------
    # 2. Collect audio files (single file OR folder)
    # ------------------------------------------------------------------
    audio_input = args.audio_input
    if os.path.isfile(audio_input):
        audio_files = [audio_input]
    elif os.path.isdir(audio_input):
        pattern = os.path.join(audio_input, '**', '*.[Ww][Aa][Vv]') if args.recursive \
                  else os.path.join(audio_input, '*.[Ww][Aa][Vv]')
        audio_files = sorted(glob.glob(pattern, recursive=args.recursive))
        if not audio_files:
            raise FileNotFoundError(f"No .WAV/.wav files found in: {audio_input}")
        print(f"Found {len(audio_files)} audio file(s) in: {audio_input}")
    else:
        raise FileNotFoundError(f"--audio_input path not found: {audio_input}")

    # ------------------------------------------------------------------
    # 3. Load ensemble once (shared across all files)
    # ------------------------------------------------------------------
    ensemble_models = load_ensemble_models(args.ensemble_dir, args.label_csv)
    print(f"Loaded {len(ensemble_models)} models for ensembling.")

    # ------------------------------------------------------------------
    # 4. Process each file
    # ------------------------------------------------------------------
    for file_idx, audio_file in enumerate(audio_files, start=1):
        file_start = time.time()
        print(f"\n[{file_idx}/{len(audio_files)}] Processing: {audio_file}")

        results = []
        model_results = [[] for _ in range(len(ensemble_models))]

        sr_for_time = 16000 if args.resample else None
        if sr_for_time is None:
            _, sr_for_time = torchaudio.load(audio_file)

        for mel_spectrogram_data, start, end in process_audio_file(
                audio_file, args.window_size, args.step_size,
                args.resample, apply_filter=False):

            with torch.no_grad():
                probs_list = []
                for m_idx, model in enumerate(ensemble_models):
                    logits = model(mel_spectrogram_data)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    probs_list.append(probs)
                    
                    if not is_ensemble:
                        model_results[m_idx].append({
                            'start_time': round(start / sr_for_time, 2),
                            'end_time':   round(end   / sr_for_time, 2),
                            burp_label:   round(float(probs[0]), 4),
                        })

            if is_ensemble:
                ensemble_probs = np.mean(probs_list, axis=0)
                burp_prob = ensemble_probs[0]
                results.append({
                    'start_time': round(start / sr_for_time, 2),
                    'end_time':   round(end   / sr_for_time, 2),
                    burp_label:   round(float(burp_prob), 4),
                })

        # ---- Write per-window CSV and possibly run NMS ----
        audio_name = os.path.basename(audio_file).rsplit('.', 1)[0]
        fieldnames = ['start_time', 'end_time', burp_label]

        if is_ensemble:
            csv_output_path = os.path.join(args.exp_dir, f'{audio_name}.csv')
            with open(csv_output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"  Ensemble CSV : {csv_output_path}  ({len(results)} windows)")
            
            if is_nms:
                process_csv(csv_output_path, args.exp_dir, prob_col=burp_label, 
                            threshold=args.threshold, smooth_window=args.smooth)
        else:
            for m_idx in range(len(ensemble_models)):
                csv_output_path = os.path.join(args.exp_dir, f'{audio_name}_model_{m_idx}.csv')
                with open(csv_output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(model_results[m_idx])
                print(f"  Model {m_idx} CSV : {csv_output_path}  ({len(model_results[m_idx])} windows)")
                
                if is_nms: # Process NMS for EACH model's CSV
                    process_csv(csv_output_path, args.exp_dir, prob_col=burp_label, 
                                threshold=args.threshold, smooth_window=args.smooth)

        print(f"  File processing time: {time.time() - file_start:.1f}s")

    print(f"\n=== All done — {len(audio_files)} file(s) processed in "
          f"{time.time() - start_time:.1f}s total ===")

if __name__ == '__main__':
    main()