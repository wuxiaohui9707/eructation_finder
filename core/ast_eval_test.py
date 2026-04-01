import argparse
import os
import json
import csv
import numpy as np
import torch
import torchaudio
from ast_model import ASTModelVis
from ast_data_module import resample_audio
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, average_precision_score

def make_features(waveform, sr, mel_bins, target_length=500, 
                  norm_mean=4.2677393, norm_std=4.5689974):
    """
    Replicate exactly what AudiosetDataset._wav2fbank does:
      1. subtract waveform mean  (matches line 271 of ast_data_module.py)
      2. kaldi fbank
      3. pad / trim
      4. normalize: (fbank - norm_mean) / (norm_std * 2)

    norm_mean and norm_std must be the POSITIVE values passed to --dataset-mean
    and --dataset-std (i.e. 4.2677393 and 4.5689974 for the default AudioSet stats).
    """
    # Step 1: subtract waveform mean (same as _wav2fbank line 271)
    waveform = waveform - waveform.mean()

    # Step 2: compute fbank
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
    )

    # Step 3: pad or trim
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p), mode='constant')
    elif p < 0:
        fbank = fbank[:target_length, :]

    # Step 4: normalize  (note: norm_mean is POSITIVE, e.g. 4.2677393)
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

def visualize_masked_mel_spectrogram(filename, feats_data, att_list, save_dir, layer_idx=11):
    """
    Visualize Mel Spectrogram with Attention Map

    Parameters:
    - filename: Filename for the title
    - feats_data: Original Mel Spectrogram data
    - att_list: List containing attention maps
    - save_dir: Directory to save the visualization
    - layer_idx: Index of the layer of interest, default is 11
    """
    # Assuming feats_data is the original Mel Spectrogram
    mel_spectrogram = np.abs(feats_data[0].t().cpu().numpy())

    # Extract attention map of the specified layer
    att_map = att_list[layer_idx].data.cpu().numpy()
    att_map = att_map[0]
    att_map = np.mean(att_map[:, 0:2, :], axis=1)
    att_map = att_map[:, 2:].reshape(12, 12, 49)  # shape: (12 heads, 12, 49)

    # Initialize an empty attention map to accumulate attention maps from all heads
    att_map_combined = np.zeros((mel_spectrogram.shape[0], mel_spectrogram.shape[1]))

    # Iterate over each head
    for head_idx in range(12):
        # Extract attention map of the current head
        att_map_head = att_map[head_idx]

        # Resize attention map to match the Mel Spectrogram
        att_map_resized = zoom(att_map_head, (mel_spectrogram.shape[0] / att_map_head.shape[0], mel_spectrogram.shape[1] / att_map_head.shape[1]))

        # Accumulate the attention map of the current head into the combined attention map
        att_map_combined += np.abs(att_map_resized) / 12  # Each head contributes 1/12 of the weight

    # Normalize the combined attention map
    att_map_combined = (att_map_combined - att_map_combined.min()) / (att_map_combined.max() - att_map_combined.min())

    # Apply the combined attention map as a mask to the Mel Spectrogram
    masked_spectrogram = mel_spectrogram * np.abs(att_map_combined)
    save_dir = os.path.join(save_dir, 'attention_maps')
    os.makedirs(save_dir, exist_ok=True)

    # Visualize the final masked Mel Spectrogram
    plt.figure(figsize=(6, 3))
    plt.imshow(masked_spectrogram, origin='lower', cmap='viridis', aspect='auto')
    plt.title(f'{filename}')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency')
    plt.colorbar(label='Masked Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}_masked.png'))
    plt.close()

def evaluate_test_set(args, model_path, test_json, label_csv, exp_dir):
    """evaluate_test_set"""
    # load label mappings
    label_dict = {}
    with open(label_csv, 'r') as f:
        #index	mid	display_name
        #0	/m/POS	burp
        #1	/m/NEG	nonburp
        reader = csv.DictReader(f)
        for row in reader:
            label_dict[int(row['index'])] = row['display_name']

    # load model — must match the exact same architecture used during training
    model = ASTModelVis(
        label_dim=args.label_dim,
        fstride=args.fstride,
        tstride=args.tstride,
        input_fdim=args.input_fdim,
        input_tdim=args.input_tdim,
        imagenet_pretrain=args.imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain,
        audioset_pretrain_path=args.audioset_pretrain_path,
        model_size=args.model_size,
    )
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
    else:
        model.load_state_dict(checkpoint)
    model.to('cuda')
    model.eval()

    # load test data
    with open(test_json, 'r') as f:
        test_data = json.load(f)['data']

    results = []
    true_labels = []
    predicted_labels = []

    for item in test_data:
        audio_name = os.path.basename(item['wav']).replace('.wav', '')
        row_label = item['labels'] #/m/POS or /m/NEG
        if row_label == '/m/POS':
            row_label = 'burp'
        elif row_label == '/m/NEG':
            row_label = 'nonburp'
        else:
            raise ValueError(f"Unknown label: {row_label}")
        waveform, sr = resample_audio(item['wav'], target_sample_rate=16000, resample=args.resample)
        
        mel_spectrogram = make_features(
            waveform, sr,
            mel_bins=args.input_fdim,
            target_length=args.input_tdim,
            norm_mean=args.dataset_mean,
            norm_std=args.dataset_std,
        )
        mel_tensor = mel_spectrogram.unsqueeze(0).to('cuda').float()

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            predicted_label_idx = np.argmax(probs)
            predicted_label = label_dict[predicted_label_idx] # burp or nonburp

        class_probs = {label_dict[i]: probs[i] for i in range(len(probs))}
        results.append({
            'audio_name': audio_name,
            'row_label': row_label,  
            'predicted_label': predicted_label,
            'probability': probs[predicted_label_idx],
            **class_probs
        })

        true_labels.append(row_label)
        predicted_labels.append(predicted_label)

        if args.plot_attention:
            att_list = model.forward_visualization(mel_tensor)
            visualize_masked_mel_spectrogram(audio_name, mel_spectrogram, att_list, exp_dir)

    # save results to csv
    csv_path = os.path.join(exp_dir, 'test_results.csv')
    fieldnames = ['audio_name', 'row_label', 'predicted_label', 'probability'] + list(label_dict.values())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Test results saved to {csv_path}")

    # calculate evaluation metrics
    precision_per_class = precision_score(true_labels, predicted_labels, average=None, labels=list(label_dict.values()))
    recall_per_class = recall_score(true_labels, predicted_labels, average=None, labels=list(label_dict.values()))
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, labels=list(label_dict.values()))

    # calculate mAP
    y_true_binary = np.array([1 if label == 'burp' else 0 for label in true_labels])
    y_scores = np.array([result['burp'] for result in results])
    map_score = average_precision_score(y_true_binary, y_scores)

    # save evaluation metrics
    metrics_path = os.path.join(exp_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        for idx, class_name in enumerate(label_dict.values()):
            f.write(f'Class: {class_name}\n')
            f.write(f'  Precision: {precision_per_class[idx]:.4f}\n')
            f.write(f'  Recall:    {recall_per_class[idx]:.4f}\n')
            f.write(f'  F1-score:  {f1_per_class[idx]:.4f}\n\n')

        f.write(f'mAP (Average Precision): {map_score:.4f}\n\n')
        f.write('Detailed Classification Report:\n')
        f.write(classification_report(true_labels, predicted_labels, target_names=label_dict.values()))
    print(f"Evaluation metrics saved to {metrics_path}")

# def str2bool(v):
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate AST model on test set")
    parser.add_argument("--model-path", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("--test-json", type=str, required=True, help="path to the test set JSON file")
    parser.add_argument("--label-csv", type=str, required=True, help="path to the label CSV file")
    parser.add_argument("--exp-dir", type=str, required=True, help="experiment directory")
    parser.add_argument("--resample", type=str2bool, default=False, help="resample audio")
    parser.add_argument("--sample-rate", type=int, default=16000, help="target sample rate")
    parser.add_argument("--plot-attention", type=str2bool, default=False, help="plot attention maps")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    evaluate_test_set(args, args.model_path, args.test_json, args.label_csv, args.exp_dir)