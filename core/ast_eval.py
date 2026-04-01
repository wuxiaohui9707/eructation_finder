import argparse
import os
import json
import csv
import numpy as np
import torch
import torchaudio
import sys
import os

# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.models.ast_model import ASTModelVis
from core.data_module import resample_audio, get_custom_mel_filterbank
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

def make_features(waveform, sr, mel_bins, target_length=500, freq_division_mode='uniform', split_freq=1000):
    """
    Create mel-spectrogram features with support for different frequency division modes.
    
    Args:
        waveform: Audio waveform tensor
        sr: Sample rate
        mel_bins: Number of mel bins
        target_length: Target time dimension
        freq_division_mode: 'uniform' or 'split_1khz'
        split_freq: Split frequency for split_1khz mode (default 1000 Hz)
    """
    if freq_division_mode == 'uniform':
        # Use standard kaldi fbank
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
        )
    elif freq_division_mode == 'split_1khz':
        # Use custom mel filterbank with split at split_freq
        n_fft = 512
        win_length = int(sr * 0.025)  # 25ms window
        hop_length = int(sr * 0.010)  # 10ms hop (frame_shift=10ms)
        
        # Compute STFT
        stft = torch.stft(
            waveform[0],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Compute power spectrogram
        power_spec = torch.abs(stft) ** 2
        
        # Get custom mel filterbank
        mel_fb = get_custom_mel_filterbank(
            sample_rate=sr,
            n_fft=n_fft,
            n_mels=mel_bins,
            mode='split_1khz',
            split_freq=split_freq
        )
        
        # Apply mel filterbank
        mel_spec = torch.matmul(mel_fb, power_spec)
        
        # Convert to log scale (to match kaldi fbank output)
        fbank = torch.log(mel_spec + 1e-6).transpose(0, 1)
    else:
        raise ValueError(f"Unknown freq_division_mode: {freq_division_mode}")

    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
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

# Parse arguments
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to the pre-trained model checkpoint')
    parser.add_argument('--data_json', type=str, required=True, help='Path to the JSON file containing audio data')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to the CSV file containing label names')
    parser.add_argument('--input_tdim', type=int, default=500, help='Input time dimension')
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--plot_attention', type=bool, default=True, help='Whether to plot attention maps')
    parser.add_argument('--resample', type=bool, default=True, help='Resample audio to 16kHz')
    parser.add_argument('--freq_division_mode', type=str, default='uniform', choices=['uniform', 'split_1khz'], help="Frequency division mode")
    parser.add_argument('--split_freq', type=int, default=1000, help='Split frequency for split_1khz mode (Hz)')

    return parser.parse_args()

def main():
    args = get_args()

    os.makedirs(args.exp_dir, exist_ok=True)

    # Load label mappings
    label_dict = {}
    with open(args.label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_dict[int(row['index'])] = row['display_name']

    # Load model
    model = ASTModelVis()
    checkpoint = torch.load(args.pretrained_model)
    if 'state_dict' in checkpoint:  # Check if it's a PyTorch Lightning checkpoint
        model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()})
    else:  # Standard PyTorch state_dict
        model.load_state_dict(checkpoint)
    model.to(torch.device('cuda'))
    model.eval()

    # Load data
    with open(args.data_json, 'r') as f:
        data = json.load(f)['data']

    results = []
    for item in data:
        audio_name = os.path.basename(item['wav']).replace('.wav', '')
        waveform, sr = resample_audio(item['wav'], target_sample_rate=16000, resample=args.resample)
        mel_spectrogram = make_features(
            waveform, sr, mel_bins=128, 
            freq_division_mode=args.freq_division_mode,
            split_freq=args.split_freq
        )
        mel_spectrogram_data = mel_spectrogram.expand(1, args.input_tdim, 128).float() 
        mel_spectrogram_data = mel_spectrogram_data.to(torch.device('cuda'),dtype=torch.float32)
        
        # Model inference
        with torch.no_grad():
            mel_tensor = torch.tensor(mel_spectrogram).unsqueeze(0).to(torch.device('cuda'),dtype=torch.float32)
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            predicted_label = np.argmax(probs)

            att_list = model.forward_visualization(mel_spectrogram_data)

        class_probs = {label_dict[i]: probs[i] for i in range(len(probs))}

        # Save results
        results.append({
            'audio_name': audio_name,
            'raw_label': item['labels'],
            'predicted_label': label_dict[predicted_label],
            'probability': probs[predicted_label],
            **class_probs
        })

        # Plot attention maps if required
        if args.plot_attention:
            visualize_masked_mel_spectrogram(audio_name, mel_spectrogram_data, att_list, args.exp_dir)

    fieldnames = ['audio_name', 'raw_label', 'predicted_label', 'probability'] + list(class_probs.keys())

    # Write results to CSV
    csv_output_path = os.path.join(args.exp_dir, 'evaluation_results.csv')
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f'Evaluation results saved to {csv_output_path}')
if __name__ == '__main__':
    main()