import numpy as np
import matplotlib.pyplot as plt
import librosa
import torchaudio
import torch
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.data_module import get_custom_mel_filterbank

def load_audio(audio_path, sr=None):
    """
    Load audio file and return waveform and sample rate
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform - waveform.mean()
    
    # If target sample rate is provided, resample if needed
    if sr is not None and sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
        sample_rate = sr
    
    print(f"Audio length: {waveform.shape[1]/sample_rate:.2f} seconds, Sample rate: {sample_rate} Hz")
    return waveform, sample_rate

def create_fbank_spectrogram(waveform, sample_rate, n_mels=128, freq_division_mode='uniform', split_freq=1000):
    """
    Create fbank spectrogram using torchaudio.compliance.kaldi.fbank
    to match the AST implementation, with support for different frequency division modes.
    
    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate
        n_mels: Number of mel bins
        freq_division_mode: 'uniform' or 'split_1khz'
        split_freq: Split frequency for split_1khz mode (default 1000 Hz)
    """
    if freq_division_mode == 'uniform':
        # Use standard kaldi fbank
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=n_mels,
            dither=0.0,
            frame_shift=10
        )
    elif freq_division_mode == 'split_1khz':
        # Use custom mel filterbank with split at split_freq
        n_fft = 512
        win_length = int(sample_rate * 0.025)  # 25ms window
        hop_length = int(sample_rate * 0.010)  # 10ms hop (frame_shift=10ms)
        
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
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            mode='split_1khz',
            split_freq=split_freq
        )
        
        # Apply mel filterbank
        mel_spec = torch.matmul(mel_fb, power_spec)
        
        # Convert to log scale (to match kaldi fbank output)
        fbank = torch.log(mel_spec + 1e-6).transpose(0, 1)
    else:
        raise ValueError(f"Unknown freq_division_mode: {freq_division_mode}")
    
    return fbank.numpy()

def visualize_ast_patches(fbank, patch_size=16, overlap=6, figsize=(10, 10)):
    """
    Visualize AST patch splitting method with corrected adjacent patch pattern
    and adjusted text positions
    """
    # Create custom colormap (using viridis as requested)
    ast_cmap = 'viridis'
        
    # Calculate effective stride
    stride = patch_size - overlap
    
    # Set up the figure with equal height ratios
    fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot original fbank spectrogram
    img0 = axs[0].imshow(fbank.T, origin='lower', aspect='auto', cmap=ast_cmap)
    axs[0].set_title('Original Fbank Spectrogram', fontsize=14)
    axs[0].set_xlabel('Time Frames')
    axs[0].set_ylabel('Frequency Bins')
    fig.colorbar(img0, ax=axs[0])
    
    # Plot fbank spectrogram with patch overlay
    img1 = axs[1].imshow(fbank.T, origin='lower', aspect='auto', cmap=ast_cmap)
    axs[1].set_title(f'AST Patch Splitting (Size={patch_size}×{patch_size}, Stride={stride}, Overlap={overlap})', fontsize=14)
    axs[1].set_xlabel('Time Frames')
    axs[1].set_ylabel('Frequency Bins')
    fig.colorbar(img1, ax=axs[1])

    # Get spectrogram dimensions
    time_frames, freq_bins = fbank.shape
    
    # Calculate number of patches along each dimension
    num_time_patches = (time_frames - patch_size) // stride + 1
    num_freq_patches = (freq_bins - patch_size) // stride + 1
    
    # Create and store patch indices
    patch_indices = []
    
    # ----- IMPROVED ADJACENT PATCH VISUALIZATION -----
    # Choose a starting point in the middle of the spectrogram
    start_time_idx = min(time_frames // 3, time_frames - (2 * stride + patch_size))
    start_freq_idx = min(freq_bins // 3, freq_bins - (2 * stride + patch_size))
    
    # Define colors for the patches
    colors = ['#00ff00', '#ff0000', '#0000ff', '#ffff00']
    
    # Create adjacent patches to show the proper overlap pattern
    adjacent_patches = [
        (start_time_idx, start_freq_idx),                  # Base patch
        (start_time_idx + stride, start_freq_idx),         # Right adjacent (time dimension)
        (start_time_idx, start_freq_idx + stride),         # Top adjacent (frequency dimension)
        (start_time_idx + stride, start_freq_idx + stride) # Diagonal adjacent
    ]
    
    # Draw the four adjacent patches
    for i, (x_start, y_start) in enumerate(adjacent_patches):
        # Store patch indices
        patch_indices.append((x_start, y_start))
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_start, y_start), 
            patch_size, 
            patch_size, 
            linewidth=2, 
            edgecolor=colors[i % len(colors)], 
            facecolor='none',
            alpha=0.8
        )
        axs[1].add_patch(rect)
        
        # Label the patches
        axs[1].text(
            x_start + patch_size/2, 
            y_start + patch_size/2, 
            f'{i+1}', 
            color='white', 
            ha='center', 
            va='center',
            fontsize=12, 
            fontweight='bold'
        )
    
    # Add arrows to show the overlap regions
    # Horizontal overlap (between patch 1 and 2)
    axs[1].annotate('', 
                   xy=(start_time_idx + patch_size, start_freq_idx + patch_size/2),
                   xytext=(start_time_idx + stride, start_freq_idx + patch_size/2),
                #    arrowprops=dict(arrowstyle='|-|', color='white', lw=2)
                   )
    # axs[1].text(start_time_idx + stride + (overlap/2), 
    #            start_freq_idx + patch_size/2 + 5, 
    #            f'Overlap={overlap}px', 
    #            color='white', 
    #            ha='center', 
    #            va='bottom',
    #            fontsize=9,
    #            fontweight='bold',
    #            bbox=dict(facecolor='black', alpha=0.6))
    
    # Vertical overlap (between patch 1 and 3)
    axs[1].annotate('', 
                   xy=(start_time_idx + patch_size/2, start_freq_idx + patch_size),
                   xytext=(start_time_idx + patch_size/2, start_freq_idx + stride),
                #    arrowprops=dict(arrowstyle='|-|', color='white', lw=2)
                   )
    # axs[1].text(start_time_idx + patch_size/2 - 5, 
    #            start_freq_idx + stride + (overlap/2), 
    #            f'Overlap={overlap}px', 
    #            color='white', 
    #            ha='right', 
    #            va='center',
    #            fontsize=9,
    #            fontweight='bold',
    #            bbox=dict(facecolor='black', alpha=0.6))
    
    # Calculate position for the legend - directly below the example patches
    # Get the center position of the patches in data coordinates
    patch_center_x = start_time_idx + stride/2 + patch_size/2
    patch_center_y = start_freq_idx - 20  # Position below the patches
    
    # Convert data coordinates to axes coordinates for text placement
    # We need to get the bounds of the axes to do this conversion
    x_min, x_max = axs[1].get_xlim()
    y_min, y_max = axs[1].get_ylim()
    
    # Position in axes coordinates (normalized from 0-1)
    text_x = (patch_center_x - x_min) / (x_max - x_min)
    text_y = (patch_center_y - y_min) / (y_max - y_min)
    
    # Make sure the text is visible by clamping to a reasonable range
    text_y = max(0.1, min(text_y, 0.1))
    
    # Add legend explanation for adjacent patches below the examples
    axs[1].text(
        text_x, 0.02, 
        "Adjacent Patch Pattern:\n"
        "1: Base Patch\n"
        "2: Adjacent in Time\n"
        "3: Adjacent in Frequency\n"
        "4: Diagonal Adjacent", 
        transform=axs[1].transAxes, 
        fontsize=10, 
        va='bottom', 
        ha='center',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
        color='white'
    )
    
    # Add patch statistics to the bottom right
    axs[1].text(
        0.98, 0.02, 
        f"Patch Size: {patch_size}×{patch_size}\n"
        f"Overlap: {overlap}\n"
        f"Stride: {stride}\n"
        f"Time Dimension Patches: {num_time_patches}\n"
        f"Frequency Dimension Patches: {num_freq_patches}\n"
        f"Total Patches: {num_time_patches * num_freq_patches}", 
        transform=axs[1].transAxes, 
        fontsize=10, 
        va='bottom', 
        ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Add example patch in the first plot
    # Choose an informative area for the example patch
    patch_x = time_frames // 3
    patch_y = freq_bins // 3
    
    # Ensure patch is fully within the spectrogram
    if patch_x + patch_size > time_frames:
        patch_x = time_frames - patch_size
    if patch_y + patch_size > freq_bins:
        patch_y = freq_bins - patch_size
    
    # Create a square inset axes for the patch example (1:1 aspect ratio)
    # Calculate position based on axes coordinates
    ax_inset = axs[0].inset_axes([0.7, 0.6, 0.25, 0.25], transform=axs[0].transAxes)
    
    # Extract the patch
    example_patch = fbank[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size].T
    
    # Display the patch with square aspect ratio (1:1)
    ax_inset.imshow(example_patch, cmap=ast_cmap, aspect='equal', origin='lower')
    ax_inset.set_title('16×16 Patch Example', fontsize=10, color='white')
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.patch.set_alpha(0.7)
    ax_inset.patch.set_facecolor('black')
    
    # Mark this patch location in the original plot
    rect_highlight = patches.Rectangle(
        (patch_x, patch_y), 
        patch_size, 
        patch_size, 
        linewidth=2, 
        edgecolor='white', 
        facecolor='none'
    )
    axs[0].add_patch(rect_highlight)
    
    # # Add explanation of patch extraction
    # plt.figtext(
    #     0.5, 0.01, 
    #     "AST extracts overlapping 16×16 patches with stride=10 (overlap=6 pixels).\n"
    #     "Adjacent patches share 6-pixel overlapping regions to better capture features across patch boundaries.",
    #     ha='center', 
    #     fontsize=10, 
    #     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    # )
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    
    return fig, fbank, patch_indices, stride, patch_size

def visualize_patch_sequence(fbank, patch_indices, patch_size, max_patches=16, figsize=(15, 15)):
    """
    Visualize the sequence of patches
    """
    # Create custom colormap
    colors = [(0.5, 0, 0.65), (0.65, 0.15, 0.8), (1, 0.4, 0.6), (1, 0.8, 0.2)]
    ast_cmap = 'viridis'
    
    # Limit the number of patches to display
    max_patches = min(max_patches, len(patch_indices))
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(max_patches)))
    
    # Create figure
    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle('AST Patch Sequence Examples', fontsize=16)
    
    # Make 2D axes array compatible with 1D or 0D cases
    if grid_size == 1:
        axs = np.array([[axs]])
    elif len(axs.shape) == 1:
        axs = axs.reshape(grid_size, 1)
    
    # Plot each patch
    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        ax = axs[row, col]
        
        if i < max_patches:
            # Get patch coordinates
            x_start, y_start = patch_indices[i]
            
            # Extract patch - note the transpose
            patch = fbank[x_start:x_start+patch_size, y_start:y_start+patch_size].T
            
            # Display patch
            ax.imshow(patch, cmap=ast_cmap, aspect='auto', origin='lower')
            ax.set_title(f'Patch #{i+1}: ({x_start},{y_start})', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig

def visualize_patch_coverage(fbank, patch_size=16, overlap=6, figsize=(12, 6)):
    """
    Visualize patch coverage heatmap
    """
    # Calculate effective stride
    stride = patch_size - overlap
    
    # Get spectrogram dimensions
    time_frames, freq_bins = fbank.shape
    
    # Calculate number of patches along each dimension
    num_time_patches = (time_frames - patch_size) // stride + 1
    num_freq_patches = (freq_bins - patch_size) // stride + 1
    
    # Create patch coverage heatmap
    coverage = np.zeros((freq_bins, time_frames))
    
    # Calculate how many patches cover each position
    for i in range(num_time_patches):
        for j in range(num_freq_patches):
            x_start = i * stride
            y_start = j * stride
            coverage[y_start:y_start+patch_size, x_start:x_start+patch_size] += 1
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(coverage, cmap='hot', aspect='auto', origin='lower')
    plt.colorbar(im, label='Patch Coverage Count')
    ax.set_title(f'AST Patch Coverage Heatmap (Size={patch_size}, Overlap={overlap})', fontsize=14)
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency Bins')
    
    # Add explanation
    plt.figtext(
        0.5, 0.01, 
        f"Heatmap shows how many patches cover each pixel.\n"
        f"Darker areas indicate higher coverage, meaning these locations are shared by multiple patches.\n"
        f"Patch overlap allows AST to better capture features that span patch boundaries.", 
        ha="center", 
        fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def visualize_overlap_comparison(fbank, patch_size=16, figsize=(15, 12)):
    """
    Compare different overlap sizes for patch splitting
    """
    # Define a consistent frame rate for visualization
    # 10ms is the default frame shift in AST's fbank implementation
    frame_rate = 10  # ms per frame
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Comparison of Different Overlap Sizes for AST Patch Splitting', fontsize=16)
    
    overlap_values = [0, 4, 8, 12]
    for i, overlap_val in enumerate(overlap_values):
        row, col = i // 2, i % 2
        stride_val = patch_size - overlap_val
        
        # Calculate patch counts
        num_time_patches = (fbank.shape[0] - patch_size) // stride_val + 1
        num_freq_patches = (fbank.shape[1] - patch_size) // stride_val + 1
        
        # Plot spectrogram
        img = axs[row, col].imshow(fbank.T, origin='lower', aspect='auto', cmap='magma')
        
        # Add some example patches
        for ti in range(min(3, num_time_patches)):
            for fi in range(min(5, num_freq_patches)):
                if (fi + ti) % 2 == 0:
                    x_start = ti * stride_val
                    y_start = fi * stride_val
                    
                    rect = patches.Rectangle(
                        (x_start, y_start), 
                        patch_size, 
                        patch_size, 
                        linewidth=1, 
                        edgecolor='cyan', 
                        facecolor='none',
                        alpha=0.7
                    )
                    axs[row, col].add_patch(rect)
        
        axs[row, col].set_title(f'Overlap={overlap_val}, Stride={stride_val}, Patches={num_time_patches*num_freq_patches}')
        axs[row, col].set_xlabel('Time Frames')
        axs[row, col].set_ylabel('Frequency Bins')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    return fig

def main(audio_path, figsize = (10,12), n_mels=128, patch_size=16, overlap=6, target_sr=None, output_prefix="ast_", freq_division_mode='uniform', split_freq=1000):
    """
    Main function: Execute all visualizations
    """
    # 1. Load audio
    waveform, sample_rate = load_audio(audio_path, sr=target_sr)
    
    # 2. Generate fbank spectrogram
    fbank = create_fbank_spectrogram(
        waveform, sample_rate, n_mels=n_mels, 
        freq_division_mode=freq_division_mode, 
        split_freq=split_freq
    )
    
    # 3. Draw AST patch splitting figure
    fig1, fbank_data, patch_indices, stride, patch_size = visualize_ast_patches(
        fbank, patch_size=patch_size, overlap=overlap, figsize=figsize
    )
    fig1.savefig(f"{output_prefix}patch_overlay.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved AST patch overlay visualization to: {output_prefix}patch_overlay.png")
    
    # 4. Draw patch sequence display
    fig2 = visualize_patch_sequence(
        fbank_data, patch_indices, patch_size, max_patches=16, figsize=figsize
    )
    fig2.savefig(f"{output_prefix}patch_sequence.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved patch sequence visualization to: {output_prefix}patch_sequence.png")
    
    # 5. Draw patch coverage heatmap
    fig3 = visualize_patch_coverage(
        fbank_data, patch_size=patch_size, overlap=overlap, figsize=(12, 6)
    )
    fig3.savefig(f"{output_prefix}patch_coverage.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved patch coverage heatmap to: {output_prefix}patch_coverage.png")
    
    # 6. Compare different overlap sizes
    fig4 = visualize_overlap_comparison(
        fbank_data, patch_size=patch_size, figsize=figsize
    )
    fig4.savefig(f"{output_prefix}overlap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved overlap comparison visualization to: {output_prefix}overlap_comparison.png")
    
    print("\nAll visualizations completed and saved!")
    
    # Return visualization for display (can be shown in notebook environments)
    return visualize_ast_patches(
        fbank_data, patch_size=patch_size, overlap=overlap, figsize=figsize
    )[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AST Spectrogram Patch Splitting Visualization Tool')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bins')
    parser.add_argument('--patch_size', type=int, default=16, help='AST patch size')
    parser.add_argument('--overlap', type=int, default=6, help='AST patch overlap pixels')
    parser.add_argument('--sample_rate', type=int, default=None, help='Target sample rate (optional)')
    parser.add_argument('--output', type=str, default="ast_", help='Output file prefix')
    parser.add_argument('--freq_division_mode', type=str, default='uniform', choices=['uniform', 'split_1khz'], help="Frequency division mode")
    parser.add_argument('--split_freq', type=int, default=1000, help='Split frequency for split_1khz mode (Hz)')
    
    args = parser.parse_args()
    main(args.audio, n_mels=args.n_mels, patch_size=args.patch_size, 
         overlap=args.overlap, target_sr=args.sample_rate, output_prefix=args.output,
         freq_division_mode=args.freq_division_mode, split_freq=args.split_freq)