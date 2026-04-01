import argparse
import os
import json
import glob
import numpy as np
import torch
import torchaudio
import pandas as pd
import sys
import os
# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.models.ast_model import ASTModel
from core.data_module import resample_audio
import h5py
import time
from tqdm import tqdm
import traceback

class ASTFeatureExtractor(ASTModel):
    def extract_features(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        
        return x
    
    def extract_dist_features(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = x[:, 1]
        return x
    
    def extract_cls_features(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        return x[:, 0]
    
    def extract_sequence_features(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        return x[:, 2:]

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

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

def process_single_file(item, models, device, args, item_idx):
    """
    Process a single audio file with error handling
    Returns: (features, metadata, error_info)
    """
    try:
        filename = item['filename']
        class_label = item['class']
        folder_path = item['folder_path']
        file_path = item['wav']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None, None, f"File not found: {file_path}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None, None, f"Empty file: {file_path}"
        
        # Load and process audio
        waveform, sr = resample_audio(file_path, target_sample_rate=16000, resample=args.resample)
        
        # Check waveform validity
        if waveform is None or waveform.shape[0] == 0:
            return None, None, f"Invalid waveform from file: {file_path}"
        
        # Create mel spectrogram
        mel_spec = make_features(waveform, sr, mel_bins=128)
        
        # Check mel_spec validity
        if mel_spec is None or torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            return None, None, f"Invalid mel spectrogram for file: {file_path}"
        
        mel_spec = mel_spec.unsqueeze(0).to(device)

        # Extract features from all models
        feature_list = []
        with torch.no_grad():
            for model_idx, model in enumerate(models):
                try:
                    features = model.extract_dist_features(mel_spec) # Extract features using the model
                    if features is None or torch.isnan(features).any() or torch.isinf(features).any():
                        return None, None, f"Invalid features from model {model_idx} for file: {file_path}"
                    feature_list.append(features.cpu().numpy())
                except Exception as e:
                    return None, None, f"Model {model_idx} failed for file {file_path}: {str(e)}"

        # Average features across models
        avg_features = np.mean(feature_list, axis=0)
        
        # Check final features validity
        if np.isnan(avg_features).any() or np.isinf(avg_features).any():
            return None, None, f"Invalid averaged features for file: {file_path}"

        metadata = {
            'filename': filename,
            'class': class_label,
            'index': item_idx,
            'audio_path': file_path,
        }
        
        return avg_features[0], metadata, None
        
    except Exception as e:
        error_msg = f"Unexpected error processing {item.get('filename', 'unknown')}: {str(e)}\n{traceback.format_exc()}"
        return None, None, error_msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, 
                       help='Directory containing all the model checkpoints')
    parser.add_argument('--model_pattern', type=str, default='*/checkpoints/best_model*.ckpt', 
                       help='Pattern to match model checkpoint files')
    parser.add_argument('--data_json', type=str, required=True, 
                       help='Path to the JSON file containing audio data')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save extracted features')
    parser.add_argument('--resample', type=bool, default=True, 
                       help='Resample audio to 16kHz')
    args = parser.parse_args()
    
    start_time = time.time()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all model checkpoints
    model_paths = glob.glob(os.path.join(args.model_dir, args.model_pattern))
    if not model_paths:
        raise ValueError(f"No model checkpoints found with pattern: {os.path.join(args.model_dir, args.model_pattern)}")
    
    print(f"Found {len(model_paths)} model checkpoints:")
    for path in model_paths:
        print(f"  - {path}")
    
    # Load data
    with open(args.data_json, 'r') as f:
        data = json.load(f)['data']
    print(f"Loaded {len(data)} audio files for feature extraction.")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all models
    models = []
    for model_idx, model_path in enumerate(model_paths):
        print(f"  Loading model {model_idx+1}/{len(model_paths)}...")
        
        # Create a new model instance
        model = ASTFeatureExtractor(label_dim=2)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()})
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        models.append(model)

    print(f"All models loaded in {time.time() - start_time:.2f} seconds.")

    # Feature extraction with error handling
    all_features = []
    metadata = []
    failed_files = []
    successful_files = []
    
    total_files = len(data)
    progress_bar = tqdm(data, desc="Extracting features", unit="file")
    
    for item_idx, item in enumerate(progress_bar):
        features, meta, error = process_single_file(item, models, device, args, item_idx)
        
        if features is not None and meta is not None and error is None:
            all_features.append(features)
            # Update index to reflect actual position in successful files
            meta['index'] = len(all_features) - 1
            metadata.append(meta)
            successful_files.append(item)
        else:
            failed_files.append({
                'original_index': item_idx,
                'filename': item.get('filename', 'unknown'),
                'class': item.get('class', 'unknown'),
                'path': item.get('wav', 'unknown'),
                'error': error
            })
        
        # Update progress bar description
        progress_bar.set_description(f"Extracting features (Success: {len(all_features)}, Failed: {len(failed_files)})")

    # Print summary
    print(f"\n{'='*50}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total files in JSON: {total_files}")
    print(f"Successfully processed: {len(all_features)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Success rate: {len(all_features)/total_files*100:.1f}%")
    
    # Print failed files details
    if failed_files:
        print(f"\n{'='*50}")
        print(f"FAILED FILES DETAILS")
        print(f"{'='*50}")
        
        # Group failed files by category
        failed_by_category = {}
        for failed in failed_files:
            category = failed['class']
            if category not in failed_by_category:
                failed_by_category[category] = []
            failed_by_category[category].append(failed)
        
        for category, files in failed_by_category.items():
            print(f"\n{category} category: {len(files)} failed files")
            for failed in files:
                print(f"  - {failed['filename']}")
                print(f"    Error: {failed['error']}")
                print(f"    Path: {failed['path']}")
                print(f"    File exists: {os.path.exists(failed['path'])}")
                if os.path.exists(failed['path']):
                    print(f"    File size: {os.path.getsize(failed['path'])} bytes")
                print()
        
        # Save failed files log
        failed_files_log = os.path.join(args.output_dir, 'failed_files.json')
        with open(failed_files_log, 'w') as f:
            json.dump(failed_files, f, indent=2)
        print(f"Failed files log saved to: {failed_files_log}")

    if len(all_features) == 0:
        print("ERROR: No features were successfully extracted!")
        return

    # Convert features to numpy array
    features_array = np.array(all_features)
    print(f"\nExtracted features shape: {features_array.shape}")
    
    # Convert metadata to pandas DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Save features and metadata
    feature_file = os.path.join(args.output_dir, 'ast_features.npz')
    np.savez(feature_file, 
             features=features_array, 
             filenames=metadata_df['filename'].values,
             classes=metadata_df['class'].values,
             indices=metadata_df['index'].values)
    
    print(f'Features extracted and saved to {feature_file}')
    print(f'Feature shape: {features_array.shape}')
    
    # Save metadata as CSV for easier inspection
    metadata_file = os.path.join(args.output_dir, 'ast_features_metadata.csv')
    metadata_df.to_csv(metadata_file, index=False)
    print(f'Metadata saved to {metadata_file}')
    
    # Create a combined file with both features and metadata (as HDF5)
    try:
        h5_file = os.path.join(args.output_dir, 'ast_features_with_metadata.h5')
        with h5py.File(h5_file, 'w') as f:
            # Save features
            f.create_dataset('features', data=features_array)
            
            # Save metadata (as string datasets)
            dt = h5py.special_dtype(vlen=str)
            filenames = metadata_df['filename'].values
            classes = metadata_df['class'].values
            
            f.create_dataset('filenames', data=filenames, dtype=dt)
            f.create_dataset('classes', data=classes, dtype=dt)
            f.create_dataset('indices', data=metadata_df['index'].values)
            
            # Add the number of samples per class as attributes
            class_counts = metadata_df['class'].value_counts().to_dict()
            for cls, count in class_counts.items():
                f.attrs[f'count_{cls}'] = count
            
            # Save some feature extraction parameters
            f.attrs['num_models'] = len(model_paths)
            f.attrs['feature_dim'] = features_array.shape[1]
            f.attrs['total_samples'] = len(metadata_df)
            f.attrs['failed_files_count'] = len(failed_files)
        
        print(f'Combined features and metadata saved to {h5_file}')
    except ImportError:
        print("h5py is not available, skipping creation of HDF5 file")

    print(f"\nFeature extraction completed in {time.time() - start_time:.2f} seconds.")
    print(f"Average time per successful file: {(time.time() - start_time) / len(all_features):.2f} seconds")
    print("Done.")

if __name__ == '__main__':
    main()