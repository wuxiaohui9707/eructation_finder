import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import h5py
import pandas as pd

def visualize_tsne(features_file, output_dir, perplexity=30, n_iter=1000):
    """
    Visualize features using t-SNE

    Args:
        features_file: NPZ or H5 file containing features and metadata
        output_dir: Directory to save visualization results
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features and metadata
    if features_file.endswith('.npz'):
        data = np.load(features_file, allow_pickle=True)
        features = data['features']
        filenames = data['filenames']
        classes = data['classes']
    elif features_file.endswith('.h5'):
        with h5py.File(features_file, 'r') as f:
            features = f['features'][:]
            filenames = f['filenames'][:]
            classes = f['classes'][:]
    else:
        raise ValueError("Feature file must be .npz or .h5 format")
        
    print(f"Feature shape: {features.shape}")
    print(f"Number of classes: {len(np.unique(classes))}")

    # Print class distribution statistics
    unique_classes, class_counts = np.unique(classes, return_counts=True)
    print("\n=== Class Statistics ===")
    for cls, count in zip(unique_classes, class_counts):
        if isinstance(cls, bytes):
            cls_str = cls.decode('utf-8')
        else:
            cls_str = str(cls)
        print(f"{cls_str}: {count} points")
    print(f"Total: {len(classes)} points")
    print("========================\n")

    if len(features.shape) == 3:
        # If features are 3D, reshape to 2D
        features = np.mean(features, axis=1)  # Average over the patch dimension
        print(f"Reshaped feature shape: {features.shape}")
    
    # Apply t-SNE dimensionality reduction
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Create class-to-color and marker mapping
    unique_classes = np.unique(classes)
    # colors list: ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = ['blue', 'red', 'green', 'purple', 'gray']
    # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10), dpi=500)
    
    label_map = {
    'FN': 'False Negative',
    'FP': 'False Positive',
    'TP': 'True Positive',
    'burp': 'burp',
    'nonburp': 'nonburp'
    }

    for i, cls in enumerate(unique_classes):
        if isinstance(cls, bytes):
            cls_str = cls.decode('utf-8')
        else:
            cls_str = str(cls)
        
        idx = classes == cls
        point_count = np.sum(idx)
        print(f"Plotting class: {cls_str} ({point_count} points)")
        
        color = colors[i]
        
        # Get human-readable label
        label = label_map.get(cls_str, cls_str)
        # Add point count to legend
        label_with_count = f"{label} ({point_count})"
            
        plt.scatter(
            features_tsne[idx, 0], 
            features_tsne[idx, 1], 
            c=[color], 
            # marker=marker,
            label=label_with_count,
            alpha=0.7,
            s=70
        )
    
    # plt.title(f't-SNE Visualization (perplexity={perplexity})', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save image
    output_file = os.path.join(output_dir, f'tsne_visualization_p{perplexity}_n{n_iter}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Save t-SNE coordinates and metadata
    tsne_data = pd.DataFrame({
        'filename': filenames,
        'class': classes,
        'tsne_x': features_tsne[:, 0],
        'tsne_y': features_tsne[:, 1]
    })
    
    tsne_data_file = os.path.join(output_dir, 'tsne_coordinates.csv')
    tsne_data.to_csv(tsne_data_file, index=False)
    print(f"t-SNE coordinates saved to {tsne_data_file}")
    
    # Save class statistics to file
    stats_file = os.path.join(output_dir, 'class_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Class Statistics:\n")
        f.write("=" * 20 + "\n")
        for cls, count in zip(unique_classes, class_counts):
            if isinstance(cls, bytes):
                cls_str = cls.decode('utf-8')
            else:
                cls_str = str(cls)
            f.write(f"{cls_str}: {count} points\n")
        f.write(f"Total: {len(classes)} points\n")
    print(f"Class statistics saved to {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for AST features")
    parser.add_argument("--features_file", type=str, required=True,
                       help="Path to NPZ or H5 file containing features and metadata")
    parser.add_argument("--output_dir", type=str, default="./tsne_results",
                       help="Directory to save visualization results")
    parser.add_argument("--perplexity", type=float, default=30,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--n_iter", type=int, default=1000,
                       help="Number of t-SNE iterations")
    
    args = parser.parse_args()
    visualize_tsne(args.features_file, args.output_dir, args.perplexity, args.n_iter)

if __name__ == "__main__":
    main()