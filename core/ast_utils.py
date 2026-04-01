import numpy as np
from scipy import stats
from sklearn import metrics
from scipy import signal
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def low_pass_filter(waveform_np, sample_rate, cutoff=1024):
    """
    Apply a low-pass filter to an audio signal.
    """
    nyquist = 0.5 * sample_rate
    cutoff_freq = cutoff / nyquist
    
    # Butterworth filter with order 5
    b, a = signal.butter(5, cutoff_freq, btype='low', analog=False)
    
    filtered_waveform_np = signal.filtfilt(b, a, waveform_np)
    
    return filtered_waveform_np

def calculate_stats(output, target):
    """Calculate comprehensive statistics including mAP, AUC, F1, Recall, Precision
    
    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array (multi-label) or 1d array (single-label)
    
    Returns:
      stats_dict: {
        'global': {
            'accuracy': ...,
            'f1_macro': ...,
            'recall_macro': ...,
            'precision_macro': ...,
            'auc_ovo': ...
        },
        'classes': [
            {
                'AP': ..., 
                'AUC': ..., 
                'F1': ..., 
                'Recall': ..., 
                'Precision': ...
            },
            ...
        ]
      }
    """
    # print("output type: ", type(output),'shape:',output.shape) 
    # print("target type: ", type(target),'shape:',target.shape)

    # Convert to numpy if torch.Tensor
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    
    # Convert target to proper format
    if target.ndim == 1:  # Single-label
        classes_num = output.shape[1]
        target_onehot = np.eye(classes_num)[target]
        true_labels = target
    else:  # Multi-label
        target_onehot = target
        true_labels = np.argmax(target, axis=1) if target.shape[1] > 1 else target[:, 0]
    
    pred_labels = np.argmax(output, axis=1)
    pred_probs = torch.nn.functional.softmax(torch.Tensor(output), dim=1).numpy()

    stats_dict = {
        'global': {},
        'classes': []
    }

    # Global metrics
    stats_dict['global']['accuracy'] = metrics.accuracy_score(true_labels, pred_labels)
    stats_dict['global']['f1_macro'] = metrics.f1_score(true_labels, pred_labels, average='macro')
    stats_dict['global']['recall_macro'] = metrics.recall_score(true_labels, pred_labels, average='macro')
    stats_dict['global']['precision_macro'] = metrics.precision_score(true_labels, pred_labels, average='macro')
    
    # For binary classification, calculate AUC with ovo
    if pred_probs.shape[1] == 2:
        stats_dict['global']['auc_ovo'] = metrics.roc_auc_score(
            true_labels, pred_probs[:, 1], multi_class='ovo'
        )
    else:
        stats_dict['global']['auc_ovo'] = metrics.roc_auc_score(
            true_labels, pred_probs, multi_class='ovo', average='macro'
        )

    # Per-class metrics
    for k in range(output.shape[1]):
        # Handle multi-label vs single-label
        if target_onehot.shape[1] > 1:
            y_true = target_onehot[:, k]
        else:
            y_true = (true_labels == k).astype(int)

        y_prob = output[:, k]
        
        # Avoid NaN when no positive samples
        if np.sum(y_true) == 0:
            class_stats = {
                'AP': 0.0,
                'AUC': 0.0,
                'F1': 0.0,
                'Recall': 0.0,
                'Precision': 0.0
            }
        else:
            # Threshold-based metrics
            y_pred = (y_prob >= 0.5).astype(int)
            
            # AUC and AP
            try:
                auc = metrics.roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = 0.0
                
            ap = metrics.average_precision_score(y_true, y_prob)

            # F1, Recall, Precision
            f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
            recall = metrics.recall_score(y_true, y_pred, zero_division=0)
            precision = metrics.precision_score(y_true, y_pred, zero_division=0)

            class_stats = {
                'AP': ap,
                'AUC': auc,
                'F1': f1,
                'Recall': recall,
                'Precision': precision
            }

        stats_dict['classes'].append(class_stats)

    return stats_dict