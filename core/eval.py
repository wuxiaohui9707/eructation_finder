import numpy as np
from sklearn import metrics


def evaluate(predictions, targets):
    """
    Calculate evaluation metrics for audio classification.

    Args:
        predictions: numpy array of shape (N, num_classes) - model logits or probabilities
                     Column 0 = burp (/m/POS), Column 1 = nonburp (/m/NEG)
        targets:     numpy array of shape (N, num_classes) - ground truth one-hot labels

    Returns:
        dict with overall (macro) metrics and per-class (burp / nonburp) metrics.
    """
    # Convert logits to probabilities using sigmoid
    if predictions.max() > 1 or predictions.min() < 0:
        predictions = 1 / (1 + np.exp(-predictions))  # sigmoid

    # ── mAP ──────────────────────────────────────────────────────────────────
    try:
        average_precision = metrics.average_precision_score(
            targets, predictions, average='macro'
        )
    except Exception:
        average_precision = 0.0

    # ── AUC ──────────────────────────────────────────────────────────────────
    try:
        auc = metrics.roc_auc_score(targets, predictions, average='macro')
    except Exception:
        auc = 0.0

    # Binary predictions (threshold = 0.5)
    predictions_binary = (predictions > 0.5).astype(int)

    # ── Accuracy ─────────────────────────────────────────────────────────────
    accuracy = metrics.accuracy_score(targets, predictions_binary)

    # ── Macro precision / recall / F1 ────────────────────────────────────────
    try:
        precision = metrics.precision_score(targets, predictions_binary, average='macro', zero_division=0)
        recall    = metrics.recall_score(   targets, predictions_binary, average='macro', zero_division=0)
        f1        = metrics.f1_score(       targets, predictions_binary, average='macro', zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0

    # ── Per-class precision / recall / F1 ────────────────────────────────────
    # Column order matches label_index.csv: 0 = burp, 1 = nonburp
    try:
        prec_cls = metrics.precision_score(targets, predictions_binary, average=None, zero_division=0)
        rec_cls  = metrics.recall_score(   targets, predictions_binary, average=None, zero_division=0)
        f1_cls   = metrics.f1_score(       targets, predictions_binary, average=None, zero_division=0)
        precision_burp,    precision_nonburp    = float(prec_cls[0]), float(prec_cls[1])
        recall_burp,       recall_nonburp       = float(rec_cls[0]),  float(rec_cls[1])
        f1_burp,           f1_nonburp           = float(f1_cls[0]),   float(f1_cls[1])
    except Exception:
        precision_burp = precision_nonburp = 0.0
        recall_burp    = recall_nonburp    = 0.0
        f1_burp        = f1_nonburp        = 0.0

    stats = {
        # ── Overall (macro) ───────────────────────────────────────────────────
        'AP':                average_precision,
        'auc':               auc,
        'accuracy':          accuracy,
        'precision':         precision,        # macro
        'recall':            recall,           # macro
        'f1':                f1,               # macro
        # ── Positive class: burp ──────────────────────────────────────────────
        'precision_burp':    precision_burp,
        'recall_burp':       recall_burp,
        'f1_burp':           f1_burp,
        # ── Negative class: nonburp ───────────────────────────────────────────
        'precision_nonburp': precision_nonburp,
        'recall_nonburp':    recall_nonburp,
        'f1_nonburp':        f1_nonburp,
    }

    return stats


def d_prime(auc):
    """
    Calculate d-prime (d') from AUC.
    d' is a measure of sensitivity index in signal detection theory.
    """
    if auc >= 1.0:
        return np.inf
    if auc <= 0.0:
        return -np.inf

    from scipy.stats import norm
    d_prime = norm.ppf(auc) * np.sqrt(2.0)
    return d_prime
