"""
run.py — Training script for PANNs (Cnn14) audio classification.

Uses PyTorch Lightning for training with cow-independent cross-validation.

Usage:
    python run.py \
        --data-train <train.json> \
        --data-val <val.json> \
        --data-eval <test.json> \
        --label-csv <label_index.csv> \
        --exp-dir <output_dir> \
        [--mel-bins 64] \
        [--panns-pretrained-path <path_to_Cnn14_mAP=0.431.pth>]
"""

import argparse
import sys
import os
import json
import pickle
import torch
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.data_module import AudioDataModule
from core.models.models import get_model_class
from core.eval import evaluate
import glob
import pandas as pd
import numpy as np


class PANNsLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for PANNs (Cnn14) audio classification."""

    def __init__(self, model_class, args):
        super().__init__()
        self.save_hyperparameters(args)

        # PANNs (Cnn14) constructor
        kwargs = {
            'classes_num': args.label_dim,
            'mel_bins': args.mel_bins,
            'pretrained_path': getattr(args, 'panns_pretrained_path', None) or None,
        }

        self.model = model_class(**kwargs)

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # For collecting predictions
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        """Forward pass wrapper."""
        # Ensure input has channel dimension [Batch, 1, Time, Freq]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        fbank, labels = batch
        logits = self(fbank)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fbank, labels = batch
        logits = self(fbank)
        loss = self.loss_fn(logits, labels)

        self.validation_step_outputs.append({
            'logits': logits.detach().cpu(),
            'labels': labels.detach().cpu()
        })

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self._calculate_metrics(self.validation_step_outputs, prefix='val')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        fbank, labels = batch
        logits = self(fbank)
        loss = self.loss_fn(logits, labels)

        self.test_step_outputs.append({
            'logits': logits.detach().cpu(),
            'labels': labels.detach().cpu()
        })

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self._calculate_metrics(self.test_step_outputs, prefix='test')
        self.test_step_outputs.clear()

    def _calculate_metrics(self, step_outputs, prefix):
        if not step_outputs:
            return

        all_logits = torch.cat([x['logits'] for x in step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in step_outputs], dim=0)

        stats = evaluate(all_logits.numpy(), all_labels.numpy())

        self.log(f'{prefix}_mAP', stats['AP'], prog_bar=True)
        self.log(f'{prefix}_AUC', stats['auc'], prog_bar=True)
        self.log(f'{prefix}_accuracy', stats['accuracy'], prog_bar=True)
        self.log(f'{prefix}_f1', stats['f1'], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.0,
            betas=(0.95, 0.999)
        )

        if hasattr(self.hparams, 'warmup_epochs') and self.hparams.warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < self.hparams.warmup_epochs:
                    return (epoch + 1) / self.hparams.warmup_epochs
                else:
                    decay_epochs = epoch - self.hparams.warmup_epochs
                    decay_epochs = max(0, decay_epochs - self.hparams.lrscheduler_start)
                    num_decays = decay_epochs // self.hparams.lrscheduler_step
                    return self.hparams.lrscheduler_decay ** num_decays

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
            }
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.lrscheduler_step,
                gamma=self.hparams.lrscheduler_decay
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
            }


class PrintMetricsCallback(Callback):
    """Custom callback to print metrics at the end of each epoch."""
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        metrics_str = f"Epoch {current_epoch}: "
        if 'train_loss_epoch' in metrics:
            metrics_str += f"train_loss={metrics['train_loss_epoch']:.4f} "
        print(metrics_str)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        val_metrics = []
        for k, v in metrics.items():
            if k.startswith('val_'):
                val_metrics.append(f"{k}={v:.4f}")

        if val_metrics:
            print(f"Epoch {current_epoch} validation: " + " | ".join(val_metrics))
            print("-" * 80)


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(
        description='PANNs (Cnn14) Training for Audio Classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data-train", type=str, required=True, help="training data json")
    parser.add_argument("--data-val", type=str, required=True, help="validation data json")
    parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
    parser.add_argument("--label-csv", type=str, required=True, help="csv with class labels")
    parser.add_argument("--label-dim", type=int, default=2, help="number of classes")
    parser.add_argument("--resample", type=str2bool, default=False, help="resample audio")
    parser.add_argument("--sample-rate", type=int, default=16000, help="target sample rate")
    parser.add_argument("--filter", type=str2bool, default=False, help="apply low-pass filter")
    parser.add_argument("--cutoff-freq", type=int, default=1024, help="cutoff frequency")

    # PANNs model arguments
    parser.add_argument("--mel-bins", type=int, default=64,
                        help="number of mel bins (64 for official Cnn14 pretrained)")
    parser.add_argument("--target-length", type=int, default=500, help="target time length")
    parser.add_argument("--panns-pretrained-path", type=str, default='',
                        help="path to official Cnn14 AudioSet checkpoint (.pth); "
                             "blank = train from scratch. Requires --mel-bins 64.")

    # Training arguments
    parser.add_argument("--exp-dir", type=str, required=True, help="experiment directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="number of warmup epochs")
    parser.add_argument("--lrscheduler-start", type=int, default=0, help="epoch to start lr scheduler")
    parser.add_argument("--lrscheduler-step", type=int, default=1, help="lr scheduler step")
    parser.add_argument("--lrscheduler-decay", type=float, default=0.5, help="lr scheduler decay")
    parser.add_argument("--min-epochs", type=int, default=1, help="minimum number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--fp16", type=str2bool, default=False, help="enable mixed precision training")

    # Audio processing arguments
    parser.add_argument("--freqm", type=int, default=0, help="frequency mask length")
    parser.add_argument("--timem", type=int, default=0, help="time mask length")
    parser.add_argument("--mixup", type=float, default=0, help="mixup ratio")
    parser.add_argument("--dataset-mean", type=float, default=-4.2677393, help="spectrogram mean")
    parser.add_argument("--dataset-std", type=float, default=4.5689974, help="spectrogram std")
    parser.add_argument("--noise", type=str2bool, default=False, help="add noise")
    parser.add_argument("--freq-division-mode", type=str, default='uniform', choices=['uniform', 'split_1khz'], help="frequency division mode: 'uniform' or 'split_1khz'")
    parser.add_argument("--split-freq", type=int, default=1000, help="split frequency for split_1khz mode (Hz)")

    # Evaluation arguments
    parser.add_argument("--watch-metric", type=str, default="mAP", help="evaluation metric to monitor")
    parser.add_argument("--eval-test", type=str2bool, default=False, help="evaluate test set after training")
    parser.add_argument("--eval-only", type=str2bool, default=False,
                        help="skip training, only re-evaluate using existing checkpoint")

    # Keep model argument for compatibility with shell scripts
    parser.add_argument("--model", type=str, default='panns', help="model architecture (panns)")

    args = parser.parse_args()
    return args


def evaluate_test_set(args, checkpoint_path, test_json, label_csv, output_dir):
    """Independent test set evaluation function."""
    print(f"\n{'='*60}")
    print(f"Evaluating test set: {test_json}")

    model_class = get_model_class(args.model)
    pl_module = PANNsLightningModule.load_from_checkpoint(
        checkpoint_path,
        model_class=model_class,
        args=args
    )
    pl_module.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pl_module = pl_module.to(device)

    # Load test data
    with open(test_json, 'r') as fp:
        test_data = json.load(fp)['data']

    # Config
    audio_conf = {
        'num_mel_bins': args.mel_bins,
        'target_length': args.target_length,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False,
        'resample': args.resample,
        'sample_rate': args.sample_rate,
        'filter': args.filter,
        'cutoff_freq': args.cutoff_freq,
        'freq_division_mode': args.freq_division_mode,
        'split_freq': args.split_freq,
    }
    from core.data_module import AudioDataset
    test_dataset = AudioDataset(test_data, audio_conf, label_csv)

    # Inference loop
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            data, label = test_dataset[i]

            # Mel-spec: (Time, Freq) -> (1, Time, Freq) if needed
            if data.ndim == 2:
                data = data.unsqueeze(0)
            data = data.unsqueeze(0).to(device)  # Add batch dim

            logits = pl_module(data)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(label.numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    stats = evaluate(all_logits, all_labels)

    # Save results
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=4)

    csv_path = os.path.join(output_dir, 'test_results.csv')
    pd.DataFrame([stats]).to_csv(csv_path, index=False)

    print(f"Test Results saved to {output_dir}")
    return stats


def main():
    args = get_args()

    # ── Eval-only mode: load existing checkpoint, skip training ───────────
    if args.eval_only:
        ckpt_list = glob.glob(f"{args.exp_dir}/checkpoints/best_model*.ckpt")
        if not ckpt_list:
            print(f"[eval-only] No checkpoint found in {args.exp_dir}/checkpoints/ — skipping.")
            return
        print(f"[eval-only] Using checkpoint: {ckpt_list[0]}")
        evaluate_test_set(args, ckpt_list[0], args.data_eval, args.label_csv, args.exp_dir)
        return

    os.makedirs(args.exp_dir, exist_ok=True)
    with open(f"{args.exp_dir}/args.pkl", "wb") as f:
        pickle.dump(args, f)

    # Audio Configs (mel-spectrogram based)
    train_audio_conf = {
        'num_mel_bins': args.mel_bins,
        'target_length': args.target_length,
        'freqm': args.freqm,
        'timem': args.timem,
        'mixup': args.mixup,
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': args.noise,
        'resample': args.resample,
        'sample_rate': args.sample_rate,
        'filter': args.filter,
        'cutoff_freq': args.cutoff_freq,
        'freq_division_mode': args.freq_division_mode,
        'split_freq': args.split_freq,
    }

    val_audio_conf = train_audio_conf.copy()
    val_audio_conf.update({'freqm': 0, 'timem': 0, 'mixup': 0, 'noise': False})

    # Data Module
    data_module = AudioDataModule(
        train_json_file=args.data_train,
        val_json_file=args.data_val,
        test_json_file=args.data_eval if args.data_eval else args.data_val,
        label_csv=args.label_csv,
        audio_conf=train_audio_conf,
        eval_audio_conf=val_audio_conf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model_class = get_model_class(args.model)
    pl_module = PANNsLightningModule(model_class, args)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.exp_dir}/checkpoints",
        filename="best_model",
        monitor=f"val_{args.watch_metric}",
        mode="max",
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor=f"val_{args.watch_metric}",
        patience=args.patience,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        min_epochs=args.min_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 0,
        logger=TensorBoardLogger(args.exp_dir, name="lightning_logs"),
        callbacks=[checkpoint_callback, early_stop_callback, PrintMetricsCallback()],
        precision="16-mixed" if args.fp16 else "32-true",
        enable_progress_bar=True
    )

    print(f"\nTraining PANNs | LR: {args.lr} | Batch: {args.batch_size}")
    trainer.fit(pl_module, data_module)

    # Test
    best_model_path = glob.glob(f"{args.exp_dir}/checkpoints/best_model*.ckpt")[0]
    if args.eval_test:
        evaluate_test_set(args, best_model_path, args.data_eval, args.label_csv, args.exp_dir)

if __name__ == "__main__":
    main()
