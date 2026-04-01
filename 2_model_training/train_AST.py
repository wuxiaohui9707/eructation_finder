import argparse
import os
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import os
import glob

# Add project root to python path to allow importing from core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.data_module import AudioDataModule
from core.models.ast_model import ASTLightning
from core.ast_eval_test import evaluate_test_set

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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data arguments
    parser.add_argument("--data-train", type=str, required=True, help="training data json")
    parser.add_argument("--data-val", type=str, required=True, help="validation data json")
    parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
    parser.add_argument("--label-csv", type=str, required=True, help="csv with class labels")
    parser.add_argument("--label-dim", type=int, default=2, help="number of classes")
    parser.add_argument("--resample", type=str2bool, default=False, help="resample audio")
    parser.add_argument("--sample-rate", type=int, default=16000, help="target sample rate")
    
    # Model arguments
    parser.add_argument("--model-size", type=str, default='base384', help="model size configuration (e.g., base384)")
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used") # didn't use this
    parser.add_argument("--fstride", type=int, default=10, help="frequency stride")
    parser.add_argument("--tstride", type=int, default=10, help="time stride")
    parser.add_argument("--input-tdim", type=int, default=500, help="input time dimension")
    parser.add_argument("--input-fdim", type=int, default=128, help="input frequency dimension")
    parser.add_argument("--imagenet-pretrain", type=str2bool, default=True, help="use ImageNet pretrained weights")
    parser.add_argument("--audioset-pretrain", type=str2bool, default=False, help="use AudioSet pretrained weights")
    parser.add_argument("--audioset-pretrain-path", type=str, default='',
                        help="path to AudioSet pretrained weight file (.pth)")
    parser.add_argument("--fp16", type=str2bool, default=False, help="enable mixed precision training")
    
    # Training arguments
    parser.add_argument("--exp-dir", type=str, required=True, help="experiment directory")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--warmup", type=bool, default=False, help="warmup or not")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="number of warmup epochs")
    parser.add_argument("--lrscheduler-start", type=int, default=0, help="epoch to start lr scheduler")
    parser.add_argument("--lrscheduler-step", type=int, default=1, help="lr scheduler step")
    parser.add_argument("--lrscheduler-decay", type=float, default=0.5, help="lr scheduler decay")
    
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
    parser.add_argument("--watch-metric", type=str, default="mAP", help="evaluation metric")
    parser.add_argument("--loss-fn", type=str, default=None, help="loss function")

    # Evaluatie test set
    parser.add_argument("--eval-test", type=str2bool, default=False, help="evaluate test set")
    parser.add_argument("--plot-attention", type=str2bool, default=False, help="plot attention maps")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Create experiment directory
    os.makedirs(args.exp_dir, exist_ok=True)
    with open(f"{args.exp_dir}/args.pkl", "wb") as f:
        pickle.dump(args, f)
        
    # Configure audio processing
    train_audio_conf = {
        'num_mel_bins': args.input_fdim,
        'target_length': args.input_tdim,
        'freqm': args.freqm,
        'timem': args.timem,
        'mixup': args.mixup,
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': args.noise,
        'resample': args.resample,
        'sample_rate': args.sample_rate,
        'freq_division_mode': args.freq_division_mode,
        'split_freq': args.split_freq,
    }
    
    val_audio_conf = {
        'num_mel_bins': args.input_fdim,
        'target_length': args.input_tdim,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False,
        'resample': args.resample,
        'sample_rate': args.sample_rate,
        'freq_division_mode': args.freq_division_mode,
        'split_freq': args.split_freq,
    }
    
    # Initialize data module
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
    
    # Initialize model
    model = ASTLightning(args)

    monitor_metric = f"val_{model.hparams.watch_metric}"

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.exp_dir}/checkpoints",
        filename="best_model",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=15,
        mode="max",
        check_on_train_epoch_end=False,
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  
        devices=1 if torch.cuda.is_available() else 0,              
        logger=TensorBoardLogger(args.exp_dir, name="lightning_logs"),
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed" if args.fp16 else "32-true",
        enable_progress_bar=False,
        num_sanity_val_steps=0
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    #get the best model file path
    best_model_path = glob.glob(f"{args.exp_dir}/checkpoints/best_model*.ckpt")[0]

    # Save final model
    #trainer.save_checkpoint(f"{args.exp_dir}/checkpoints/final_model.ckpt")

    # Evaluate test set
    if args.eval_test:
        evaluate_test_set(args, f"{args.exp_dir}/checkpoints/best_model.ckpt", args.data_eval, args.label_csv, args.exp_dir)

if __name__ == "__main__":
    main()
