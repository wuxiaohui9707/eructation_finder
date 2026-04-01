import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_
from utils import calculate_stats, AverageMeter
from torch.nn import functional as F
import numpy as np
import os
from torchmetrics import Accuracy, AveragePrecision, AUROC, F1Score
from torch.cuda.amp import GradScaler
import csv
from pytorch_toolbelt.losses import FocalLoss

# orginal model
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    def __init__(self, label_dim=2, fstride=10, tstride=10, input_fdim=128, input_tdim=500,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384',
                 audioset_pretrain_path=None):
        super().__init__()
        self.label_dim = label_dim
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self._init_model(model_size, imagenet_pretrain, audioset_pretrain, audioset_pretrain_path)

    def _init_model(self, model_size, imagenet_pretrain, audioset_pretrain, audioset_pretrain_path):
        if not audioset_pretrain:
            self.v = self._load_pretrained_model(model_size, imagenet_pretrain)
            self._init_from_imagenet(imagenet_pretrain)
            self._init_positional_embedding(imagenet_pretrain)
        else:
            self._init_from_audioset(model_size, audioset_pretrain_path)

    def _load_pretrained_model(self, model_size, imagenet_pretrain):
        model_mapping = {
            'tiny224': 'vit_deit_tiny_distilled_patch16_224',
            'small224': 'vit_deit_small_distilled_patch16_224',
            'base224': 'vit_deit_base_distilled_patch16_224',
            'base384': 'vit_deit_base_distilled_patch16_384'
        }
        if model_size not in model_mapping:
            raise Exception('Invalid model size.')
        return timm.create_model(model_mapping[model_size], pretrained=imagenet_pretrain)

    def _init_from_imagenet(self, imagenet_pretrain):
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim), 
            nn.Linear(self.original_embedding_dim, self.label_dim)
        )
        f_dim, t_dim = self.get_shape(self.fstride, self.tstride, self.input_fdim, self.input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        self._adjust_projection_layer(imagenet_pretrain)

    def _adjust_projection_layer(self, imagenet_pretrain):
        new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
        if imagenet_pretrain:
            new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

    def _init_positional_embedding(self, imagenet_pretrain):
        f_dim, t_dim = self.get_shape(self.fstride, self.tstride, self.input_fdim, self.input_tdim)
        if imagenet_pretrain:
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2)
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, f_dim * t_dim).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            self.v.pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            trunc_normal_(self.v.pos_embed, std=.02)

    def _init_from_audioset(self, model_size, audioset_pretrain_path):
        """
        Load weights from a locally saved AudioSet-pretrained AST checkpoint
        (e.g. audioset_10_10_0.4593.pth from MIT PSDS).
        That checkpoint was trained with fstride=10, tstride=10, input_tdim=1024
        on 527 AudioSet classes.  We:
          1. Build the backbone the same way as ImageNet init (pretrained=True)
             so patch-embed and pos-embed are already adapted to our input shape.
          2. Overwrite every matching weight from the AudioSet checkpoint.
          3. Re-initialise the classification head for our own label_dim.
        """
        if not os.path.exists(audioset_pretrain_path):
            raise FileNotFoundError(
                f'AudioSet pretrain weight file not found: {audioset_pretrain_path}')

        # Step 1: build the backbone architecture only (no ImageNet download needed).
        #         The AudioSet checkpoint already contains the fully-trained weights
        #         (originally trained from DeiT-ImageNet and fine-tuned on AudioSet),
        #         so we just need the model structure here; weights get overwritten below.
        self.v = self._load_pretrained_model(model_size, imagenet_pretrain=False)
        self._init_from_imagenet(imagenet_pretrain=False)
        self._init_positional_embedding(imagenet_pretrain=False)

        # Step 2: load AudioSet checkpoint and overwrite backbone weights
        print(f'Loading AudioSet pretrained weights from {audioset_pretrain_path}')
        audioset_sd = torch.load(audioset_pretrain_path, map_location='cpu')

        # The checkpoint may be wrapped under 'audio_model' or stored flat
        if isinstance(audioset_sd, dict) and 'audio_model' in audioset_sd:
            audioset_sd = audioset_sd['audio_model']

        # Build a mapping: strip the leading 'module.' prefix if present
        audioset_sd = {
            (k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in audioset_sd.items()
        }

        # Our model's state_dict uses 'v.*' for the backbone and 'mlp_head.*' for the head
        own_sd = self.state_dict()
        matched, skipped = 0, 0
        new_sd = {}
        for own_key in own_sd:
            if own_key.startswith('mlp_head'):
                # Always keep our own randomly-initialised head
                new_sd[own_key] = own_sd[own_key]
                continue
            # Strip 'v.' prefix to form the AudioSet key
            if own_key.startswith('v.'):
                ast_key = own_key[len('v.'):]
            else:
                ast_key = own_key

            if ast_key in audioset_sd and audioset_sd[ast_key].shape == own_sd[own_key].shape:
                new_sd[own_key] = audioset_sd[ast_key]
                matched += 1
            else:
                new_sd[own_key] = own_sd[own_key]
                skipped += 1

        self.load_state_dict(new_sd)
        print(f'AudioSet weights loaded: {matched} matched, {skipped} skipped/shape-mismatch.')

        # Step 3: re-initialise only the classification head for our label_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, self.label_dim)
        )

    def forward(self, x):
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
        return self.mlp_head(x)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=500):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list

class ASTLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # CSV file path
        self.csv_file = os.path.join(self.hparams.exp_dir, 'validation_results.csv')
        self._init_csv_file()  # create and write CSV header

        # Create AST model
        self.model = ASTModel(
            label_dim=self.hparams.label_dim,
            fstride=self.hparams.fstride,
            tstride=self.hparams.tstride,
            input_fdim=self.hparams.input_fdim,
            input_tdim=self.hparams.input_tdim,
            model_size=self.hparams.model_size,
            imagenet_pretrain=self.hparams.imagenet_pretrain,
            audioset_pretrain=self.hparams.audioset_pretrain,
            audioset_pretrain_path=self.hparams.audioset_pretrain_path,
        )

        # Loss function: BCE for binary, CE for multi-class
        self.loss_fn = self._get_adaptive_loss()

        # Initialize metrics
        self._init_metrics()
        self.scaler = GradScaler()
        self.train_step_outputs = []
        self.val_step_outputs = []

        # For best model tracking
        self.best_metric_value = 0.0
        self.best_model_path = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        audio_input, labels = batch
        outputs = self(audio_input)
        loss = self._calculate_loss(outputs, labels)

        self.train_loss.update(loss.item(), audio_input.size(0))
        self.log('train_loss', self.train_loss.avg, prog_bar=True)

        self.train_step_outputs.append({
            'loss': loss.detach(),
            'preds': outputs.detach(),
            'labels': labels.detach()
        })
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch < self.hparams.warmup_epochs and self.hparams.warmup:
            lr_scale = min(1.0, (self.current_epoch + 1) / self.hparams.warmup_epochs)
            for pg in self.optimizers().param_groups:
                pg['lr'] = lr_scale * self.hparams.lr


    def on_train_epoch_end(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)

        all_preds = torch.cat([x['preds'] for x in self.train_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.train_step_outputs])

        # convert one-hot to class index
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            all_labels = torch.argmax(all_labels, dim=1)  # one-hot to class index

        # calculate metrics
        if self.hparams.label_dim <= 2:  # binary
            all_preds = torch.sigmoid(all_preds)
            train_f1_macro = self.val_f1_macro((all_preds > 0.5).int(), all_labels)
        else:  # multi-class
            train_f1_macro = self.val_f1_macro(torch.argmax(all_preds, dim=1), all_labels)

        # log metrics
        self.log('train_f1_macro', train_f1_macro, prog_bar=True)
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        audio_input, labels = batch
        outputs = self(audio_input)
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            pass
        else:
            labels = labels.float()
        loss = self._calculate_loss(outputs, labels)

        # Update val loss
        self.val_loss.update(loss.item(), audio_input.size(0))
        self.log('val_loss', self.val_loss.avg, prog_bar=True)

        # Compute probabilities and predictions
        if self.hparams.label_dim <= 2:
            # Binary classification => sigmoid
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
        else:
            # Multi-class => softmax
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

        # Update each metric
        self.val_acc.update(preds, labels.long())
        self.val_map.update(probs, labels.long())
        self.val_f1_macro.update(preds, labels.long())
        self.val_auc.update(probs, labels.long())

        self.val_step_outputs.append({'loss': loss.detach()})
        return loss

    def on_validation_epoch_end(self):
        # Compute all metrics
        val_acc = self.val_acc.compute()
        val_mAP = self.val_map.compute()
        val_f1_macro = self.val_f1_macro.compute()
        val_auc = self.val_auc.compute()

        # Print logs
        print(f"\nValidation - Epoch {self.current_epoch}")
        print(f"val_loss: {self.val_loss.avg:.4f}")
        print(f"val_acc:  {val_acc:.4f}")
        print(f"val_mAP:  {val_mAP:.4f}")
        print(f"val_f1_macro:  {val_f1_macro:.4f}")
        print(f"val_auc:  {val_auc:.4f}")

        # Log metrics to TensorBoard
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_mAP', val_mAP, prog_bar=False)
        self.log('val_f1_macro', val_f1_macro, prog_bar=False)
        self.log('val_auc', val_auc, prog_bar=False)

        # Choose which metric to watch for best model saving
        if self.hparams.watch_metric == 'acc':
            metric_val = val_acc
        elif self.hparams.watch_metric == 'mAP':
            metric_val = val_mAP
        elif self.hparams.watch_metric == 'f1_macro':
            metric_val = val_f1_macro
        elif self.hparams.watch_metric == 'auc':
            metric_val = val_auc
        else:
            raise ValueError("watch_metric must be one of ['acc','mAP','f1_macro','auc']")

        # If the current metric is better than best_metric_value, save model
        if metric_val > self.best_metric_value:
            self.best_metric_value = metric_val
            self._save_best_model(metric_val)

        # Save validation results to CSV (acc, mAP, f1, auc, etc.)
        self._save_validation_results_to_csv(
            epoch=self.current_epoch,
            val_loss=self.val_loss.avg,
            val_acc=val_acc.item(),
            val_mAP=val_mAP.item(),
            val_f1_macro=val_f1_macro.item(),
            val_auc=val_auc.item()
        )

        # Reset all validation metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_map.reset()
        self.val_f1_macro.reset()
        self.val_auc.reset()
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-7,
            betas=(0.95, 0.999)
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(range(self.hparams.lrscheduler_start, 1000, self.hparams.lrscheduler_step)),
                gamma=self.hparams.lrscheduler_decay
            ),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def _init_csv_file(self):
        """
        Initialize the CSV file and write the header row.
        """
        if not os.path.exists(self.hparams.exp_dir):
            os.makedirs(self.hparams.exp_dir)

        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'val_loss', 'val_acc', 'val_mAP', 'val_f1_macro', 'val_auc'])

    def _save_best_model(self, metric_val):
        """
        Save the current model checkpoint if the chosen watch_metric is improved.
        Remove the previous best checkpoint if exists.
        """
        ckpt_path = os.path.join(
            self.hparams.exp_dir,
            f'best_model_epoch{self.current_epoch}_{self.hparams.watch_metric}{metric_val:.4f}.ckpt'
        )
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)

        torch.save(self.model.state_dict(), ckpt_path)
        self.best_model_path = ckpt_path
        print(f'Saved best model by [{self.hparams.watch_metric}={metric_val:.4f}] at {ckpt_path}')

    def _save_validation_results_to_csv(self, epoch, val_loss, val_acc, val_mAP, val_f1_macro, val_auc):
        """
        Save validation metrics of the current epoch to the CSV file.
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, val_loss, val_acc, val_mAP, val_f1_macro, val_auc])

    def _calculate_loss(self, outputs, labels):
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            return self.loss_fn(outputs, torch.argmax(labels.long(), dim=1))
        return self.loss_fn(outputs, labels)

    def _get_adaptive_loss(self):
        if self.hparams.label_dim <= 2:  # binary
            return nn.BCEWithLogitsLoss()
        else:  # multi-class
            return nn.CrossEntropyLoss()

    def _init_metrics(self):
        """initialize metrics"""
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()

        if self.hparams.label_dim <= 2:
            # binary
            self.val_acc = Accuracy(task='binary')
            self.val_map = AveragePrecision(task='binary')
            self.val_f1_macro = F1Score(task='binary', average='macro')
            self.val_auc = AUROC(task='binary')
        else:
            # multi-class
            self.val_acc = Accuracy(task='multiclass', num_classes=self.hparams.label_dim)
            self.val_map = AveragePrecision(task='multiclass', num_classes=self.hparams.label_dim)
            self.val_f1_macro = F1Score(task='multiclass', num_classes=self.hparams.label_dim, average='macro')
            self.val_auc = AUROC(task='multiclass', num_classes=self.hparams.label_dim)