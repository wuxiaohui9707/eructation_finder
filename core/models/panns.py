import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and pooling"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14(nn.Module):
    """
    PANNs Cnn14 architecture for audio classification.

    Supports loading official AudioSet-pretrained weights (Cnn14_mAP=0.431.pth).
    When pretrained_path is provided:
      - Model is first built with 527 output classes to match the checkpoint.
      - Weights are loaded (mel_bins must be 64, the official pretrained spec).
      - The final fc_audioset layer is replaced with a new Linear(2048, classes_num).

    Input : log-mel spectrogram  [Batch, 1, Time, mel_bins]
    Output: classification logits [Batch, classes_num]

    Official pretrained requirements:
        sample_rate : 32000 Hz
        mel_bins    : 64
        window_size : 1024 samples
        hop_size    : 320 samples
        fmin / fmax : 50 / 14000 Hz
    """

    # Number of output classes in the official AudioSet checkpoint
    _AUDIOSET_CLASSES = 527

    def __init__(self, classes_num, mel_bins=64, pretrained_path=None):
        super(Cnn14, self).__init__()

        self.mel_bins = mel_bins
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        # Build with AudioSet class count initially so pretrained weights load cleanly
        self.fc_audioset = nn.Linear(2048, self._AUDIOSET_CLASSES, bias=True)

        self.init_weight()

        # ── Load pretrained AudioSet weights ─────────────────────────────────
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # ── Replace final classification layer for target task ────────────────
        # Always replace so the final layer is freshly initialised for fine-tuning,
        # even when classes_num == _AUDIOSET_CLASSES.
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        init_layer(self.fc_audioset)

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _load_pretrained(self, pretrained_path):
        """
        Load PANNs official checkpoint.

        Expected checkpoint formats (both are handled):
          • {'model': state_dict, ...}   – official Zenodo format
          • bare state_dict              – plain torch.save(model.state_dict())
        """
        print(f"[PANNs] Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Unwrap nested checkpoint dict if necessary
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Validate mel_bins compatibility
        bn0_w_key = 'bn0.weight'
        if bn0_w_key in state_dict:
            pretrained_mel_bins = state_dict[bn0_w_key].shape[0]
            if pretrained_mel_bins != self.mel_bins:
                raise ValueError(
                    f"[PANNs] mel_bins mismatch: pretrained checkpoint expects "
                    f"{pretrained_mel_bins} mel bins, but model is initialised with "
                    f"{self.mel_bins}. Set --mel-bins {pretrained_mel_bins} in train.sh."
                )

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # fc_audioset will mismatch if classes differ – that's expected and fine
        unexpected_filtered = [k for k in unexpected if 'fc_audioset' not in k]
        missing_filtered    = [k for k in missing    if 'fc_audioset' not in k]
        if unexpected_filtered:
            print(f"[PANNs]  Unexpected keys (non-fc_audioset): {unexpected_filtered}")
        if missing_filtered:
            print(f"[PANNs]  Missing keys (non-fc_audioset): {missing_filtered}")
        print("[PANNs] Pretrained weights loaded successfully (fc_audioset will be re-initialised).")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, input):
        """
        Input : (batch_size, 1, time_steps, freq_bins)
        Output: (batch_size, classes_num)
        """
        x = input

        # Batch-normalise the mel-frequency axis
        x = x.transpose(1, 3)   # (B, freq, time, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (B, 1, time, freq)

        # CNN blocks with average pooling
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global temporal + freq pooling  (time-dim mean, then max+mean over time)
        x = torch.mean(x, dim=3)        # (B, 2048, time')
        (x1, _) = torch.max(x, dim=2)   # (B, 2048)
        x2 = torch.mean(x, dim=2)       # (B, 2048)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(embedding)

        return clipwise_output
