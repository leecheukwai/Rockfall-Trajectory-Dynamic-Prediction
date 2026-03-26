import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout=0.2):
        super().__init__()
        # Two weight-norm convolutions with BatchNorm, ReLU, Dropout
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=1, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=1, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class ConvPyramidTCN(nn.Module):
    def __init__(self, in_channels=3, channel_sizes=None,
                 kernel_size=3, dropout=0.2, pyramid_kernel=3):
        super().__init__()
        if channel_sizes is None:
            channel_sizes = [16, 32, 64]
        self.levels = len(channel_sizes)
        self.blocks = nn.ModuleList()
        self.pyramid_convs = nn.ModuleList()
        # Build pyramid: channels increase, length halves
        for i, out_ch in enumerate(channel_sizes):
            in_ch = in_channels if i == 0 else channel_sizes[i-1]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            # Temporal convolution block
            self.blocks.append(
                TemporalBlock(in_ch, out_ch,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=padding,
                              dropout=dropout)
            )
            # Strided conv to halve length
            self.pyramid_convs.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(out_ch, out_ch,
                                  kernel_size=pyramid_kernel,
                                  stride=2,
                                  padding=pyramid_kernel//2)
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU()
                )
            )
        # Final output channels = last of channel_sizes
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, in_channels, L]
        for block, pyramid_conv in zip(self.blocks, self.pyramid_convs):
            x = block(x)
            x = pyramid_conv(x)

        # After all levels, channels = channel_sizes[-1], length ≈ L / 2^levels
        x = self.global_pool(x)  # -> [B, final_channels, 1]
        return x

# Test
if __name__ == "__main__":
    B, L = 8, 600
    x = torch.randn(B, 3, L)
    model = ConvPyramidTCN(in_channels=3,
                           channel_sizes=[8, 16, 32, 64],
                           kernel_size=3,
                           dropout=0.1,
                           pyramid_kernel=3)
    y = model(x)
    print(y.shape)  # torch.Size([8, 64, 1])
