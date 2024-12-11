import torch.nn as nn

from .parts import (
    DownsampleBlock,
    ResidualBlock,
    UpsampleBlock,
    UpsampleBlockWithAttention,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features=[64, 128, 256, 512],
        concatenate_features=False,
        use_logits=True,
    ):
        super().__init__()
        self.concatenate_features = concatenate_features
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_logits = use_logits
        self.sigmoid = nn.Sigmoid()

        # Down part of UNet
        for feature in features:
            self.downs.append(DownsampleBlock(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                UpsampleBlock(
                    feature * 2,
                    feature,
                    concatenate_features=self.concatenate_features,
                )
            )

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for i, up in enumerate(self.ups):
            skip = skip_connections[i]
            x = up(x, skip)

        x = self.final_conv(x)

        if self.use_logits:
            return x
        else:
            return self.sigmoid(x)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features=[64, 128, 256, 512],
        concatenate_features=False,
    ):
        super().__init__()
        self.concatenate_features = concatenate_features
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DownsampleBlock(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                UpsampleBlockWithAttention(
                    feature * 2,
                    feature,
                    concatenate_features=self.concatenate_features,
                ),
            )

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for i, up in enumerate(self.ups):
            skip = skip_connections[i]
            x = up(x, skip)

        return self.final_conv(x)
