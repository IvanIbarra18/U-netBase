# -*- coding: utf-8 -*-
"""U-Net

author: Masahiro Hayashi

This script defines the model architecture of U-Net.
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by optional BatchNorm and ReLU."""

    def __init__(self,in_channels: int,out_channels: int,batch_norm: bool = False,) -> None:
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetClassic(nn.Module):
    """Implementation of the U-Net architecture
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self,n_classes: int,n_channels: int = 1,batch_norm: bool = False,) -> None:
        super().__init__()

        if n_classes < 1 or n_channels < 1:
            raise ValueError("n_classes and n_channels must be positive.")

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.batch_norm = batch_norm
        self.filter_sizes = [64, 128, 256, 512, 1024]

        # Encoder: the first block does not use max pooling.
        encoder_blocks = []
        in_channels = n_channels

        for i, out_channels in enumerate(self.filter_sizes):
            layers = []

            if i > 0:
                layers.append(nn.MaxPool2d(kernel_size=2))

            layers.append(DoubleConv(in_channels,out_channels,batch_norm=batch_norm,))

            encoder_blocks.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.encoder = nn.ModuleList(encoder_blocks)

        # Decoder: upsample, concatenate the skip connection,
        # then apply two convolutions.
        decoder_blocks = []
        in_channels = self.filter_sizes[-1]

        for out_channels in reversed(self.filter_sizes[:-1]):
            decoder_blocks.append(
                nn.ModuleDict({
                    "up": nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,),
                    "conv": DoubleConv(in_channels,out_channels,batch_norm=batch_norm,),
                })
            )

            in_channels = out_channels

        self.decoder = nn.ModuleList(decoder_blocks)

        # Map the final feature maps to the requested output channels.
        self.segment = nn.Conv2d(
            self.filter_sizes[0],
            n_classes,
            kernel_size=1,
        )

    @staticmethod
    def center_crop(encoder_features: torch.Tensor,target: torch.Tensor,) -> torch.Tensor:
        """Center-crop encoder features to match decoder dimensions."""

        diff_h = encoder_features.size(2) - target.size(2)
        diff_w = encoder_features.size(3) - target.size(3)

        if diff_h < 0 or diff_w < 0:
            raise ValueError(
                f"Encoder features {tuple(encoder_features.shape)} "
                f"are smaller than decoder features {tuple(target.shape)}."
            )

        top = diff_h // 2
        left = diff_w // 2

        return encoder_features[:,:,top:top + target.size(2),left:left + target.size(3),]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contracting path: retain feature maps for skip connections.
        skip_connections = []

        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)

        # Expanding path: upsample, crop, concatenate, and convolve.
        for i, block in enumerate(self.decoder):
            x = block["up"](x)

            skip = skip_connections[-(i + 2)]
            skip = self.center_crop(skip, x)

            x = torch.cat((skip, x), dim=1)
            x = block["conv"](x)

        return self.segment(x)