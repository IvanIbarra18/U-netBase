
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with padding, optional BatchNorm, and ReLU."""

    def __init__(self,in_channels: int,out_channels: int,batch_norm: bool = True,) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=not batch_norm,)
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=not batch_norm,)
        )

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetModern(nn.Module):
    """
    U-Net with same-padded convolutions.

    Input and output have the same spatial dimensions.
    Designed for binary segmentation with logits as output.
    """

    def __init__(self,n_classes: int = 1,n_channels: int = 3,base_channels: int = 32,batch_norm: bool = True,) -> None:
        super().__init__()

        if n_classes < 1 or n_channels < 1 or base_channels < 1:
            raise ValueError("n_classes, n_channels, and base_channels must be positive.")

        self.n_channels = n_channels
        self.n_classes = n_classes

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        # Encoder
        self.enc1 = DoubleConv(n_channels, channels[0], batch_norm)
        self.enc2 = DoubleConv(channels[0], channels[1], batch_norm)
        self.enc3 = DoubleConv(channels[1], channels[2], batch_norm)
        self.enc4 = DoubleConv(channels[2], channels[3], batch_norm)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(channels[3], channels[3] * 2, batch_norm)

        # Decoder
        self.up4 = nn.ConvTranspose2d(channels[3] * 2, channels[3],kernel_size=2, stride=2)
        self.dec4 = DoubleConv(channels[3] * 2, channels[3], batch_norm)

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2],kernel_size=2, stride=2)
        self.dec3 = DoubleConv(channels[2] * 2, channels[2], batch_norm)

        self.up2 = nn.ConvTranspose2d(channels[2], channels[1],kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1], batch_norm)

        self.up1 = nn.ConvTranspose2d(channels[1], channels[0],kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0], batch_norm)

        self.segment = nn.Conv2d(channels[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] < 16 or x.shape[-1] < 16:
            raise ValueError(
                "Input height and width must be at least 16 pixels "
                "for the four pooling stages."
            )

        # Save encoder features for skip connections.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Upsample, concatenate, and decode.
        d4 = self.up4(b, output_size=e4.size())
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4, output_size=e3.size())
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3, output_size=e2.size())
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2, output_size=e1.size())
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.segment(d1)
