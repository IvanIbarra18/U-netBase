"""Joint image/mask transformations for segmentation.

Inputs from BasicDataset:
    image: (C, H, W), float32, normally in [0, 1]
    mask:  (1, H, W), float32 binary mask in {0, 1}

Geometric transforms are applied consistently to image and mask.
Photometric transforms affect the image only.
"""
import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PairTransformDataset(Dataset):
    """Apply the same spatial transformations to each image and its mask."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Load one paired image and mask as tensors through BasicDataset.
        sample = self.dataset[index]
        image, mask = sample["image"], sample["mask"]

        # Apply paired augmentations to the image and mask, when configured.
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}


def _pair_size(size):
    if isinstance(size, int):
        size = (size, size)
    if not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError("crop_size must be an integer or a pair (height, width)")
    height, width = int(size[0]), int(size[1])
    if height < 1 or width < 1:
        raise ValueError("crop dimensions must be positive")
    return height, width


class DoubleCompose:
    """Compose transforms that accept and return (image, mask)."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

    def __repr__(self):
        names = ", ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([{names}])"


class DoubleToFloat:
    """Convert an integer image to float32 in [0, 1]; the mask is unchanged.

    Place it after DoubleRandomCrop so only the cropped region is converted.
    Float images pass through untouched.
    """

    def __call__(self, image, mask):
        if not image.is_floating_point():
            max_value = torch.iinfo(image.dtype).max
            image = image.to(torch.float32).div_(max_value)
        return image, mask

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class DoubleRandomCrop:
    """Randomly crop the same spatial region from an image and its mask.
        Based on the random crop implementation in:
        https://github.com/cosmic-cortex/pytorch-UNet
    """

    def __init__(self, crop_size):
        self.crop_size = _pair_size(crop_size)

    def __call__(self, image, mask):
        if image.ndim != 3 or mask.ndim != 3:
            raise ValueError(
                f"Expected image and mask as (C,H,W) tensors; got "
                f"{tuple(image.shape)} and {tuple(mask.shape)}"
            )
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"Image/mask spatial sizes differ: "
                f"{tuple(image.shape[-2:])} vs {tuple(mask.shape[-2:])}"
            )

        crop_h, crop_w = self.crop_size
        height, width = image.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError(
                f"Crop {self.crop_size} is larger than sample spatial size "
                f"({height}, {width}). Choose a smaller crop."
            )

        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        return (
            TF.crop(image, top, left, crop_h, crop_w),
            TF.crop(mask, top, left, crop_h, crop_w),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"


class DoubleHorizontalFlip:
    """Flip image and mask horizontally with probability p."""

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image, mask = TF.hflip(image), TF.hflip(mask)
        return image, mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class DoubleVerticalFlip:
    """Flip image and mask vertically with probability p."""

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image, mask = TF.vflip(image), TF.vflip(mask)
        return image, mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class DoubleElasticTransform:
    """Apply the same smooth random deformation to image and binary mask.
        - alpha: controls the magnitude of the displacement in pixels.
        - sigma: controls the smoothness of the displacement field.
        - p: probability of applying the deformation.
        - randinit: when enabled, randomizes alpha between 100 and 300, and sigma between 10 and 15.
    """

    def __init__(self, alpha=250.0, sigma=10.0, p=0.5, randinit=True):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.p = float(p)
        self.randinit = bool(randinit)

    def __call__(self, image, mask):
        if random.random() >= self.p or self.alpha == 0:
            return image, mask
        if image.ndim != 3 or mask.ndim != 3:
            raise ValueError("Expected image and mask tensors with shape (C,H,W) and (1,H,W)")
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("Image and mask must have matching spatial dimensions")

        _, height, width = image.shape
        device = image.device
        dtype = image.dtype
        alpha = random.uniform(100.0, 300.0) if self.randinit else self.alpha
        sigma = random.uniform(10.0, 15.0) if self.randinit else self.sigma

        # Build a low-resolution random field, then upsample it to obtain
        # smooth displacements without requiring SciPy or NumPy conversions.
        min_dim = max(2, int(round(min(height, width) / max(sigma, 1.0))))
        field_h = max(2, min(height, min_dim))
        field_w = max(2, min(width, min_dim))
        displacement = torch.rand(
            1, 2, field_h, field_w, device=device, dtype=torch.float32
        ) * 2.0 - 1.0
        displacement = F.interpolate(
            displacement, size=(height, width), mode="bicubic", align_corners=True
        )
        displacement = displacement * alpha

        # Normalize pixel displacements to grid_sample coordinates [-1, 1].
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        dx = displacement[:, 0] * (2.0 / max(width - 1, 1))
        dy = displacement[:, 1] * (2.0 / max(height - 1, 1))
        grid = base_grid + torch.stack((dx[0], dy[0]), dim=-1).unsqueeze(0)

        image_out = F.grid_sample(
            image.unsqueeze(0), grid, mode="bilinear",
            padding_mode="border", align_corners=True
        ).squeeze(0)
        mask_out = F.grid_sample(
            mask.unsqueeze(0).float(), grid, mode="nearest",
            padding_mode="border", align_corners=True
        ).squeeze(0)
        mask_out = (mask_out > 0.5).to(dtype=mask.dtype)
        return image_out.to(dtype=dtype), mask_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma}, "
            f"p={self.p}, randinit={self.randinit})"
        )


class GaussianNoise:
    """Add Gaussian noise to an image tensor only."""

    def __init__(self, mean=0.0, std=0.03, p=0.5, clamp=True):
        if std < 0:
            raise ValueError("std must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.mean = mean
        self.std = std
        self.p = p
        self.clamp = clamp

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        noise = torch.randn_like(image) * self.std + self.mean
        result = image + noise
        return result.clamp(0.0, 1.0) if self.clamp else result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, "
            f"p={self.p}, clamp={self.clamp})"
        )