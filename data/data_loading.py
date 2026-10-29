import logging
from pathlib import Path
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from os.path import splitext
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self,images_dir,mask_dir,mask_suffix='',image_dtype=np.float32,mask_dtype=np.float32,):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        self.image_dtype = image_dtype
        self.mask_dtype = mask_dtype

        if not self.images_dir.is_dir():
            raise RuntimeError(f'Image directory does not exist: {self.images_dir}')

        if not self.mask_dir.is_dir():
            raise RuntimeError(f'Mask directory does not exist: {self.mask_dir}')

        self.ids = sorted(
            splitext(f.name)[0]
            for f in self.images_dir.iterdir()
            if f.is_file() and not f.name.startswith('.')
        )

        if not self.ids:
            raise RuntimeError(f'No input files found in {self.images_dir}')

        # Index file paths once so __getitem__ does not scan the directories.
        self.image_files = self._index_files(self.images_dir)
        self.mask_files = self._index_files(self.mask_dir)

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    @staticmethod
    def _index_files(directory):
        files = {}
        for f in directory.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                files.setdefault(splitext(f.name)[0], []).append(f)
        return files

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def load_image(filename):
        filename = Path(filename)
        ext = filename.suffix.lower()

        if ext == '.npy':
            return Image.fromarray(np.load(filename))

        if ext in ('.pt', '.pth'):
            data = torch.load(filename,map_location='cpu',weights_only=True)

            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

            return Image.fromarray(data)

        return Image.open(filename)

    @staticmethod
    def preprocess(pil_img, is_mask=False, normalize=True):
        """Convert a PIL image to a (C, H, W) array.

        With normalize=False the image keeps its original dtype (e.g. uint8)
        so it can be cropped before the more expensive float conversion.
        """
        arr = np.asarray(pil_img)

        if is_mask:
            if arr.ndim == 3:
                if arr.shape[-1] == 1:
                    arr = arr[..., 0]
                else:
                    raise ValueError('Binary masks must be grayscale or have one channel')

            # Threshold in the original dtype; same result as scaling to
            # [0, 1] and comparing with 0.5, without a float copy.
            threshold = 127.5 if arr.max() > 1 else 0.5
            arr = (arr > threshold)[np.newaxis, ...]

        else:
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            else:
                arr = arr.transpose((2, 0, 1))

            if normalize:
                arr = arr.astype(np.float32)

                if arr.max() > 1:
                    arr /= 255.0

        return arr

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_files = self.image_files.get(name, [])
        mask_files = self.mask_files.get(name + self.mask_suffix, [])

        assert len(img_files) == 1, (f'Expected one image for {name}, found {img_files}')
        assert len(mask_files) == 1, (f'Expected one mask for {name}, found {mask_files}')

        with (self.load_image(img_files[0]) as img, self.load_image(mask_files[0]) as mask):
            if img.size != mask.size:
                raise ValueError(
                    f'Image and mask {name} have different sizes: '
                    f'{img.size} vs {mask.size}'
                )

            # Integer image dtypes skip normalization; DoubleToFloat converts
            # them to float in [0, 1] after cropping.
            normalize = not np.issubdtype(self.image_dtype, np.integer)
            image = self.preprocess(img, is_mask=False, normalize=normalize)
            mask = self.preprocess(mask, is_mask=True)

        image = image.astype(self.image_dtype, copy=False)
        mask = mask.astype(self.mask_dtype, copy=False)

        return {
            # Copy only when the array is a non-contiguous or read-only view.
            'image': torch.from_numpy(np.require(image, requirements=('C', 'W'))),
            'mask': torch.from_numpy(np.require(mask, requirements=('C', 'W'))),
        }