import random

import numpy as np
import torch
from tifffile import imread
from torch.utils.data import DataLoader, Dataset


class ImagePatchDataset(Dataset):
    def __init__(
        self,
        image_paths,
        patch_size=64,
        batch_size=12,
        augmentations=None,
        pair_transform=None,
        percentile_norms=(0.1, 99.9),
        device="cpu",
    ):
        """
        Args:
            image_paths: List of paths to the images (either single 2D or 2D
            timelapse).
            patch_size: Tuple (height, width) for the random patches.
            augmentations: Optional augmentations to apply on the patches.
        """
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.pair_transform = pair_transform
        self.percentile_norms = percentile_norms
        self.device = device
        # load images
        self.images = self._load_and_flatten_images(image_paths)

    def _load_and_flatten_images(self, paths):
        images = []
        for path in paths:
            img = self._load_image(path)
            if isinstance(img, list):
                images.extend(img)
            else:
                images.append(img)
        return images

    def _load_image(self, path):
        img = imread(path)
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            return img
        elif img.ndim == 3:
            Nt, Ny, Nx = img.shape
            # for timelapses return a list of each frame
            imgmin = img.min(axis=(-2, -1), keepdims=True)
            imgmax = img.max(axis=(-2, -1), keepdims=True)

            img = img - imgmin
            img = img / (imgmax - imgmin)

            return [img[i] for i in range(Nt)]
        else:
            raise ValueError("Only 2D and 2D timelapse is supported")

    def _get_random_patch(self, image):
        Ny, Nx = image.shape

        patch_h, patch_w = self.patch_size, self.patch_size

        top = random.randint(0, Ny - patch_h)
        left = random.randint(0, Nx - patch_w)

        patch = image[top : top + patch_h, left : left + patch_w]

        return patch

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if isinstance(image, list):
            timepoint = random.choice(range(len(image)))
            image = image[timepoint]

        patch = self._get_random_patch(image)

        if self.augmentations:
            patch = self.augmentations(patch)

        patch = torch.from_numpy(patch).float().to(self.device)

        patch = patch.unsqueeze(0)  # add channel dimension

        if self.pair_transform:
            input_image, target_image = self.pair_transform(patch)
            return input_image, target_image
        else:
            return patch


class PairImagesTransform:
    def __init__(self):
        """splits an image into two"""
        pass

    def __call__(self, patch):
        """given a patch, return (input_patch, target_patch)"""

        if not torch.is_tensor(patch):
            patch = torch.from_numpy(patch).float()

        q1 = patch[:, 0::2, 0::2]
        q2 = patch[:, 0::2, 1::2]
        q3 = patch[:, 1::2, 0::2]
        q4 = patch[:, 1::2, 1::2]

        left = (q1 + q3) / 2.0
        right = (q2 + q4) / 2.0

        return left, right


class SimpleAugmentations:
    def __init__(self):
        pass

    def __call__(self, patch):
        if random.random() > 0.5:
            patch = np.fliplr(patch)
        if random.random() > 0.5:
            patch = np.flipud(patch)
        if random.random() > 0.5:
            patch = np.transpose(patch, axes=(1, 0))

        # return patch copy to ensure arrays are contiguous in memory
        # flipping sets the stride to be negative
        return patch.copy()


def get_dataloader(
    image_paths,
    patch_size=64,
    batch_size=12,
    augmentations=None,
    pair_transform=None,
    percentile_norms=(0.1, 99.9),
):
    dataset = ImagePatchDataset(
        image_paths,
        patch_size=patch_size,
        augmentations=augmentations,
        pair_transform=pair_transform,
        percentile_norms=percentile_norms,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
