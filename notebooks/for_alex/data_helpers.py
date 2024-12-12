from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class MaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_paths = sorted(list(self.image_dir.glob("*.tif")))

        self.mask_paths = []

        for img_path in self.image_paths:
            mask_path = self.mask_dir / img_path.name.replace(
                "Straightened", "Mask_Straightened"
            )
            if not mask_path.exists():
                print(f"Mask not found for {img_path.name}")
                # remove image from list
                self.image_paths.remove(img_path)
            else:
                self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image and corresponding mask path
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read image and mask
        image = tifffile.imread(img_path)

        # normalize image to percentile
        ilow, ihigh = np.percentile(image, (1.0, 99.0))
        image = (image - ilow) / (ihigh - ilow)
        mask = tifffile.imread(mask_path) > 1

        # Convert to torch tensors and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            combined = torch.cat([image, mask], dim=0)
            combined = self.transform(combined)
            image = combined[0].unsqueeze(0)
            mask = combined[1].unsqueeze(0)

        return (
            image,
            mask,
            str(img_path.name),
        )  # Added filename for verification


# Define transforms
def get_train_transforms():
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
        ]
    )


def get_val_transforms():
    return None


def create_dataloaders(image_dir, mask_dir, batch_size=8, num_workers=4):
    train_dataset = MaskDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_train_transforms(),
    )

    val_dataset = MaskDataset(
        image_dir=image_dir, mask_dir=mask_dir, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=padder_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=padder_fn,
    )

    return train_loader, val_loader


def get_valid_dimensions(size, factor=32):
    """Round up to the nearest multiple of factor

    2^5 = 32 (e.g. can be halved 5 times)

    """
    return ((size + factor - 1) // factor) * factor


def padder_fn(batch):
    # Unzip the batch into separate lists
    images, masks, fnames = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # ensure valid dimensions
    target_h = get_valid_dimensions(max_h)
    target_w = get_valid_dimensions(max_w)

    # Pad each image/mask to max dimensions
    padded_images = []
    padded_masks = []

    for img, mask in zip(images, masks):
        # Calculate padding needed
        pad_h = target_h - img.shape[1]
        pad_w = target_w - img.shape[2]

        # Padding convention
        # last to first dimension: (W, H, C)
        padding = (0, pad_w, 0, pad_h, 0, 0)

        # Apply padding
        padded_images.append(F.pad(img, padding))
        padded_masks.append(F.pad(mask, padding))

    # Stack into tensors
    images_tensor = torch.stack(padded_images)
    masks_tensor = torch.stack(padded_masks)

    return images_tensor, masks_tensor, fnames
