"""Dual Augmentation Dataset and DataLoaders for Remote Sensing Change Detection."""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Prevent OpenCV from spawning its own thread pool inside every DataLoader worker
# (on 2-vCPU machines such as Google Colab this oversubscription slows training a lot).
cv2.setNumThreads(0)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _gauss_noise(p: float):
    """GaussNoise compatible with both Albumentations 1.x (var_limit) and >=1.4.18/2.x (std_range)."""
    try:
        return A.GaussNoise(std_range=(0.02, 0.1), p=p)
    except TypeError:
        # Legacy API: variance in absolute pixel units (std ~5-25 on a 0-255 scale)
        # pyrefly: ignore [unexpected-keyword]
        return A.GaussNoise(var_limit=(25.0, 625.0), p=p)


def get_transforms(split: str = "train"):
    """
    Constructs the Albumentations transformation pipelines:
    1. Synchronous Geometric Transforms: Applied simultaneously to T1, T2, and Mask.
    2. Asynchronous Radiometric Transforms: Applied independently to T1 or T2.
    3. Normalization: ImageNet standard mean & std.
    """
    if split == "train":
        # Synchronous geometric transforms (Both T1, T2 and Mask undergo identical geometry)
        geometric_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={"image2": "image"},
        )

        # Asynchronous photometric transforms (Independent per temporal image to simulate seasonal/lighting differences)
        radiometric_t1 = A.Compose(
            [
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
                A.RandomBrightnessContrast(p=0.4),
                _gauss_noise(p=0.3),
            ]
        )

        radiometric_t2 = A.Compose(
            [
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
                A.RandomBrightnessContrast(p=0.4),
                _gauss_noise(p=0.3),
            ]
        )
    else:
        # Validation / Test: No augmentations
        geometric_transform = None
        radiometric_t1 = None
        radiometric_t2 = None

    # Normalization (0-255 RGB -> Standard normalized float tensor)
    norm_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return {
        "geometric": geometric_transform,
        "radiometric_t1": radiometric_t1,
        "radiometric_t2": radiometric_t2,
        "normalize": norm_transform,
    }


class DualAugmentationDataset(Dataset):
    """
    Dataset loader for bitemporal remote sensing change detection.
    Reads triplets:
      - T1 image (A)
      - T2 image (B)
      - Ground Truth mask (label: 0=no change, 1=change)
    """
    def __init__(self, root_dir: str, split: str = "train", transforms=None):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transforms = transforms or get_transforms(split)

        self.a_dir = os.path.join(self.root_dir, "A")
        self.b_dir = os.path.join(self.root_dir, "B")
        self.lbl_dir = os.path.join(self.root_dir, "label")

        if not os.path.exists(self.a_dir):
            raise FileNotFoundError(f"Directory not found: {self.a_dir}")

        if not os.path.exists(self.b_dir):
            raise FileNotFoundError(f"Directory not found: {self.b_dir}")

        # Collect image filenames (only keep samples that exist in both A and B)
        a_files = sorted(f for f in os.listdir(self.a_dir) if f.lower().endswith(IMG_EXTENSIONS))
        b_files = set(os.listdir(self.b_dir))
        self.filenames = [f for f in a_files if f in b_files]

        missing_b = len(a_files) - len(self.filenames)
        if missing_b > 0:
            print(f"[Dataset Warning] {missing_b} files in {self.a_dir} have no counterpart in B and were skipped.")

        if os.path.isdir(self.lbl_dir):
            lbl_files = set(os.listdir(self.lbl_dir))
            missing_lbl = sum(1 for f in self.filenames if f not in lbl_files)
            if missing_lbl > 0:
                print(f"[Dataset Warning] {missing_lbl} samples in '{split}' have no label (treated as no-change).")

        if len(self.filenames) == 0:
            print(f"[Dataset Warning] No images found in {self.a_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path_a = os.path.join(self.a_dir, filename)
        path_b = os.path.join(self.b_dir, filename)
        path_lbl = os.path.join(self.lbl_dir, filename)

        # Read RGB images
        img_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
        if img_a is None:
            raise ValueError(f"Could not read image: {path_a}")
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
        if img_b is None:
            raise ValueError(f"Could not read image: {path_b}")
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        # Read Ground Truth binary mask
        if os.path.exists(path_lbl):
            mask = cv2.imread(path_lbl, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read label: {path_lbl}")
            # Ensure binary 0 or 1 (labels are 0/255; threshold guards against JPEG/resize artefacts)
            mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros(img_a.shape[:2], dtype=np.uint8)

        # 1. Apply Synchronous Geometric Transformations
        if self.transforms["geometric"] is not None:
            augmented = self.transforms["geometric"](
                image=img_a,
                image2=img_b,
                mask=mask,
            )
            img_a = augmented["image"]
            img_b = augmented["image2"]
            mask = augmented["mask"]

        # Keep unnormalized versions for visualization debugging if needed
        raw_a = img_a.copy()
        raw_b = img_b.copy()

        # 2. Apply Asynchronous Radiometric Transformations (Only in train)
        if self.transforms["radiometric_t1"] is not None:
            img_a = self.transforms["radiometric_t1"](image=img_a)["image"]
        if self.transforms["radiometric_t2"] is not None:
            img_b = self.transforms["radiometric_t2"](image=img_b)["image"]

        # 3. Normalize & convert to PyTorch tensors
        norm_a = self.transforms["normalize"](image=img_a)["image"]
        norm_b = self.transforms["normalize"](image=img_b)["image"]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]

        return {
            "t1": norm_a,
            "t2": norm_b,
            "mask": mask_tensor,
            "raw_t1": raw_a,
            "raw_t2": raw_b,
            "raw_mask": mask,
            "filename": filename,
        }


def get_dataloaders(config: dict):
    """Factory creating train and validation DataLoaders from configuration dict."""
    processed_dir = config["data"]["processed_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = int(config["data"].get("num_workers", 2))
    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dataset = DualAugmentationDataset(
        root_dir=processed_dir,
        split="train",
        transforms=get_transforms("train"),
    )
    val_dataset = DualAugmentationDataset(
        root_dir=processed_dir,
        split="val",
        transforms=get_transforms("val"),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"Training set is empty: {train_dataset.a_dir}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"Validation set is empty: {val_dataset.a_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # drop_last avoids BatchNorm crashing on a final batch of size 1,
        # but must not drop the *only* batch on tiny datasets.
        drop_last=len(train_dataset) > batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader
