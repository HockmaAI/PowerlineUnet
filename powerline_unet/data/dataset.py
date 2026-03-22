"""
Dataset for Powerline segmentation with modern configuration support.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch

from ..config.config import config


class PowerlineDataset(Dataset):
    """
    Dataset for power line segmentation.
    
    Handles loading images and binary masks, with support for augmentations.
    """
    
    def __init__(self, img_dir=None, mask_dir=None, transform=None, is_train=True):
        """
        Args:
            img_dir: Directory containing images
            mask_dir: Directory containing masks
            transform: Albumentations transform pipeline
            is_train: Whether this is training dataset (affects augmentations)
        """
        self.config = config
        
        # Use config paths if not provided
        if img_dir is None:
            img_dir = self.config.paths.train_images if is_train else self.config.paths.val_images
        if mask_dir is None:
            mask_dir = self.config.paths.train_masks if is_train else self.config.paths.val_masks
            
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.is_train = is_train
        
        # Get image files
        self.images = [f for f in os.listdir(self.img_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.images.sort()
        
        # Color palette for mask conversion (background, powerline)
        self.palette = np.array([[120, 120, 120], [0, 255, 255]])
        
        # Setup transforms
        self.transform = transform or self._get_default_transform(is_train)
        
        print(f"Loaded {len(self.images)} images from {self.img_dir}")
    
    def _get_default_transform(self, is_train=True):
        """Get default augmentation pipeline."""
        if is_train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.7),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # Validation transforms (no heavy augmentation)
            return A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
    
    def _convert_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert RGB mask to binary mask."""
        # Check if mask matches powerline color
        mask_binary = np.all(np.equal(mask, self.palette[1]), axis=-1).astype(np.float32)
        return mask_binary
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image and mask pair."""
        img_path = self.img_dir / self.images[idx]
        mask_filename = os.path.splitext(self.images[idx])[0] + ".png"
        mask_path = self.mask_dir / mask_filename
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        
        # Convert mask to binary
        mask_binary = self._convert_mask(mask)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask_binary)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension
        
        return image, mask


# Convenience functions
def get_train_dataloader(batch_size=None):
    """Get training dataloader."""
    if batch_size is None:
        batch_size = config.training.batch_size
        
    dataset = PowerlineDataset(is_train=True)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )


def get_val_dataloader(batch_size=None):
    """Get validation dataloader."""
    if batch_size is None:
        batch_size = config.training.batch_size
        
    dataset = PowerlineDataset(is_train=False)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
