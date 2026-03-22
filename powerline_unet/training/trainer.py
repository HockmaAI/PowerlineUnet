"""
Modern training module for PowerlineUnet.

This provides a clean, configurable training loop using the new configuration system.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple

from ..config.config import config
from ..models.unet import create_model
from ..data.dataset import PowerlineDataset
from ..losses.hybrid_loss import HybridLoss


class Trainer:
    """Modern trainer for PowerlineUnet with configuration support."""
    
    def __init__(self, config=None):
        self.config = config or config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logging()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def _setup_logging(self):
        """Setup logging for training."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def build(self):
        """Build all training components."""
        self.logger.info(f"Building trainer on device: {self.device}")
        
        # Create model
        self.model = create_model(self.config).to(self.device)
        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Loss function
        self.criterion = HybridLoss(
            bce_weight=self.config.training.bce_weight,
            dice_weight=self.config.training.dice_weight
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            verbose=True
        )
        
        self.logger.info("Trainer built successfully")
        return self
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        train_dataset = PowerlineDataset(
            img_dir=self.config.paths.train_images,
            mask_dir=self.config.paths.train_masks,
            transform=None  # Will be added in data augmentation phase
        )
        
        val_dataset = PowerlineDataset(
            img_dir=self.config.paths.val_images,
            mask_dir=self.config.paths.val_masks,
            transform=None
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        metrics = {'iou': 0.0, 'f1': 0.0}
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # TODO: Add proper metric calculation
                # metrics = compute_metrics(outputs, masks)
        
        return {
            'val_loss': total_loss / len(dataloader),
            **metrics
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        save_path = Path(self.config.paths.model_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        torch.save(checkpoint, save_path / "latest.pth")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, save_path / "best.pth")
            self.logger.info(f"New best model saved! Metrics: {metrics}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
            
        train_loader, val_loader = self.get_dataloaders()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.scheduler.step(val_metrics['val_loss'])
            
            self.logger.info(f"Train Loss: {train_loss".4f"}, Val Loss: {val_metrics['val_loss']".4f"}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=False)
            
        self.logger.info("Training completed!")
        return self.model
