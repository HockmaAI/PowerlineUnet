"""
Hybrid loss function combining BCE and Dice loss for power line segmentation.

This addresses the severe class imbalance in power line detection 
(power lines are very thin compared to background).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Dice loss.
    
    The original implementation used 70% BCE + 30% Dice.
    This version makes the weights configurable.
    """
    
    def __init__(self, bce_weight: float = 0.7, dice_weight: float = 0.3, 
                 pos_weight: float = 15.5, smooth: float = 1.0):
        """
        Args:
            bce_weight: Weight for BCE loss (default 0.7)
            dice_weight: Weight for Dice loss (default 0.3) 
            pos_weight: Positive weight for BCE to handle imbalance
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # BCE with positive class weighting for imbalance
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid loss.
        
        Args:
            pred: Model predictions (logits)
            target: Ground truth masks
            
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        # Weighted combination
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss)
        
        return total_loss


def create_loss(config=None):
    """Create loss function from config or with defaults."""
    if config is None:
        return HybridLoss()
    
    return HybridLoss(
        bce_weight=config.training.bce_weight,
        dice_weight=config.training.dice_weight,
        pos_weight=15.5  # Can be made configurable too
    )


# For backward compatibility during migration
def dice_loss(pred, target, smooth=1):
    """Standalone dice loss function (for compatibility)."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
