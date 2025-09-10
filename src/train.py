import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import torch.nn.functional as F
from model import UNet
from data import PowerlineDataset

# not using google colab
google_drive_base = os.path.abspath(os.pardir)

# declare global variables
# Use forward slashes for paths in Google Colab
img_root = google_drive_base + '/data/train'
train_imgs = img_root + '/train/train_imgs'
val_imgs = img_root + '/val/val_imgs'
train_gt = img_root + '/train/train_gt'
val_gt = img_root + '/val/val_gt'
test_imgs = img_root + '/test/test_imgs'
test_gt = img_root + '/test/test_gt'

# print the file count
print(len(os.listdir(train_imgs)), len(os.listdir(val_imgs)),
      len(os.listdir(train_gt)), len(os.listdir(val_gt)))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = UNet(in_channels=3, out_channels=1).to(device)



# Load pre-trained weights from best_model.pth for transfer learning
pretrained_path = google_drive_base + '/models/best_finetuned_model.pth'
model.load_state_dict(torch.load(pretrained_path))
print(f"Loaded pre-trained weights from {pretrained_path}")

model_save_path = google_drive_base + '/models/trained_model_5_9_25.pth'

# Optional: Freeze some layers (e.g., encoder) if you only want to fine-tune later layers
# For UNet, you might freeze the encoder part (adjust based on your UNet implementation)
# for name, param in model.named_parameters():
#      if "encoder" in name:  # Adjust this condition based on your UNet structure
#          param.requires_grad = False


# # Freeze encoder layers (enc1, enc2, enc3, enc4)
# for layer in [model.enc1, model.enc2, model.enc3, model.enc4]:
#     for param in layer.parameters():
#         param.requires_grad = False

# # Verify which parameters are frozen
# print("Parameters with requires_grad=True (trainable):")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)



def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class HybridLoss(nn.Module):
    def __init__(self, pos_weight=20.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device='cuda'))
        self.dice_weight = 0.3

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)
        return (1 - self.dice_weight) * bce + self.dice_weight * dice

criterion = HybridLoss(pos_weight=15.5)


# Optimizer and Scheduler
# optimizer = optim.Adam(model.parameters(), lr=0.00005)
optimizer = optim.SGD(model.parameters(), lr=0.00015, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)

batch_size = 16

# Dataset and Dataloader
train_dataset = PowerlineDataset(train_imgs, train_gt)
val_dataset = PowerlineDataset(val_imgs, val_gt)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def compute_metrics(pred, target, threshold=0.1):
    # Convert predictions to binary using the threshold
    pred = (torch.sigmoid(pred) > threshold).to(torch.bool)  # Convert to boolean for bitwise operations
    target = target.to(torch.bool)  # Ensure target is also boolean

    # Compute intersection, union, etc.
    intersection = (pred & target).sum()  # Bitwise AND for intersection
    union = (pred | target).sum()  # Bitwise OR for union

    # Calculate IoU
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Calculate TP, FP, FN for precision, recall, and F1
    tp = intersection  # True Positives (intersection)
    fp = (pred & ~target).sum()  # False Positives
    fn = (~pred & target).sum()  # False Negatives

    # Calculate precision, recall, and F1
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return iou, f1, precision, recall

# Training Loop with Gradient Accumulation
num_epochs = 150
best_iou = 0.0
patience = 50
trigger_times = 0
accumulation_steps = 4  # This is the key: 4 * 16 = 64


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad() # Clear gradients at the start of the epoch

    for i, (images, masks) in enumerate(train_dataloader):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Normalize loss to account for accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights only after accumulating gradients for `accumulation_steps`
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps  # Revert normalization for correct loss logging

    # Handle any remaining gradients if the dataset size is not divisible by accumulation_steps
    if (len(train_dataloader) % accumulation_steps != 0):
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_iou, val_f1, _, _ = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            iou, f1, _, _ = compute_metrics(outputs, masks, threshold=0.1)
            val_iou += iou.item()
            val_f1 += f1.item()
        val_iou /= len(val_dataloader)
        val_f1 /= len(val_dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f}")

    # Learning Rate Scheduling
    scheduler.step(train_loss)

    # Early Stopping
    if val_iou > best_iou:
        best_iou = val_iou
        trigger_times = 0 # Correct logic: reset patience counter
        torch.save(model.state_dict(), model_save_path)
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Visualize every 5 epochs
    if epoch % 5 == 0:
        # Use torch.no_grad() to save memory and computations during inference
        with torch.no_grad():
            # Get one batch of data
            images, masks = next(iter(val_dataloader))

            # Take the first image and mask from the batch
            # This is where we ensure the tensor has the correct shape
            image = images[0].to(device)
            mask = masks[0].to(device)

            # Add a batch dimension of size 1 for a single image
            # The shape of 'image' should be [3, 512, 512]
            # The shape of 'image.unsqueeze(0)' will be [1, 3, 512, 512]
            pred = model(image.unsqueeze(0))

        # Get the predicted mask by applying a sigmoid and threshold
        pred = (torch.sigmoid(pred.squeeze(0)) > 0.5).float()

        # plot out the gt and prediction
        train_dataset.plot_sample(mask, pred)




# Load best model
# model.load_state_dict(torch.load('best_finetuned_model.pth'))
