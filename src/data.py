import os
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class PowerlineDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.palette = np.array([[120,120,120], [0,255,255]]) # background, powerline

        # List of image filenames
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

        # # Define augmentation pipeline
        if transform is None:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.9),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Resize(512, 512, interpolation=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.InvertImg(p=0.4), 
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def _convert_mask(self, mask):
        mask = np.array(mask)  # Convert PIL image to numpy array
        mask_binary = np.all(np.equal(mask, self.palette[1]), axis=-1).astype(np.float32)  # 1 for powerline, 0 for background
        return mask_binary  # No tensor conversion here, let albumentations handle it


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.images[idx])[0] + ".png")

        # Load image and mask as numpy arrays
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))

        # Convert mask to binary
        mask_binary = self._convert_mask(mask)

        # Apply augmentations to both image and mask
        if self.transform:
            augmented = self.transform(image=image, mask=mask_binary)
            image = augmented['image']  # Already a tensor
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension (1, H, W), still numpy -> tensor

        return image, mask


    def plot_sample(self, mask, pred):
        mask = mask.squeeze()
        pred = pred.squeeze()

        # Move tensors to CPU and convert to numpy for matplotlib
        mask_np = mask.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # Create an empty RGB array for the color-coded result
        height, width = mask_np.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Define the colors
        WHITE = [255, 255, 255] # True Positive
        BLACK = [0, 0, 0]       # True Negative
        RED = [255, 0, 0]       # False Positive
        GREY = [128, 128, 128]  # False Negative

        colored_mask[(pred_np == 1) & (mask_np == 1)] = WHITE
        colored_mask[(pred_np == 0) & (mask_np == 0)] = BLACK
        colored_mask[(pred_np == 1) & (mask_np == 0)] = RED
        colored_mask[(pred_np == 0) & (mask_np == 1)] = GREY

        # Display the results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1);
        plt.imshow(image.permute(1, 2, 0).cpu());
        plt.title("Original Image")

        plt.subplot(1, 3, 2);
        plt.imshow(mask_np, cmap='gray');
        plt.title("Target Mask")

        plt.subplot(1, 3, 3);
        plt.imshow(colored_mask);
        plt.title("TP: White | TN: Black | FP: Red | FN: Grey")

        plt.show()


