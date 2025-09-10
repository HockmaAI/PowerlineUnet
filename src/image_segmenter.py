import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from model import UNet
from torchvision import transforms

class ImageSegmenter:
    """A class for running image segmentation inference with tiling."""
    
    def __init__(self, model_path=None, tile_size=512, overlap=64, target_size=(256, 256), device=None):
        """Initialize the segmenter with model and inference parameters."""
        self.tile_size = tile_size
        self.overlap = overlap
        self.target_size = target_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.model = self._load_model(model_path)
        self.transform = None  # Set externally if needed
        print(f"Initialized ImageSegmenter on device: {self.device}")

    def _load_model(self, model_path):
        """Load a trained PyTorch model from the specified path."""
        if model_path is None:
            basedir = os.path.abspath(os.pardir)
            model_path = os.path.join(basedir, 'models', 'trained_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        # Initialize your model architecture (replace UNet with your actual model)
        model = UNet(in_channels=3, out_channels=1)  # Adjust in_channels, out_channels
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)  # Load weights into model
        model.to(self.device)
        model.eval()
        return model

    def _detect_bright_sky(self, image_np, threshold=0.3, brightness_threshold=180,
                           blue_threshold=150, very_bright_threshold=0.02):
        """
        Detect if the image contains a bright sky by focusing on the upper half.
        Args:
            image_np: NumPy array of shape [H, W, 3] (RGB)
            threshold: Fraction of upper half pixels that must be bright
            brightness_threshold: Minimum intensity for average RGB
            blue_threshold: Minimum blue channel value for sky-like regions
            very_bright_threshold: Fraction of very bright pixels for sun detection
        Returns:
            bool: True if bright sky detected
        """
        height, width, _ = image_np.shape
        upper_half = image_np[:height//2, :, :]
        avg_intensity = upper_half.mean(axis=2)
        blue_intensity = upper_half[:, :, 2]
        bright_pixels = np.sum((avg_intensity > brightness_threshold) | (blue_intensity > blue_threshold))
        total_pixels = upper_half.shape[0] * upper_half.shape[1]
        very_bright_pixels = np.sum(avg_intensity > 240)
        has_bright_spot = (very_bright_pixels / total_pixels) > very_bright_threshold        
        return ((bright_pixels / total_pixels) > threshold) or has_bright_spot

    def _preprocess_image(self, image_path):
        """Preprocess an image for inference, applying CLAHE if bright sky detected and normalization."""
        image_np = np.array(Image.open(image_path).convert("RGB"))
        is_bright_sky = self._detect_bright_sky(image_np)    
        
        # Apply CLAHE for bright sky
        if is_bright_sky:
            clahe_transform = A.Compose([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0)
            ])
            image_np = clahe_transform(image=image_np)['image']
            # image_np[:] = 255 - image_np
            print(f"Bright sky detected in {image_path}, applied CLAHE")

        # Apply main transform if provided (albumentations)
        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented['image']
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = normalize(image)
        
        return image, image_np.shape, is_bright_sky

    def _predict_single(self, image_path, output_path):
        """Run inference on a single image and save the output."""
        # Load and preprocess image
        image, (H_orig_np, W_orig_np, _), is_bright_sky = self._preprocess_image(image_path)
        C, H_orig, W_orig = image.shape
        image = image.to(self.device)

        # Handle small images by resizing
        if H_orig < self.tile_size or W_orig < self.tile_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.tile_size, self.tile_size),
                                  mode='bilinear', align_corners=False)
            H_padded, W_padded = self.tile_size, self.tile_size
        else:
            # Pad to ensure full tiles
            pad_h = self.tile_size - (H_orig % self.tile_size) if H_orig % self.tile_size != 0 else 0
            pad_w = self.tile_size - (W_orig % self.tile_size) if W_orig % self.tile_size != 0 else 0
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            H_padded, W_padded = image.shape[1], image.shape[2]

        # Compute stride
        stride = self.tile_size - self.overlap
        buffer = None
        count = torch.zeros((H_padded, W_padded), device=self.device)

        # Generate start positions
        start_hs = list(range(0, H_padded - self.tile_size + 1, stride))
        if start_hs and start_hs[-1] + self.tile_size < H_padded:
            start_hs.append(H_padded - self.tile_size)
        start_ws = list(range(0, W_padded - self.tile_size + 1, stride))
        if start_ws and start_ws[-1] + self.tile_size < W_padded:
            start_ws.append(W_padded - self.tile_size)

        # Process tiles
        with torch.no_grad():
            if H_padded == self.tile_size and W_padded == self.tile_size:
                tile_input = image.unsqueeze(0)
                outputs = self.model(tile_input)
                prob = torch.sigmoid(outputs)
                buffer = prob.squeeze(0)
                count[:] = 1
            else:
                for start_h in start_hs:
                    for start_w in start_ws:
                        tile = image[:, start_h:start_h + self.tile_size, start_w:start_w + self.tile_size]
                        tile_input = tile.unsqueeze(0)
                        outputs = self.model(tile_input)
                        prob = torch.sigmoid(outputs).squeeze(0)
                        if buffer is None:
                            buffer = torch.zeros((prob.shape[0], H_padded, W_padded), device=self.device)
                        buffer[:, start_h:start_h + self.tile_size, start_w:start_w + self.tile_size] += prob
                        count[start_h:start_h + self.tile_size, start_w:start_w + self.tile_size] += 1

        # Average and crop
        final_prob = buffer / count.unsqueeze(0).clamp(min=1e-6)
        if H_orig >= self.tile_size and W_orig >= self.tile_size:
            final_prob = final_prob[:, pad_top:H_orig + pad_top, pad_left:W_orig + pad_left]

        # Post-process and save
        output = final_prob.cpu().numpy()
        threshold = 0.0025 if is_bright_sky else 0.15
        output = (output > threshold).astype(np.uint8) * 255  # Binary mask
        # output = (output > 0.01).astype(np.uint8) * 255  # Binary mask
        output = output.squeeze()  # Adjust based on your model's output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)
        print(f"Saved prediction to {output_path}")

    def predict(self, input_path=None, output_path=None):
        """Run inference on a single image or batch of images."""
        if input_path and output_path:
            # Single image inference
            self._predict_single(input_path, output_path)
        else:
            # Batch inference
            input_dir = os.path.join(os.path.abspath(os.getcwd()), 'data', 'inference', 'in')
            output_dir = os.path.join(os.path.abspath(os.getcwd()), 'data', 'inference', 'out')
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
            for filename in os.listdir(input_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.png')
                    self._predict_single(input_path, output_path)
