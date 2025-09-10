import os
import argparse
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
from image_segmenter import ImageSegmenter

def run_single_inference(model_path=None, input_path=None, output_path=None):
    """Run inference on a single image using ImageSegmenter.
    
    Args:
        model_path (str, optional): Path to the trained model file.
        input_path (str): Path to the input image.
        output_path (str): Path to save the output mask.
    """
    segmenter = ImageSegmenter(model_path=model_path)
    segmenter.predict(input_path=input_path, output_path=output_path)
    print(f"Completed inference on {input_path}")

def run_batch_inference(model_path=None):
    """Run inference on all images in data/inference/in using ImageSegmenter.
    
    Args:
        model_path (str, optional): Path to the trained model file.
    """
    segmenter = ImageSegmenter(model_path=model_path)
    segmenter.predict()
    print("Completed batch inference. Check data/inference/out for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for image segmentation.")
    parser.add_argument("--model_path", type=str, default="models/best_finetuned_model.pth", help="Path to trained model file (default: ../models/best_pretrained_model.pth)")
    parser.add_argument("--input_path", type=str, default=None, help="Path to single input image (optional)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save single output mask (optional)")
    args = parser.parse_args()

    if args.input_path and args.output_path:
        run_single_inference(args.model_path, args.input_path, args.output_path)
    else:
        run_batch_inference(args.model_path)
