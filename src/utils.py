# -*- coding: utf-8 -*-
"""Utilities for working with images and devices."""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def get_device():
    """
    Determine available device (CUDA, MPS or CPU).

    Returns:
        torch.device: Device for computations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    return device


def get_image_loader(imsize, device):
    """
    Create image loader.

    Args:
        imsize: Image size
        device: Device for loading

    Returns:
        Image loading function
    """
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])

    def image_loader(image_name):
        image = Image.open(image_name)
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    return image_loader


def save_image(tensor, path):
    """
    Save tensor as image.

    Args:
        tensor: Image tensor
        path: Path to save
    """
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)
    print(f"Image saved to: {path}")


def imshow(tensor, title=None):
    """
    Display tensor as image.

    Args:
        tensor: Image tensor
        title: Image title
    """
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

