# -*- coding: utf-8 -*-
"""Main function for image stylization."""

import torch
import torchvision.models as models

from .style_transfer import (
    run_style_transfer,
    CNN_NORMALIZATION_MEAN,
    CNN_NORMALIZATION_STD
)
from .utils import get_device, get_image_loader


def stylize_image(content_path, style_path,
                  output_size=512,
                  num_steps=300,
                  style_weight=1_000_000,
                  content_weight=1):
    """
    Stylize content image using style from another image.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_size: Output image size (default: 512)
        num_steps: Number of optimization steps (default: 300)
        style_weight: Style loss weight (default: 1_000_000)
        content_weight: Content loss weight (default: 1)

    Returns:
        torch.Tensor: Stylized image
    """
    # Determine device
    device = get_device()

    # Determine image size based on device
    if device.type == 'cuda':
        imsize = output_size
    elif device.type == 'mps':
        imsize = output_size
    else:
        imsize = min(output_size, 128)  # Use smaller size for CPU

    # Create image loader
    image_loader = get_image_loader(imsize, device)

    # Load images
    print(f"Loading content image from: {content_path}")
    content_img = image_loader(content_path)

    print(f"Loading style image from: {style_path}")
    style_img = image_loader(style_path)

    # Check that sizes match
    assert style_img.size() == content_img.size(), \
        "Style and content images must have the same size"

    # Initialize input image as copy of content
    input_img = content_img.clone()

    # Load pre-trained VGG19 model
    print("Loading VGG19 model...")
    cnn = (models.vgg19(weights=models.VGG19_Weights.DEFAULT)
           .features.to(device).eval())

    # Run style transfer
    output = run_style_transfer(
        cnn,
        torch.tensor(CNN_NORMALIZATION_MEAN).to(device),
        torch.tensor(CNN_NORMALIZATION_STD).to(device),
        content_img,
        style_img,
        input_img,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight
    )

    print("Style transfer completed!")
    return output
