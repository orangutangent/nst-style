# -*- coding: utf-8 -*-
"""Module with models for computing style and content losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """Module for computing content loss."""

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Detach target content from gradient computation
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """Module for computing style loss."""

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """Module for normalizing input images."""

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Reshape mean and std to [C x 1 x 1] format
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Normalize image
        return (img - self.mean) / self.std


def gram_matrix(input):
    """
    Compute Gram matrix for input tensor.

    Args:
        input: Tensor of shape [batch, channels, height, width]

    Returns:
        Normalized Gram matrix
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # Reshape F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # Compute Gram product

    # Normalize Gram matrix values
    # by dividing by the number of elements in each feature map
    return G.div(a * b * c * d)

