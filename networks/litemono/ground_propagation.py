from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from layers import *


class GroundPropagation(nn.Module):
    """
    Ground Propagation Module for refining depth features in dynamic objects.

    Args:
        depth_channel_ratio (int): Ratio for selecting depth-aware channels
        clipping_ratio (float): Ratio for clipping normalization (0-1)
        iterative_num (int): Base number of propagation iterations
    """

    def __init__(self, depth_channel_ratio=8, clipping_ratio=0.3, iterative_num=32):
        super(GroundPropagation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.depth_channel_ratio = depth_channel_ratio
        self.clipping_ratio = clipping_ratio
        self.iterative_num = iterative_num

    def clipping_normalization(self, vec, clipping_range=0.3):
        """
        Normalize input vector using clipping-based normalization.

        Args:
            vec (torch.Tensor): Input tensor of shape (B, C, H, W)
            clipping_range (float): Clipping threshold ratio

        Returns:
            torch.Tensor: Normalized tensor with values in [0, 1]
        """
        batch_size, channels, height, width = vec.shape

        # Calculate max values per channel
        max_vals = vec.reshape(batch_size, channels, -1).max(dim=-1)[0]
        max_vals = max_vals.reshape(batch_size, channels, 1, 1)

        # Avoid division by zero
        max_vals = max_vals * clipping_range
        max_vals[max_vals == 0] = 1.0

        # Normalize and clip
        normalized_vec = vec / max_vals
        normalized_vec = torch.clamp(normalized_vec, 0.0, 1.0)

        return normalized_vec

    def depth_aware_channel_selection(self, x):
        """
        Select depth-aware channels based on similarity with pseudo depth maps.

        Args:
            x (torch.Tensor): Input features of shape (B, C, H, W)

        Returns:
            torch.Tensor: Indices of selected depth-aware channels
        """
        batch_size, channels, height, width = x.shape

        def create_pseudo_depth_maps():
            """Create pseudo disparity and depth maps for similarity computation."""
            # Create linear disparity map from top to bottom
            disparity_map = torch.linspace(0.1, 1.0, height, requires_grad=False)
            disparity_map = disparity_map.view(1, height, 1).expand(batch_size, height, width)
            disparity_map = disparity_map.unsqueeze(1).repeat_interleave(channels, 1)

            return disparity_map.to(x.device)

        # Generate pseudo depth maps
        pseudo_disp = create_pseudo_depth_maps()
        pseudo_depth = 1.0 - pseudo_disp

        # Normalize input features
        normalized_input = torch.sigmoid(x)

        # Compute similarity scores
        flat_input = normalized_input.transpose(0, 1).reshape(channels, -1)
        flat_disp = pseudo_disp.transpose(0, 1).reshape(channels, -1)
        flat_depth = pseudo_depth.transpose(0, 1).reshape(channels, -1)

        disp_similarity = self.cosine_similarity(flat_input, flat_disp)
        depth_similarity = self.cosine_similarity(flat_input, flat_depth)

        # Select top channels based on similarity
        num_selected = channels // self.depth_channel_ratio
        disp_indices = torch.argsort(disp_similarity, dim=-1, descending=True)
        depth_indices = torch.argsort(depth_similarity, dim=-1, descending=True)

        # Combine disparity and depth aware channels
        selected_indices = torch.cat([
            disp_indices[:num_selected],
            depth_indices[:num_selected]
        ])

        return selected_indices

    def forward(self, input_features, dynamic_masks):
        """
        Forward pass of ground propagation.

        Args:
            input_features (torch.Tensor): Input features (B, C, H, W)
            dynamic_masks (torch.Tensor): Dynamic object masks (B, H_mask, W_mask)

        Returns:
            torch.Tensor: Refined features with ground propagation
        """
        batch_size, channels, height, width = input_features.shape

        # Select depth-aware channels
        depth_indices = self.depth_aware_channel_selection(input_features)
        selected_features = input_features[:, depth_indices]
        num_selected = len(depth_indices)

        # Prepare dynamic masks
        dynamic_masks = dynamic_masks.float()
        resized_masks = transforms.Resize(
            (height, width),
            interpolation=Image.NEAREST
        )(dynamic_masks)
        expanded_masks = resized_masks.unsqueeze(1).repeat_interleave(num_selected, 1)

        # Calculate adaptive iteration number based on feature resolution
        adaptive_iterations = int(self.iterative_num * height / dynamic_masks.shape[1])

        # Initialize propagation with padding
        propagated_features = F.pad(selected_features, (0, 0, 0, 1), mode='replicate')
        original_features = selected_features

        # Iterative ground propagation
        for iteration in range(adaptive_iterations):
            # Propagate features downward
            propagated_features = (
                    expanded_masks * propagated_features[:, :, 1:] +
                    (1 - expanded_masks) * original_features
            )
            # Add padding for next iteration
            propagated_features = F.pad(propagated_features, (0, 0, 0, 1), mode='replicate')

        # Remove padding and compute fusion weights
        propagated_features = propagated_features[:, :, :-1]

        feature_diff = torch.abs(propagated_features - original_features)
        fusion_weights = self.clipping_normalization(feature_diff, self.clipping_ratio)
        fusion_weights = fusion_weights.max(dim=1, keepdim=True)[0]

        # Fuse propagated and original features
        output_features = input_features.clone()
        refined_features = (
                fusion_weights * propagated_features +
                (1 - fusion_weights) * original_features
        )
        output_features[:, depth_indices] = refined_features

        return output_features
