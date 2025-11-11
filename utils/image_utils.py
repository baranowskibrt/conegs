#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def save_image(tensor, path, permute=True, mean_channel=False, scale=True):
    if permute:
        tensor = tensor.permute(1, 2, 0)
    if mean_channel:
        tensor = tensor.mean(0)
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.cpu().numpy()
    if scale:
        tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor).save(path)

def save_mask(mask_tensor, path):
    mask_np = mask_tensor.float().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(path)

def save_selected(mask, indices, path):
    mask = mask.cpu()
    selected_image = torch.zeros_like(mask)
    selected_mask = torch.zeros(mask.sum()).bool()
    selected_mask[indices.cpu()] = True
    selected_image[torch.nonzero(mask, as_tuple=True)] = selected_mask
    save_mask(selected_image, path)

def save_depth_map(depth_tensor, path, max_depth=50, invert=True, log_eps=1e-2):
    """
    Save a depth map with log scaling and nice colormap (like Depth Anything style).

    Parameters:
    - depth_tensor: torch.Tensor (H, W) with depth values
    - path: output image path
    - max_depth: clip depth values to this maximum
    - invert: if True, makes closer objects brighter
    - log_eps: small epsilon added before log to avoid log(0)
    """
    # Clamp and normalize
    depth_tensor = torch.clamp(depth_tensor, min=log_eps, max=max_depth)
    depth_tensor = torch.log(depth_tensor + log_eps)
    min_val, max_val = 0, 6
    depth_tensor = (depth_tensor - min_val) / (max_val - min_val)
    depth_tensor = torch.clamp(depth_tensor, 0.0, 1.0)

    if invert:
        depth_tensor = 1.0 - depth_tensor

    depth_np = depth_tensor.cpu().numpy()

    # Use clean colormap like turbo
    colormap = cm.get_cmap('turbo')
    colored_depth = colormap(depth_np)[..., :3]
    colored_depth = (colored_depth * 255).astype(np.uint8)

    Image.fromarray(colored_depth).save(path)