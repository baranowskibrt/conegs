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

import math

import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            antialiasing=False
        )
    except Exception as e:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            # antialiasing=False
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    results = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": results[0],
            "viewspace_points": screenspace_points,
            "visibility_filter" : results[1] > 0,
            "radii": results[1],
            "depth": results[2] if len(results) > 2 else None,
            }



def generate_and_render(params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, include_existing=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    means3D = params["means3D"]
    opacity = params["opacity"]
    cone_scaling = params["cone_scaling"]
    cone_rotation = params["cone_rotation"]

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

 
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    colors_precomp = None
    cov3D_precomp = None
    shs = None

    len_samples = len(means3D)

    if "colors_precomp" in params:
        colors_precomp = params["colors_precomp"]
    else:
        if params.get("features_rest") is None:
            shs = None
            colors_precomp = params["features_dc"].reshape(-1, 3)
        else:
            shs = torch.cat((params["features_dc"], params["features_rest"]), dim=1)
            colors_precomp = None


    if len(pc.get_xyz) and include_existing:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz.detach() - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)

        means3D = torch.concatenate([means3D, pc.get_xyz], dim=0)
        shs = torch.concatenate([shs, pc.get_features], dim=0)
        opacity = torch.concatenate([opacity, pc.get_opacity], dim=0)
        if pc.align_cfg.predict_scaling and pc.align_cfg.predict_rotation:
            scaling = params["scaling"]
            rotations = params["rotations"]
            cone_scaling = torch.concatenate([cone_scaling.detach(), pc.get_scaling], dim=0)
            cone_rotation = torch.concatenate([cone_rotation.detach(), pc.get_rotation], dim=0)
            scaling = torch.cat((scaling, pc.get_scaling), dim=0)
            rotations = torch.cat((rotations, pc.get_rotation), dim=0)
        else:
            cone_scaling = torch.concatenate([cone_scaling.detach(), pc.get_scaling], dim=0)
            cone_rotation = torch.concatenate([cone_rotation.detach(), pc.get_rotation], dim=0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rendered_image, radii = rasterizer(
    means3D = means3D,
    means2D = screenspace_points,
    shs = shs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = cone_scaling,
    rotations = cone_rotation,
    cov3D_precomp = cov3D_precomp)[:2]

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0)[len_samples:],
            "radii": radii[len_samples:],
            "optimized_samples_index": len_samples,
            "shape_image": None,
    }
