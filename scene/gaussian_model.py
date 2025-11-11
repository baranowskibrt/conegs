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

import os
import random

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from utils.general_utils import (build_scaling_rotation, get_expon_lr_func, inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud

from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, return_whole_matrix=False):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if return_whole_matrix:
                return actual_covariance
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, start_sh_degree : int = 3):
        self.active_sh_degree = start_sh_degree
        self.max_sh_degree = sh_degree  
        self.features_rest_len = 3 * ((sh_degree + 1)**2) - 3
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.n_input_dims = 3

        self.scaling_adjustment = 0
        self.pruned_xyz = torch.empty((0, 3), device="cuda", requires_grad=False)

        self.stashed_xyz = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_features_dc = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_features_rest = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_opacities = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_scaling = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_rotation = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_weights = torch.empty(0, device="cuda", requires_grad=False)

        self.mult = 1

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity) * self.mult

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, opt):
        self.percent_dense = opt.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.max_points = opt.max_points
        self.iterations = opt.iterations
        self.contraction_scaling = self.spatial_lr_scale / opt.contraction_scaling_mult if opt.contraction_scaling_mult else 1

        l = [
            {'params': [self._xyz], 'lr': opt.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        ]

        l.append({'params': [self._features_dc], 'lr': opt.features_dc_lr, "name": "f_dc"})
        l.append({'params': [self._features_rest], 'lr': opt.features_rest_lr, "name": "f_rest"})
        l.append({'params': [self._opacity], 'lr': opt.opacity_lr, "name": "opacity"})
        l.append({'params': [self._scaling], 'lr': opt.scaling_lr, "name": "scaling"})
        l.append({'params': [self._rotation], 'lr': opt.rotation_lr, "name": "rotation"})


        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_hash_model(self, path, estimator=None):
        torch.save({
            'model': self.hash_model.state_dict(),
            'estimator': estimator.state_dict(),
        }, path)
        torch.save(estimator, path.replace(".pt", "_estimator.pt"))
    
    def load_hash_model(self, path, estimator=None):
        checkpoint = torch.load(path)
        self.hash_model.load_state_dict(checkpoint['model'])
        # estimator.load_state_dict(checkpoint['estimator'])
        return torch.load(path.replace(".pt", "_estimator.pt"))

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "model":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == "model":
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def initialize_new_points(self, use_cone_radius=False, max_points=None, max_cone_radius=None, opacity_init_value=0.1, num_to_select=None):
        if not len(self.stashed_xyz) > 1:
            return

        # Limit for multinomial sampling
        if len(self.stashed_weights) >= 2**24:
            indices = torch.randperm(len(self.stashed_weights), device="cuda")[:2**24]
            self.stashed_xyz = self.stashed_xyz[indices]
            self.stashed_features_dc = self.stashed_features_dc[indices]
            self.stashed_features_rest = self.stashed_features_rest[indices]
            self.stashed_opacities = self.stashed_opacities[indices]
            self.stashed_scaling = self.stashed_scaling[indices]
            self.stashed_rotation = self.stashed_rotation[indices]
            self.stashed_weights = self.stashed_weights[indices]

        if max_points:
            print("Num to add", max_points - len(self._xyz), "stashed", len(self.stashed_xyz))

        if num_to_select:
            selected_indices = torch.multinomial(self.stashed_weights, min(num_to_select, len(self.stashed_xyz)), replacement=False)
            self.stashed_xyz = self.stashed_xyz[selected_indices]
            self.stashed_features_dc = self.stashed_features_dc[selected_indices]
            self.stashed_features_rest = self.stashed_features_rest[selected_indices]
            self.stashed_opacities = self.stashed_opacities[selected_indices]
            self.stashed_scaling = self.stashed_scaling[selected_indices]
            self.stashed_rotation = self.stashed_rotation[selected_indices]

        if max_points and len(self._xyz) + len(self.stashed_xyz) > max_points:
            num_added = max_points - len(self._xyz)
            self.stashed_xyz = self.stashed_xyz[:num_added]
            self.stashed_features_dc = self.stashed_features_dc[:num_added]
            self.stashed_features_rest = self.stashed_features_rest[:num_added]
            self.stashed_opacities = self.stashed_opacities[:num_added]
            self.stashed_scaling = self.stashed_scaling[:num_added]
            self.stashed_rotation = self.stashed_rotation[:num_added]


        if use_cone_radius:
            rots = torch.zeros((len(self.stashed_xyz), 4), device="cuda")
            rots[:, 0] = 1

            scales = self.scaling_activation(self.stashed_scaling)
            scales = torch.log(scales)
            opacities = inverse_sigmoid(opacity_init_value * torch.ones((self.stashed_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.concatenate((self.stashed_xyz, self._xyz), dim=0)).float().cuda(), 0.0000001)[:self.stashed_xyz.shape[0]]
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
            scales = torch.clamp_max(scales, torch.exp(self.stashed_scaling))
            scales = torch.log(scales)

            rots = torch.zeros((len(self.stashed_xyz), 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(opacity_init_value * torch.ones((self.stashed_xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        self.densification_postfix(self.stashed_xyz, self.stashed_features_dc, self.stashed_features_rest, opacities, scales, rots, reset_params=False)

        self.stashed_xyz = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_features_dc = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_features_rest = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_opacities = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_scaling = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_rotation = torch.empty(0, device="cuda", requires_grad=False)
        self.stashed_weights = torch.empty(0, device="cuda", requires_grad=False)

    

    def stash_new_points(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, weights):
        self.stashed_xyz = torch.cat((self.stashed_xyz, new_xyz), dim=0)
        self.stashed_features_dc = torch.cat((self.stashed_features_dc, new_features_dc), dim=0)
        self.stashed_features_rest = torch.cat((self.stashed_features_rest, new_features_rest), dim=0)
        self.stashed_opacities = torch.cat((self.stashed_opacities, new_opacity), dim=0)
        self.stashed_scaling = torch.cat((self.stashed_scaling, new_scaling), dim=0)
        self.stashed_rotation = torch.cat((self.stashed_rotation, new_rotation), dim=0)
        self.stashed_weights = torch.cat((self.stashed_weights, weights), dim=0)


    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        return optimizable_tensors
