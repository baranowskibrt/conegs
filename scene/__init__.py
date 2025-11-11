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

import json
import os
import random

import numpy as np
import torch
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.graphics_utils import fov2focal
from utils.nerf_utils import get_gs2nerf_transform
from utils.system_utils import mkdir_p, searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        stack_train_images=True,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.n_views = args.n_views

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, n_views=args.n_views
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        # Rescale on the whole dataset to keep consistent with nerfacc
        c2w_train = np.array([c.cam2worlds.cpu().numpy() for c in self.getTrainCameras()])
        c2w_test = np.array([c.cam2worlds.cpu().numpy() for c in self.getTestCameras()])
        c2w = np.concatenate([c2w_train.squeeze(1), c2w_test.squeeze(1)], axis=0)
        self.gs2nerf_T, self.gs2nerf_scale = get_gs2nerf_transform(c2w, strict_scaling=False)

        for resolution_scale in resolution_scales:
            c2w = np.einsum("nij, ki -> nkj", c2w, self.gs2nerf_T)
            c2w[:, :3, 3] *= self.gs2nerf_scale
            c2w[:, :, 1:3] *= -1

            c2w_train = c2w[: len(self.train_cameras[resolution_scale])]
            c2w_test = c2w[len(self.train_cameras[resolution_scale]) :]

            for i, c in enumerate(self.train_cameras[resolution_scale]):
                c.cam2worlds = torch.tensor(c2w_train[i]).to(c.data_device).float()
            for i, c in enumerate(self.test_cameras[resolution_scale]):
                c.cam2worlds = torch.tensor(c2w_test[i]).to(c.data_device).float()

            self.gs2nerf_T = torch.tensor(self.gs2nerf_T).float().cuda()
            self.inv_gs2nerf_T = torch.linalg.inv(self.gs2nerf_T)
            self.gs2nerf_scale = torch.tensor(self.gs2nerf_scale).float().cuda()

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.gaussians.spatial_lr_scale = self.cameras_extent
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )

        self.stacked_images = []
        self.stacked_pix2cams = []
        self.stacked_cam2worlds = []

        if stack_train_images:
            self.biggest_resolution = torch.tensor([0, 0, 0])
            for camera in self.train_cameras[1.0]:
                self.stacked_images.append(camera.original_image)
                self.biggest_resolution = torch.max(
                    self.biggest_resolution, torch.tensor(camera.original_image.shape)
                )
                self.stacked_pix2cams.append(camera.pix2cams)
                self.stacked_cam2worlds.append(camera.cam2worlds)

            self.image_height = self.biggest_resolution[1]
            self.image_width = self.biggest_resolution[2]

            for i in range(len(self.stacked_images)):
                if (self.biggest_resolution != torch.tensor(self.stacked_images[i].shape)).any():
                    self.stacked_images[i] = torch.nn.functional.interpolate(
                        self.stacked_images[i].unsqueeze(0),
                        size=(self.biggest_resolution[1], self.biggest_resolution[2]),
                    ).squeeze(0)
                    K = torch.zeros((3, 3)).to(self.train_cameras[1.0][i].pix2cams.device)
                    K[0, 0] = fov2focal(self.train_cameras[1.0][i].FoVx, self.image_width)
                    K[1, 1] = fov2focal(self.train_cameras[1.0][i].FoVy, self.image_height)
                    K[0, 2] = self.image_width / 2
                    K[1, 2] = self.image_height / 2
                    K[2, 2] = 1
                    self.stacked_pix2cams[i] = K.inverse()
            self.stacked_images = torch.stack(self.stacked_images).cuda()
            self.stacked_pix2cams = torch.stack(self.stacked_pix2cams)
            self.stacked_cam2worlds = torch.stack(self.stacked_cam2worlds)
            self.num_pixels = self.image_height * self.image_width
        else:
            self.image_height = self.train_cameras[1.0][0].original_image.shape[1]
            self.image_width = self.train_cameras[1.0][0].original_image.shape[2]
            for i in range(len(self.getTrainCameras())):
                self.getTrainCameras()[i].original_image = self.getTrainCameras()[i].original_image.cuda()
            for i in range(len(self.getTestCameras())):
                self.getTestCameras()[i].original_image = self.getTestCameras()[i].original_image.cuda()                

            self.num_pixels = torch.tensor(self.image_height * self.image_width)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        mkdir_p(point_cloud_path)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
