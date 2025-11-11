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
from argparse import ArgumentParser
from os import makedirs

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from gaussian_renderer import GaussianModel, render
from omegaconf import DictConfig, OmegaConf
from scene import Scene
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import save_depth_map


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    check_rendering_time = True

    if check_rendering_time:
        rendering_time = []
        for _ in range(100):
            for view in views:
                torch.cuda.synchronize()
                iter_start.record()
                rendering = render(view, gaussians, pipeline, background)["render"]
                iter_end.record()
                torch.cuda.synchronize()
                elapsed_time = iter_start.elapsed_time(iter_end)
                rendering_time.append(elapsed_time)

        avg_time = sum(rendering_time) / len(rendering_time)
        print("Average rendering time: ", avg_time, 1/avg_time * 1000, gaussians._xyz.shape[0], torch.tensor(rendering_time).min() , torch.tensor(rendering_time).max(), )

def render_sets(dataset, iteration : int, pipeline, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


@hydra.main(version_base=None, config_name="config")
def main(cfg : DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.gaussian_model.source_path = os.path.abspath(cfg.gaussian_model.source_path)
    print("Rendering " + cfg.gaussian_model.model_path, ", for ", cfg.gaussian_model.source_path)

    render_sets(cfg.gaussian_model, -1, cfg.pipeline, True, False)


if __name__ == "__main__":
    main()