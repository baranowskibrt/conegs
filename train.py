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
import uuid
from random import randint

import hydra
import torch
import wandb
from gaussian_renderer import render
from omegaconf import DictConfig, OmegaConf
from scene import GaussianModel, Scene
from scene.nerf_model import NeRFModel
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.nerf_utils import get_num_rays_to_cast


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(cfg):
    first_iter = 0

    tb_writer = prepare_output_and_logger(cfg.gaussian_model)
    gaussians = GaussianModel(cfg.gaussian_model.sh_degree)
    scene = Scene(cfg.gaussian_model, gaussians, stack_train_images=cfg.nerf_model.stack_images)

    gaussians.training_setup(cfg.optimization)
    bg_color = [1, 1, 1] if cfg.gaussian_model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)


    nerf_model = NeRFModel(cfg.nerf_model)
    nerf_model.pretrain_model(cfg, gaussians, scene, background)
    nerf_model.save(os.path.join(cfg.gaussian_model.model_path, "nerf_model.pth"))

    nerf_model.prop_optimizer.zero_grad(set_to_none=True)
    nerf_model.model_optimizer.zero_grad(set_to_none=True)

    # We don't need batching across images after pretraining
    del scene.stacked_images, scene.stacked_cam2worlds, scene.stacked_pix2cams
    if not cfg.nerf_model.nerf_train_during_3dgs:
        del nerf_model.prop_optimizer, nerf_model.model_optimizer, nerf_model.prop_scheduler, nerf_model.model_scheduler, nerf_model.estimator.optimizer, nerf_model.estimator.scheduler, nerf_model.estimator.prop_cache

    # Non-batched cameras loaded later to save GPU memory
    for i in range(len(scene.getTrainCameras())):
        scene.getTrainCameras()[i].original_image = scene.getTrainCameras()[i].original_image.cuda()
    for i in range(len(scene.getTestCameras())):
        scene.getTestCameras()[i].original_image = scene.getTestCameras()[i].original_image.cuda()

    num_rays_to_cast = get_num_rays_to_cast(len(gaussians._xyz), gaussian_percentage_increase=cfg.optimization.gaussian_percentage_increase, nerf_train_during_3dgs=cfg.nerf_model.nerf_train_during_3dgs)

    nerf_model.estimator_params["n_rays"] = num_rays_to_cast
    nerf_model.estimator_params["stratified"] = False
    nerf_model.estimator_params["requires_grad"] = False

    if cfg.checkpoint:
        (model_params, first_iter) = torch.load(cfg.checkpoint)
        gaussians.restore(model_params, cfg.optimization)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, cfg.optimization.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, cfg.optimization.iterations + 1):
        iter_start.record()
        xyz_lr = gaussians.update_learning_rate(iteration)

        if iteration % cfg.optimization.SH_increase_iter == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == cfg.debug_from:
            cfg.pipeline.debug = True
        bg = torch.rand((3), device="cuda") if cfg.optimization.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, cfg.pipeline, bg)
        image = render_pkg["render"]

        l1_full = torch.abs((image - gt_image))
        ssim_full = ssim(image, gt_image)
        Ll1 = l1_full.mean()

        loss = (1.0 - cfg.optimization.lambda_dssim) * Ll1 + cfg.optimization.lambda_dssim * (1.0 - ssim_full.mean())
        opacity = gaussians._opacity if cfg.optimization.use_preactivation_opacity_penalty else torch.abs(gaussians.get_opacity)
        loss = loss + cfg.optimization.opacity_penalty * opacity.mean()

        num_pixels = viewpoint_cam.original_image.shape[1] * viewpoint_cam.original_image.shape[2]
        weights = (1.0 - cfg.optimization.densification_lambda_dssim) * l1_full.mean(0) + cfg.optimization.densification_lambda_dssim * (1.0 - ssim_full.mean(0))
        weights = torch.clamp_min(weights, 0.0)
        
        loss.backward()
        iter_end.record()


        with torch.no_grad():
            if cfg.optimization.use_3dgs_densification and iteration < cfg.optimization.densify_until_iter:
                radii = render_pkg["radii"]
                visibility_filter = render_pkg["visibility_filter"]
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg.get("optimized_samples_index", 0))

                if iteration > cfg.optimization.densify_from_iter and iteration % 100 == 0:
                    size_threshold = 20 if iteration > 3000 else None
                    gaussians.densify_and_prune(0.0002, 0.005, scene.cameras_extent, size_threshold, max_points=cfg.optimization.max_points)
                
                if iteration % 3000 == 0:
                    gaussians.reset_opacity()

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == cfg.optimization.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), cfg.test_iterations, scene, render, (cfg.pipeline, background))
            if (iteration in cfg.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)


        if cfg.optimization.densify_from_iter < iteration and iteration < cfg.optimization.densify_until_iter and (cfg.optimization.gaussian_percentage_increase or cfg.optimization.max_points):
            nerf_model.cast_rays_during_optimization(weights, num_rays_to_cast, viewpoint_cam, cfg, gaussians, scene, bg, iteration)

        with torch.no_grad():
            if cfg.optimization.densify_from_iter < iteration and iteration % 100 == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)

                if iteration < cfg.optimization.densify_until_iter:
                    gaussians.prune_points(dead_mask)

                if cfg.optimization.gaussian_percentage_increase or cfg.optimization.max_points:
                    num_gaussians = len(gaussians._xyz)

                    gaussians.initialize_new_points(use_cone_radius=True, max_points=cfg.optimization.max_points, opacity_init_value=cfg.optimization.opacity_init_value)
                    num_added = len(gaussians._xyz) - num_gaussians
                    print("Num added", num_added)

                    if not cfg.optimization.max_points:
                        num_added = None

                    num_rays_to_cast = get_num_rays_to_cast(len(gaussians._xyz), num_added=num_added, gaussian_percentage_increase=cfg.optimization.gaussian_percentage_increase, nerf_train_during_3dgs=cfg.nerf_model.nerf_train_during_3dgs)

        with torch.no_grad():

            if (iteration in cfg.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    if cfg.nerf_model.log_nerf_renders:
        for i in [4, 7]:
            nerf_model.render_test(scene, background, scene.getTrainCameras()[i], img_id=f"final_{i}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('GPU Usage', torch.cuda.max_memory_allocated() / 1e9, iteration)
    if iteration % 10 == 0:
        wandb.log({"iter_time": elapsed, "GPU Usage": torch.cuda.max_memory_allocated() / 1e9}, step=iteration)
        torch.cuda.reset_peak_memory_stats()

    if iteration in testing_iterations and len(scene.gaussians._xyz) > 0:
        print("Num gaussians", scene.gaussians.get_xyz.shape[0])
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameras()},
                                {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                i = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1 = torch.abs((image - gt_image)).mean(0, keepdim=True)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_scalar(config['name'] + "_view_{}/l1_loss".format(viewpoint.image_name), l1.mean().item(), iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/loss".format(viewpoint.image_name), l1[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if (idx < 5):
                        logged = {
                                config['name'] + "_view_{}/l1_loss".format(viewpoint.image_name): l1.mean().item(),
                                config['name'] + "_view_{}/loss".format(viewpoint.image_name): [wandb.Image(l1)],
                             }
                        logged[config['name'] + "_view_{}/render".format(viewpoint.image_name)] = [wandb.Image(image)]
                        if iteration == testing_iterations[0]:
                            logged[config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name)] = [wandb.Image(gt_image)]
                        wandb.log(logged, step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    i += 1

                psnr_test /= i
                l1_test /= i     
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                wandb.log({config['name'] + '/loss_viewpoint - l1_loss ': l1_test, config['name'] + '/loss_viewpoint - psnr ': psnr_test}, step=iteration)


        
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.gaussian_model.source_path = os.path.join(cfg.gaussian_model.source_path, cfg.scene_name)
    cfg.gaussian_model.source_path = os.path.abspath(cfg.gaussian_model.source_path)
    print("Optimizing " + cfg.gaussian_model.model_path, ", for ", cfg.gaussian_model.source_path)

    # Initialize system state (RNG)
    safe_state(cfg.quiet)

    wandb.init(
        mode=None if cfg.enable_wandb else "disabled",
        name=cfg.run_name.replace("/", "_"),
        project=cfg.log_project_name.replace("/", "_"),
        dir=cfg.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(start_method="fork", _disable_stats=True),
    )
    print(OmegaConf.to_yaml(cfg))

    cfg.optimization.resolution = float(cfg.optimization.resolution)
    os.makedirs(cfg.gaussian_model.model_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.gaussian_model.model_path, "config.yaml"))
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    training(cfg)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
