import itertools
import os
from random import randint

import hydra
import torch
import torch.nn.functional as F
from gaussian_renderer import generate_and_render
from scene.dataset_readers import storePly
from tqdm import tqdm
from utils.camera_utils import nerf2gs_point_transform
from utils.general_utils import inverse_sigmoid
from utils.image_utils import (save_depth_map, save_image, save_mask,
                               save_selected)
from utils.ngp import NGPDensityField, NGPRadianceField, NGPRadianceFieldwithSH
from utils.ray_utils import pixels_to_rays
from utils.sh_utils import RGB2SH, SH2RGB

from nerfacc.estimators.prop_net import (PropNetEstimator,
                                         get_proposal_requires_grad_fn)
from nerfacc.scan import inclusive_prod
from nerfacc.volrend import render_transmittance_from_density, rendering


class NeRFModel:
    def __init__(self, nerf_cfg):
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device="cuda")

        if nerf_cfg.predict_SH:
            self.hash_model = NGPRadianceFieldwithSH(
                aabb=aabb, unbounded=nerf_cfg.unbounded, use_viewdirs=False
            ).to("cuda")
        else:
            self.hash_model = NGPRadianceField(
                aabb=aabb, unbounded=nerf_cfg.unbounded, use_viewdirs=True
            ).to("cuda")

        self.proposal_networks = [
            NGPDensityField(
                aabb=aabb,
                unbounded=nerf_cfg.unbounded,
                n_levels=5,
                max_resolution=128,
            ).to("cuda"),
            NGPDensityField(
                aabb=aabb,
                unbounded=nerf_cfg.unbounded,
                n_levels=5,
                max_resolution=256,
            ).to("cuda"),
        ]

        self.estimator_params = {
            "prop_samples": [256, 96],
            "num_samples": nerf_cfg.num_samples,
            "n_rays": nerf_cfg.pretraining_num_rays,
            "near_plane": nerf_cfg.near_plane,
            "far_plane": nerf_cfg.far_plane,
            "sampling_type": nerf_cfg.sampling_type,
        }

        self.model_optimizer = hydra.utils.instantiate(
            nerf_cfg.optimizer, params=self.hash_model.parameters()
        )
        self.prop_optimizer = hydra.utils.instantiate(
            nerf_cfg.optimizer,
            params=itertools.chain(
                *[p.parameters() for p in self.proposal_networks],
            ),
        )

        self.prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.prop_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.prop_optimizer,
                    milestones=[
                        nerf_cfg.num_iters_pretrain // 2,
                        nerf_cfg.num_iters_pretrain * 3 // 4,
                        nerf_cfg.num_iters_pretrain * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )

        self.model_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.model_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.model_optimizer,
                    milestones=[
                        nerf_cfg.num_iters_pretrain // 2,
                        nerf_cfg.num_iters_pretrain * 3 // 4,
                        nerf_cfg.num_iters_pretrain * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )

        self.estimator = PropNetEstimator(self.prop_optimizer, self.prop_scheduler).to("cuda")
        self.grad_scaler = torch.amp.GradScaler("cuda", 2**10)
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

        self.gaussian_density_avg = None
        # self.trained_hash_model_path = nerf_cfg.trained_hash_model_path
        self.nerf_cfg = nerf_cfg

        if self.nerf_cfg.trained_hash_model_path:
            self.load()

    def proposal_network_pass(
        self,
        viewpoint_cam,
        scene,
        pixels_x=None,
        pixels_y=None,
        use_stacked=False,
        image_id=None,
    ):
        if use_stacked:
            origins, directions, normalized_dirs, pixel_width_mult = pixels_to_rays(
                pixels_x.flatten(),
                pixels_y.flatten(),
                scene.stacked_pix2cams[image_id],
                scene.stacked_cam2worlds[image_id],
            )
        else:
            origins, directions, normalized_dirs, pixel_width_mult = viewpoint_cam.get_rays(
                pixels_x, pixels_y
            )

        origins = origins[..., None, :]
        pixel_width_mult = pixel_width_mult[..., None, :]
        normalized_dirs = normalized_dirs[..., None, :]

        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = origins
            t_dirs = normalized_dirs
            tdist = (t_starts + t_ends)[..., None] / 2.0
            positions = t_origins + t_dirs * tdist
            sigmas = proposal_network(positions)
            if self.nerf_cfg.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        t_starts, t_ends = self.estimator.sampling(
            prop_sigma_fns=[lambda *args: prop_sigma_fn(*args, p) for p in self.proposal_networks],
            **self.estimator_params,
        )

        return {
            "t_starts": t_starts,
            "t_ends": t_ends,
            "normalized_dirs": normalized_dirs,
            "origins": origins,
            "pixel_width_mult": pixel_width_mult,
            "unnormalized_directions": directions,
        }

    def predict_and_render(self, origins, normalized_dirs, t_starts, t_ends, render_bkgd, with_directions=True):
        render_pkg = {"render": None, "shape_image": None}

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins
            t_dirs = normalized_dirs.repeat_interleave(t_starts.shape[-1], dim=-2)

            tdist = (t_starts + t_ends)[..., None] / 2.0
            means = t_origins + t_dirs * tdist
            render_pkg["tdist"] = tdist
            if not with_directions:
                t_dirs = torch.zeros_like(t_dirs)
            predictions = self.hash_model(means, t_dirs)
            rgb, sigmas = predictions["rgb"], predictions["density"]

            if self.nerf_cfg.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            render_pkg["xyz"] = means
            if "features_rest" in predictions:
                render_pkg["features_rest"] = predictions["features_rest"].reshape(*sigmas.squeeze(-1).shape, -1, 3)
                render_pkg["features_dc"] = predictions["features_dc"].reshape(*sigmas.squeeze(-1).shape, -1,  3)
            render_pkg["sigmas"] = sigmas
            render_pkg["rgb"] = rgb
            rgb = rgb.reshape(*sigmas.squeeze(-1).shape, 3)
            sigmas = sigmas.squeeze(-1)
            return rgb, sigmas

        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=self.estimator_params["n_rays"],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        extras["depth"] = depth
        return rgb, extras, render_pkg

    def pretrain_iteration(
        self,
        max_cone_radius,
        gaussians,
        scene,
        viewpoint_cam,
        bg,
        pipeline,
        is_densification_iteration=False,
        iteration=0,
        pretrain_with_gaussians=False,
    ):
        if pretrain_with_gaussians or not self.nerf_cfg.stack_images:
            image_id = None
            pixels_x = torch.randint(
                0,
                viewpoint_cam.original_image.shape[2],
                size=(self.estimator_params["n_rays"],),
                device="cuda",
            )
            pixels_y = torch.randint(
                0,
                viewpoint_cam.original_image.shape[1],
                size=(self.estimator_params["n_rays"],),
                device="cuda"
            )

            gt_color = viewpoint_cam.original_image.permute(1, 2, 0)[pixels_y, pixels_x]
        else:
            pixels_x = torch.randint(
                0,
                scene.image_width,
                size=(self.estimator_params["n_rays"],),
                device=scene.stacked_images.device,
            )
            pixels_y = torch.randint(
                0,
                scene.image_height,
                size=(self.estimator_params["n_rays"],),
                device=scene.stacked_images.device,
            )
            image_id = torch.randint(
                0,
                len(scene.stacked_images),
                size=(self.estimator_params["n_rays"],),
                device=scene.stacked_images.device,
            )
            gt_color = scene.stacked_images[image_id, :, pixels_y, pixels_x]

        params = self.proposal_network_pass(
            viewpoint_cam,
            scene,
            pixels_x,
            pixels_y,
            use_stacked=self.nerf_cfg.stack_images,
            image_id=image_id,
        )

        origins = params["origins"]
        normalized_dirs = params["normalized_dirs"]
        if pretrain_with_gaussians:
            t_dirs = normalized_dirs.repeat_interleave(params["t_starts"].shape[-1], dim=-2)

            tdist = (params["t_starts"] + params["t_ends"])[..., None] / 2.0
            means = origins + t_dirs * tdist
            predictions = self.hash_model(means, t_dirs)
            rgb, sigmas = predictions["rgb"], predictions["density"]

            if self.nerf_cfg.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            rgb = rgb.reshape(*sigmas.squeeze(-1).shape, 3)

            original_xyz = means
            means = means.reshape(-1, 3)
            stds = params["pixel_width_mult"] / 2 * tdist / scene.gs2nerf_scale
            params["cone_scaling"] = stds.reshape(-1, 1).repeat(1, 3)

            rotation = torch.zeros(len(means), 4).cuda()
            rotation[:, 0] = 1
            params["cone_rotation"] = rotation
            params["means3D"] = nerf2gs_point_transform(means, scene.gs2nerf_scale, scene.inv_gs2nerf_T)
            trans, params["opacity"] = render_transmittance_from_density(
                params["t_starts"], params["t_ends"], sigmas.squeeze(-1)
            )
            alphas = params["opacity"]
            params["opacity"] = params["opacity"].reshape(-1, 1)
            params["colors_precomp"] = rgb.reshape(-1, 3)
            render_pkg = generate_and_render(
                params,
                viewpoint_cam,
                gaussians,
                scene,
                pipeline,
                bg,
                is_densification_iteration=is_densification_iteration,
                iteration=iteration,
                include_existing=True,
            )
            render_pkg["xyz"] = original_xyz
            render_pkg["tdist"] = tdist
            rgb = render_pkg["render"].permute(1, 2, 0)[pixels_y, pixels_x]
            extras = {"trans": trans}
        else:
            rgb, extras, render_pkg = self.predict_and_render(
                origins,
                normalized_dirs,
                params["t_starts"],
                params["t_ends"],
                bg,
                with_directions=not is_densification_iteration
            )
            alphas = extras["alphas"]
            sigmas = render_pkg["sigmas"]

        loss = F.smooth_l1_loss(rgb, gt_color)

        with torch.no_grad():
            if is_densification_iteration:
                scaling_mult = max_cone_radius
                

                weights = torch.ones(len(render_pkg["xyz"])).cuda()
                if "features_rest" in render_pkg:
                    features_dc, features_rest = (
                        render_pkg["features_dc"],
                        render_pkg["features_rest"],
                    )
                else:
                    features_dc, features_rest = None, None

                xyz = self.select_for_densification(
                    gaussians,
                    scene,
                    render_pkg["xyz"],
                    rgb,
                    # gt_color,
                    alphas,
                    weights,
                    render_pkg["tdist"],
                    params["pixel_width_mult"],
                    params["unnormalized_directions"],
                    scaling_mult=scaling_mult,
                    cumsum_jiggle=self.nerf_cfg.cumsum_jiggle,
                    features_dc=features_dc,
                    features_rest=features_rest,
                    # min_opacity=0.05,
                    densities=sigmas,
                )
                


        return loss, extras["trans"]

    def pretrain_model(self, cfg, gaussians, scene, background):
        max_selected = cfg.optimization.num_gaussians_initialized
        
        if cfg.optimization.max_points:
            max_selected = min(
                max_selected,
                cfg.optimization.max_points,
            )

        # To ensure going through the whole training set
        n_rays_densification = min(
            (max_selected // len(scene.getTrainCameras())) + 1,
            self.estimator_params["n_rays"],
        )

        approx_num_iters = (
            cfg.nerf_model.num_iters_pretrain + int(max_selected // n_rays_densification) + 1
        )

        progress_bar = tqdm(range(1, approx_num_iters), desc="Initialization training progress")

        viewpoint_stack = None
        loss_sum = 0
        train_iter = True

        # 5000 more to rely on the break condition and allow the possibility of parsing out points
        for iteration in range(1, approx_num_iters + 5000):

            if iteration == cfg.nerf_model.num_iters_pretrain:
                train_iter = False

            if train_iter:
                self.train()
            else:
                self.eval()

            if not viewpoint_stack or cfg.nerf_model.num_iters_pretrain == iteration:
                viewpoint_stack = scene.getTrainCameras().copy()

            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            bg = (
                torch.rand((3), device="cuda") if cfg.optimization.random_background else background
            )

            proposal_requires_grad = self.proposal_requires_grad_fn(iteration)
            is_densification_iteration = iteration > cfg.nerf_model.num_iters_pretrain

            self.estimator_params["stratified"] = not is_densification_iteration
            self.estimator_params["requires_grad"] = proposal_requires_grad and not self.nerf_cfg.trained_hash_model_path and train_iter

            if is_densification_iteration:
                self.estimator_params["n_rays"] = n_rays_densification

            loss, trans = self.pretrain_iteration(
                10000000000,
                gaussians,
                scene,
                viewpoint_cam,
                bg,
                cfg.pipeline,
                is_densification_iteration=is_densification_iteration,
                iteration=iteration,
                pretrain_with_gaussians=self.nerf_cfg.pretrain_with_gaussians,
            )

            self.estimator.update_every_n_steps(
                trans, self.estimator_params["requires_grad"], loss_scaler=1024
            )
            if train_iter and not self.nerf_cfg.trained_hash_model_path:
                self.grad_scaler.scale(loss).backward()
                self.model_optimizer.step()
                self.model_scheduler.step()
            self.model_optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
                loss_sum = 0

            if len(gaussians._xyz) >= max_selected or len(gaussians.stashed_xyz) >= max_selected:
                break

        gaussians.initialize_new_points(
            False,
            cfg.optimization.max_points,
            num_to_select=cfg.optimization.num_gaussians_initialized
        )

        self.eval()

        with torch.no_grad():
            gaussians._features_dc[:] = RGB2SH(
                torch.clamp(SH2RGB(gaussians._features_dc), 0.0, 1.0)
            )


    def select_for_densification(
        self,
        gaussians,
        scene,
        xyz,
        color,
        opacities,
        weights,
        tdist,
        pixel_width_mult,
        unnormalized_directions,
        cumsum_opacity_threshold=0.5,
        scaling_mult=1.0,
        cumsum_jiggle=False,
        features_dc=None,
        features_rest=None,
        min_opacity=0,
        densities=None,
        xyz_depth = None,
        color_samples=None,
    ):
        trans = inclusive_prod((1 - opacities.squeeze(-1)))
        if cumsum_jiggle:
            cumsum_opacity_threshold = torch.rand(len(trans), 1, device=trans.device) * 0.48 + 0.02
        reaches_threshold = trans < cumsum_opacity_threshold
        indexes = torch.argmax(reaches_threshold.int(), dim=1)
        opacity_mask = opacities[torch.arange(len(indexes)), indexes] > min_opacity
        indexes = indexes[opacity_mask]

        xyz = xyz[opacity_mask][torch.arange(len(indexes)), indexes]
        xyz = nerf2gs_point_transform(xyz, scene.gs2nerf_scale, scene.inv_gs2nerf_T)

        if features_dc is not None and features_rest is not None:
            features_dc = features_dc[opacity_mask].reshape(len(indexes), -1, 1, 3)[
                torch.arange(len(indexes)), indexes
            ]
            features_rest = features_rest[opacity_mask].reshape(
                len(indexes), -1, gaussians.features_rest_len // 3, 3
            )[torch.arange(len(indexes)), indexes]
        else:
            if color_samples is not None:
                color = color_samples[opacity_mask][torch.arange(len(indexes)), indexes]
            else:
                color = color[opacity_mask]
            
            features_dc = RGB2SH(color).reshape(-1, 1, 3)
            # features_dc = RGB2SH(color[opacity_mask]).reshape(-1, 1, 3)
            features_rest = torch.zeros(len(indexes), gaussians.features_rest_len // 3, 3).cuda()
        stds = (
            pixel_width_mult.squeeze(1)[opacity_mask]
            * tdist[opacity_mask][torch.arange(len(indexes)), indexes]
            / unnormalized_directions[opacity_mask].norm(dim=1, keepdim=True)
            / scene.gs2nerf_scale
        )
        scaling = stds.repeat(1, 3)
        rotation = torch.zeros(len(indexes), 4).cuda()
        rotation[:, 0] = 1
        opacity = inverse_sigmoid(
            0.1 * torch.ones((len(indexes), 1), dtype=torch.float, device="cuda")
        )
        if xyz_depth is not None:
            xyz = xyz_depth
        gaussians.stash_new_points(
            xyz,
            features_dc.reshape(-1, 1, 3),
            features_rest.reshape(-1, gaussians.features_rest_len // 3, 3),
            opacity,
            gaussians.scaling_inverse_activation(scaling * scaling_mult),
            rotation,
            weights[opacity_mask],
        )
        return xyz

    def cast_rays_during_optimization(
        self,
        weights,
        num_rays_to_cast,
        viewpoint_cam,
        cfg,
        gaussians,
        scene,
        bg,
        iteration,
    ):
        gt_image = viewpoint_cam.original_image

        if self.nerf_cfg.nerf_train_during_3dgs:
            self.hash_model.train()
            for p in self.proposal_networks:
                p.train()
            self.estimator.train()
            self.estimator_params["requires_grad"] = self.proposal_requires_grad_fn(
                iteration + 20000
            )
        else:
            self.estimator_params["requires_grad"] = False

        self.estimator_params["n_rays"] = num_rays_to_cast
        weights_flattened = weights.view(-1)
        selected = torch.multinomial(
            weights_flattened, self.estimator_params["n_rays"], replacement=False
        )
        width = gt_image.shape[2]

        pixels_y, pixels_x = selected // width, selected % width
        params = self.proposal_network_pass(viewpoint_cam, scene, pixels_x, pixels_y)
        rgb, extras, render_pkg = self.predict_and_render(
            params["origins"],
            params["normalized_dirs"],
            params["t_starts"],
            params["t_ends"],
            bg,
            # with_directions=True
            with_directions=False
        )

        selected_gt = gt_image.permute(1, 2, 0)[pixels_y, pixels_x]
        l1_nerf = torch.abs(rgb - selected_gt)
        extras["depth"] = 1 / (extras["depth"] / scene.gs2nerf_scale)

        if self.nerf_cfg.nerf_train_during_3dgs:
            self.estimator.update_every_n_steps(
                extras["trans"],
                self.estimator_params["requires_grad"],
                loss_scaler=1024,
            )

            l1 = F.smooth_l1_loss(rgb, selected_gt, reduction="mean")
            self.grad_scaler.scale(l1).backward()

            self.model_optimizer.step()
            self.model_optimizer.zero_grad(set_to_none=True)

            self.model_scheduler.step()

        with torch.no_grad():
            l1_diff = torch.ones_like(l1_nerf)
            scaling_mult = cfg.optimization.densification_gaussian_size
            
            if "features_rest" in render_pkg:
                features_dc, features_rest = (
                    render_pkg["features_dc"],
                    render_pkg["features_rest"],
                )
            else:
                features_dc, features_rest = None, None
            xyz = self.select_for_densification(
                gaussians,
                scene,
                render_pkg["xyz"],
                rgb,
                # selected_gt,
                extras["alphas"],
                l1_diff.mean(1),
                render_pkg["tdist"],
                params["pixel_width_mult"],
                params["unnormalized_directions"],
                scaling_mult=scaling_mult,
                cumsum_jiggle=self.nerf_cfg.cumsum_jiggle,
                features_dc=features_dc,
                features_rest=features_rest,
                densities=render_pkg["sigmas"],
                xyz_depth=None,
                # color_samples=render_pkg["rgb"]
                )
            self.estimator_params["requires_grad"] = False
            return pixels_y, pixels_x, extras["depth"].detach()

    def log_images(
        self,
        iteration,
        image=None,
        l1_full_image=None,
        l1_diff=None,
        gt_image=None,
        mask=None,
        selected_rays=None,
        valid_rays=None,
        image_no_rays=None,
        pixels_x=None,
        pixels_y=None,
        rays_rgb=None,
        depth=None,
    ):
        os.makedirs("../images", exist_ok=True)

        prefix = f"../images/{iteration}"

        if image is not None:
            save_image(image.detach(), f"{prefix}_image.png")

        if l1_full_image is not None:
            save_image(l1_full_image, f"{prefix}_l1_full.png", permute=False, mean_channel=True)

        if l1_diff is not None and mask is not None:
            l1 = torch.zeros_like(mask).float()
            normalized_diff = l1_diff.mean(1) / l1_diff.mean(1).max()
            l1[mask.bool()] = normalized_diff
            l1 = torch.clamp(l1, 0, 1)
            save_mask(l1, f"{prefix}_l1_diff.png")

        if gt_image is not None:
            save_image(gt_image, f"{prefix}_gt_image.png")

        if mask is not None:
            save_mask(mask, f"{prefix}_mask.png")

        if mask is not None and selected_rays is not None:
            save_selected(mask, selected_rays, f"{prefix}_selected_rays.png")

        if mask is not None and valid_rays is not None:
            save_selected(mask, valid_rays, f"{prefix}_valid_rays.png")

        if image_no_rays is not None:
            save_image(image_no_rays, f"{prefix}_image_no_rays.png")

        if pixels_x is not None and pixels_y is not None:
            pixel_mask = torch.zeros(gt_image.shape[1], gt_image.shape[2])
            pixel_mask[pixels_y, pixels_x] = 1
            save_mask(pixel_mask, f"{prefix}_selected_rays.png")

            if rays_rgb is not None:
                image_rays = torch.zeros(3, gt_image.shape[1], gt_image.shape[2])
                image_rays[:, pixels_y, pixels_x] = rays_rgb.T.detach().cpu()
                save_image(image_rays, f"{prefix}_image_rays.png")

        if depth is not None:
            if depth.ndim == 3:
                depth = depth.squeeze(0)  # In case it's [1, H, W]
            save_depth_map(depth, f"{prefix}_depth.png")

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.hash_model.state_dict(),
                "proposal_networks": 
                    [p.state_dict() for p in self.proposal_networks],
            },
            path,
        )

    def load(self):
        checkpoint = torch.load(self.nerf_cfg.trained_hash_model_path)
        self.hash_model.load_state_dict(checkpoint["model_state_dict"])
        for p, state_dict in zip(self.proposal_networks, checkpoint["proposal_networks"]):
            p.load_state_dict(state_dict)

    @torch.no_grad()
    def render_test(self, scene, bg, viewpoint_cam=None, save_images=True, img_id=None):
        
        height, width = (
            viewpoint_cam.original_image.shape[1],
            viewpoint_cam.original_image.shape[2],
        )
        num_pixels = height * width
        rendered_image = torch.zeros_like(viewpoint_cam.original_image).view(3, -1).cuda()
        rendered_depth = torch.zeros((1, num_pixels)).cuda()

        flat_indices = torch.arange(num_pixels).cuda()
        prev_n_rays = self.estimator_params["n_rays"]
        self.estimator_params["n_rays"] = 10000
        batches = flat_indices.split(self.estimator_params["n_rays"])

        for batch in batches:
            pixels_y, pixels_x = batch // width, batch % width
            self.estimator_params["n_rays"] = len(batch)

            params = self.proposal_network_pass(
                viewpoint_cam, scene, pixels_x=pixels_x, pixels_y=pixels_y, 
            )
            rgb, extras, render_pkg = self.predict_and_render(
                params["origins"],
                params["normalized_dirs"],
                params["t_starts"],
                params["t_ends"],
                bg,
            )

            rendered_image[:, batch] = rgb.T
            rendered_depth[:, batch] = extras["depth"].squeeze(1) / scene.gs2nerf_scale

        rendered_image = rendered_image.view(viewpoint_cam.original_image.shape)
        rendered_depth = rendered_depth.view(viewpoint_cam.original_image.shape[1:])
        self.log_images(
            0 if img_id is None else f"rendered_{img_id}",
            image=rendered_image,
            gt_image=viewpoint_cam.original_image,
            depth=rendered_depth,
        )

        psnr = None
        loss = F.smooth_l1_loss(
            rendered_image, viewpoint_cam.original_image.cuda(), reduction="mean"
        )
        psnr = 10 * torch.log10(1 / loss)
        print("Loss", loss, "PSNR", psnr)
        psnr = psnr.item()
        self.estimator_params["n_rays"] = prev_n_rays
        return rendered_image, rendered_depth, psnr

    def train(self):
        self.hash_model.train()
        for p in self.proposal_networks:
            p.train()
        self.estimator.train()

    def eval(self):
        self.hash_model.eval()
        for p in self.proposal_networks:
            p.eval()
        self.estimator.eval()