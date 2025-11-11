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

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="eval")
parser.add_argument("--common_args", type=str, default="")
args, _ = parser.parse_known_args()

# if not args.skip_training or not args.skip_rendering:
parser.add_argument('--mipnerf360', "-m360", required=False, type=str, default=None)
parser.add_argument("--tanksandtemples", "-tat", required=False, type=str, default=None)
parser.add_argument("--deepblending", "-db", required=False, type=str,  default=None)
parser.add_argument("--ommo", "-ommo", required=False, type=str, default=None)
args = parser.parse_args()

mipnerf360_outdoor_scenes = ["bicycle", "garden", "stump"] if args.mipnerf360 else []
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"] if args.mipnerf360 else []
tanks_and_temples_scenes = ["truck", "train"] if args.tanksandtemples else []
deep_blending_scenes = ["drjohnson", "playroom"] if args.deepblending else []
ommo_scenes = ["ommo_01", "ommo_03", "ommo_05", "ommo_06", "ommo_10", "ommo_13", "ommo_14", "ommo_15"] if args.ommo else []

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(ommo_scenes)



if not args.skip_training:
    common_args = " quiet=true test_iterations=[] enable_wandb=false " + args.common_args
    
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system(f"python train.py --config-name defaults.yaml scene_name={scene} run_name={args.output_path}  gaussian_model.model_path={os.path.join(args.output_path, scene)} gaussian_model.images=images_4 {common_args} gaussian_model.source_path={args.mipnerf360}")
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system(f"python train.py --config-name defaults.yaml scene_name={scene} run_name={args.output_path}  gaussian_model.model_path={os.path.join(args.output_path, scene)} gaussian_model.images=images_2 {common_args} gaussian_model.source_path={args.mipnerf360}")
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system(f"python train.py --config-name defaults.yaml scene_name={scene} run_name={args.output_path}  gaussian_model.model_path={os.path.join(args.output_path, scene)} {common_args} gaussian_model.source_path={args.tanksandtemples} gaussian_model.resolution=1")
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system(f"python train.py --config-name defaults.yaml scene_name={scene} run_name={args.output_path}  gaussian_model.model_path={os.path.join(args.output_path, scene)} {common_args} gaussian_model.source_path={args.deepblending}")
    for scene in ommo_scenes:
        source = args.ommo + "/" + scene
        os.system(f"python train.py --config-name defaults.yaml scene_name={scene} run_name={args.output_path}  gaussian_model.model_path={os.path.join(args.output_path, scene)} {common_args} gaussian_model.source_path={args.ommo}")

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    for scene in ommo_scenes:
        all_sources.append(args.ommo + "/" + scene)

    common_args = " --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        pth = os.path.join(args.output_path, scene)
        os.system(f"python render.py --config-path {pth}")

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    print(scenes_string)
    os.system(f"python metrics.py --model_paths " + scenes_string)