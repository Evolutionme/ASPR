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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import json

from utils.loss_utils import gaussian
from utils.frequency_calibration import fit_frequency_calibration, apply_frequency_calibration


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    method = "ours_{}{}".format(iteration, args.render_suffix)
    render_path = os.path.join(model_path, name, method, "renders")
    gts_path = os.path.join(model_path, name, method, "gt")

    total_time = 0.0

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        rendering = render_fastgs(view, gaussians, pipeline, background, args.mult)["render"]
        if args.frequency_coefficient is not None:
            rendering = apply_frequency_calibration(rendering, args.frequency_coefficient, args.frequency_kernel_size)
        end_time = time.time()
        total_time += (end_time - start_time)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        print(f"Gaussian number: {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        args.frequency_coefficient = None
        if args.frequency_calibration != "none":
            args.frequency_coefficient, calibration_stats = fit_frequency_calibration(
                scene.getTrainCameras(),
                lambda view: render_fastgs(view, gaussians, pipeline, background, args.mult)["render"],
                mode=args.frequency_calibration,
                max_views=args.frequency_calibration_views,
                kernel_size=args.frequency_kernel_size,
                min_coefficient=args.frequency_min_coefficient,
                max_coefficient=args.frequency_max_coefficient,
            )
            args.frequency_coefficient = args.frequency_coefficient * args.frequency_coefficient_scale
            calibration_stats["coefficient_scale"] = args.frequency_coefficient_scale
            calibration_stats["coefficient"] = args.frequency_coefficient.detach().cpu().tolist()
            print(f"Frequency calibration mode={args.frequency_calibration}, stats={calibration_stats}")
            calibration_log = os.path.join(
                dataset.model_path,
                f"frequency_calibration{args.render_suffix or '_default'}.json",
            )
            with open(calibration_log, "w", encoding="utf-8") as log_file:
                json.dump(calibration_stats, log_file, indent=2)

        # gaussians._scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling * 0.5)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--render_suffix", type=str, default="")
    parser.add_argument("--frequency_calibration", choices=("none", "scalar", "channel"), default="none")
    parser.add_argument("--frequency_calibration_views", type=int, default=32)
    parser.add_argument("--frequency_kernel_size", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--frequency_min_coefficient", type=float, default=-0.75)
    parser.add_argument("--frequency_max_coefficient", type=float, default=0.0)
    parser.add_argument("--frequency_coefficient_scale", type=float, default=2.0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
