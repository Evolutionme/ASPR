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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, method_filter=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            scene_path = Path(scene_dir)
            direct_method = (scene_path / "renders").is_dir() and (scene_path / "gt").is_dir()
            result_dir = scene_path if direct_method else scene_path
            scene_key = str(scene_path)
            print("Scene:", scene_dir)
            full_dict[scene_key] = {}
            per_view_dict[scene_key] = {}
            full_dict_polytopeonly[scene_key] = {}
            per_view_dict_polytopeonly[scene_key] = {}

            if direct_method:
                method_dirs = [(scene_path.name, scene_path)]
            else:
                test_dir = scene_path / "test"
                method_dirs = [
                    (method, test_dir / method)
                    for method in os.listdir(test_dir)
                    if method_filter is None or method == method_filter
                ]

            for method, method_dir in method_dirs:
                print("Method:", method)

                full_dict[scene_key][method] = {}
                per_view_dict[scene_key][method] = {}
                full_dict_polytopeonly[scene_key][method] = {}
                per_view_dict_polytopeonly[scene_key][method] = {}

                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_key][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_key][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(result_dir / "results.json", 'w') as fp:
                json.dump(full_dict[scene_key], fp, indent=True)
            with open(result_dir / "per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_key], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--method', type=str, default=None,
                        help='Evaluate only this method directory under each model path.')
    args = parser.parse_args()
    evaluate(args.model_paths, args.method)
