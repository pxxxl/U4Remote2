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
import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
import logging

from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
# import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_w
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, Scene_of_frame, GaussianModel_dec, Scene_decoder, Scene_dec_render
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.encodings import anchor_round_digits, Q_anchor, encoder_anchor, get_binary_vxl_size

from lpipsPyTorch import lpips

bit2MB_scale = 8 * 1024 * 1024
run_codec = True

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def decoding_I_frame(args_param, dataset, opt, pipe, dataset_name, logger=None, ply_path=None, init=True):

    gaussians = GaussianModel_dec(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        mode = "I_frame"
    )

    if init: # True
        scene = Scene_decoder(dataset, gaussians)
    # else: # not available
    #     scene = Scene_of_frame(dataset, gaussians, ply_path=ply_path, init_points=args_param.init_points)
    iteration = opt.iterations
    bit_stream_path = os.path.join(args_param.model_path, f"iteration_{iteration}",'bitstreams')
    log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path)
    logger.info(log_info)
    scene.save(iteration)




def decoding_P_frame(args_param, dataset, opt, pipe, dataset_name, logger=None, ply_path=None):

    gaussians = GaussianModel_dec(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        decoded_version=run_codec, # True
        mode="P_frame",
    )
    scene = Scene_decoder(dataset, gaussians)
    scene.gaussians.initial_for_P_frame(args_param.ntc_cfg, args_param.stage)
    iteration = opt.iterations
    bit_stream_path = os.path.join(args_param.model_path, f"iteration_{iteration}",'bitstreams')
    log_info = scene.gaussians.conduct_decoding_for_ntc(pre_path_name=bit_stream_path)
    logger.info(log_info)

    scene.save(iteration)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    psnr_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if name == "pre_test" and idx > 6:
            return None

        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        #
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        render_image = torch.clamp(rendering.to("cuda"), 0.0, 1.0)
        psnr_view = psnr(render_image, gt_image).mean().double()
        psnr_list.append(psnr_view)

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    print('testing_float_psnr=:', sum(psnr_list) / len(psnr_list))

    return t_list, visible_count_list


def render_sets(args_param, dataset : ModelParams, iteration : int, pipeline : PipelineParams, dataset_name=None, logger=None, x_bound_min=None, x_bound_max=None):
    with torch.no_grad():
        gaussians = GaussianModel_dec(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            decoded_version=run_codec,
        )
        scene = Scene_dec_render(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        
        if x_bound_min is not None:
            gaussians.x_bound_min = x_bound_min
            gaussians.x_bound_max = x_bound_max

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        # test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
        test_fps = 1.0 / torch.tensor(t_test_list).mean()
        logger.info(f'Test FPS: {test_fps.item():.5f}')

    return visible_count


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


def evaluate(model_paths, visible_count=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
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

        logger.info(f"model_paths: {model_paths}")
        logger.info("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")



        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
    # last
    psnr_f = torch.tensor(psnrs).mean()
    ssim_f= torch.tensor(ssims).mean()
    lpips_f = torch.tensor(lpipss).mean()
    return psnr_f, ssim_f, lpips_f

def get_logger(path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

def release_logger():
    # release
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--init_name", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    assert args.config_path is not None, "Please provide a config path"
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    scene = config["scene"]
    frame_start = config["frame_start"]
    frame_end = config["frame_end"]
    GOF = config["GOF"]
    lmbda = config["lmbda"]
    P_lmbda = config["P_lmbda"]

    base_source_path = config["base_source_path"]
    base_model_path = config["base_model_path"]

    psnr_list, ssim_list, lpips_list = {}, {}, {}
    psnr_sum, ssim_sum, lpips_sum = 0, 0, 0

    serializable_namespace = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    json_namespace = json.dumps(serializable_namespace)
    os.makedirs(os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}"), exist_ok = True)
    with open(os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", "cfg_args.json"), 'w') as f:
        f.write(json_namespace)

    for frame in range(frame_start, frame_end+1):

        print("Frame:", frame)

        for key, value in config["model_params"].items():
                setattr(args, key, value)
        args.source_path = os.path.join(base_source_path, scene, f"frame{frame:06d}")
        dataset = args.source_path.split('/')[-1]
        args.lmbda = lmbda

        # I_frame
        if frame % GOF == 0: 
            args.model_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame:06d}")
            logger = get_logger(args.model_path)

            if frame == 0: # init_frame
                for key, value in config["Init_frame_params"].items():
                    setattr(args, key, value)
            else:
                for key, value in config["I_frame_params"].items():
                    setattr(args, key, value)
                args.ref_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame-1:06d}")
            
            logger.info("Decoding " + args.model_path)
            logger.info(f'args: {args}')
            decoding_I_frame(args, lp.extract(args), op.extract(args), pp.extract(args), dataset, logger, init=(frame==0))

        # P_frame
        else:
            config_P = config["P_frame_params"]
            args.ntc_cfg = config_P["ntc_cfg"]
            args.P_lmbda = P_lmbda

            # stage1
            release_logger()
            args.stage = "stage1"
            args.model_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame:06d}_offsets")
            os.makedirs(args.model_path, exist_ok=True)
            logger = get_logger(args.model_path)

            for key, value in config_P["stage1"].items():
                if key != "first_ref_iteration":
                    setattr(args, key, value)
            if (frame-1) % GOF == 0:
                args.ref_iter = config_P["stage1"]["first_ref_iteration"]
            args.ref_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame-1:06d}")

            logger.info("Decoding " + args.model_path)
            logger.info(f'args: {args}')
            decoding_P_frame(args, lp.extract(args), op.extract(args), pp.extract(args), dataset, logger)

            # stage2
            release_logger()
            args.stage = "stage2"
            args.model_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame:06d}")
            os.makedirs(args.model_path, exist_ok=True)
            logger = get_logger(args.model_path)

            for key, value in config_P["stage2"].items():
                setattr(args, key, value)
            
            args.ref_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame:06d}_offsets")

            logger.info("Decoding " + args.model_path)
            logger.info(f'args: {args}')
            decoding_P_frame(args, lp.extract(args), op.extract(args), pp.extract(args), dataset, logger)

        # All done
        logger.info("\Decoding complete.")

        # rendering
        logger.info(f'\nStarting Rendering~')
        for iteration in args.save_iterations:
            visible_count = render_sets(args, lp.extract(args), iteration, pp.extract(args), logger=logger)
        logger.info("\nRendering complete.")

        # calc metrics
        logger.info("\n Starting evaluation...")
        psnr_f, ssim_f, lpips_f = evaluate(args.model_path, visible_count=visible_count, logger=logger)
        psnr_list[f"{frame:06d}"] = psnr_f.item()
        ssim_list[f"{frame:06d}"] = ssim_f.item()
        lpips_list[f"{frame:06d}"] = lpips_f.item()
        psnr_sum += psnr_f
        ssim_sum += ssim_f
        lpips_sum += lpips_f
        logger.info("\nEvaluating complete.")

        release_logger()

    num_frames = frame_end - frame_start + 1
    mean_list = {"psnr":(psnr_sum/num_frames).item(), "ssim":(ssim_sum/num_frames).item(), "lpips":(lpips_sum/num_frames).item()}
    with open(os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frames_results.json"), 'w') as fp:
        json.dump({"mean":mean_list, "psnr":psnr_list, "ssim":ssim_list, "lpips":lpips_list}, fp, indent=True)