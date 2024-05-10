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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from torchvision.utils import make_grid
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from scene.colmap_loader import qvec2rotmat, rotmat2qvec
import numpy as np
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import imageio
import json

from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from scene.cameras import Camera

def get_gs_def_tr_from_real_tr(t,r):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = r
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    w2c = np.linalg.inv(Rt)
    R = w2c[:3, :3].transpose()
    T = w2c[:3, 3]

    return T, R

class Camera_Light(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, w, h,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_Light, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = w
        self.image_height = h

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def render_ring_cameras(dataset : ModelParams, iteration : int, pipeline : PipelineParams, frame_number: int, ring_radius: float, fps: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if args.point_cloud:
            gaussians.load_ply(args.point_cloud, reset_basis_dim=1)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        model_path = dataset.model_path
        iteration = scene.loaded_iter
        views = scene.getTrainCameras()
        newnewView = []
        for i in range(frame_number):
            theta = 2 * np.pi * i / frame_number
            newnewView.append( Camera(colmap_id=i, R=qvec2rotmat(np.array([1,0,0,0])), T=np.array([ring_radius*np.cos(theta),ring_radius*np.sin(theta),0]), 
                  FoVx=views[0].FoVx, FoVy=views[0].FoVy, 
                  image=views[0].original_image, gt_alpha_mask=None,
                  image_name=views[0].image_name, uid=i))

        # makedirs(os.path.join(model_path, name, "ours_{}".format(iteration), "video_gen"), exist_ok=True)
        allpics = []
        for idx, view in enumerate(tqdm(newnewView, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            # pic = (rendering * 255).clamp(0,255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            grid = make_grid(rendering, nrow=8, padding=2, pad_value=0,
                     normalize=False, range=None, scale_each=False)
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            pic = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
            allpics.append(pic)

        # Save video
        with imageio.get_writer(os.path.join(model_path, f"render_{iteration}.mp4"), fps=fps) as video:
            for image in allpics:
                frame = imageio.core.asarray(image)
                video.append_data(frame)

def render_path_cameras(dataset : ModelParams, iteration : int, pipeline : PipelineParams, dense_path_path , fps: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # if args.point_cloud:
        #     gaussians.load_ply(args.point_cloud, reset_basis_dim=1)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        model_path = dataset.model_path
        iteration = scene.loaded_iter
        views = scene.getTrainCameras()
        newnewView = []

        init_view = views[0]
        fovx = init_view.FoVx
        fovy = init_view.FoVy
        R = init_view.R
        T = init_view.T
        w = init_view.image_width
        h = init_view.image_height

        # now_view = Camera_Light(R=R, T=T, FoVx=fovx, FoVy=fovy, w=w, h=h)
        # rendering = render(now_view, gaussians, pipeline, background)["render"]
        # torchvision.utils.save_image(rendering, "xxxx.png")

        # read the dense path
        dense_file = os.path.join(dense_path_path, "DensePath.json")
        # Open the JSON file
        with open(dense_file) as json_file:
            tsrs = json.load(json_file)
        newnewView = []
        for onetr in tsrs:
            r = np.array(onetr["r"])
            t = np.array(onetr["t"])
            T, R = get_gs_def_tr_from_real_tr(t, r)
            newnewView.append( Camera_Light(R=R, T=T, FoVx=fovx, FoVy=fovy, w=w, h=h))
        
        allpics = []
        for idx, view in enumerate(tqdm(newnewView, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            # pic = (rendering * 255).clamp(0,255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            grid = make_grid(rendering, nrow=8, padding=2, pad_value=0,
                     normalize=False, range=None, scale_each=False)
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            pic = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
            allpics.append(pic)
        
        # Save video
        with imageio.get_writer(os.path.join(model_path, f"render_{iteration}_custom_path.mp4"), fps=fps) as video:
            for image in allpics:
                frame = imageio.core.asarray(image)
                video.append_data(frame)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-pf", "--path_folder", default='mypaths', help='Folder for path')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_path_cameras(model.extract(args), args.iteration, pipeline.extract(args), args.path_folder, 10)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # dataset = model.extract(args)
    # iteration = args.iteration
    # pipeline = pipeline.extract(args)
    # skip_train = args.skip_train
    # skip_test = args.skip_test


    # with torch.no_grad():
    #     gaussians = GaussianModel(dataset.sh_degree)
    #     scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    #     bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    #     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #     views = scene.getTrainCameras()

    #     init_view = views[0]
    #     fovx = init_view.FoVx
    #     fovy = init_view.FoVy
    #     R = init_view.R
    #     T = init_view.T
    #     w = init_view.image_width
    #     h = init_view.image_height

    #     # now_view = Camera_Light(R=R, T=T, FoVx=fovx, FoVy=fovy, w=w, h=h)
    #     # rendering = render(now_view, gaussians, pipeline, background)["render"]
    #     # torchvision.utils.save_image(rendering, "xxxx.png")

    #     # read the dense path
    #     dense_path = os.path.join("mypaths", "DensePath.json")
    #     tsrs = json.load(dense_path)
    #     newnewView = []
    #     for onetr in tsrs:
    #         r = np.array(onetr["r"])
    #         t = np.array(onetr["t"])
    #         T, R = get_gs_def_tr_from_real_tr(t, r)
    #         newnewView.append( Camera_Light(R=R, T=T, FoVx=fovx, FoVy=fovy, w=w, h=h))
        
    #     allpics = []
    #     for idx, view in enumerate(tqdm(newnewView, desc="Rendering progress")):
    #         rendering = render(view, gaussians, pipeline, background)["render"]
    #         # pic = (rendering * 255).clamp(0,255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    #         grid = make_grid(rendering, nrow=8, padding=2, pad_value=0,
    #                  normalize=False, range=None, scale_each=False)
    #         # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    #         pic = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    #         allpics.append(pic)
        
    #     # Save video
    #     with imageio.get_writer(os.path.join(model_path, f"render_{iteration}.mp4"), fps=fps) as video:
    #         for image in allpics:
    #             frame = imageio.core.asarray(image)
    #             video.append_data(frame)
        # newnewView.append( Camera(colmap_id=i, R=qvec2rotmat(np.array([1,0,0,0])), T=np.array([ring_radius*np.cos(theta),ring_radius*np.sin(theta),0]), 
        #           FoVx=views[0].FoVx, FoVy=views[0].FoVy, 
        #           image=views[0].original_image, gt_alpha_mask=None,
        #           image_name=views[0].image_name, uid=i))

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
