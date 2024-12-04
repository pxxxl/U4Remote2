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
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_dec import GaussianModel_dec
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        self.x_bound = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
            self.x_bound = 1.3
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "checkpoint.pth"))
        else:
            # self.gaussians.init_pcd_bound(scene_info.point_cloud.points)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class Scene_of_frame:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, shuffle=True, resolution_scales=[1.0], ply_path=None, init_points=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.ref_path = args.ref_path
        self.ref_iter = args.ref_iter

        if self.ref_iter == -1:
            self.ref_iter = searchForMaxIteration(os.path.join(self.ref_path, "point_cloud"))
        
        print("Loading reference model at iteration {} from {}".format(self.ref_iter, self.ref_path))

        self.train_cameras = {}
        self.test_cameras = {}

        self.x_bound = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
            self.x_bound = 1.3
        else:
            assert False, "Could not recognize scene type!"

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        
        if self.gaussians.mode == "P_frame":
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.ref_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.ref_iter),
                                                        "point_cloud.ply"))
        
            self.gaussians.load_mlp_checkpoints(os.path.join(self.ref_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.ref_iter),
                                                        "checkpoint.pth"), False)

        elif self.gaussians.mode == "I_frame":            
            self.gaussians.load_pcd_bound(os.path.join(self.ref_path,
                                                                "point_cloud",
                                                                "iteration_" + str(self.ref_iter),
                                                                "checkpoint.pth"))
                                                                
            pcd = self.gaussians.load_anchor_to_ply(os.path.join(self.ref_path,
                                                                "point_cloud",
                                                                "iteration_" + str(self.ref_iter),
                                                                "point_cloud.ply"), init_points)
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if self.gaussians.mode == "I_frame":
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:
            self.gaussians.save_ply_for_P_frame(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.save_ntc_checkpoints(os.path.join(point_cloud_path, "NTC.pth"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class Scene_decoder:

    gaussians : GaussianModel_dec

    def __init__(self, args : ModelParams, gaussians : GaussianModel_dec, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.ref_path = args.ref_path
        self.ref_iter = args.ref_iter

        self.test_cameras = {}

        self.x_bound = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
            self.x_bound = 1.3
        else:
            assert False, "Could not recognize scene type!"

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        
        if self.gaussians.mode == "P_frame":
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.ref_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.ref_iter),
                                                        "point_cloud.ply"))
        
            self.gaussians.load_mlp_checkpoints(os.path.join(self.ref_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.ref_iter),
                                                        "checkpoint.pth"), False)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if self.gaussians.mode == "I_frame":
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:
            self.gaussians.save_ply_for_P_frame(os.path.join(point_cloud_path, "point_cloud.ply"))
            # self.gaussians.save_ntc_checkpoints(os.path.join(point_cloud_path, "NTC.pth"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class Scene_dec_render:

    gaussians : GaussianModel_dec

    def __init__(self, args : ModelParams, gaussians : GaussianModel_dec, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}

        self.x_bound = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
            self.x_bound = 1.3
        else:
            assert False, "Could not recognize scene type!"

        # if not self.loaded_iter:
        #     if ply_path is not None:
        #         with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #             dest_file.write(src_file.read())
        #     else:
        #         with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #             dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # self.cameras_extent = scene_info.nerf_normalization["radius"]


        for resolution_scale in resolution_scales:
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
        self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "checkpoint.pth"))

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]