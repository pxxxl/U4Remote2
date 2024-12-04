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
import time
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric, build_rotation)
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, Q_anchor, \
    encoder_anchor, decoder_anchor, \
    encoder, decoder, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size, \
    read_uints, read_ints, read_floats, read_list, \
    write_uints, write_ints, write_floats, write_list


class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]

        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        
        return out_i

class GaussianModel_dec(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 mode: str='I_frame',
                 ):
        super().__init__()
        
        self.mode = mode

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)

        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_ntc_params(self):
        params = []
        
        params.append(self.ntc.encoding_xyz.params)
        params.append(self.ntc.encoding_xy.params)
        params.append(self.ntc.encoding_xz.params)
        params.append(self.ntc.encoding_yz.params)
        
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid
    
    @property
    def get_mask_mlp(self):
        return self.mlp_mask

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features
    
    def get_ntc(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]

        mask = (x >= 0) & (x <= 1)
        mask = mask.all(dim=1)
        features = self.ntc(x[mask])
        features = self.ntc_mlp(features)

        d_feat = torch.full((x.shape[0], self.feat_dim), 0.0, dtype=torch.float32, device="cuda")
        d_offsets = torch.full((x.shape[0], 3*self.n_offsets), 0.0, dtype=torch.float32, device="cuda")
        
        if self.stage == "stage1":
            d_offsets[mask] = features
        elif self.stage == "stage2":
            d_feat[mask] = features

        d_offsets = d_offsets.reshape(-1, self.n_offsets, 3)

        return d_feat, d_offsets

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def initial_for_P_frame(self, ntc_cfg, stage):

        self.ntc = mix_3D2D_encoding(
                n_features=ntc_cfg['n_features_per_level'],
                resolutions_list=ntc_cfg["resolutions_list"],
                log2_hashmap_size=ntc_cfg["log2_hashmap_size"],
                resolutions_list_2D=ntc_cfg["resolutions_list_2D"],
                log2_hashmap_size_2D=ntc_cfg["log2_hashmap_size_2D"],
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()

        self.stage = stage
        if stage == "stage1":
            output_dim = 3*self.n_offsets
        elif stage == "stage2":
            output_dim = self.feat_dim
                
        self.ntc_mlp = nn.Sequential(
            nn.Linear(self.ntc.output_dim, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, output_dim),
        ).cuda()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        # l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        # for i in range(self._rotation.shape[1]):
        #     l.append('rot_{}'.format(i))
        return l
    
    def save_ply_for_P_frame(self, path):
        mkdir_p(os.path.dirname(path))

        d_feat, d_offsets = self.get_ntc(self._anchor)

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = (self._anchor_feat + d_feat).detach().cpu().numpy()
        offset = (self._offset + d_offsets).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, scale), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # conduct_encoding: self._scaling = activation(self._scaling)
        scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, scale), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        # rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        # rots = np.zeros((anchor.shape[0], len(rot_names)))
        # for idx, attr_name in enumerate(rot_names):
        #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(False))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(False))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        if not self.decoded_version:
            print("inverse scaling")
            scales = self.scaling_inverse_activation(torch.tensor(scales, dtype=torch.float, device="cuda"))
        else:
            scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        # self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))


    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        torch.save({
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'encoding_xyz': self.encoding_xyz.state_dict(),
            'grid_mlp': self.mlp_grid.state_dict(),
            'x_bound_min': self.x_bound_min,
            'x_bound_max': self.x_bound_max,
        }, path)


    def load_mlp_checkpoints(self, path, load_hash_grid=True):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if load_hash_grid:
            self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
            self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])

        if 'x_bound_min' in checkpoint:
            self.x_bound_min = checkpoint['x_bound_min']
            self.x_bound_max = checkpoint['x_bound_max']


    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, patched_infos=None, read_from_bits=True):
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')

        if read_from_bits:
            header_b_name = os.path.join(pre_path_name, 'header.b')
            with open(header_b_name, 'rb') as f:
                N_full = read_uints(f, 1)[0]
                N = read_uints(f, 1)[0]
                MAX_batch_size = read_uints(f, 1)[0]
                x_bound_min_x = read_floats(f, 1)[0]
                x_bound_min_y = read_floats(f, 1)[0]
                x_bound_min_z = read_floats(f, 1)[0]
                x_bound_max_x = read_floats(f, 1)[0]
                x_bound_max_y = read_floats(f, 1)[0]
                x_bound_max_z = read_floats(f, 1)[0]
                # anchor_infos_list = read_list(f)
                min_feat_list = read_list(f)
                max_feat_list = read_list(f)
                min_scaling_list = read_list(f)
                max_scaling_list = read_list(f)
                min_offsets_list = read_list(f)
                max_offsets_list = read_list(f)
                prob_hash = read_floats(f, 1)[0]

            # self.mlp_opacity = torch.load(os.path.join(pre_path_name, 'mlp_opacity.b')).cuda()
            # self.mlp_cov = torch.load(os.path.join(pre_path_name, 'mlp_cov.b')).cuda()
            # self.mlp_color = torch.load(os.path.join(pre_path_name, 'mlp_color.b')).cuda()
            # self.mlp_grid = torch.load(os.path.join(pre_path_name, 'mlp_grid.b')).cuda()

            self.mlp_opacity.load_state_dict(torch.load(os.path.join(pre_path_name, 'mlp_opacity.b')))
            self.mlp_cov.load_state_dict(torch.load(os.path.join(pre_path_name, 'mlp_cov.b')))
            self.mlp_color.load_state_dict(torch.load(os.path.join(pre_path_name, 'mlp_color.b')))
            self.mlp_grid.load_state_dict(torch.load(os.path.join(pre_path_name, 'mlp_grid.b')))

            self.x_bound_min = torch.tensor([[x_bound_min_x, x_bound_min_y, x_bound_min_z]]).cuda()
            self.x_bound_max = torch.tensor([[x_bound_max_x, x_bound_max_y, x_bound_max_z]]).cuda()

        else:
            [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash] = patched_infos
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)


        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        if self.ste_binary:
            p = torch.zeros_like(self.get_encoding_params()).to(torch.float32)
            p[...] = prob_hash
            hash_embeddings = decoder(p.view(-1), hash_b_name)  # {-1, 1}
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.b')).cuda()

        for s in range(steps):
            min_feat = min_feat_list[s]
            max_feat = max_feat_list[s]
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat_decoded = decoder_gaussian(mean, scale, Q_feat, file_name=feat_b_name, min_value=min_feat, max_value=max_feat)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            offsets_decoded = decoder_gaussian(mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name, min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N_full, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N_full, 6], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        # assert self._anchor_feat.shape == _anchor_feat.shape
        self._anchor_feat = nn.Parameter(_anchor_feat)
        # assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        # assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        # assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)


        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_decoding_for_ntc(self, pre_path_name, patched_infos=None, read_from_bits=True):
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ntc ...')

        if read_from_bits:
            header_b_name = os.path.join(pre_path_name, 'header.b')
            with open(header_b_name, 'rb') as f:
                prob_ntc = read_floats(f, 1)[0]

            # self.ntc_mlp = torch.load(os.path.join(pre_path_name, 'ntc_mlp.b')).cuda()
            self.ntc_mlp.load_state_dict(torch.load(os.path.join(pre_path_name, 'ntc_mlp.b')))
        else:
            [prob_ntc] = patched_infos

        ntc_b_name = os.path.join(pre_path_name, 'ntc.b')


        if self.ste_binary:
            p = torch.zeros_like(self.get_ntc_params()).to(torch.float32)
            p[...] = prob_ntc
            ntc_embeddings = decoder(p.view(-1), ntc_b_name)  # {-1, 1}
            ntc_embeddings = ntc_embeddings.view(-1, self.n_features_per_level)


        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        print('Start replacing parameters with decoded ones...')

        if self.ste_binary:
            # len_2D = self.ntc.encoding_xy.params.shape[0]
            # self.ntc.encoding_xy.params = nn.Parameter(ntc_embeddings[0:len_2D])
            # self.ntc.encoding_xz.params = nn.Parameter(ntc_embeddings[len_2D:len_2D*2])
            # self.ntc.encoding_yz.params = nn.Parameter(ntc_embeddings[len_2D*2:len_2D*3])

            len_3D = self.ntc.encoding_xyz.params.shape[0]
            len_2D = self.ntc.encoding_xy.params.shape[0]
            self.ntc.encoding_xyz.params = nn.Parameter(ntc_embeddings[0:len_3D])
            self.ntc.encoding_xy.params = nn.Parameter(ntc_embeddings[len_3D:len_3D+len_2D])
            self.ntc.encoding_xz.params = nn.Parameter(ntc_embeddings[len_3D+len_2D:len_3D+len_2D*2])
            self.ntc.encoding_yz.params = nn.Parameter(ntc_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

