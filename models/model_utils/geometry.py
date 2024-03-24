""" implement 
1. Volume SDF MLPs for NeuS
2. Volume Density MLPs for NeRF
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities.rank_zero import _get_rank

import models
from models.base import BaseModel
from models.model_utils.utils import scale_anything, get_activation, cleanup, chunk_batch
from models.model_utils.network_utils import get_encoding, get_mlp, get_encoding_with_network


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z)
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts.to(_get_rank())
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(_get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')   

    def forward_level(self, points):
        raise NotImplementedError
    
    def isosurface_(self, vmin, vmax):
        grid_verts = self.helper.grid_vertices()
        grid_verts = torch.stack([
            scale_anything(grid_verts[...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(grid_verts[...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(grid_verts[...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)

        def batch_func(x):
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv
        
        level = chunk_batch(batch_func, self.config.isosurface.chunk, grid_verts)
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine


@models.register('volume-density')
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config)
        self.radius = self.config.radius
    
    def forward(self, points):
        points = scale_anything(points, (-self.radius, self.radius), (0, 1))
        out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1], self.n_output_dims).float()
        density, feature = out[...,0], out
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        if 'feature_activation' in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature
    
    def forward_level(self, points):
        points = scale_anything(points, (-self.radius, self.radius), (0, 1))
        density = self.encoding_with_network(points.reshape(-1, self.n_input_dims)).reshape(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        return -density      


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim 
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.radius = self.config.radius
        self.grad_type = self.config.grad_type
    
    def forward(self, points, with_grad=True, with_feature=True):
        if len(points)==0:
            sdf,grad,feature = torch.zeros([0,1]).to(points),torch.zeros([0,3]).to(points),torch.zeros([0,self.n_output_dims]).to(points)
        else:
            points = scale_anything(points, (-self.radius, self.radius), (0, 1))
            with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
                with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                    if with_grad and self.grad_type == 'analytic':
                        if not self.training:
                            points = points.clone() # points may be in inference mode, get a copy to enable grad
                        points.requires_grad_(True)
                    out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                    sdf, feature = out[...,0], out
                    if 'sdf_activation' in self.config:
                        sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                    if 'feature_activation' in self.config:
                        feature = get_activation(self.config.feature_activation)(feature)               
                    if with_grad:
                        if self.grad_type == 'analytic':
                            grad = torch.autograd.grad(
                                sdf, points, grad_outputs=torch.ones_like(sdf),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )[0]
                        elif self.grad_type == 'finite_difference':
                            eps = 0.001
                            points_d = torch.stack([
                                points + torch.as_tensor([eps, 0.0, 0.0]).to(points),
                                points + torch.as_tensor([-eps, 0.0, 0.0]).to(points),
                                points + torch.as_tensor([0.0, eps, 0.0]).to(points),
                                points + torch.as_tensor([0.0, -eps, 0.0]).to(points),
                                points + torch.as_tensor([0.0, 0.0, eps]).to(points),
                                points + torch.as_tensor([0.0, 0.0, -eps]).to(points)
                            ], dim=0).clamp(0, 1)
                            points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(6, *points.shape[:-1]).float()
                            grad = torch.stack([
                                0.5 * (points_d_sdf[0] - points_d_sdf[1]) / eps,
                                0.5 * (points_d_sdf[2] - points_d_sdf[3]) / eps,
                                0.5 * (points_d_sdf[4] - points_d_sdf[5]) / eps,
                            ], dim=-1)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv
    
    def forward_level(self, points):
        points = scale_anything(points, (-self.radius, self.radius), (0, 1))
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf        
        
