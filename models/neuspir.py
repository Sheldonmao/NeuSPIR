""" full model for NeuS (geometry) + nvdiffrect(light/material) """
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.model_utils.utils import chunk_batch, reflect, safe_normalize
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching
from nerfacc import rendering as vol_rendering
from models.PIR_render import light, material

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)

@models.register('neuspir')
class NeuSPIRModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.material = material.VolumeMaterial(self.config.material)
        self.light = light.create_trainable_env_rnd(self.config.light) # create an EnvrionmentLight with base of shape 6xHxWx3
        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def setup_aabb(self,z_scale):
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius*z_scale, self.config.radius, self.config.radius, self.config.radius*z_scale], dtype=torch.float32))


    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)
        

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_(self, rays, material_branch=True,texture_branch=True,env_light=None):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        # print('rays.shape:',rays.shape)

        sdf_samples = []
        sdf_grad_samples = []
        feature_samples =[]

        def alpha_fn(t_starts, t_ends, ray_indices):
            # print('alpha_fn t_starts.shape:',t_starts.shape)
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad = self.geometry(positions, with_grad=True, with_feature=False)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            return alpha[...,None]

        def rgbt_alpha_fn(t_starts, t_ends, ray_indices):
            # print('rgb_alpha_fn t_starts.shape:',t_starts.shape)
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            # print('sdf',sdf_grad,feature)
            sdf_samples.append(sdf)
            sdf_grad_samples.append(sdf_grad)
            feature_samples.append(feature)

            if len(t_starts)==0:
                return torch.zeros([0,6]).to(t_starts),torch.zeros([0,1]).to(t_starts)
            else:
                dists = t_ends - t_starts
                normal = F.normalize(sdf_grad, p=2, dim=-1)
                alpha = self.get_alpha(sdf, normal, t_dirs, dists)
                # Note: need to expand. By Shi Mao
                rgb_t = self.texture(feature, t_dirs, normal) 
                value_t = torch.concat([rgb_t,normal],dim=-1) # # (N_samples,6)

                return value_t, alpha[...,None]

        def rgbm_alpha_fn(t_starts, t_ends, ray_indices):
            # print('rgb_alpha_fn t_starts.shape:',t_starts.shape)
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)

            if not texture_branch:
                sdf_samples.append(sdf)
                sdf_grad_samples.append(sdf_grad)
                feature_samples.append(feature)
            # with torch.no_grad():
            #     _, feature_jitter = self.geometry(positions+0, with_grad=False, with_feature=True)

            if len(t_starts)==0:
                return torch.zeros([0,9]).to(t_starts),torch.zeros([0,1]).to(t_starts)
            else:
                dists = t_ends - t_starts
                normal = F.normalize(sdf_grad, p=2, dim=-1) # (N_samples,3)
                alpha = self.get_alpha(sdf, normal, t_dirs, dists)
                # material branch with detached fatures
                view_dir = t_dirs*-1

                if self.material.detach:
                    normal,alpha,view_dir,feature = normal.detach(),alpha.detach(),view_dir.detach(),feature.detach()

                material = self.material(feature)  # (N_samples,6)
                # material_jitter = self.mateial(feature_jitter) # (N_samples,6)
                rgb_m = env_light.shade(view_dir,normal,kd=material[...,:3],ks=material[...,3:6]) # (N_samples,3)
                value_m = torch.concat([rgb_m,material],dim=-1) # (N_samples,9)

                return value_m, alpha[...,None]

        with torch.no_grad():
            # packed_info: (N_rays,2) first column: index of the first sample of each ray, second column: number of samples of each ray
            # t_starts: (N_samples,1) per-sample start distance
            # t_ends: (N_samples,1) per-sample end disntance
            packed_info, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=alpha_fn,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        rv = {}
        if texture_branch: # texture branch
            value_t, opacity, depth = vol_rendering(
                packed_info,
                t_starts,
                t_ends,
                rgb_alpha_fn=rgbt_alpha_fn,
                render_bkgd=self.background_color[...,:6],
            )

            rgb_t = value_t[...,0:3] # rgb from texture model
            normal = value_t[...,3:6]
            opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

            rv.update({
                'comp_rgbt': rgb_t,
                'opacity': opacity,
                'normal':normal,
                'depth': depth,
                'rays_valid': opacity > 0,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
            })

        if material_branch:
            value_m, opacity, depth = vol_rendering(
                packed_info,
                t_starts,
                t_ends,
                rgb_alpha_fn=rgbm_alpha_fn,
                render_bkgd=self.background_color[...,:9],
            )
            
            rgb_m = value_m[...,0:3] # rgb from material model
            kd = value_m[...,3:6]
            occ = value_m[...,6:7]
            rough = value_m[...,7:8]
            metal = value_m[...,8:9]

            rv.update({
                'comp_rgbm': rgb_m,
                'rough':rough,
                'kd':kd,
                'metal':metal,
                'occ':occ
            })

            if not texture_branch:
                opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)
                rv.update({
                    'opacity': opacity,
                    'depth': depth,
                    'rays_valid': opacity > 0,
                    'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
                })

        if self.training:
            sdf_samples = torch.cat(sdf_samples, dim=0)
            sdf_grad_samples = torch.cat(sdf_grad_samples, dim=0)
            feature_samples = torch.cat(feature_samples,dim=0)

            mat_samples = self.material(feature_samples)  # (N_samples,6)
            mat_jitter_samples = self.material(feature_samples + torch.randn(feature_samples.shape).to(feature_samples)*0.01)  # (N_samples,6)
            rv.update({
                'sdf_samples': sdf_samples,
                'sdf_grad_samples': sdf_grad_samples,
                'feature_samples': feature_samples,
                'mat_samples':mat_samples,
                'mat_jitter_samples':mat_jitter_samples
            })

        return rv

    def forward(self, rays,material_branch=True,texture_branch=True, env_light=None):
        if env_light==None:
            if material_branch:
                self.light.build_mips() # important to update mips for each batch
            env_light = self.light

        if self.training:
            out = self.forward_(rays, material_branch,texture_branch,env_light)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays, material_branch=material_branch,texture_branch=texture_branch,env_light=env_light)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out, branch='texture'):
        losses = {}
        if branch == 'texture':
            # losses.update(self.geometry.regularizations(out))
            losses.update(self.texture.regularizations(out))
        elif branch == 'material':
            losses.update(self.material.regularizations(out))
            losses.update(self.light.regularizations(out))
        return losses

