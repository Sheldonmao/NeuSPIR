""" full model for NeuS (geometry) + nvdiffrect(light/material) """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from models.base import BaseModel
from models.model_utils.utils import chunk_batch, reflect, safe_normalize
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching
from nerfacc import rendering as vol_rendering
from models.PIR_render import light, material
import os
from utils.typing import *
from models.model_utils.ray_tracing import RayTracing
from models.model_utils import sg_utils

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

def hemisphere_sampling(
    n_samples: Int,
    surface_normal: Float[Tensor,"n_surface_pts 3"]
) -> Float[Tensor, "n_surface_pts*n_samples 3"]:

    # randomly sample directions on upper hemi-sphere
    azimuth = torch.rand(n_samples) * 2 * torch.pi
    elevation = torch.rand(n_samples)* torch.pi/2 # sampling on hemi-sphere with elevation larger than 0
    dirs:Float[Tensor,"n_samples 3"] = torch.stack(
        [
            torch.cos(elevation)*torch.cos(azimuth),
            torch.cos(elevation)*torch.sin(azimuth),
            torch.sin(elevation)
        ],
        dim=-1
    ).to(surface_normal.device)

    # adjust each direction according to surface normal
    z_axis = torch.tensor([[0.,0.,1.]]).to(surface_normal.device)
    surface_x = torch.cross(surface_normal,z_axis)
    # adjust to make all tangent value is valid
    invalid_mask = torch.norm(surface_x,dim=-1)==0
    if sum(invalid_mask)>0: 
        surface_x[invalid_mask] = torch.tensor([1.,0.,0.]).to(surface_normal.device)
    surface_x = F.normalize(surface_x,dim=-1)
    surface_y = torch.cross(surface_normal,surface_x)

    n_surface_pts = surface_normal.shape[0]
    dir_map = torch.stack([surface_x,surface_y,surface_normal],dim=-1) # (n_surface_pts,3,3)
    
    adjusted_dirs = torch.matmul(dir_map,dirs.T.unsqueeze(0)).permute(0,2,1) # (n_surface_pts, n_samples,3)
    return adjusted_dirs.float()


def query_indir_illum(
    lgtSGs: Float[Tensor, "n_surface_pts n_lgt_sgs 7/5"], 
    sample_dirs: Float[Tensor, "n_surface_pts n_dirs 3"]
) -> Float[Tensor,"n_surface_pts n_dirs 3/1"]:
    n_dirs = sample_dirs.shape[1]
    nlobe = lgtSGs.shape[1]
    lgtSGs = lgtSGs.unsqueeze(-3).expand(-1, n_dirs, -1, -1)
    sample_dirs = sample_dirs.unsqueeze(-2).expand(-1, -1, nlobe, -1)
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = lgtSGs[..., 3:4]
    lgtSGMus = lgtSGs[..., 4:]  # positive values

    pred_radiance = lgtSGMus * torch.exp(
        lgtSGLambdas * (torch.sum(sample_dirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    pred_radiance = torch.sum(pred_radiance, dim=2)
    return pred_radiance


@models.register('neuspir-indirect')
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
        
        if os.path.isfile(self.config.convert_from): 
            state_dict = torch.load(self.config.convert_from)['state_dict']
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            msg = self.load_state_dict(state_dict, strict=False)
            print("previous goemtry/material/raidiance weights loaded with msg: {}".format(msg))

        self.indirect = models.make(self.config.indirect.name, self.config.indirect)
        self.visibility = models.make(self.config.visibility.name, self.config.visibility)
        self.ray_tracer = RayTracing(self.scene_aabb)
    
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

    def batch_forward_rgbt(self,pts: Float[Tensor,"n 3"], dirs: Float[Tensor,"n 3"]) -> Float[Tensor, "n 3"]:
        sdf, sdf_grad, feature = self.geometry.to(pts)(pts, with_grad=True, with_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        rgb_t = self.texture.to(feature)(feature, dirs, normal) 
        
        return rgb_t

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
                return torch.zeros([0,13]).to(t_starts),torch.zeros([0,1]).to(t_starts)
            else:
                dists = t_ends - t_starts
                normal = F.normalize(sdf_grad, p=2, dim=-1) # (N_samples,3) normalized
                alpha = self.get_alpha(sdf, normal, t_dirs, dists)
                # material branch with detached fatures
                view_dir = t_dirs*-1 # (N_samples,3) normalized
                NdotV = torch.clamp(torch.sum(view_dir*normal, -1, keepdim=True), min=1e-4)
                ref_dir = F.normalize(2*(NdotV)*normal - view_dir) # # (N_samples,3) normalized

                if self.material.detach:
                    normal,alpha,view_dir,feature = normal.detach(),alpha.detach(),view_dir.detach(),feature.detach()

                material = self.material(feature)  # (N_samples,5)
                
                # get occlusion
                occSGs = self.visibility(positions)
                occ = sg_utils.hemisphere_int_SGs(occSGs, normal)/np.pi
                # occ = torch.clamp(occ*10,0,1)
                
                # assign ks, ks
                kd = material[:,:3]
                if material.shape[1]==5:
                    ks = torch.concat([occ,material[:,3:5]],dim=-1)
                elif material.shape[1]==6:
                    ks = torch.concat([occ,material[:,4:6]],dim=-1)
                else:
                    raise ValueError("material shape not supported")
                
                # occlusion for reflection direction
                reflect_occ = query_indir_illum(occSGs, ref_dir.unsqueeze(-2))[...,0] # (N_samples,1)
                # print(reflect_occ.shape)
                # direct illumination, use pre-integrated Rendering
                rgb_direct = env_light.shade(view_dir,normal,kd=kd,ks=ks,reflect_occ=reflect_occ) # (N_samples,3)

                # indirect illumination, use SG rendering
                lgtSGs = self.indirect(positions)
                # finalSGs = sg_utils.lambda_trick_sgs(lgtSGs,occSGs)
                # finalSGs = lgtSGs

                rgb_indirect = sg_utils.render_with_sg(positions, normal, view_dir, 
                   lgtSGs, specular_reflectance=ks[:,2:3], roughness=ks[:,1:2], diffuse_albedo=kd,
                   comp_vis=False, VisModel=None)
                rgb_indirect['sg_rgb'] = rgb_indirect['sg_specular_rgb'] * reflect_occ + rgb_indirect['sg_diffuse_rgb']  * ks[:,0:1]

                # rgb_m = rgb_direct + rgb_indirect['sg_rgb']

                value_m = torch.concat([rgb_direct,rgb_indirect['sg_rgb'],kd,ks],dim=-1) # (N_samples,12)
                if value_m.shape[-1]!=12: # assure valid result if rgb_indirect['sg_rgb'] is empty
                    rgb_indirect['sg_rgb'] = torch.zeros_like(rgb_direct)
                    value_m = torch.concat([rgb_direct,rgb_indirect['sg_rgb'],kd,ks],dim=-1) # (N_samples,12)
                
                value_m = torch.concat([value_m,reflect_occ],dim=-1) # (N_samples,13)
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
                'rays_valid': opacity > 0.5,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
            })

        if material_branch:
            value_m, opacity, depth = vol_rendering(
                packed_info,
                t_starts,
                t_ends,
                rgb_alpha_fn=rgbm_alpha_fn,
                render_bkgd=self.background_color[...,:13],
            )
            
            rgb_direct = value_m[...,0:3] # rgb from material model
            rgb_indirect = value_m[...,3:6] # rgb from material model
            rgb_m = rgb_direct + rgb_indirect # rgb from material model
            kd = value_m[...,6:9]
            occ = value_m[...,9:10]
            rough = value_m[...,10:11]
            metal = value_m[...,11:12]
            reflect_occ = value_m[...,12:13]
            opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

            rv.update({
                'comp_rgbm': rgb_m,
                'comp_rgb_direct': rgb_direct,
                'comp_rgb_indirect': rgb_indirect,
                'rough':rough,
                'kd':kd,
                'metal':metal,
                'occ':occ,
                'reflect_occ':reflect_occ
            })

            if not texture_branch:
                rv.update({
                    'opacity': opacity,
                    'depth': depth,
                    'rays_valid': opacity > 0.5,
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
            
            # supervise occlusion and indirect illuminaiton term
            surface_pts:Float[Tensor, "n_surface_pts 3"] = (rays_o + rays_d * depth.unsqueeze(1))[opacity>0.5] # surface point, i.e. where the camera ray hits the surface
            
            
            # query indirect illumination SGs for surface point
            indirect_sgs = torch.ones(surface_pts.shape[0], self.indirect.num_lgt_sgs, 7).to(surface_pts.device)
            indirect_sgs[:, :, -3:] = 0
            if surface_pts.shape[0] > 0:
                indirect_sgs= self.indirect(surface_pts)

            trace_radiance = self.trace_radiance(surface_pts,self.config.sample_dirs)
            pred_indirect_rgb = query_indir_illum(indirect_sgs, trace_radiance['sample_dirs'])

            rv.update({
                'sample_dirs': trace_radiance['sample_dirs'],
                'gt_indirect_rgb': trace_radiance['trace_radiance'],
                'pred_indirect_rgb': pred_indirect_rgb,
                'gt_occ': trace_radiance['gt_occ'],
                'pred_occ': trace_radiance['pred_occ']
            })

        return rv
    
    def trace_radiance(self, surface_pts, n_dirs=16):
        n_surface_pts = surface_pts.shape[0]
        surface_sdf, surface_sdf_grad = self.geometry(surface_pts, with_grad=True, with_feature=False)
        surface_normal:Float[Tensor, "n_surface_pts 3"] = F.normalize(surface_sdf_grad,dim=-1) # nomalized surface normal vector
        sample_dirs:Float[Tensor, "n_surface_pts n_dirs 3"] = hemisphere_sampling(n_dirs,surface_normal) # sample directions on upper hemisphere
        sample_dirs = sample_dirs.detach()
        
        trace_radiance = torch.zeros(n_surface_pts, n_dirs, 3).cuda()
        gt_occ = torch.zeros(n_surface_pts, n_dirs, 1).bool().cuda()
        pred_occ = torch.zeros(n_surface_pts, n_dirs, 2).cuda()

        if n_surface_pts > 0:
            # find second intersections
            with torch.no_grad():
                # surface_pts = surface_pts + 0.1*sample_dirs
                sec_points, sec_net_object_mask, sec_dist = self.ray_tracer(
                                    sdf=lambda x: self.geometry(x, with_grad=False, with_feature=False),
                                    cam_loc=surface_pts,
                                    object_mask=torch.ones(surface_pts.shape[0] * n_dirs).bool().to(surface_pts.device),
                                    ray_directions=sample_dirs)

            hit_points = sec_points[sec_net_object_mask]
            hit_viewdirs = -sample_dirs.reshape(-1, 3)[sec_net_object_mask]

            # get traced radiance
            mask_radiance = torch.zeros_like(sec_points).cuda()
            mask_radiance[sec_net_object_mask] = self.batch_forward_rgbt(hit_points, hit_viewdirs)
            trace_radiance = mask_radiance.reshape(surface_pts.shape[0], n_dirs, 3)
            
            # get visibility from network
            # input_p = surface_pts.unsqueeze(1).expand(-1, n_dirs, 3) # repeat for n_dirs
            occ_sgs = self.visibility(surface_pts)
            # query_vis = sg_utils.hemisphere_int_SGs(vis_sgs, surface_normal)
            pred_occ:Float[Tensor,"n_pts n_dirs 1"] = query_indir_illum(occ_sgs, sample_dirs)
            gt_occ:Float[Tensor,"n_pts n_dirs 1"] = sec_net_object_mask.reshape(surface_pts.shape[0], n_dirs, 1)

        output = {'trace_radiance': trace_radiance,
                  'sample_dirs': sample_dirs,
                  'gt_occ': gt_occ,
                  'pred_occ': pred_occ
                  }

        return output

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


