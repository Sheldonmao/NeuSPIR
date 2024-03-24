import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.model_utils.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy, SSIM
from lpips import LPIPS
from systems.utils import parse_optimizer, parse_scheduler

from models.PIR_render.util import cubemap_to_latlong
from models.PIR_render.light import load_env
import models.PIR_render.renderutils as ru
import time
import imageio


# helper functions
def valid_random_patch(index, img_wh, patch_size,all_fg_masks,device):
    ''' crop valid (masked as foreground) fixed-sized patches from images.

    args:
        index (int): previous index, needed to expand
        img_wh (list): list of image [width, height]
        patch_size (int): eg for a 8x8 patch, patch_size=8
        all_fg_masks (torch.tensor): with shape (N, H, W), foreground mask for all images
        device (str?): device of the tensor to be assigend to.
    '''
    index = index.repeat([patch_size**2,1]).T.flatten() #(N_rays//64 x 64) -> (N_rays)
    
    x_list,y_list = [],[]
    for idx in index[::patch_size**2]:
        x = torch.randint(0, img_wh[0]-patch_size, size=(1,), device=device)
        y = torch.randint(0, img_wh[1]-patch_size, size=(1,), device=device) 
        while all_fg_masks[idx, y, x]<=0.00001:
            x = torch.randint(0, img_wh[0]-patch_size, size=(1,), device=device)
            y = torch.randint(0, img_wh[1]-patch_size, size=(1,), device=device) 
        x_list.append(x)
        y_list.append(y)
    x = torch.concat(x_list) # (N_rays//64)
    y = torch.concat(y_list) # (N_rays//64)

    dx = torch.arange(patch_size,device=device) # (8)
    dy = torch.arange(patch_size,device=device) # (8)
    dx,dy = torch.meshgrid(dx,dy,indexing='ij')  # (8x8), (8x8)
    
    x = (x[:,None] + dx.reshape(1,patch_size**2)).flatten() # (N_rays//64 x 64) -> (N_rays)
    y = (y[:,None] + dy.reshape(1,patch_size**2)).flatten() # (N_rays//64 x 64) -> (N_rays)

    return index,x,y

def kl_Bern(rho_hat,rho=0.05):
    ''' KL divergence of two Bernoulli distributions: latent code rho_hat and target distribution rho
    inspired by: https://github.com/zju3dv/InvRender/blob/45e6cdc5e3c9f092b5d10e2904bbf3302152bb2f/code/model/loss.py

    args:
        rho_hat (torch.tensor): with shape (N,D). latent code Bernoulli distribution (before sigmoid that transform the value to a distribution)
        rho (float): Target Bernoulli distribution, denoted by the probability of rho==1, if the target distrinbution is sparse, rho~=0, e.g. rho=0.05
    '''
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

@systems.register('neusPIRIndirect-system')
class NeuSPIRIndirectSystem(BaseSystem):
    """ NeuS-PIR system

    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        """ prepare data for the model 


        """
        self.automatic_optimization = False # use manual optimization 
        self.material_branch=False # default setting material branch unused
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
            'lpips':LPIPS(net='alex')
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch, material_branch=True, texture_branch=True,env_light=None):
        out = self.model(batch['rays'],material_branch,texture_branch,env_light)
        return out
    
    def preprocess_data(self, batch, stage):
        device = self.dataset.all_images.device

        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays//128,), device=device)
                # index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=device)
            else:
                # index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=device)
                raise NotImplementedError('not implemented for single image')
        if stage in ['train']:
            index,x,y = valid_random_patch(index, self.dataset.img_wh, 8, self.dataset.all_fg_masks, device)
            index_add = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays - len(index),), device=device)
            x_add = torch.randint(0, self.dataset.w, size=(len(index_add),), device=device)
            y_add = torch.randint(0, self.dataset.h, size=(len(index_add),), device=device)

            index = torch.concat([index,index_add])
            x = torch.concat([x,x_add])
            y = torch.concat([y,y_add])

            # x = torch.randint(
            #     0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            # )
            # y = torch.randint(
            #     0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            # )

            c2w = self.dataset.all_c2w[index]
            directions = self.dataset.pixels_to_rays(y,x,index)
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1)
        else:
            if hasattr(self.dataset,'directions'):
                c2w = self.dataset.all_c2w[index][0]
                directions = self.dataset.directions
            else:
                c2w = self.dataset.all_c2w[index]
                directions = self.dataset.all_dirs[0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index].view(-1)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background == 'white':
                self.model.background_color = torch.ones((13,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background == 'random':
                self.model.background_color = torch.rand((13,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background == 'black':
                self.model.background_color = torch.zeros((13,), dtype=torch.float32, device=self.rank)
                # rgb = torch.rand((3,), dtype=torch.float32, device=self.rank)
                # zeros = torch.zeros((6,), dtype=torch.float32, device=self.rank)
                # self.model.background_color = torch.cat([rgb,zeros],dim=0)
            else:
                raise NotImplementedError
        else:
            if self.config.model.background == 'black':
                self.model.background_color = torch.zeros((13,), dtype=torch.float32, device=self.rank)
            else:
                self.model.background_color = torch.ones((13,), dtype=torch.float32, device=self.rank)
        
        rgb = rgb * fg_mask[...,None] + self.model.background_color[:3] * (1 - fg_mask[...,None])
        
        # update light_probe if available
        if hasattr(self.dataset, 'all_light_prob'):
            batch.update({
                'light_prob':self.dataset.all_light_prob[index]
            })
        
        if hasattr(self.dataset, 'all_albedo_images'):
            albedo = self.dataset.all_albedo_images[index].view(-1, self.dataset.all_albedo_images.shape[-1])
            albedo = albedo * fg_mask[...,None] + self.model.background_color[:3] * (1 - fg_mask[...,None])
            batch.update({
                'gt_albedo':albedo
            })

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })
    

    def loss_rgb(self,pred_rgb,gt_rgb,name='rgbt'):
        loss_rgb_mse = F.mse_loss(pred_rgb, gt_rgb)
        loss_rgb_mse_w = loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)
        self.log(f'train/loss_{name}_mse', loss_rgb_mse)
        self.log(f'train_w/loss_{name}_mse', loss_rgb_mse_w)
        loss = loss_rgb_mse_w

        loss_rgb_l1 = F.l1_loss(pred_rgb, gt_rgb)
        loss_rgb_l1_w = loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)
        self.log(f'train/loss_{name}_l1', loss_rgb_l1)
        self.log(f'train_w/loss_{name}_l1', loss_rgb_l1_w)
        loss += loss_rgb_l1_w

        return loss

    def training_step(self, batch, batch_idx):
        if self.global_step>=self.config.system.material_start_steps:
            self.material_branch=True

        out = self(batch, material_branch = self.material_branch, texture_branch = True)
        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        loss += self.loss_rgb(out['comp_rgbt'], batch['rgb'], 'rgbt')

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        loss_eikonal_w = loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        self.log('train/loss_eikonal', loss_eikonal)
        self.log('train_w/loss_eikonal', loss_eikonal_w)
        loss += loss_eikonal_w
        
        opacity = torch.clamp(out['opacity'], 1.e-3, 1.-1.e-3)
        # opacity = torch.clamp(out['opacity'], 0., 1.)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        loss_mask_w = loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.use_mask else 0.0)
        self.log('train/loss_mask', loss_mask)
        self.log('train_w/loss_mask', loss_mask_w)
        loss += loss_mask_w

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        loss_sparsity_w = loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)
        self.log('train/loss_sparsity', loss_sparsity)
        self.log('train_w/loss_sparsity', loss_sparsity_w)
        loss += loss_sparsity_w

        # enforce sparsity on latent code learned from spatial position
        loss_sparse_feat = kl_Bern(out['feature_samples'][...,1:],0.05) # drop the first channel since it is SDF
        loss_sparse_feat_w = loss_sparse_feat * self.C(self.config.system.loss.lambda_sparse_feat)
        self.log('train/loss_sparse_feat', loss_sparse_feat)
        self.log('train_w/loss_sparse_feat', loss_sparse_feat_w)
        loss += loss_sparse_feat_w

        # enforce smoothness on lantent code learnded from spatial position
        loss_smooth_feat = F.l1_loss(out['mat_samples'],out['mat_jitter_samples'])
        loss_smooth_feat_w = loss_smooth_feat * self.C(self.config.system.loss.lambda_smooth_feat)
        self.log('train/loss_smooth_feat', loss_smooth_feat)
        self.log('train_w/loss_smooth_feat_w', loss_smooth_feat_w)
        loss += loss_smooth_feat_w

        # enforce indirect illumination 
        loss_indirect_rgb_mse = F.mse_loss(out['pred_indirect_rgb'], out['gt_indirect_rgb'].detach())
        loss_indirect_rgb_mse_w = loss_indirect_rgb_mse * self.C(self.config.system.loss.lambda_indirect_rgb_mse)
        self.log(f'train/loss_indirect_rgb_mse', loss_indirect_rgb_mse)
        self.log(f'train_w/loss_indirect_rgb_mse', loss_indirect_rgb_mse_w)
        loss += loss_indirect_rgb_mse_w

        # enforce indirect occlusion of different directions 
        loss_occ_mse = F.mse_loss(out['pred_occ'], out['gt_occ'].float().detach())
        loss_occ_mse_w = loss_occ_mse * self.C(self.config.system.loss.lambda_occ_mse)
        self.log(f'train/loss_occ_mse', loss_occ_mse)
        self.log(f'train_w/loss_occ_mse', loss_occ_mse_w)
        loss += loss_occ_mse_w

        losses_model_reg = self.model.regularizations(out,'texture')
        for name, value in losses_model_reg.items():
            self.log(f'train_reg/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        
        self.log('train/loss_t', loss.item(), prog_bar=True)       
        
        # self.manual_backward(loss)
        # opt_t.step()
        # sch_t.step()

        if self.material_branch:
            # import matplotlib.pyplot as plt # testing patch code 
            # patch_end = out['kd'].shape[0]//128*64
            # rgb = batch['rgb'][:patch_end].reshape(-1,8,8,3) # N x 64 x 3
            # for idx in range(rgb.shape[0]):
            #     plt.imshow(rgb[idx].cpu().numpy())
            #     plt.savefig(f'rgb_patch{idx}')
            #     plt.cla()
            loss_m = self.loss_rgb(out['comp_rgbm'], batch['rgb'], 'rgbm')
            for name, value in self.model.regularizations(out,'material').items():
                self.log(f'train_reg/loss_{name}', value)
                loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
                loss_m += loss_
            self.log('train/loss_m', loss_m.item(), prog_bar=True)
            loss += loss_m

        # schedule optimaztion steps
        opt_t, opt_m = self.optimizers()
        sch_t, sch_m = self.lr_schedulers()

        opt_t.zero_grad()
        opt_m.zero_grad()
        self.manual_backward(loss)
        opt_t.step()
        sch_t.step()
        if self.material_branch:
            opt_m.step()
            sch_m.step()
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        # self.eval()
        with torch.no_grad():
            # update optional external light prob 
            if 'light_prob' in batch.keys():
                light = load_env(batch['light_prob'])
                out = self(batch,env_light=light)
            else:
                out = self(batch)

            # visualize output 
            if hasattr(self.dataset, 'img_wh'):
                W, H = self.dataset.img_wh
            else:
                W, H = self.config.dataset.img_wh
            
            img_gt = batch['rgb'].view(H, W, 3).cpu()
            img_tex = out['comp_rgbt'].view(H, W, 3).cpu()
            img_mat = out['comp_rgbm'].view(H, W, 3).cpu()
            img_direct = out['comp_rgb_direct'].view(H, W, 3).cpu()
            img_indirect = out['comp_rgb_indirect'].view(H, W, 3).cpu()
            kd = out['kd'].view(H, W, 3)
            
            occ = out['occ'].view(H,W)
            rough = out['rough'].view(H, W)
            metal = out['metal'].view(H,W)
            normal = out['normal'].view(H,W,3)
            reflect_occ = out['reflect_occ'].view(H,W)

            depth = out['depth'].view(H, W)
            opacity = out['opacity'].view(H, W)
            F0 = metal.view(H,W,1) * kd + (1-metal.view(H,W,1))*0.04
            self.save_image_grid(f"it{self.global_step}-val/{batch['index'][0].item()}.png", [
                {'name': 'img_gt','type': 'rgb', 'img': img_gt, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'img_tex','type': 'rgb', 'img': img_tex, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'img_mat','type': 'rgb', 'img': img_mat, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'img_direct','type': 'rgb', 'img': img_direct, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'img_indirect','type': 'rgb', 'img': img_indirect, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'normal','type': 'rgb', 'img': normal, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'depth','type': 'grayscale', 'img': depth, 'kwargs': {}},
                {'name': 'opacity','type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
                {'name': 'kd','type': 'rgb', 'img': kd, 'kwargs': {'data_format': 'HWC'}},
                {'name': 'occ','type': 'grayscale', 'img': occ, 'kwargs': {'data_range': (0, 1)}},
                {'name': 'reflect_occ','type': 'grayscale', 'img': reflect_occ, 'kwargs': {'data_range': (0, 1)}},
                {'name': 'rough','type': 'grayscale', 'img': rough, 'kwargs': {'data_range': (0, 1)}},
                {'name': 'metal','type': 'grayscale', 'img': metal, 'kwargs': {'data_range': (0, 1)}},
                {'name': 'F0','type': 'rgb', 'img': F0, 'kwargs': {'data_format': 'HWC'}},
            ])
            
            # calculate quants
            quant_dict = {'index': batch['index']}
            for name,criteria in self.criterions.items():
                quant_dict[f'tex_{name}'] = criteria(img_tex.permute(2,0,1)[None], img_gt.permute(2,0,1)[None]).flatten()
                quant_dict[f'mat_{name}'] = criteria(img_mat.permute(2,0,1)[None], img_gt.permute(2,0,1)[None]).flatten()

            return quant_dict
    
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        self.save_env_maps()
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = step_out  # {'psnrt': step_out['psnrt'],'psnrm': step_out['psnrm']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {}
                        for k,v in step_out.items():
                            if k!='index':
                                out_set[index[0].item()][k] = v[oi]
                        # out_set[index[0].item()] = {'psnrt': step_out['psnrt'][oi],'psnrm': step_out['psnrm'][oi]}
            for metric in out_set[0].keys():
                metric_value = torch.mean(torch.stack([o[metric] for o in out_set.values()]))
                self.log(f'val/{metric}', metric_value, prog_bar=False, rank_zero_only=True)     
                print(f'val/{metric}', metric_value)    


            # psnrt = torch.mean(torch.stack([o['psnrt'] for o in out_set.values()]))
            # psnrm = torch.mean(torch.stack([o['psnrm'] for o in out_set.values()]))
            # self.log('val/psnrt', psnrt, prog_bar=True, rank_zero_only=True)         
            # self.log('val/psnrm', psnrm, prog_bar=True, rank_zero_only=True)   

    def test_step(self, batch, batch_idx):
        # # save mesh and env_maps when first loaded
        if batch['index'][0]==0:
            self.save_env_maps([0.05,0.1,0.2,0.5,1])
        #     mesh = self.model.isosurface()
        #     self.save_mesh(
        #         f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
        #         mesh['v_pos'],
        #         mesh['t_pos_idx'],
        #     )
        # update optional external light prob 
        if 'light_prob' in batch.keys():
            env_light = load_env(batch['light_prob'])
            out = self(batch,env_light=env_light)
        else:
            out = self(batch)
        
        if hasattr(self.dataset, 'img_wh'):
            W, H = self.dataset.img_wh
        else:
            W, H = self.config.dataset.img_wh
        
        img_gt = batch['rgb'].view(H, W, 3).cpu()
        img_tex = out['comp_rgbt'].view(H, W, 3).cpu()
        img_mat = out['comp_rgbm'].view(H, W, 3).cpu()
        img_direct = out['comp_rgb_direct'].view(H, W, 3).cpu()
        img_indirect = out['comp_rgb_indirect'].view(H, W, 3).cpu()
        kd = out['kd'].view(H, W, 3)
        if 'gt_albedo' in batch.keys():
            albedo_gt = batch['gt_albedo'].view(H, W, 3).cpu()
            fg_mask = batch['fg_mask'].view(H,W,1)
            kd_scale_factor = batch['gt_albedo'][batch['fg_mask'].bool()].mean(dim=0) / out['kd'][batch['fg_mask'].bool()].mean(dim=0)
            kd_scaled = (kd*kd_scale_factor*fg_mask + self.model.background_color[:3] * (1 - fg_mask)).cpu()
        else:
            albedo_gt = torch.ones_like(kd).cpu()
            kd_scaled = kd.cpu()
        
        occ = out['occ'].view(H,W)
        rough = out['rough'].view(H, W)
        metal = out['metal'].view(H,W)
        normal = out['normal'].view(H,W,3)
        reflect_occ = out['reflect_occ'].view(H,W)
        F0 = metal.view(H,W,1) * kd + (1-metal.view(H,W,1))*0.04

        depth = out['depth'].view(H, W)
        opacity = out['opacity'].view(H, W)
        print(f"opacity range:[{opacity.min()},{opacity.max()}]")
        print(f"F0 range:[{F0.min()},{F0.max()}]")
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'name': 'img_gt','type': 'rgb', 'img': img_gt, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'img_tex','type': 'rgb', 'img': img_tex, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'img_mat','type': 'rgb', 'img': img_mat, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'img_direct','type': 'rgb', 'img': img_direct, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'img_indirect','type': 'rgb', 'img': img_indirect, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'normal','type': 'rgb', 'img': normal, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'depth','type': 'grayscale', 'img': depth, 'kwargs': {}},
            {'name': 'opacity','type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
            {'name': 'albedo_gt','type': 'rgb', 'img': albedo_gt, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'kd','type': 'rgb', 'img': kd, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'kd_scaled','type': 'rgb', 'img': kd_scaled, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'occ','type': 'grayscale', 'img': occ, 'kwargs': {'data_range': (0, 1)}},
            {'name': 'reflect_occ','type': 'grayscale', 'img': reflect_occ, 'kwargs': {'data_range': (0, 1)}},
            {'name': 'rough','type': 'grayscale', 'img': rough, 'kwargs': {'data_range': (0, 1)}},
            {'name': 'metal','type': 'grayscale', 'img': metal, 'kwargs': {'data_range': (0, 1)}},
            {'name': 'F0','type': 'rgb', 'img': F0, 'kwargs': {'data_format': 'HWC'}},
        ])
        # calculate quants
        quant_dict = {'index': batch['index']}
        for name,criteria in self.criterions.items():
            quant_dict[f'tex_{name}'] = criteria(img_tex.permute(2,0,1)[None], img_gt.permute(2,0,1)[None]).flatten()
            quant_dict[f'mat_{name}'] = criteria(img_mat.permute(2,0,1)[None], img_gt.permute(2,0,1)[None]).flatten()
            quant_dict[f'albedo_{name}'] = criteria(albedo_gt.permute(2,0,1)[None], kd_scaled.permute(2,0,1)[None]).flatten()
        print('test_quantities:',quant_dict)
        return quant_dict
    
    def predict_step(self,batch,batch_idx):
        """ predict using the model, while optionally relight by novel illumination
        """
        if hasattr(self.dataset, 'img_wh'):
            W, H = self.dataset.img_wh
        else:
            W, H = self.config.dataset.img_wh

        if self.config.predict.with_background:
            self.model.background_color=torch.zeros_like(self.model.background_color) # zeros the background

        #############################  render using old env_light ##################################
        ori_env_map = cubemap_to_latlong(self.model.light.base, [H, H*2])
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-env_map_ori.hdr"),ori_env_map.detach().cpu().numpy())
        out = self(batch)
        gt_img = batch['rgb'].view(H, W, 3)
        imgt = out['comp_rgbt'].view(H, W, 3)
        ori_imgm = out['comp_rgbm'].view(H, W, 3)

        if self.config.predict.with_background:
            env_bg = self.model.light.render_bg(batch['rays']).view(H, W, 3)
            opacity = out['opacity'].view(H, W, 1)     
            imgt = imgt*opacity + env_bg*(1-opacity)
            ori_imgm = ori_imgm*opacity + env_bg*(1-opacity)
            gt_img = gt_img*opacity + env_bg*(1-opacity)

        ################################### render using new env_light  ##################################
        # old_light = self.model.light.clone()
        if 'light_prob' in batch.keys():
            env_light = load_env(batch['light_prob'])
        else:
            env_light = load_env(self.config.predict.hdri,scale=1.0) 
        env_map = cubemap_to_latlong(env_light.base, [H, H*2])
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-env_map_changed.hdr"),env_map.detach().cpu().numpy())

        out = self(batch,env_light=env_light)
        # gt_img = batch['rgb'].view(H, W, 3)
        # imgt = out['comp_rgbt'].view(H, W, 3)
        imgm = out['comp_rgbm'].view(H, W, 3)

        if self.config.predict.with_background:
            env_bg = env_light.render_bg(batch['rays']).view(H, W, 3)
            imgm = imgm*opacity + env_bg*(1-opacity)
        
        # env_map = env_map/(env_map.median()*2)
        opacity = out['opacity'].view(H, W)  
        self.save_image_grid(f"it{self.global_step}-pred/{batch['index'][0].item()}.png", [
            {'name': 'gt_img','type': 'rgb', 'img': gt_img, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'ori_imgm','type': 'rgb', 'img': ori_imgm, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'imgm','type': 'rgb', 'img': imgm, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'ori_env_map','type': 'rgb', 'img': ori_env_map, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'env_map','type': 'rgb', 'img': env_map, 'kwargs': {'data_format': 'HWC'}},
            {'name': 'opacity','type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
        ],rows=1)

        self.save_image_grid(f"it{self.global_step}-pred-orienv/{batch['index'][0].item()}.png", [
            {'name': 'ori_imgm','type': 'rgb', 'img': ori_imgm, 'kwargs': {'data_format': 'HWC'}},
        ],rows=1)

        self.save_image_grid(f"it{self.global_step}-pred-newenv/{batch['index'][0].item()}.png", [
            {'name': 'imgm','type': 'rgb', 'img': imgm, 'kwargs': {'data_format': 'HWC'}},
        ],rows=1)

        psnrt = self.criterions['psnr'](out['comp_rgbt'], batch['rgb'])
        psnrm = self.criterions['psnr'](out['comp_rgbm'], batch['rgb'])

        # self.model.light = old_light.clone()

        return {
            'psnrt': psnrt,
            'psnrm': psnrm,
            'index': batch['index']
        }      
    
    def on_predict_end(self):
        """ save result in mp4 file """
        self.save_img_sequence(f"it{self.global_step}-pred",f"it{self.global_step}-pred",'(\d+)\.png',save_format='mp4',fps=10)  
        self.save_img_sequence(f"it{self.global_step}-pred-orienv",f"it{self.global_step}-pred-orienv",'(\d+)\.png',save_format='mp4',fps=10)  
        self.save_img_sequence(f"it{self.global_step}-pred-newenv",f"it{self.global_step}-pred-newenv",'(\d+)\.png',save_format='mp4',fps=10)  

    def save_env_maps(self,rough_list=[]): # rough_list: 0.05,0.1,0.2,0.5
        with torch.no_grad():
            env_map = cubemap_to_latlong(self.model.light.base, [512, 1024])
            v_min,v_max = env_map.min(),env_map.max()
            print('before env_map min max',v_min,v_max)
            self.log('env_map/min', v_min, rank_zero_only=True)
            self.log('env_map/max', v_max, rank_zero_only=True)
            
            # if required, save hdri first
            imageio.imwrite(self.get_save_path(f"it{self.global_step}-env_map.hdr"),env_map.detach().cpu().numpy())
            for rough in rough_list:
                rough_env_map = ru.specular_cubemap(self.model.light.base, rough, 0.99)
                rough_env_map = cubemap_to_latlong(rough_env_map, [512, 1024])
                imageio.imwrite(self.get_save_path(f"it{self.global_step}-env_map_rough{rough}.hdr"),rough_env_map.detach().cpu().numpy())

            env_map = env_map/(env_map.median()*2)
            print('after env_map min max',env_map.min(),env_map.max())
            self.save_image_grid(f"it{self.global_step}-env_map.png", [
                {'name': 'env_map','type': 'rgb', 'img': env_map, 'kwargs': {'data_format': 'HWC'}}
            ],rows=1)
            self.save_image_grid(f"it{self.global_step}-env_map_hdr.png", [
                {'name': 'env_map_adjusted','type': 'rgb', 'img': env_map/env_map.max(), 'kwargs': {'data_format': 'HWC'}}
            ],rows=1)

    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        if self.trainer.is_global_zero:
            mesh = self.model.isosurface()
            self.save_mesh(
                f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
                mesh['v_pos'],
                mesh['t_pos_idx'],
            )
            # out = self.all_gather(out) # note this gathering processing may generate error when trying to retrive results from other devices
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = step_out
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {}
                        for k,v in step_out.items():
                            out_set[index[0].item()][k] = v[oi]

            for metric in out_set[0].keys():
                if metric != 'index':
                    metric_value = torch.mean(torch.stack([o[metric] for o in out_set.values()]))
                    self.log(f'test/{metric}', metric_value, prog_bar=False, rank_zero_only=True)     
                    print(f'test/{metric}', metric_value)  

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )     

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })    

        if 'scheduler1' in self.config.system:
            optim1 = parse_optimizer(self.config.system.optimizer1, self.model)
            ret2 = {
                'optimizer': optim1,
                'lr_scheduler': parse_scheduler(self.config.system.scheduler1, optim1),
            }
            return (ret,ret2)
        
        return ret