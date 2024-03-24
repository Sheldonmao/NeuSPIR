import torch
import torch.nn as nn

from models.model_utils.network_utils import get_mlp


class VolumeMaterial(nn.Module):
    ''' get material properties from location features, created by Shi Mao
    '''
    def __init__(self, config):
        super(VolumeMaterial, self).__init__()
        self.config = config
        self.detach = self.config.detach
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_input_dims = self.config.input_feature_dim 

        # notes on output directions:
        # if self.n_output_dims==7 represent roughness,diffuse(red,gree,blue), specular(red,gree,blue)
        # if self.n_output_dims==6 represent diffuse(red,gree,blue), specular(occlusion,roughness,metalness)
        # if self.n_output_dims==5 represent diffuse(red,gree,blue), specular(roughness,metalness), meaning no occlusion is included
        self.n_output_dims = self.config.output_feature_dim 
        
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    

        self.network = network
    
    def forward(self, features):
        # print('features.shape',features.shape)
        network_inp = features.view(-1, features.shape[-1]) # N_samples x feature_channel
        # print('network_inp.shape',network_inp.shape)
        material = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        
        material = torch.sigmoid(material) # limit the material to range (0,1)
        return material

    def update_step(self, epoch, global_step):
        pass

    def regularizations(self, out):
        patch_end = out['kd'].shape[0]//128*64

        # opacity = out['opacity'][:patch_end].reshape(-1,64) # N x 64
        # mask = opacity>0.001  # N x 64 
        kd = out['kd'][:patch_end].reshape(-1,64,3) # N x 64 x 3
        rough = out['rough'][:patch_end].reshape(-1,64) # N x 64
        metal = out['metal'][:patch_end].reshape(-1,64) # N x 64
        
        occ = out['occ'][out['opacity']>0.8] #  valid occlusion

        # kd_masked = masked_tensor(kd,mask[:,:,None].repeat(1,1,3))
        # rough_masked = masked_tensor(rough,mask)
        # metal_masked = masked_tensor(metal,mask)

        return {
            'kd_smooth':kd.std(dim=1).mean(),
            'rough_smooth':rough.std(dim=1).mean(),
            'metal_smooth':metal.std(dim=1).mean(),
            'occ_mean': occ.mean()
            }
        # return {}



class VolumeMaterialSep(nn.Module):
    ''' get material properties from location features, seperating occlusion and material network created by Shi Mao
    '''
    def __init__(self, config):
        super(VolumeMaterialSep, self).__init__()
        self.config = config
        self.detach = self.config.detach
        self.n_input_dims = self.config.input_feature_dim 

        # notes on output features: n_output_dims==6 represent diffuse(red,gree,blue), specular(occlusion,roughness,metalness)
        # notes on input features: [0] for sdf value, [1:65] for semantic feature, [65:n_input_features] for non-semantic features

        self.mat_net = get_mlp(self.n_input_dims, 5, self.config.mlp_network_config)      # material depend on 
        self.occ_net = get_mlp(self.n_input_dims-65, 1, self.config.mlp_network_config)   # occ only depend on non-semantic features
    
    def forward(self, features):
        # print('features.shape',features.shape)
        network_inp = features.view(-1, features.shape[-1]) # N_samples x feature_channel
        # print('network_inp.shape',network_inp.shape)
        material = self.mat_net(network_inp).view(*features.shape[:-1], 5).float()
        material = torch.sigmoid(material) # limit the material to range (0,1)

        occ = self.occ_net(network_inp[:,-(self.n_input_dims-65):]).view(*features.shape[:-1], 1).float()
        occ = torch.sigmoid(occ) # limit the occ to range (0,1)
        
        # represent in order: diffuse(red,gree,blue), specular(occlusion,roughness,metalness)
        material = torch.cat([material[...,:3],occ,material[...,3:5]],dim=-1) 
        return material

    def update_step(self, epoch, global_step):
        pass

    def regularizations(self, out):
        patch_end = out['kd'].shape[0]//128*64

        # opacity = out['opacity'][:patch_end].reshape(-1,64) # N x 64
        # mask = opacity>0.001  # N x 64 
        kd = out['kd'][:patch_end].reshape(-1,64,3) # N x 64 x 3
        rough = out['rough'][:patch_end].reshape(-1,64) # N x 64
        metal = out['metal'][:patch_end].reshape(-1,64) # N x 64
        
        occ = out['occ'][out['opacity']>0.8] #  valid occlusion

        # kd_masked = masked_tensor(kd,mask[:,:,None].repeat(1,1,3))
        # rough_masked = masked_tensor(rough,mask)
        # metal_masked = masked_tensor(metal,mask)

        return {
            'kd_smooth':kd.std(dim=1).mean(),
            'rough_smooth':rough.std(dim=1).mean(),
            'metal_smooth':metal.std(dim=1).mean(),
            'occ_mean': occ.mean()
            }
        # return {}
