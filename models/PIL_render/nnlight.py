import torch
import torch.nn as nn
import numpy as np

import models
from models.model_utils.utils import get_activation
from models.model_utils.network_utils import get_encoding, get_mlp
EPS = 1e-7

# Helper functions for Defining FiLM-SIREN Network
class Sine(nn.Module):
    """
    Sine activation function with w0 scaling support.
    Args:
        w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
    """
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim,w0=1.):
        super(FiLMLayer,self).__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.sine = Sine(w0)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        # freq = freq.unsqueeze(1).expand_as(x)
        # phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


# Helper function for env_map visualization

def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor, r=1) -> torch.Tensor:
    """
    The referrence plane is the Cartesian xy plane
    phi is inclination from the z direction.
    theta is measured from the Cartesian x axis (so that the axis has theta = +90°).

    Args:
        theta: is azimuth [0, 2pi)
        phi: is inclination [0, Pi],
        r: is length  [0, inf)

    Returns:
        The cartesian vector (x,y,z)
    """
    # x = r * torch.sin(phi) * torch.sin(theta)
    # y = r * torch.cos(phi)
    # z = r * torch.sin(phi) * torch.cos(theta)

    x = r * torch.sin(phi) * torch.cos(theta) # sintheta*cosphi,  
    y = -r * torch.sin(phi) * torch.sin(theta)  # -sintheta*sinphi,
    z = r * torch.cos(phi)  #  costheta

    return torch.concat([x, y, z], -1)

def uv_to_spherical(uvs: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uvs[..., 0], 0, 1)  
    v = torch.clamp(uvs[..., 1], 0, 1)

    theta = torch.clamp((2 * u * np.pi), 0, 2 * np.pi - EPS,)  # [0, 2*pi)
    phi = torch.clamp(np.pi * v, 0, np.pi)  # [0, pi]

    return torch.stack([theta, phi], -1)


def uv_to_direction(uvs: torch.Tensor) -> torch.Tensor:
    spherical = uv_to_spherical(uvs)
    theta = spherical[..., 0:1]
    phi = spherical[..., 1:2]
    return spherical_to_cartesian(theta, phi)


def shape_to_uv(height: int, width: int) -> torch.Tensor:
    # UV
    # 0,0              1,0
    # 0,1              1,1
    us, vs = torch.meshgrid(
        torch.linspace(
            0.0 + 0.5 / width,
            1.0 - 0.5 / width,
            width,
        ),
        torch.linspace(
            0.0 + 0.5 / height,
            1.0 - 0.5 / height,
            height,
        ),indexing='xy'
    )  # Use pixel centers
    return torch.stack([us, vs], -1).float()

# sub-module for injecting Roughness information
class RoughNet(nn.Module):
    def __init__(self,config):
        super(RoughNet,self).__init__()
        self.config = config
        self.FiLM_width = self.config.FiLM_width
        self.rough_encoding = get_encoding(1, self.config.rough_encoding_config) # need check
        rough_emb_dim = self.rough_encoding.encoding.out_dim
        self.rough_network = nn.Sequential(
            nn.Linear(rough_emb_dim,self.FiLM_width),
            nn.ELU(),
            nn.Linear(self.FiLM_width,self.FiLM_width*2)
        )
    
    def forward(self,roughness):
        
        rough_emb = self.rough_encoding(roughness)
        rough_params = self.rough_network(rough_emb)
        B,N = rough_params.shape
        rough_params = rough_params.reshape(B,2,self.FiLM_width)

        rough_phi_shift, rough_freq = rough_params[:,0],rough_params[:,1]
        return rough_phi_shift, rough_freq

# sub-module for injection light embedding
class LightNet(nn.Module):
    def __init__(self,config):
        super(LightNet,self).__init__()
        self.config = config
        self.FiLM_width = self.config.FiLM_width
        self.FiLM_depth = self.config.FiLM_depth
        self.light_emb_dim = self.config.light_emb_dim
              
        net_list = [nn.Linear(self.light_emb_dim,self.FiLM_width),
                    nn.ELU()]
        for i in range(config.light_depth):
            net_list+=[nn.Linear(self.FiLM_width,self.FiLM_width),
                nn.ELU()]
        net_list+=[nn.Linear(self.FiLM_width,self.FiLM_width*self.FiLM_depth*2)]
        self.light_network = nn.Sequential(*net_list)
    
    def forward(self,light_emb):
        ''' query light params

        light_emb: shape = (light_emb_dim)
        '''
        light_params = self.light_network(light_emb)
        light_params = light_params.reshape(2,self.FiLM_depth,self.FiLM_width)

        light_phi_shift, light_freq = light_params[0],light_params[1]
        return light_phi_shift, light_freq

# Define neural pre-integrated model
@models.register('nn-pil')
class NNPIL(nn.Module):
    ''' realize Pre-integrated lighting by neural network 
        (using FILM-SIREN as Nerual_PIL do). Created by Shi Mao
    '''
    def __init__(self, config):
        super(NNPIL, self).__init__()
        self.config = config
        FiLM_depth = self.config.FiLM_depth
        FiLM_width = self.config.FiLM_width
        light_emb_dim = self.config.light_emb_dim
        self.register_parameter('light_emb', nn.Parameter(torch.zeros(light_emb_dim)))
        
        # define light network that map lighting embedding to phase_shift and frequency for FiLM
        self.light_network = LightNet(self.config)
        
        # define conditional network that map roughness to phase_shift and frequency for FiLM
        self.rough_network = RoughNet(self.config)
        
        # define main FiLM layer
        FiLM_list = [FiLMLayer(3, FiLM_width)]
        for idx in range(FiLM_depth):
            FiLM_list.append(FiLMLayer(FiLM_width,FiLM_width))
        self.FiLM_network = nn.ModuleList(FiLM_list)
        print(self.FiLM_network)
        self.FiLM_network.apply(frequency_init(2.))
        self.FiLM_network[0].apply(first_layer_sine_init)

        self.final_layer = nn.Linear(FiLM_width,3) 
        
    
    def forward(self, light_dirs, roughness):
        ''' 
        Inputs:
            light_dirs: torch (N_sample,3)
            roughness: torch (N_sample,1)
            light_emb, torch (light_emb_dim)
        '''
        light_phi_shift, light_freq = self.light_network(self.light_emb) # (FiLM_depth,FiLM_width)
        rough_phi_shift, rough_freq = self.rough_network(roughness) # both (N_samples,FiLM_width)


        x = light_dirs
        for i, film_layer in enumerate(self.FiLM_network[:-1]):
            x = film_layer(x, light_freq[None,i], light_phi_shift[None, i])

        # final FiLM Layer should be conditional on roughness
        x = self.FiLM_network[-1](x, rough_freq, rough_phi_shift)

        x = self.final_layer(x)

        # lift x to hdr:
        x = torch.exp(nn.functional.relu(x))-1
        # x = nn.functional.relu(x)
        return x
    
    def eval_env_map(self,roughness=0.01, H=128,W=256):
        uvs = shape_to_uv(H, W)  # (H, W, 2)
        directions = uv_to_direction(uvs)  # (H, W, 3)
        light_dirs = directions.reshape(-1, 3).to(self.light_emb.device) # (H*W, 3)
        roughness = torch.ones_like(light_dirs[:,:1])*roughness

        env_map_flat = self.forward(light_dirs, roughness)
        env_map= env_map_flat.reshape(H, W, 3)

        return env_map,light_dirs.reshape(H,W,3)
    
    def regularizations(self, out):
        n=3
        std=0
        for rough in torch.rand(n):
            env_map,_ = self.eval_env_map(rough)
            std += env_map.std()
        
        return {'light_white':std/n}