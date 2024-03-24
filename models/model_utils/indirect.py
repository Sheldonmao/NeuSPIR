""" implement volume radiance mlp """



import torch
import torch.nn as nn

import models
from models.model_utils.utils import get_activation
from models.model_utils.network_utils import get_encoding, get_mlp
import numpy as np

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

@models.register('sphere-gaussians')
class SphereGaussians(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.multires= config.get('multires', 0)
        self.dims = config.get('dims',[128,128,128,128])
        self.num_lgt_sgs = config.get("num_lgt_sgs",24) # number of light Sphere-Gaussians
        self.num_mus = config.get("num_mus",3) # number of light Sphere-Gaussians

        input_dim = 3
        self.embed_fn = None
        # add multi-resolution sin and cos encoding for 3D point's positional embedding
        if self.multires > 0:
            self.embed_fn, input_dim = get_embedder(self.multires)
        self.actv_fn = nn.ReLU()

        print('indirect illumination SG network size: ', self.dims)
        lobe_layer = []
        dim = input_dim
        for i in range(len(self.dims)):
            lobe_layer.append(nn.Linear(dim, self.dims[i]))
            lobe_layer.append(self.actv_fn)
            dim = self.dims[i]
        lobe_layer.append(nn.Linear(dim, self.num_lgt_sgs * (3+self.num_mus)))
        self.lobe_layer = nn.Sequential(*lobe_layer)
        self.to("cuda")

    def forward(self, points):
        if self.embed_fn is not None:
            points = self.embed_fn.to(points)(points)

        batch_size = points.shape[0]
        output = self.lobe_layer.to(points)(points).reshape(batch_size, self.num_lgt_sgs, 3+self.num_mus)

        lgt_lobes = torch.sigmoid(output[..., :2])
        theta, phi = lgt_lobes[..., :1] * 2 * np.pi, lgt_lobes[..., 1:2] * 2 * np.pi
        
        lgt_lobes = torch.cat(
            [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)
        
        lambda_mu = output[..., 2:]
        lambda_mu[..., :1] = torch.sigmoid(lambda_mu[..., :1]) * 30 + 0.1 # lobe sharpness (0.1, 30.1)
        lambda_mu[..., 1:] = self.actv_fn(lambda_mu[..., 1:]) # lobe rgb mu

        lgt_sgs = torch.cat([lgt_lobes, lambda_mu], axis=-1) # Float[Tensor, "n_pts 4+self.num_mus"]
        
        return lgt_sgs


# @models.register('visibility')
# class VisNetwork(nn.Module):
#     def __init__(self, config): 
#         super().__init__()
#         self.points_multires= config.get('points_multires', 10)
#         self.dirs_multires= config.get('dirs_multires', 4)
#         self.dims = config.get('dims',[128,128,128,128])
        
#         p_input_dim = 3
#         self.p_embed_fn = None
#         if self.points_multires > 0:
#             self.p_embed_fn, p_input_dim = get_embedder(self.points_multires)
        
#         dir_input_dim = 3
#         self.dir_embed_fn = None
#         if self.dirs_multires > 0:
#             self.dir_embed_fn, dir_input_dim = get_embedder(self.dirs_multires)

#         self.actv_fn = nn.ReLU()

#         vis_layer = []
#         dim = p_input_dim + dir_input_dim
#         for i in range(len(self.dims)):
#             vis_layer.append(nn.Linear(dim, self.dims[i]))
#             vis_layer.append(self.actv_fn)
#             dim = self.dims[i]
#         vis_layer.append(nn.Linear(dim, 2))
#         self.vis_layer = nn.Sequential(*vis_layer)
#         self.to("cuda")

#     def forward(self, points, view_dirs):
#         if self.p_embed_fn is not None:
#             points = self.p_embed_fn(points)
#         if self.dir_embed_fn is not None:
#             view_dirs = self.dir_embed_fn(view_dirs)

#         vis = self.vis_layer(torch.cat([points, view_dirs], -1))

#         return vis