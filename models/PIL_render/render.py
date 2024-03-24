import models
import os
import imageio
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_lightning.utilities.rank_zero import _get_rank

def saturate(x, low=0.0, high=1.0):
    return torch.clamp(x, low, high)

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = saturate(x)

    switch_val = 0.04045
    return torch.where(
        x >= switch_val,
        torch.pow((torch.clamp(x, min=switch_val) + 0.055) / 1.055, 2.4),
        x / 12.92,
    )


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    with torch.name_scope("linear_to_srgb"):
        x = saturate(x)

        switch_val = 0.0031308
        return torch.where(
            x >= switch_val,
            1.055 * torch.pow(torch.clamp(x, min=switch_val), 1.0 / 2.4) - 0.055,
            x * 12.92,
        )


@models.register('pil-render')
class PILRender(nn.Module):
    def __init__(self,config):
        super(PILRender,self).__init__()

        print('loading LUT from:',config.brdf_lut_path)
        brdf_map = imageio.imread(config.brdf_lut_path, format="HDR-FI")
        self.rank = _get_rank()
        self.brdf_lut = transforms.ToTensor()(brdf_map)[None,...].to(self.rank) # (1,C,H,W)
    
    def fresnel_schlick_roughness(self, ndotv, f0, roughness):
        return f0 + (torch.clamp(1.0 - roughness, min=f0) - f0) * torch.pow(
                torch.clamp(1.0 - ndotv, min=0.0), 5.0
            )
    
    def forward(self,view_dir,normal,diffuse_irradiance,specular_irradiance,material):
        roughness,diffuse, specular = material[:,:1],material[:,1:4],material[:,4:7]
        return self.render(view_dir, normal, diffuse_irradiance, specular_irradiance, diffuse, specular, roughness)

    def render(
        self,
        view_dir: torch.Tensor,
        normal: torch.Tensor,
        diffuse_irradiance: torch.Tensor,
        specular_irradiance: torch.Tensor,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        roughness: torch.Tensor,
    ):
        """Performs the pre-integrated rendering.

        Args:
            view_dir (torch.Tensor(float32), [batch, 3]): View vector pointing
                away from the surface.
            normal (torch.Tensor(float32), [batch, 3]): Normal vector of the
                surface.
            diffuse_irradiance (torch.Tensor(float32), [batch, 3]): The diffuse
                preintegrated irradiance.
            specular_irradiance (torch.Tensor(float32), [batch, 3]): The specular
                preintegrated .
            diffuse (torch.Tensor(float32), [batch, 3]): The diffuse material
                parameter.
            specular (torch.Tensor(float32), [batch, 3]): The specular material
                parameter.
            roughness (torch.Tensor(float32), [batch, 1]): The roughness material
                parameter.

        Returns:
            rendered_rgb (torch.Tensor(float32), [batch, 3]): The rendered result.
        """
        normal = torch.where(normal == torch.zeros_like(normal), view_dir, normal)
        ndotv = (normal*view_dir).sum(dim=-1,keepdim=True)

        lin_diffuse = srgb_to_linear(diffuse)
        lin_specular = srgb_to_linear(specular)

        F = self.fresnel_schlick_roughness(
            torch.clamp(ndotv, min=0.0), lin_specular, roughness
        )
        kS = F
        kD = 1.0 - kS

        diffuse = diffuse_irradiance * lin_diffuse

        # Evaluate specular
        # (u,v) is defined as (ndotv, roughness)
        # (0,1) +---------+ (1, 1)
        #       |         |                    /\
        #       |         |           roughness |
        #       |         |
        # (0,0) +---------+ (1, 0)
        #         ndotv ->

        # We access in ij coordinates
        # (i,j)
        # (0,0) +---------+ (1, 0)
        #       |         |
        #       |         |
        #       |         |
        # (0,1) +---------+ (1, 1)

        # So the start is top left instead of bottom left:
        #       -> (1-roughness)
        # Also we swap from x,y indexing to ij
        envBrdfCoords = torch.concat(
            [
                (1 - roughness), # in hight
                torch.clamp(ndotv, min=0.0), # in width
            ],
            -1,
        ) # (B,2)

        envBrdf = nn.functional.grid_sample(self.brdf_lut, envBrdfCoords[None,None, ...],
            align_corners=True)[0,:,0]  # (C,B) Remove fake batch & Height dimension
        specular = specular_irradiance * (F * envBrdf[0][:,None] + envBrdf[1][:,None])

        # Joined ambient light
        rgb = kD * diffuse + specular

        rgb = torch.clamp(rgb,0,1)
        return rgb