import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import torchvision.transforms.functional as tff

from . import util
from . import renderutils as ru

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base,brdf_lut_path='models/PIR_render/bsdf_256_256.bin'):
        '''
        input:
            base: torch.tensor with shape 6xHxWx3, highest-resolution env_map
        '''
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.brdf_lut_path = brdf_lut_path
        self.register_parameter('env_base', self.base)
        

    def xfm(self, mtx):
        ''' store transform matrix for vector transformation **unused**
        '''
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        ''' build mips for env_map
        '''
        with torch.no_grad():
            self.clamp_(min=0.0) # make sure the env_map is in valid range
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizations(self,out):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return {'light_white': torch.mean(torch.abs(self.base - white))}
        
    def shade(self, view_dir, normal, kd, ks, specular=True,reflect_occ=1):
        ''' perform shading of a point.
        
        input: 
            view_dir: viewing direction pointing from surface to camera (N_samples x 3)
            normal: normal direction (N_samples x 3)
            kd: diffuse albedo, channels are RGB (N_samples x 3)
            ks: specular parameters,channels are (occlusion,roughness,metallic) (N_samples x 3)
            specular: weather render using specular lobe or not
            reflect_occ: occlusion of the reflected direction, if determined, should have shape (N_sample x 1)
        '''
        assert ks.shape[1]==3 and kd.shape[1]==3 
        if specular:
            roughness = ks[..., 1:2] # y component
            metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(view_dir, normal))
        nrmvec = normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec[None,None,:,:].contiguous(), filter_mode='linear', boundary_mode='cube')[0,0]
        diffuse = torch.clamp(diffuse,min=0)
        shaded_col = diffuse * diff_col * (1.0 - ks[..., 0:1]) # Modulate by ambient occlusion *hemisphere visibility* 

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(view_dir, normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile(self.brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv[None,None,:,:], filter_mode='linear', boundary_mode='clamp')[0,0]

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness[None,None,:,:])
            spec = dr.texture(self.specular[0][None, ...], reflvec[None,None,:,:].contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')[0,0]
            spec = torch.clamp(spec,min=0)
            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance * (1.0 - reflect_occ)

        shaded_col = shaded_col  
        shaded_col = torch.clamp(shaded_col,0,1)
        return util.rgb_to_srgb(shaded_col)
    
    def render_bg(self, rays):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        env_bg = dr.texture(self.base[None, ...], rays_d[None,None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0,0]
        return env_bg


######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0,contrast=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda') # H x W x 3
    latlong_img = tff.adjust_contrast(latlong_img.permute(2,0,1),contrast).permute(1,2,0)*scale 
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

# mod env hdr by adding block hight light
def mod_env_hdr(light, x=0, y=0,bs=100,black=False,intensity=1):
    '''
        x in range(-1,1) indicates the vertical location of bright block
        y in range(-1,1) indicates the horizontal location of bright block
        bs indicates the bright block size 
    '''
    latlong_img = util.cubemap_to_latlong(light.base, [512, 1024])
    max_val = intensity #latlong_img.max()
    if black:
        latlong_img[:,:,:]=0
    if type(bs)==int:
        bs_x = bs_y = bs
    else:
        bs_x,bs_y = bs
    H,W,_ = latlong_img.shape
    x_start = int((W-bs_x)/2*(x+1))
    y_start = int((H-bs_y)/2*(y+1))

    latlong_img[y_start:y_start+bs_y,x_start:x_start+bs_x] = max_val

    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

# mod env hdr by horizontally moving the light
def mod_env_hdr_mov(light,idx, period):
    latlong_img = util.cubemap_to_latlong(light.base, [512, 1024])
    div = int(idx/period*1024)
    left_part = latlong_img[:,:div].clone()
    latlong_img[:,:1024-div] = latlong_img[:,div:] 
    latlong_img[:,1024-div:] = left_part
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

# mod env hdr by cycling the darkness
def mod_env_hdr_cycle(light, idx, period = 20):
    '''
    '''
    latlong_img = util.cubemap_to_latlong(light.base, [512, 1024])
    idx = np.abs(idx % (period*2) - period)
    latlong_img = latlong_img / period * idx
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l


def load_env(fn, scale=1.0,contrast=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr" or os.path.splitext(fn)[1].lower() == ".png" :
        return _load_env_hdr(fn, scale,contrast)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(config, scale=0.5, bias=0.25):

    base = torch.rand(6, config.base_res, config.base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base,config.brdf_lut_path)
      
