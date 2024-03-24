import torch
import torch.nn.functional as F
import numpy as np
from utils.typing import *

TINY_NUMBER = 1e-6


def hemisphere_int_SGs(
    sgs:Float[Tensor,"n_pts n_sgs 4+n_mus"],
    normal:Float[Tensor,"n_pts 3"]
) -> Float[Tensor,"n_pts n_mus"]:

    lobes = sgs[...,:3]
    lambdas= sgs[...,3:4]
    mus = sgs[...,4:]
    cos_beta = torch.sum(lobes * normal.unsqueeze(1), dim=-1, keepdim=True)
    
    int_rgbs = hemisphere_int(lambdas, cos_beta) * mus 
    int_rgbs = int_rgbs.sum(dim=-2) 

    return int_rgbs



def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    
    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s

def norm_axis(x):
    return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

def lambda_trick_sgs(sgs1,sgs2):
    lobe1 = sgs1[...,:3]
    lambda1 = sgs1[...,3:4]
    mu1 = sgs1[...,4:]

    lobe2 = sgs2[...,:3]
    lambda2 = sgs2[...,3:4]
    mu2 = sgs2[...,4:]

    if mu1.shape[-1]!=mu2.shape[-1]:
        if mu1.shape[-1]==1:
            mu1 = mu1.expand(mu2.shape)
        elif mu2.shape[-1]==1:
            mu2 = mu2.expand(mu1.shape)
        else:
            raise ValueError("None of the SGs has mono Mu")

    final_lobes, final_lambdas, final_mus = lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2)

    return torch.concat([final_lobes,final_lambdas,final_mus],dim=-1)

def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    # for insurance
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)
    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus

#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg_MC(normal, viewdirs, 
                   lgtSGs, specular_reflectance, roughness, diffuse_albedo,
                   comp_vis=True, VisModel=None):
    '''
    :param normal: [batch_size, 3]; ----> camera; must have unit norm
    :param viewdirs: [batch_size, 3]; ----> camera; must have unit norm
    :param lgtSGs: [batch_size, M, 7]
    :param specular_reflectance: [1, 1]; 
    :param roughness: [batch_size, 1]; values must be positive
    :param diffuse_albedo: [batch_size, 3]; values must lie in [0,1]
    '''
    pass


def render_with_sg(points, normal, viewdirs, 
                   lgtSGs, specular_reflectance, roughness, diffuse_albedo,
                   comp_vis=True, VisModel=None):
    '''
    :param points: [batch_size, 3]
    :param normal: [batch_size, 3]; ----> camera; must have unit norm
    :param viewdirs: [batch_size, 3]; ----> camera; must have unit norm
    :param lgtSGs: [batch_size, M, 7]
    :param specular_reflectance: [1, 1]; 
    :param roughness: [batch_size, 1]; values must be positive
    :param diffuse_albedo: [batch_size, 3]; values must lie in [0,1]
    '''

    M = lgtSGs.shape[1]
    dots_shape = list(normal.shape[:-1])

    ########################################
    # light
    ########################################

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) # sharpness
    origin_lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    
    ########################################
    # specular color
    ########################################
    normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
    viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3]).detach()  # [dots_shape, M, 3]
    
    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 2. / (roughness * roughness * roughness * roughness)  # [dots_shape, 1]
    brdfSGLambdas = inv_roughness_pow4.unsqueeze(1).expand(dots_shape + [M, 1])
    mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape + [3])  # [dots_shape, 1] ---> [dots_shape, 3]
    brdfSGMus = mu_val.unsqueeze(1).expand(dots_shape + [M, 3])

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus  # [..., M, 3]

    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)

    specular_reflectance = specular_reflectance.unsqueeze(1).expand(dots_shape + [M, 3])
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    k = k.unsqueeze(1).expand(dots_shape + [M, 1])
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    vis_shadow = torch.zeros(dots_shape[0], 3).cuda()
    if comp_vis:
        # light SG visibility
        light_vis = get_diffuse_visibility(points, normal[:, 0, :], VisModel,
                                           lgtSGLobes[0], lgtSGLambdas[0], nsamp=32)
        light_vis = light_vis.permute(1, 0).unsqueeze(-1).expand(dots_shape +[M, 3])

        # BRDF SG visibility
        brdf_vis = get_specular_visibility(points, normal[:, 0, :], viewdirs[:, 0, :], 
                                           VisModel, warpBrdfSGLobes[:, 0], warpBrdfSGLambdas[:, 0], nsamp=16)
        brdf_vis = brdf_vis.unsqueeze(-1).unsqueeze(-1).expand(dots_shape + [M, 3])

        # using brdf vis if sharper
        # vis_brdf_mask = (warpBrdfSGLambdas > lgtSGLambdas).expand(dots_shape + [M, 3])
        # spec_vis = torch.zeros(dots_shape + [M, 3]).cuda()
        # spec_vis[vis_brdf_mask] = brdf_vis[vis_brdf_mask]
        # spec_vis[~vis_brdf_mask] = light_vis[~vis_brdf_mask]
        # vis_shadow = torch.mean(spec_vis, axis=1).squeeze()
        lgtSGMus = origin_lgtSGMus * brdf_vis
        vis_shadow = torch.mean(light_vis, axis=1).squeeze()
    else:
        lgtSGMus = origin_lgtSGMus

    # multiply with light sg
    final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                         warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    # [..., M, K, 3]
    specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    specular_rgb = specular_rgb.sum(dim=-2)
    specular_rgb = torch.clamp(specular_rgb, min=0.)


    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    # diffuse visibility
    if comp_vis:
        lgtSGMus = origin_lgtSGMus * light_vis
    else:
        lgtSGMus = origin_lgtSGMus 
    #  * (1. - specular_reflectance)
    diffuse = (diffuse_albedo).unsqueeze(-2).expand(dots_shape + [M, 3]) # / np.pi
    diffuse = diffuse * (1. - specular_reflectance)
    # multiply with light sg
    final_lobes = lgtSGLobes
    final_lambdas = lgtSGLambdas
    final_mus = lgtSGMus * diffuse

    # now multiply with clamped cosine, and perform hemisphere integral
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
                    final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    diffuse_rgb = diffuse_rgb.sum(dim=-2)
    diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    # for debugging
    if torch.isnan(specular_rgb).sum() > 0:
        import ipdb; ipdb.set_trace()
    if torch.isnan(diffuse_rgb).sum() > 0:
        import ipdb; ipdb.set_trace()

    # combine diffue and specular rgb
    rgb = specular_rgb + diffuse_rgb
    ret = {'sg_rgb': rgb,
           'sg_specular_rgb': specular_rgb,
           'sg_diffuse_rgb': diffuse_rgb,
           'vis_shadow': vis_shadow}

    return ret