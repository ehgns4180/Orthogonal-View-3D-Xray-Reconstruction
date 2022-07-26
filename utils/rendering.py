import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import time
import numpy as np


def get_near_far(near,far,i) :
    # middle = (near+far)/2
    # eta = max(3 - 2 * i/1000,1)
    # near = middle + (near - middle) * eta
    # far = middle + (far - middle) * eta
    return near,far

def get_rays(angle,imsize,device='cuda') :
    '''
    r_o,r_d : ray origin and direction. [imsize,imsize,3]
    '''
    i = torch.linspace(-1,1,imsize,device=device)
    j = torch.linspace(-1,1,imsize,device=device)

    ii,jj = torch.meshgrid([i,j])
    kk = torch.ones_like(jj) * 2
    r_o = torch.stack([ii,jj,kk],dim=-1)

    r_d = torch.zeros_like(r_o,device=device)
    r_d[...,2] = -1
    
    rad = angle * math.pi / 180 # angle -> radian
    
    sin = math.sin(rad)
    cos = math.cos(rad)
    
    rotation_matrix = torch.tensor([[cos,0.,-sin],[0.,1.,0.],[sin,0.,cos]],device=device)    

    r_o = torch.einsum('ab,xyb -> xya',rotation_matrix,r_o)
    r_d = torch.einsum('ab,xyb -> xya',rotation_matrix,r_d)
    
    return r_o.view(-1,3).unsqueeze(0), r_d.view(-1,3).unsqueeze(0) # [1,imsize**2,3]

def sample_pts(ray_o, ray_d, N_samples,near=None,far=None,device='cuda') :
    '''
    ray_o : [B,N,3]
    ray_d : [B,N,3]
    '''
    batch_size,N_rays,_ = ray_o.shape
    
    t_vals = torch.linspace(0., 1., steps=N_samples,device=device) # Random uniform sampling
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([batch_size,N_rays, N_samples]) #[B,N,N_samples]
    # Perturbing
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    t_rand = torch.rand(z_vals.shape,device=device) * 0 # * 0 not to perturb

    # stratified samples in those intervals
    z_vals = lower + (upper - lower) * t_rand
    pts = ray_o[...,None,:] + ray_d[...,None,:] * z_vals[...,:,None] # [B,N,N_samples,3]
    return pts, z_vals

def render(sigma,depth_values,ray_directions,device):
    '''
    sigma : [B,N_rays,N_samples,sigma_dim]
    depth_values : [B,N_rays,N_samples]
    ray_directions : [B,N_rays,3]
    '''
    zeros = torch.zeros(1,device=device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            zeros.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)
    sigma = sigma.squeeze(-1)
    noise = 0.0

    # sigma_a = torch.relu(sigma + noise)
    # sigma_a = F.gelu(sigma + noise)
    sigma_a = F.silu(sigma + noise)
    # sigma_a = F.softplus(sigma + noise)

    alpha = sigma_a  * dists
    weights = alpha
    
    intensity_map = weights[..., None]
    intensity_map = intensity_map.sum(dim=-2)
    
    # depth_map = weights * depth_values
    # depth_map = depth_map.sum(dim=-1)

    # acc_map = weights.sum(dim=-1) # Accumulated map

    # disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    return intensity_map

def random_ray_sample(img, n_rays,device='cuda') :
    '''
    img : {B,C,H,W}
    '''
    _,_,H,W = img.shape
    ray_idx = torch.randperm(H*W)[: n_rays]
    return ray_idx
def full_ray_sample(img,device='cuda') :
    '''
    img : {B,C,H,W}
    '''
    _,_,H,W = img.shape
    ray_idx = torch.arange(H*W,device=device)
    return ray_idx

def get_segmentation_mask(img,config,device) :
    seg_mask = torch.zeros_like(img,device=device)
    seg_mask[img>=0.01] = 1
    seg_mask1 = seg_mask[:,0:1,...,None].repeat(1,1,1,1,config.data.imsize)
    seg_mask2 = seg_mask[:,1:2,...,None].repeat(1,1,1,1,config.data.imsize).transpose(2,4).flip([2])
    
    seg_mask = seg_mask1 * seg_mask2
    
    # Dilation
    # dilation_filter = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]],dtype=torch.float32,device=device)[None,None,...]
    # seg_mask = F.conv3d(seg_mask,dilation_filter,padding=1)
    # seg_mask[seg_mask != 0] = 1
    # seg_mask = seg_mask * 0 + 1
    
    scale_factor = config.voxel.out_dim/config.data.imsize
    seg_mask = torch.round(F.upsample(seg_mask,scale_factor=scale_factor))

    return seg_mask

def get_backprojection(img,config,device) :
    proj = img * config.training.projection_scale

    bp1 = proj[:,0:1,...,None].repeat(1,1,1,1,config.data.imsize)
    bp2 = proj[:,1:2,...,None].repeat(1,1,1,1,config.data.imsize).transpose(2,4).flip([2])
    
    bp = bp1 + bp2

    return bp