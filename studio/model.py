"""Differentiable 3D Gaussians, with gsplat CUDA and a portable PyTorch renderer."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PARAMETER_NAMES = {'means':'xyz','scales':'log_scale','quats':'rotation',
                   'opacities':'opacity','colors':'color','sh_rest':'sh_rest'}


def logit(x):
    return torch.logit(x.clamp(0.001, 0.999))


def sh_basis(directions, degree):
    """Real SH bases in the same order as 3DGS (DC handled separately)."""
    x,y,z=F.normalize(directions,dim=-1).unbind(-1)
    bases=[-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x]
    if degree>=2:
        bases += [1.0925484305920792*x*y,-1.0925484305920792*y*z,
                  .31539156525252005*(2*z*z-x*x-y*y),
                  -1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)]
    return torch.stack(bases[:(degree+1)**2-1],-1)


class Gaussians(nn.Module):
    def __init__(self, xyz, colors, scale=0.08, sh_degree=0, opacity=0.1, backend='torch', color_activation='sigmoid'):
        super().__init__()
        self.xyz=nn.Parameter(torch.as_tensor(xyz,dtype=torch.float32))
        initial_colors=torch.as_tensor(colors,dtype=torch.float32)
        self.color=nn.Parameter(logit(initial_colors) if color_activation=='sigmoid' else (initial_colors-.5)/.28209479177387814 if color_activation=='sh' else initial_colors.clone())
        self.color_activation=color_activation
        scales=torch.as_tensor(scale,dtype=torch.float32,device=self.xyz.device)
        self.log_scale=nn.Parameter(torch.ones_like(self.xyz)*scales.clamp_min(1e-5).log())
        q=torch.zeros(len(xyz),4,device=self.xyz.device);q[:,0]=1
        self.rotation=nn.Parameter(q)
        self.opacity=nn.Parameter(torch.full((len(xyz),),math.log(opacity/(1-opacity)),device=self.xyz.device))
        self.sh_rest=nn.Parameter(torch.zeros(len(xyz),(sh_degree+1)**2-1,3,device=self.xyz.device)) if sh_degree or color_activation=='sh' else None
        self.sh_degree=sh_degree
        self.active_sh_degree=0
        self.backend=backend

    def parameter_map(self):
        return {key:getattr(self,name) for key,name in PARAMETER_NAMES.items() if getattr(self,name) is not None}

    def synchronize_parameters(self, params):
        # gsplat replaces Parameters and remaps Adam's moments during split/clone/prune.
        for key,value in params.items(): setattr(self,PARAMETER_NAMES[key],value)

    def covariance(self):
        w,x,y,z=F.normalize(self.rotation,dim=-1).unbind(-1)
        r=torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                       2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                       2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
        rs=r*self.log_scale.clamp(-9,2).exp()[:,None,:]
        return rs@rs.transpose(1,2)

    def base_colors(self):
        return self.color.sigmoid() if self.color_activation=='sigmoid' else self.color*.28209479177387814+.5 if self.color_activation=='sh' else self.color

    def colors_for(self,c2w):
        if self.color_activation=='sh':
            from .official import official_module
            features=torch.cat([self.color[:,None],self.sh_rest],1).transpose(1,2)
            directions=F.normalize(self.xyz-c2w[:3,3],dim=-1)
            rgb=official_module('sh_utils').eval_sh(self.active_sh_degree,features,directions)+.5
            return (rgb+self.sh_rest.sum((1,2))[:,None]*0).clamp_min(0)
        colors=self.base_colors()
        if self.sh_rest is not None:
            degree=self.active_sh_degree
            if degree:
                basis=sh_basis(self.xyz-c2w[:3,3],degree)
                colors=colors+(basis[:,:,None]*self.sh_rest[:,:basis.shape[1]]).sum(1)
            else:
                # Keep zero gradients for Adam/strategy state before the SH warmup ends.
                colors=colors+self.sh_rest.sum((1,2))[:,None]*0
        return colors.clamp_min(0)

    def render(self,c2w,fx,fy,cx,cy,width,height,background=0.025,tile=1024,return_aux=False):
        view=torch.linalg.inv(c2w)
        flip=self.xyz.new_tensor([1.,-1.,-1.])
        if self.backend=='gsplat':
            from gsplat import rasterization
            cv_view=torch.cat([view[:3]*flip[:,None],view[3:]],0)
            # Our intrinsics use integer pixel centers; gsplat samples at x+0.5, y+0.5.
            intrinsics=self.xyz.new_tensor([[fx,0,cx+.5],[0,fy,cy+.5],[0,0,1]])
            rgb,alpha,info=rasterization(
                means=self.xyz,quats=self.rotation,scales=self.log_scale.exp(),
                opacities=self.opacity.sigmoid(),colors=torch.cat([self.color[:,None],self.sh_rest],1) if self.color_activation=='sh' else self.colors_for(c2w),
                sh_degree=self.active_sh_degree if self.color_activation=='sh' else None,
                viewmats=cv_view[None],Ks=intrinsics[None],width=width,height=height,
                backgrounds=torch.as_tensor(background,dtype=self.xyz.dtype,device=self.xyz.device).expand(1,3),packed=False,
                absgrad=self.color_activation!='sh',rasterize_mode='classic')
            return (rgb[0],alpha[0,...,0],info) if return_aux else rgb[0]
        p=(self.xyz@view[:3,:3].T+view[:3,3])*flip
        z=p[:,2].clamp_min(0.01)
        uv=torch.stack([fx*p[:,0]/z+cx,fy*p[:,1]/z+cy],-1)
        j=p.new_zeros((len(p),2,3))
        j[:,0,0]=fx/z;j[:,1,1]=fy/z
        j[:,0,2]=-fx*p[:,0]/z.square();j[:,1,2]=-fy*p[:,1]/z.square()
        r=flip[:,None]*view[:3,:3]
        cov=r@self.covariance()@r.T
        cov2=j@cov@j.transpose(1,2)+torch.eye(2,device=p.device)[None]*.3
        order=p[:,2].argsort();inv=torch.linalg.inv(cov2[order]);uv=uv[order]
        opacity=self.opacity.sigmoid()[order]*(p[order,2]>.01)
        colors=self.colors_for(c2w)[order]
        yy,xx=torch.meshgrid(torch.arange(height,device=p.device),torch.arange(width,device=p.device),indexing='ij')
        pixels=torch.stack([xx,yy],-1).reshape(-1,2);pieces=[];alphas=[]
        for xy in pixels.split(tile):
            d=xy[None]-uv[:,None]
            power=torch.einsum('npi,nij,npj->np',d,inv,d)
            alpha=(opacity[:,None]*torch.exp(-.5*power)).clamp(max=.99)
            trans=torch.cumprod(torch.cat([torch.ones_like(alpha[:1]),1-alpha+1e-8],0),0)
            pieces.append((trans[:-1]*alpha).T@colors+trans[-1,:,None]*background)
            alphas.append(1-trans[-1])
        rgb=torch.cat(pieces).reshape(height,width,3)
        return (rgb,torch.cat(alphas).reshape(height,width),{}) if return_aux else rgb

    def render_batch(self,cameras,intrinsics,width,height,background=.025):
        if self.backend!='gsplat':
            results=[self.render(camera,*k,width,height,background,return_aux=True) for camera,k in zip(cameras,intrinsics)]
            return torch.stack([r[0] for r in results]),torch.stack([r[1] for r in results]),{}
        from gsplat import rasterization
        views=torch.linalg.inv(cameras).clone()
        views[:,:3]*=self.xyz.new_tensor([1.,-1.,-1.])[None,:,None]
        matrices=[]
        for fx,fy,cx,cy in intrinsics:
            matrices.append([[fx,0,cx+.5],[0,fy,cy+.5],[0,0,1]])
        colors=torch.cat([self.color[:,None],self.sh_rest],1) if self.color_activation=='sh' else torch.stack([self.colors_for(camera) for camera in cameras])
        rgb,alpha,info=rasterization(means=self.xyz,quats=self.rotation,scales=self.log_scale.exp(),
            opacities=self.opacity.sigmoid(),colors=colors,sh_degree=self.active_sh_degree if self.color_activation=='sh' else None,viewmats=views,Ks=self.xyz.new_tensor(matrices),
            width=width,height=height,backgrounds=torch.as_tensor(background,dtype=self.xyz.dtype,device=self.xyz.device).expand(len(cameras),3),
            packed=False,absgrad=self.color_activation!='sh',rasterize_mode='classic')
        return rgb,alpha[...,0],info

    def export_ply(self,path):
        xyz=self.xyz.detach().cpu().numpy()
        dc=(self.base_colors().detach().cpu().numpy()-.5)/.28209479177387814
        arrays=[xyz,np.zeros_like(xyz),dc]
        names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2']
        if self.sh_rest is not None:
            rest=self.sh_rest.detach().transpose(1,2).reshape(len(xyz),-1).cpu().numpy()
            arrays.append(rest);names += [f'f_rest_{i}' for i in range(rest.shape[1])]
        arrays += [self.opacity.detach().cpu().numpy()[:,None],self.log_scale.detach().cpu().numpy(),
                   F.normalize(self.rotation,dim=-1).detach().cpu().numpy()]
        names += ['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        with Path(path).open('w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex %d\n'%len(xyz))
            f.write(''.join('property float '+n+'\n' for n in names)+'end_header\n')
            np.savetxt(f,np.concatenate(arrays,1),fmt='%.7g')


def dssim(pred,target):
    if pred.ndim==3:pred,target=pred[None],target[None]
    x,y=pred.permute(0,3,1,2),target.permute(0,3,1,2)
    pool=lambda a:F.avg_pool2d(a,7,stride=1,padding=3)
    mx,my=pool(x),pool(y)
    vx,vy=(pool(x*x)-mx*mx).clamp_min(0),(pool(y*y)-my*my).clamp_min(0)
    cov=pool(x*y)-mx*my
    s=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return (1-s.mean())/2
