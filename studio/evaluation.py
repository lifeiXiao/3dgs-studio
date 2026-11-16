"""Fixed-camera RGB evaluation. Validation images never participate in optimization."""
import numpy as np
import torch
from .dataset import read_scene,load_images
from .model import Gaussians,dssim


def evaluation_frames(source,resolution,count=8,split=None,background=.025,official=False):
    frames,_=read_scene(source,split=split)
    indices=np.linspace(0,len(frames)-1,min(count,len(frames)),dtype=int)
    return load_images([frames[i] for i in indices],resolution,background=background,official=official)


@torch.no_grad()
def evaluate_model(model,frames,device,background=.025,quality=True):
    rows=[]
    is_official=getattr(model,'training_profile','studio')=='official'
    if is_official:background=getattr(model,'background',0.)
    for f in frames:
        pred=model.render(torch.as_tensor(f['c2w'],device=device),f['fx'],f['fy'],f['cx'],f['cy'],f['w'],f['h'],background)
        target=torch.as_tensor(f['image'],device=device)
        mse=(pred-target).square().mean()
        l1=(pred-target).abs().mean()
        if is_official:
            from .official import one_minus_ssim
            structural=one_minus_ssim(pred,target)
        else:structural=dssim(pred,target)
        foreground=torch.as_tensor(f['alpha']>.5,device=device)
        fg=(pred-target).square()[foreground].mean() if foreground.any() else mse
        row=dict(name=str(f['path'].name),loss=float(.8*l1+.2*structural),
                         psnr=float(-10*torch.log10(mse.clamp_min(1e-10))),
                         fg_psnr=float(-10*torch.log10(fg.clamp_min(1e-10))))
        if quality:
            from .quality import image_quality
            row.update(image_quality(pred,target))
        rows.append(row)
    return {**{k:float(np.mean([r[k] for r in rows])) for k in ['loss','psnr','fg_psnr']+(['ssim','lpips'] if quality else [])},
            'views':len(rows),'per_view':rows}


def load_checkpoint(path,device,backend='torch',mmap=False,model_only=False):
    checkpoint=torch.load(path,map_location='cpu' if mmap or model_only else device,weights_only=False,mmap=mmap)
    state=checkpoint['model'];rest=state.get('sh_rest')
    degree=round((rest.shape[1]+1)**.5)-1 if rest is not None else 0
    model=Gaussians(state['xyz'],torch.full_like(state['xyz'],.5),sh_degree=degree,backend=backend,color_activation=checkpoint.get('color_activation','sigmoid')).to(device)
    model.load_state_dict(state)
    model.active_sh_degree=checkpoint.get('active_sh_degree',degree)
    model.training_profile=checkpoint['config'].get('profile','studio')
    model.background=1. if checkpoint['config'].get('white_background',False) else 0. if model.training_profile=='official' else .025
    return model,checkpoint
