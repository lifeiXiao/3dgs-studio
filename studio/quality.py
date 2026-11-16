"""Image quality metrics using the pinned GraphDeco evaluation implementations."""
from functools import lru_cache
import importlib.util
import os
import sys
import torch
from .official import SOURCE,official_module
from .storage import ROOT

QUALITY_SPEC=dict(version=1,ssim='GraphDeco Gaussian 11x11 / sigma 1.5',
                  lpips='GraphDeco LPIPS / VGG16 / v0.1',
                  input='RGB float [0,1], clamped, native evaluation resolution')


def lpips_model(device):
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
    return _lpips_model(str(device))


@lru_cache(maxsize=2)
def _lpips_model(device):
    # Keep both pretrained VGG and calibrated LPIPS weights within this project.
    os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
    name='studio_graphdeco_lpips'
    if name not in sys.modules:
        path=SOURCE/'lpipsPyTorch'
        spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name,None);raise
    model=sys.modules[name].LPIPS(net_type='vgg',version='0.1').to(device).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def image_quality(pred,target):
    """Match metrics.py: float RGB [0,1] directly into GraphDeco's LPIPS module."""
    if pred.ndim==3:pred,target=pred[None],target[None]
    x=pred.clamp(0,1).permute(0,3,1,2).contiguous()
    y=target.clamp(0,1).permute(0,3,1,2).contiguous()
    ssim=official_module('loss_utils').ssim(x,y)
    perceptual=lpips_model(str(x.device))(x,y).mean()
    return dict(ssim=float(ssim),lpips=float(perceptual))
