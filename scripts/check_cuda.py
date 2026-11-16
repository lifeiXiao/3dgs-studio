from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio.runtime import configure_cuda_build
configure_cuda_build()
import torch
from studio.model import Gaussians
model=Gaussians([[0,0,-2]],[[.5,.3,.2]],backend='gsplat').cuda()
rgb,alpha,info=model.render(torch.eye(4,device='cuda'),20,20,8,8,16,16,return_aux=True)
(rgb.square().mean()).backward()
print('CUDA RASTERIZER READY',rgb.shape,alpha.max().item(),model.xyz.grad, list(info),flush=True)
