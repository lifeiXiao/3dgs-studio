import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from studio.runtime import configure_cuda_build
from studio.evaluation import load_checkpoint,evaluation_frames,evaluate_model
from studio.model import dssim
configure_cuda_build();torch.set_num_threads(4);device=torch.device('cuda')
source=Path('data/datasets/lego');frames=evaluation_frames(source,128,8);val=evaluation_frames(source,128,8,'val')
m,c=load_checkpoint('data/runs/20261116-193538-78c0f3/checkpoint.pt',device,'gsplat')
for degree in [2,0]:
    m.active_sh_degree=degree;r=evaluate_model(m,frames,device);print('SH',degree,'fixed',r['loss'],r['psnr'],flush=True)
m.active_sh_degree=2
f=frames[0];cam=torch.tensor(f['c2w'],device=device);target=torch.tensor(f['image'],device=device)
for p in m.parameters():p.requires_grad_(False)
m.color.requires_grad_(True);m.sh_rest.requires_grad_(True)
o=torch.optim.Adam([m.color,m.sh_rest],lr=.001)
for step in range(201):
    o.zero_grad();pred=m.render(cam,f['fx'],f['fy'],f['cx'],f['cy'],f['w'],f['h'])
    loss=.8*(pred-target).abs().mean()+.2*dssim(pred,target)
    if step%50==0:print('single camera',step,float(loss),float(-10*torch.log10((pred-target).square().mean())),flush=True)
    loss.backward();o.step()
