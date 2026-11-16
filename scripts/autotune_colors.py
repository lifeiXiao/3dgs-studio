"""Controlled ablation: same geometry, linear RGB, balanced multiview fitting."""
import sys,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from studio.runtime import configure_cuda_build
from studio.evaluation import load_checkpoint,evaluation_frames,evaluate_model
from studio.dataset import read_scene,load_images
from studio.model import dssim
from studio.storage import write_json
configure_cuda_build();torch.set_num_threads(4);device=torch.device('cuda')
source=Path('data/datasets/lego');frames,_=read_scene(source);frames=load_images(frames,128)
fixed=[frames[i] for i in [0,14,28,42,56,70,84,99]];val=evaluation_frames(source,128,8,'val')
tensors=[(torch.tensor(f['c2w'],device=device),torch.tensor(f['image'],device=device)) for f in frames]
results=[]
for lr in [.002,.005]:
    random.seed(42)
    m,c=load_checkpoint('data/runs/20261116-193538-78c0f3/checkpoint.pt',device,'gsplat')
    with torch.no_grad():m.color.copy_(m.color.sigmoid());m.sh_rest.zero_()
    m.color_activation='linear';m.active_sh_degree=0
    for p in m.parameters():p.requires_grad_(False)
    m.color.requires_grad_(True);o=torch.optim.Adam([m.color],lr=lr)
    order=[];history=[];start=time.monotonic()
    for step in range(1201):
        if step%200==0:
            a=evaluate_model(m,fixed,device);b=evaluate_model(m,val,device)
            r=dict(step=step,train_loss=a['loss'],train_psnr=a['psnr'],val_loss=b['loss'],val_psnr=b['psnr'])
            history.append(r);print(lr,r,flush=True)
        if step==1200:break
        o.zero_grad(set_to_none=True)
        for _ in range(4):
            if not order:order=list(range(len(frames)));random.shuffle(order)
            idx=order.pop();f=frames[idx];camera,target=tensors[idx]
            pred=m.render(camera,f['fx'],f['fy'],f['cx'],f['cy'],f['w'],f['h'])
            loss=(.8*(pred-target).abs().mean()+.2*dssim(pred,target))/4
            loss.backward()
        o.step()
    path=Path(f'data/benchmarks/color-linear-{lr}.pt')
    torch.save(dict(model=m.state_dict(),config=c['config'],step=1200,active_sh_degree=0,color_activation='linear'),path)
    results.append(dict(lr=lr,history=history,seconds=time.monotonic()-start,checkpoint=str(path)))
    write_json('data/benchmarks/color-ablation.json',results)
