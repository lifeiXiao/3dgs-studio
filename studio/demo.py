"""Deterministic synthetic multi-view data for the end-to-end smoke test."""
import json
import math
import numpy as np
from PIL import Image
import torch
from .model import Gaussians
from .storage import write_json


def create_demo(root):
    root.mkdir(parents=True,exist_ok=True); (root/'images').mkdir(exist_ok=True)
    rng=np.random.default_rng(42)
    xyz=rng.normal(size=(120,3)); xyz/=np.linalg.norm(xyz,axis=1,keepdims=True)
    xyz*=np.array([0.8,0.9,0.65]); color=np.clip((xyz+1)/2,0.05,0.95)
    model=Gaussians(xyz,color,0.12)
    with torch.no_grad(): model.opacity.fill_(2.)
    frames=[]; torch.set_num_threads(4)
    for i in range(12):
        angle=i*2*math.pi/12; pos=np.array([3*math.sin(angle),1.1,3*math.cos(angle)])
        back=pos/np.linalg.norm(pos); right=np.cross([0,1,0],back); right/=np.linalg.norm(right); up=np.cross(back,right)
        c2w=np.eye(4,dtype=np.float32); c2w[:3,:3]=np.stack([right,up,back],1); c2w[:3,3]=pos
        with torch.no_grad(): im=model.render(torch.tensor(c2w),100,100,48,48,96,96).numpy()
        name=f'images/{i:03}.png'; Image.fromarray((im*255).astype(np.uint8)).save(root/name)
        frames.append(dict(file_path=name,transform_matrix=c2w.tolist()))
    with (root/'points3D.txt').open('w') as f:
        for i,(p,c) in enumerate(zip(xyz,color)):
            f.write(f"{i+1} {' '.join(map(str,p))} {' '.join(str(round(v*255)) for v in c)} 0\n")
    write_json(root/'transforms.json',dict(fl_x=100,fl_y=100,cx=48,cy=48,w=96,h=96,frames=frames))
