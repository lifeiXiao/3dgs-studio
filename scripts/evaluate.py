"""Deterministic evaluation on the same camera sets, with foreground metrics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from studio.dataset import read_scene,load_images
from studio.model import Gaussians,dssim
from studio.storage import ROOT,write_json


from studio.evaluation import evaluate_model,evaluation_frames,load_checkpoint


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--resolution',type=int,default=96);p.add_argument('--output',default='data/benchmarks/baseline.json');a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,ckpt=load_checkpoint(a.checkpoint,device)
    source=ROOT/ckpt['config']['dataset'];results={}
    for name,split in [('train_fixed',None),('val','val')]:
        result=evaluate_model(model,evaluation_frames(source,a.resolution,8,split),device);results[name]=result;print(name,{k:v for k,v in result.items() if k!='per_view'},flush=True)
    results.update(resolution=a.resolution,checkpoint=str(a.checkpoint));write_json(a.output,results)
