"""Read-only novel-view rendering of real checkpoints, independent of training state."""
import io
import threading
from collections import OrderedDict
from pathlib import Path
from .storage import read_json
import numpy as np
from PIL import Image


class SceneViewer:
    def __init__(self):
        self.lock=threading.Lock();self.model=None;self.key=None;self.step=0;self.models=OrderedDict()

    def render(self,run_dir,matrix,fx,fy,cx,cy,width,height,transparent=False,background=(.035,.045,.055),snapshot_step=None):
        import torch
        from .runtime import configure_cuda_build
        from .evaluation import load_checkpoint
        if not self.lock.acquire(blocking=False):raise BlockingIOError('预览正在渲染，请稍后重试')
        try:
            if snapshot_step is not None:
                path=run_dir/'replay/models'/f'{snapshot_step:07d}.pt'
                if not path.is_file():raise FileNotFoundError('该步骤未保存完整三维模型；旧任务可播放多视角环绕，新训练会记录三维演化')
            else:
                candidates=[run_dir/'checkpoint.pt',run_dir/'best_checkpoint.pt']
                candidates=[p for p in candidates if p.is_file()]
                if not candidates:raise FileNotFoundError('模型快照尚未生成')
                path=max(candidates,key=lambda p:p.stat().st_mtime_ns)
            key=(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
            if key in self.models:
                self.model,self.step=self.models.pop(key);self.models[key]=(self.model,self.step);self.key=key
            elif key!=self.key:
                if not torch.cuda.is_available():raise ValueError('真实高斯新视角预览需要 CUDA，请使用几何检查或固定渲染')
                configure_cuda_build()
                try:
                    atomic=snapshot_step is not None or read_json(run_dir/'status.json',{}).get('checkpoint_atomic',False)
                    model,checkpoint=load_checkpoint(path,torch.device('cuda'),'gsplat',mmap=atomic,model_only=True)
                except (RuntimeError,EOFError,OSError):
                    if snapshot_step is not None or self.model is None or Path(self.key[0]).parent!=path.resolve().parent:
                        raise FileNotFoundError('模型快照正在更新，请稍后重试')
                else:
                    loaded_step=int(checkpoint.get('step',0))
                    if snapshot_step is not None and loaded_step!=snapshot_step:raise ValueError('三维快照步数与索引不一致')
                    model.eval()
                    for parameter in model.parameters():parameter.requires_grad_(False)
                    self.model=model;self.key=key;self.step=loaded_step
                    self.models[key]=(model,loaded_step)
                    while len(self.models)>3:self.models.popitem(last=False)
                    del checkpoint
            with torch.inference_mode():
                camera=torch.as_tensor(matrix,dtype=torch.float32,device='cuda')
                rgb,alpha,_=self.model.render(camera,fx,fy,cx,cy,width,height,background=0. if transparent else background,return_aux=True)
                if transparent:
                    straight=torch.where(alpha[...,None]>1e-6,rgb/alpha[...,None].clamp_min(1e-6),0.)
                    pixels=(torch.cat([straight.clamp(0,1),alpha[...,None].clamp(0,1)],-1)*255).round().byte().cpu().numpy()
                else:
                    # Keep HDR SH contributions until after alpha compositing. Converting
                    # to 8-bit straight-alpha first would darken bright translucent splats.
                    pixels=(rgb.clamp(0,1)*255).round().byte().cpu().numpy()
            stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG',compress_level=1)
            return stream.getvalue(),self.step,len(self.model.xyz)
        finally:self.lock.release()


def validate_camera(matrix):
    camera=np.asarray(matrix,dtype=np.float32)
    if camera.shape!=(4,4) or not np.isfinite(camera).all():raise ValueError('相机必须是有限的 4×4 camera-to-world 矩阵')
    if not np.allclose(camera[3],[0,0,0,1],atol=1e-5):raise ValueError('相机矩阵最后一行必须为 [0,0,0,1]')
    rotation=camera[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=.01) or np.linalg.det(rotation)<.99:
        raise ValueError('相机旋转必须为右手正交矩阵')
    return camera
