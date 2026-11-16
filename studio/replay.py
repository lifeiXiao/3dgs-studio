from pathlib import Path
import tempfile
from PIL import Image
from .storage import read_json


def export_gif(run_dir):
    run_dir=Path(run_dir); frames=read_json(run_dir/'replay/index.json',[])
    if len(frames)<2: raise ValueError('至少需要两个训练快照才能导出动画')
    images=[]
    for frame in frames:
        with Image.open(run_dir/'replay'/f"{frame['step']:07d}.png") as image:
            images.append(image.convert('RGB'))
    with tempfile.NamedTemporaryFile(dir=run_dir,suffix='.gif',delete=False) as f:
        temp=Path(f.name)
    try:
        images[0].save(temp,save_all=True,append_images=images[1:],duration=250,loop=0)
        temp.replace(run_dir/'training.gif')
    finally:
        temp.unlink(missing_ok=True)
    return len(images)


def save_model_snapshot(path,model,config,step):
    """Store every Gaussian and SH coefficient, with no optimizer or point sampling."""
    import os
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.pt')
    payload=dict(model={key:value.detach().cpu() for key,value in model.state_dict().items()},
                 config=config,step=int(step),active_sh_degree=model.active_sh_degree,
                 color_activation=model.color_activation,snapshot_kind='full_gaussian_model',
                 gaussians=len(model.xyz),snapshot_version=1)
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)
