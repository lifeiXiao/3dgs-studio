import csv
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from .storage import ROOT, DATASETS, RUNS, read_json, write_json, project_path
from .dataset import read_scene, EXTS
from .sfm import image_files
from .official import OFFICIAL_DEFAULTS,source_info
from .viewer import SceneViewer,validate_camera
scene_viewer=SceneViewer()

app=FastAPI(title='3DGS Studio',version='0.1.0')
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
children={}; launch_lock=threading.Lock()


@app.middleware('http')
async def local_origin(request: Request, call_next):
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin:
        parsed=urlparse(origin)
        if parsed.netloc != request.headers.get('host'):
            return JSONResponse({'detail':'仅允许本机 Studio 页面操作'},status_code=403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(request,exc): return JSONResponse({'detail':str(exc)},status_code=400)


@app.get('/')
def home(): return FileResponse(ROOT/'static/index.html')


@app.get('/api/system')
def system():
    import torch
    available=torch.cuda.is_available()
    return dict(python=sys.executable,root=str(ROOT),torch=torch.__version__,cuda=available,
                gpu=torch.cuda.get_device_name(0) if available else None,
                vram_gb=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if available else 0,
                cpu_threads=os.cpu_count(),default_device='cuda' if available else 'cpu',
                colmap=__import__('pycolmap').__version__,colmap_cuda=bool(__import__('pycolmap').has_cuda))


@app.get('/api/browse')
def browse(path: str='.'):
    folder=project_path(path)
    if not folder.is_dir(): raise HTTPException(404,'目录不存在')
    entries=[]
    for p in sorted(folder.iterdir(),key=lambda p:(not p.is_dir(),p.name.lower())):
        if p.name.startswith('.') or p.is_symlink(): continue
        entries.append(dict(name=p.name,path=str(p.relative_to(ROOT)),directory=p.is_dir(),image=p.suffix.lower() in EXTS))
    return dict(path=str(folder.relative_to(ROOT)),parent=str(folder.parent.relative_to(ROOT)) if folder!=ROOT else None,entries=entries[:1000])


class DatasetPath(BaseModel):
    path:str


def describe_dataset(path):
    try:
        frames,points=read_scene(path)
    except ValueError:
        has_pose=any((path/n).exists() for n in ['transforms.json','transforms_train.json','cameras.txt','sparse/cameras.txt','sparse/0/cameras.txt'])
        if has_pose: raise
        _,files=image_files(path)
        if not files: raise ValueError('目录中没有可用图片')
        return dict(path=str(path.relative_to(ROOT)),name=path.name,frames=len(files),points=0,ready=False,cameras=[],initialization='仅图片 · 自动运行 COLMAP 生成位姿')
    return dict(ready=True,path=str(path.relative_to(ROOT)),name=path.name,frames=len(frames),points=len(points[0]) if points else 0,
                cameras=[dict(name=f['path'].name,width=f['w'],height=f['h'],fx=f['fx'],fy=f['fy'],cx=f['cx'],cy=f['cy'],matrix=f['c2w'].tolist()) for f in frames],
                initialization='稀疏点云' if points else '无稀疏点：优先前景掩码初始化，否则随机')


@app.post('/api/datasets/inspect')
def inspect_dataset(body:DatasetPath):
    try: return describe_dataset(project_path(body.path))
    except (OSError,KeyError,IndexError) as exc: raise ValueError(f'数据读取失败：{exc}')


@app.get('/api/datasets')
def datasets():
    results=[]
    for p in sorted(DATASETS.iterdir()):
        if not p.is_dir() or p.name.startswith('.'): continue
        try: results.append(describe_dataset(p))
        except Exception as exc: results.append(dict(name=p.name,path=str(p.relative_to(ROOT)),error=str(exc)))
    return results


@app.get('/api/datasets/image')
def dataset_image(path:str,index:int=0):
    root=project_path(path)
    info=describe_dataset(root)
    if info['ready']:
        frames,_=read_scene(root); images=[f['path'] for f in frames]
    else: _,images=image_files(root)
    if not 0<=index<len(images): raise HTTPException(404,'图片不存在')
    return FileResponse(images[index])


@app.post('/api/datasets/upload')
async def upload(files:list[UploadFile]=File(...),paths:str=Form(...)):
    try: names=__import__('json').loads(paths)
    except Exception: raise ValueError('文件路径列表无效')
    if not isinstance(names,list) or len(names)!=len(files) or not files: raise ValueError('文件列表无效')
    if len(files)>1000: raise ValueError('一次最多导入 1000 个文件，更大数据集请使用项目目录选择')
    folder=DATASETS/('import-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]); folder.mkdir()
    total=0
    try:
        for file,name in zip(files,names):
            if not isinstance(name,str): raise ValueError('路径必须是文本')
            relative=Path(name.replace('\\','/'))
            if relative.is_absolute() or '..' in relative.parts or any(p.startswith('.') for p in relative.parts): raise ValueError('上传路径无效')
            if relative.suffix.lower() not in EXTS|{'.json','.txt','.bin','.ply'}: continue
            dest=folder/relative; dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('wb') as f:
                while chunk:=await file.read(1024*1024):
                    total+=len(chunk)
                    if total>4*1024**3: raise ValueError('网页上传限 4GB，更大数据请复制到项目目录后选择')
                    f.write(chunk)
        # The browser supplies paths relative to the selected folder, stripped by the client.
        result=describe_dataset(folder)
        return result
    except Exception as exc:
        shutil.rmtree(folder,ignore_errors=True)
        raise ValueError(f'导入失败：{exc}')
    finally:
        for file in files: await file.close()


@app.post('/api/datasets/lego')
def lego():
    folder=DATASETS/'lego'
    try: return describe_dataset(folder)
    except Exception:
        raise ValueError('Lego 尚未下载完成。可在项目内运行 .venv/bin/python scripts/download_lego.py')


class TrainConfig(BaseModel):
    dataset:str
    replay_3d:bool=True
    replay_3d_every:int=Field(1000,ge=100,le=100000)
    profile:str='official'
    device:str='auto'
    backend:str='auto'
    iterations:int=Field(30000,ge=1,le=1000000)
    resolution:int=Field(-1,ge=-1,le=16384)
    gaussians:int=Field(100000,ge=8,le=1000000)
    max_gaussians:int=Field(0,ge=0,le=2000000)
    batch_size:int=Field(1,ge=1,le=8)
    sh_degree:int=Field(3,ge=0,le=3)
    densify:bool=True
    seed:int=Field(0,ge=0,le=2**31-1)
    preview_every:int=Field(500,ge=1,le=10000)
    eval_every:int=Field(1000,ge=10,le=100000)
    lr:float=Field(.01,gt=0,le=.2)
    position_lr_init:float=Field(.00016,gt=0,le=.1)
    position_lr_final:float=Field(.0000016,gt=0,le=.1)
    position_lr_delay_mult:float=Field(.01,ge=0,le=1)
    position_lr_max_steps:int=Field(30000,ge=1,le=1000000)
    feature_lr:float=Field(.0025,gt=0,le=1)
    opacity_lr:float=Field(.025,gt=0,le=1)
    scaling_lr:float=Field(.005,gt=0,le=1)
    rotation_lr:float=Field(.001,gt=0,le=1)
    percent_dense:float=Field(.01,gt=0,le=1)
    lambda_dssim:float=Field(.2,ge=0,le=1)
    densification_interval:int=Field(100,ge=1,le=10000)
    opacity_reset_interval:int=Field(3000,ge=1,le=1000000)
    densify_from_iter:int=Field(500,ge=0,le=1000000)
    densify_until_iter:int=Field(15000,ge=1,le=1000000)
    densify_grad_threshold:float=Field(.0002,gt=0,le=1)
    white_background:bool=False
    random_background:bool=False
    eval:bool=False
    sfm_matching:str='exhaustive'
    sfm_max_size:int=Field(1600,ge=320,le=3200)
    prepare_only:bool=False


@app.get('/api/defaults')
def official_defaults():
    return dict(defaults=OFFICIAL_DEFAULTS,source=source_info(),
                model=dict(type='3D Gaussian Splatting',sh_coefficients=16,sh_channels=48,
                           mlp_width=None,initial_nerf_points=100000),
                resolution=dict(auto=-1,original=1,half=2,quarter=4))


def run_path(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9-]+',run_id): raise HTTPException(404,'任务不存在')
    p=RUNS/run_id
    if not (p/'config.json').exists(): raise HTTPException(404,'任务不存在')
    return p


def status_for(path):
    state=read_json(path/'status.json',dict(id=path.name,status='starting',step=0))
    if state['status'] in ('starting','running','stopping'):
        child=children.get(path.name)
        if child:
            dead=child.poll() is not None
        else:
            pid=state.get('pid'); cmd=Path(f'/proc/{pid}/cmdline')
            try: dead=path.name.encode() not in cmd.read_bytes() or b'studio.worker' not in cmd.read_bytes()
            except OSError: dead=True
        if dead:
            state.update(status='failed',error='训练进程已退出，详见 training.log',finished_at=time.time())
            write_json(path/'status.json',state)
    state['animation']=(path/'training.gif').exists()
    state['config']=read_json(path/'config.json')
    from .quality_history import quality_status
    quality=quality_status(path)
    state.update(quality_status=quality.get('status'),quality_done=quality.get('done',0),
                 quality_total=quality.get('total',0),quality_error=quality.get('error'))
    return state


@app.get('/api/runs')
def runs(): return [status_for(p) for p in sorted(RUNS.iterdir(),reverse=True) if p.is_dir() and (p/'config.json').exists()]


@app.post('/api/runs')
def start_training(config:TrainConfig):
    if config.device not in ('auto','cpu','cuda'): raise ValueError('设备必须为 auto、cuda 或 cpu')
    if config.backend not in ('auto','torch','gsplat'): raise ValueError('后端无效')
    if config.profile not in ('official','studio'):raise ValueError('训练配置无效')
    if config.resolution not in (-1,1,2,4,8) and config.resolution<16:raise ValueError('分辨率需为 -1（自动）、1（原图）、2/4/8（缩放）或目标像素宽度')
    if config.max_gaussians and config.max_gaussians<config.gaussians: raise ValueError('高斯上限不能低于初始数量')
    if (config.backend=='torch' or config.device=='cpu') and (config.gaussians>4096 or config.resolution>256):
        raise ValueError('参考 / CPU 后端限 4096 个初始高斯与 256px，请选择 CUDA gsplat')
    dataset=project_path(config.dataset); info=describe_dataset(dataset)
    if config.sfm_matching not in ('exhaustive','sequential'): raise ValueError('匹配方式无效')
    if not info['ready'] and info['frames']<3: raise ValueError('仅图片数据集至少需要 3 张照片')
    with launch_lock:
        for path in RUNS.iterdir():
            if path.is_dir() and (path/'config.json').exists() and status_for(path)['status'] in ('starting','running','stopping'):
                raise HTTPException(409,'已有训练正在运行，请先停止当前任务')
        run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]; path=RUNS/run_id
        run_cache=read_json(ROOT/'studio-local.json',{}).get('run_cache')
        if run_cache:
            actual=Path(run_cache).expanduser().resolve()/run_id;actual.mkdir(parents=True)
            path.symlink_to(actual,target_is_directory=True)
        else:path.mkdir()
        values=config.model_dump(); values['dataset']=str(dataset.relative_to(ROOT)); values['needs_sfm']=not info['ready'];values['official_source']=source_info()
        values['effective_overrides']={k:values[k] for k,v in OFFICIAL_DEFAULTS.items() if k in values and values[k]!=v}
        write_json(path/'config.json',values)
        with (path/'training.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-m','studio.worker',run_id],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[run_id]=child
        if not (path/'status.json').exists():
            write_json(path/'status.json',dict(id=run_id,status='starting',step=0,total=config.iterations,pid=child.pid,started_at=time.time()))
    return status_for(path)


@app.get('/api/runs/{run_id}')
def run_status(run_id:str): return status_for(run_path(run_id))


@app.post('/api/runs/{run_id}/stop')
def stop(run_id:str):
    path=run_path(run_id); (path/'stop.request').touch()
    return dict(message='已请求停止，将在当前迭代后保存模型')


@app.get('/api/runs/{run_id}/metrics')
def metrics(run_id:str):
    path=run_path(run_id)/'metrics.csv'
    if not path.exists(): return []
    rows=[]
    for row in csv.DictReader(io.StringIO(path.read_text())):
        try: rows.append({k:float(v) for k,v in row.items()})
        except (ValueError,TypeError): pass
    # Bound response size while retaining the latest step.
    stride=max(1,len(rows)//2000)
    return rows[::stride]+([rows[-1]] if rows and rows[-1] is not rows[::stride][-1] else [])


@app.get('/api/runs/{run_id}/evaluations')
def fixed_evaluations(run_id:str):
    from .quality_history import evaluation_rows
    return evaluation_rows(run_path(run_id))


@app.post('/api/runs/{run_id}/quality/evaluate')
def evaluate_quality_history(run_id:str):
    from .quality_history import quality_status
    path=run_path(run_id)
    with launch_lock:
        for candidate in RUNS.iterdir():
            if not candidate.is_dir():continue
            state=quality_status(candidate)
            if state.get('status')=='running':
                if candidate.name==run_id:return state
                raise HTTPException(409,'已有历史指标补算任务正在运行')
        if not any((path/name).exists() for name in ('checkpoint.pt','best_checkpoint.pt','replay/models')):
            raise ValueError('此任务尚未保存完整模型，暂时无法补算')
        with (path/'quality.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-m','studio.quality_history',run_id],cwd=ROOT,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state=dict(status='running',pid=child.pid,done=0,total=0,started_at=time.time())
        write_json(path/'quality_status.json',state)
    return state


class NovelViewRequest(BaseModel):
    c2w:list[list[float]]
    snapshot_step:int|None=Field(None,ge=0,le=1000000)
    transparent:bool=False
    background:tuple[float,float,float]=(.035,.045,.055)
    width:int=Field(960,ge=32,le=1600)
    height:int=Field(540,ge=32,le=1200)
    fx:float=Field(800,gt=0,le=100000)
    fy:float=Field(800,gt=0,le=100000)
    cx:float=Field(480,ge=0,le=1600)
    cy:float=Field(270,ge=0,le=1200)


@app.post('/api/runs/{run_id}/render-view')
def render_novel_view(run_id:str,body:NovelViewRequest):
    path=run_path(run_id);camera=validate_camera(body.c2w)
    try:
        data,step,count=scene_viewer.render(path,camera,body.fx,body.fy,body.cx,body.cy,body.width,body.height,body.transparent,body.background,body.snapshot_step)
    except BlockingIOError as exc:raise HTTPException(429,str(exc))
    except FileNotFoundError as exc:raise HTTPException(404 if body.snapshot_step is not None else 503,str(exc))
    return Response(data,media_type='image/png',headers={'Cache-Control':'no-store',
        'X-Model-Step':str(step),'X-Gaussians':str(count)})


@app.get('/api/runs/{run_id}/scene')
def scene(run_id:str): return read_json(run_path(run_id)/'scene.json',dict(points=[],colors=[],cameras=[]))


@app.get('/api/runs/{run_id}/replay')
def replay(run_id:str):
    return read_json(run_path(run_id)/'replay/index.json',[])


@app.post('/api/runs/{run_id}/replay/export')
def export_replay(run_id:str):
    from .replay import export_gif
    count=export_gif(run_path(run_id))
    return dict(frames=count,url=f'/api/runs/{run_id}/files/training.gif')


@app.get('/api/runs/{run_id}/replay/{step}/scene')
def replay_scene(run_id:str,step:int):
    if not 0<=step<=100000: raise HTTPException(404,'快照不存在')
    path=run_path(run_id)/'replay'/f'{step:07d}.json'
    if not path.exists(): raise HTTPException(404,'快照尚未生成')
    return FileResponse(path,media_type='application/json')


@app.get('/api/runs/{run_id}/replay/{step}/image')
def replay_image(run_id:str,step:int):
    if not 0<=step<=100000: raise HTTPException(404,'快照不存在')
    path=run_path(run_id)/'replay'/f'{step:07d}.png'
    if not path.exists(): raise HTTPException(404,'快照尚未生成')
    return FileResponse(path)


@app.get('/api/runs/{run_id}/log')
def log(run_id:str):
    path=run_path(run_id)/'training.log'
    if not path.exists(): return {'text':''}
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16000)); raw=f.read()
    return {'text':raw.decode(errors='replace')}


@app.get('/api/runs/{run_id}/files/{name}')
def artifact(run_id:str,name:str):
    if name not in ('render.png','target.png','gaussians.ply','checkpoint.pt','metrics.csv','config.json','training.log','training.gif','evaluations.csv','evaluation.json','best_checkpoint.pt','initial.png','comparison.png'):
        raise HTTPException(404,'文件不存在')
    path=run_path(run_id)/name
    if not path.is_file(): raise HTTPException(404,'文件尚未生成')
    return FileResponse(path,headers={'Cache-Control':'no-store'})
