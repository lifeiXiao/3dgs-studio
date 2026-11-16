"""Background training with fixed validation, CUDA splatting and adaptive density."""
import argparse
import csv
import fcntl
import math
import os
import random
import time
import traceback
import shutil
import numpy as np
from PIL import Image
import torch
from .dataset import read_scene,load_images
from .model import Gaussians,dssim
from .evaluation import evaluate_model,evaluation_frames
from .storage import ROOT,RUNS,write_json,read_json


def device_for(requested):
    if requested=='cpu': return torch.device('cpu')
    if torch.cuda.is_available():
        try:
            torch.zeros(1,device='cuda').add_(1)
            return torch.device('cuda')
        except RuntimeError:
            if requested=='cuda': raise
    if requested=='cuda': raise ValueError('CUDA 不可用，请选择自动或 CPU')
    return torch.device('cpu')


def initial_points(frames,points,count,seed):
    rng=np.random.default_rng(seed)
    if points is not None:
        valid=np.isfinite(points[0]).all(1)&np.isfinite(points[1]).all(1)
        ids=np.flatnonzero(valid)
        if len(ids)<8: raise ValueError('有效稀疏点不足')
        idx=rng.choice(ids,min(count,len(ids)),replace=False)
        return points[0][idx],np.clip(points[1][idx],.001,.999)
    centers=np.array([f['c2w'][:3,3] for f in frames])
    directions=np.array([-f['c2w'][:3,2] for f in frames])
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    a=np.eye(3)[None]-directions[:,:,None]*directions[:,None,:]
    center=np.linalg.lstsq(a.sum(0),np.einsum('nij,nj->i',a,centers),rcond=None)[0]
    radius=max(float(np.median(np.linalg.norm(centers-center,axis=1)))*.25,.1)
    masked=[f for f in frames if np.mean(f['alpha']<.05)>.05]
    if len(masked)>=3:
        views=masked[::max(1,len(masked)//16)]
        candidates=rng.uniform(-radius,radius,(max(count*64,32768),3))+center
        valid=np.ones(len(candidates),bool)
        def project(f,xyz):
            view=np.linalg.inv(f['c2w']);p=xyz@view[:3,:3].T+view[:3,3]
            depth=-p[:,2];safe=np.maximum(depth,1e-5)
            u=np.rint(f['fx']*p[:,0]/safe+f['cx']).astype(int)
            v=np.rint(-f['fy']*p[:,1]/safe+f['cy']).astype(int)
            inside=(depth>0)&(u>=0)&(u<f['w'])&(v>=0)&(v<f['h'])
            return np.clip(u,0,f['w']-1),np.clip(v,0,f['h']-1),depth,inside
        for f in views:
            u,v,depth,inside=project(f,candidates)
            valid &= inside&(f['alpha'][v,u]>.2)
        hull=candidates[valid]
        if len(hull)>=8:
            # Keep visible hull surfaces and assign colors only from front-facing samples.
            # Averaging every view colors hidden/interior points with unrelated surfaces.
            colors=np.zeros_like(hull);votes=np.zeros(len(hull))
            for f in views:
                u,v,depth,inside=project(f,hull);pixel=v*f['w']+u
                front=np.full(f['w']*f['h'],np.inf);np.minimum.at(front,pixel,depth)
                visible=inside&(depth<=front[pixel]+radius*.025)
                colors[visible]+=f['image'][v[visible],u[visible]];votes[visible]+=1
            ids=np.flatnonzero(votes>0)
            if len(ids)>=8:
                selected=rng.choice(ids,count,replace=len(ids)<count)
                xyz=hull[selected].copy()
                if len(ids)<count: xyz+=rng.normal(0,radius*.001,xyz.shape)
                return xyz.astype(np.float32),(colors[selected]/votes[selected,None]).astype(np.float32)
    return (rng.uniform(-radius,radius,(count,3))+center).astype(np.float32),rng.uniform(.2,.8,(count,3)).astype(np.float32)


def initial_scales(xyz):
    from scipy.spatial import cKDTree
    distances,_=cKDTree(xyz).query(xyz,k=min(4,len(xyz)))
    return np.sqrt(np.mean(distances[:,1:]**2,axis=1)).clip(.002,1)[:,None].astype(np.float32)*.7


def scene_snapshot(model,frames,path):
    with torch.no_grad():
        # Geometry inspection has a bounded payload; the checkpoint/PLY retain all Gaussians.
        idx=torch.linspace(0,len(model.xyz)-1,min(8000,len(model.xyz)),device=model.xyz.device).long()
        write_json(path,dict(points=model.xyz[idx].cpu().tolist(),colors=model.base_colors()[idx].clamp(0,1).cpu().tolist(),
                             scales=model.log_scale[idx].exp().cpu().tolist(),total_gaussians=len(model.xyz),
                             cameras=[dict(matrix=f['c2w'].tolist(),name=f['path'].name) for f in frames]))


def select_backend(device,requested,update):
    if requested=='torch' or device.type!='cuda':
        if requested=='gsplat': raise ValueError('gsplat 需要 CUDA GPU')
        return 'torch'
    update(stage='加载 CUDA 光栅化器（首次运行会编译）')
    try:
        from .runtime import configure_cuda_build
        configure_cuda_build()
        probe=Gaussians([[0,0,-2]],[[.5,.4,.3]],backend='gsplat').to(device)
        probe.render(torch.eye(4,device=device),20,20,8,8,16,16).mean().backward()
        return 'gsplat'
    except Exception as exc:
        if requested=='gsplat': raise
        traceback.print_exc()
        update(backend_warning=f'CUDA 光栅化器不可用，改用 PyTorch 参考后端：{exc}')
        return 'torch'


def train(run_dir):
    config=read_json(run_dir/'config.json')
    state=dict(id=run_dir.name,status='starting',step=0,total=config['iterations'],pid=os.getpid(),started_at=time.time(),checkpoint_atomic=True)
    def update(**values): state.update(values);write_json(run_dir/'status.json',state)
    update()
    try:
        lock=(RUNS/'.training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        device=device_for(config['device']);source=ROOT/config['dataset']
        official=config.get('profile','studio')=='official'
        background=(1. if config.get('white_background',False) else 0.) if official else .025
        if config.get('needs_sfm'):
            from .sfm import reconstruct
            from .storage import DATASETS
            source=DATASETS/('colmap-'+run_dir.name);update(status='running',phase='sfm')
            reconstruct(ROOT/config['dataset'],run_dir/'colmap',source,update,lambda:(run_dir/'stop.request').exists(),config.get('sfm_matching','exhaustive'),config.get('sfm_max_size',1600))
            config['prepared_dataset']=str(source.relative_to(ROOT));write_json(run_dir/'config.json',config)
            update(prepared_dataset=config['prepared_dataset'])
            if config.get('prepare_only'):
                update(status='completed',phase='prepared',stage='COLMAP 位姿已生成',finished_at=time.time());return
        update(phase='training',stage='载入影像与固定评测相机')
        frames,points=read_scene(source)
        if official and not config.get('eval',False) and (source/'transforms_test.json').exists():
            test_frames,_=read_scene(source,split='test');frames.extend(test_frames)
        frames=load_images(frames,config['resolution'],background=background,official=official)
        fixed_ids=np.linspace(0,len(frames)-1,min(8,len(frames)),dtype=int)
        fixed=[frames[i] for i in fixed_ids]
        val_frames=evaluation_frames(source,config['resolution'],8,'val',background=background,official=official) if (source/'transforms_val.json').exists() else []
        backend=select_backend(device,config.get('backend','auto'),update)
        if backend=='torch' and (config['gaussians']>4096 or config['resolution']>256):
            raise ValueError('PyTorch 参考后端限 4096 个初始高斯 / 256px；请启用 gsplat CUDA 或降低配置')
        if official:
            rng=np.random.RandomState(config['seed'])
            if points is not None:xyz,colors=points
            else:
                xyz=(rng.random_sample((config['gaussians'],3))*2.6-1.3).astype(np.float32)
                colors=(rng.random_sample((len(xyz),3))/255*.28209479177387814+.5).astype(np.float32)
        else:xyz,colors=initial_points(frames,points,config['gaussians'],config['seed'])
        degree=config.get('sh_degree',3 if official else 2) if backend=='gsplat' or official else 0
        scales=initial_scales(xyz)
        if official:
            from scipy.spatial import cKDTree
            distances,_=cKDTree(xyz).query(xyz,k=min(4,len(xyz)))
            scales=np.sqrt(np.maximum(np.mean(distances[:,1:]**2,axis=1),1e-7))[:,None].astype(np.float32)
        model=Gaussians(xyz,colors,scales,sh_degree=degree,opacity=.1,backend=backend,color_activation='sh' if official else 'linear').to(device)
        model.training_profile=config.get('profile','studio');model.background=background
        params=model.parameter_map();base_lr=config.get('lr',.01)
        camera_center=np.mean([f['c2w'][:3,3] for f in frames],axis=0)
        camera_distances=[np.linalg.norm(f['c2w'][:3,3]-camera_center) for f in frames]
        scene_scale=max(float(max(camera_distances))*1.1 if official else float(np.median(camera_distances)),.1)
        if official:
            rates=dict(means=config['position_lr_init']*scene_scale,colors=config['feature_lr'],
                       opacities=config['opacity_lr'],scales=config['scaling_lr'],quats=config['rotation_lr'],
                       sh_rest=config['feature_lr']/20)
        else:rates=dict(means=base_lr*.04,colors=base_lr*.25,opacities=base_lr*3,scales=base_lr*.5,quats=base_lr*.1,sh_rest=base_lr*.0125)
        optimizers={key:torch.optim.Adam([value],lr=rates[key],eps=1e-15 if official else 1e-8) for key,value in params.items()}
        strategy=None;strategy_state=None
        if backend=='gsplat' and config.get('densify',True):
            if official:
                from .official import make_strategy
                strategy,strategy_state=make_strategy(config,scene_scale)
            else:
                from .density import BoundedStrategy
                strategy=BoundedStrategy(max_gaussians=config.get('max_gaussians') or 65536,absgrad=True,
                    grow_grad2d=.0008,refine_start_iter=min(200,max(20,config['iterations']//10)),
                    refine_stop_iter=int(config['iterations']*.8),refine_every=100,reset_every=2500,
                    pause_refine_after_reset=200,prune_opa=.005,prune_scale3d=.2)
                strategy_state=strategy.initialize_state(scene_scale=scene_scale)
            strategy.check_sanity(params,optimizers)
        tensors=[(torch.as_tensor(f['c2w'],device=device),torch.as_tensor(f['image'],device=device),torch.as_tensor(f['alpha'],device=device)) for f in frames]
        masked=[bool(np.mean(f['alpha']<.99)>.001) for f in frames]
        batch_size=config.get('batch_size',4) if backend=='gsplat' and len({(f['w'],f['h']) for f in frames})==1 else 1
        batch_size=min(batch_size,len(frames))
        update(status='running',stage='高斯优化 · 固定训练/验证评测',device=str(device),backend=backend,
               device_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',gaussians=len(xyz),
               frames=len(frames),validation_views=len(val_frames),fixed_views=len(fixed),densify=strategy is not None,
               initialization='sparse' if points is not None else 'official random cube' if official else 'visible alpha hull / random',effective_batch_size=batch_size,profile=config.get('profile','studio'),image_width=frames[0]['w'],image_height=frames[0]['h'])
        Image.fromarray((frames[0]['image']*255).astype(np.uint8)).save(run_dir/'target.png')
        replay_dir=run_dir/'replay';replay_dir.mkdir(exist_ok=True);replay_frames=[]
        snapshot_every=config['preview_every']*max(1,math.ceil(config['iterations']/(100*config['preview_every'])))
        model_every=config['preview_every']*max(1,math.ceil(max(config.get('replay_3d_every',1000),config['iterations']/40)/config['preview_every']))
        evaluation_every=config.get('eval_every',200);row={};best_val=float('-inf');loss_ema=None;last_evaluated=-1
        start=time.monotonic()
        def checkpoint(path,step):
            temporary=path.with_suffix('.tmp.pt')
            torch.save(dict(model=model.state_dict(),optimizers={k:o.state_dict() for k,o in optimizers.items()},
                            config=config,step=step,active_sh_degree=model.active_sh_degree,backend=backend,color_activation=model.color_activation),temporary)
            os.replace(temporary,path)
        def save_preview(step,record=False):
            with torch.no_grad():
                f=frames[0];pred=model.render(tensors[0][0],f['fx'],f['fy'],f['cx'],f['cy'],f['w'],f['h'],background=background)
                image=Image.fromarray((pred.clamp(0,1).cpu().numpy()*255).astype(np.uint8))
                if step==0:image.save(run_dir/'initial.png')
                image.save(run_dir/'render.tmp.png');os.replace(run_dir/'render.tmp.png',run_dir/'render.png')
                scene_snapshot(model,frames,run_dir/'scene.json')
                save_full=config.get('replay_3d',True) and (step in (0,1) or step%model_every==0 or step==config['iterations'] or (run_dir/'stop.request').exists())
                if record or save_full:
                    stem=f'{step:07d}';image.save(replay_dir/(stem+'.png'));scene_snapshot(model,frames,replay_dir/(stem+'.json'))
                    model_path=replay_dir/'models'/(stem+'.pt')
                    if save_full:
                        required=sum(value.numel()*value.element_size() for value in model.state_dict().values())
                        if shutil.disk_usage(run_dir).free>required+1024**3:
                            from .replay import save_model_snapshot
                            save_model_snapshot(model_path,model,config,step)
                        else:update(replay_warning='存储空间不足：继续训练和图像记录，暂停新的完整三维关键帧')
                    entry=dict(step=step,loss=row.get('loss'),psnr=row.get('psnr'),model=model_path.exists(),gaussians=len(model.xyz))
                    if replay_frames and replay_frames[-1]['step']==step: replay_frames[-1]=entry
                    else: replay_frames.append(entry)
                    write_json(replay_dir/'index.json',replay_frames)
                    update(replay_3d_frames=sum(bool(frame.get('model')) for frame in replay_frames))
            update(preview_at=time.time())
        with (run_dir/'metrics.csv').open('w',buffering=1) as metrics,(run_dir/'evaluations.csv').open('w',buffering=1) as evaluations:
            mw=csv.DictWriter(metrics,fieldnames=['step','view','loss','loss_ema','l1','dssim','alpha_loss','psnr','elapsed','step_ms','gaussians','lr_position']);mw.writeheader()
            fields=['step','train_loss','train_psnr','train_fg_psnr','train_ssim','train_lpips','val_loss','val_psnr','val_fg_psnr','val_ssim','val_lpips','gaussians','elapsed']
            ew=csv.DictWriter(evaluations,fieldnames=fields);ew.writeheader()
            def evaluate(step):
                nonlocal best_val,last_evaluated
                training=evaluate_model(model,fixed,device)
                validation=evaluate_model(model,val_frames,device) if val_frames else None
                result=dict(step=step,train_loss=training['loss'],train_psnr=training['psnr'],train_fg_psnr=training['fg_psnr'],
                            train_ssim=training['ssim'],train_lpips=training['lpips'],
                            gaussians=len(model.xyz),elapsed=time.monotonic()-start)
                if validation: result.update(val_loss=validation['loss'],val_psnr=validation['psnr'],val_fg_psnr=validation['fg_psnr'],val_ssim=validation['ssim'],val_lpips=validation['lpips'])
                ew.writerow(result);last_evaluated=step
                update(fixed_loss=training['loss'],fixed_psnr=training['psnr'],eval_step=step,
                       fixed_ssim=training['ssim'],fixed_lpips=training['lpips'],
                       **({k:result[k] for k in ['val_loss','val_psnr','val_fg_psnr','val_ssim','val_lpips']} if validation else {}))
                write_json(run_dir/'evaluation.json',dict(step=step,train_fixed=training,val=validation,resolution=config['resolution']))
                score=validation['psnr'] if validation else training['psnr']
                if score>best_val:
                    best_val=score;checkpoint(run_dir/'best_checkpoint.pt',step)
                    update(best_eval_psnr=score,best_step=step)
                print(f"[eval {step}] fixed_loss={training['loss']:.6f} fixed_psnr={training['psnr']:.2f}"+(f" val_loss={validation['loss']:.6f} val_psnr={validation['psnr']:.2f}" if validation else ''),flush=True)
            save_preview(0,True);evaluate(0);last_step=0;order=[]
            for step in range(1,config['iterations']+1):
                if (run_dir/'stop.request').exists(): break
                tick=time.monotonic()
                selected=[]
                for _ in range(batch_size):
                    if not order:order=list(range(len(frames)));random.shuffle(order)
                    selected.append(order.pop())
                idx=selected[0];f=frames[idx]
                cameras=torch.stack([tensors[i][0] for i in selected])
                target=torch.stack([tensors[i][1] for i in selected]);alpha=torch.stack([tensors[i][2] for i in selected])
                intrinsics=[(frames[i]['fx'],frames[i]['fy'],frames[i]['cx'],frames[i]['cy']) for i in selected]
                model.active_sh_degree=min(degree,step//1000)
                if official:
                    progress=min(step/config['position_lr_max_steps'],1.)
                    optimizers['means'].param_groups[0]['lr']=scene_scale*math.exp(math.log(config['position_lr_init'])*(1-progress)+math.log(config['position_lr_final'])*progress)
                else:
                    lr_ratio=.01**(step/config['iterations'])
                    for name,opt in optimizers.items():
                        ratio=lr_ratio if name=='means' else .2**(step/config['iterations'])
                        opt.param_groups[0]['lr']=rates[name]*ratio
                for o in optimizers.values():o.zero_grad(set_to_none=True)
                render_bg=torch.rand(3,device=device) if official and config.get('random_background',False) else background
                pred,render_alpha,info=model.render_batch(cameras,intrinsics,f['w'],f['h'],background=render_bg)
                if strategy: strategy.step_pre_backward(params,optimizers,strategy_state,step,info)
                l1=(pred-target).abs().mean()
                if official:
                    from .official import one_minus_ssim
                    structural=one_minus_ssim(pred,target)
                else:structural=dssim(pred,target)
                mask_available=torch.as_tensor([masked[i] for i in selected],device=device)
                alpha_loss=(render_alpha[mask_available]-alpha[mask_available]).abs().mean() if mask_available.any() else pred.new_zeros(())
                weight=config.get('lambda_dssim',.2)
                loss=(1-weight)*l1+weight*structural+(0 if official else .05)*alpha_loss
                if not torch.isfinite(loss):raise RuntimeError('损失非有限值，训练已停止')
                loss.backward()
                # Official reporting runs before densification/reset/optimizer updates.
                # This also avoids recording a just-reset opacity cloud as an evaluation frame.
                if official and (step%evaluation_every==0 or step==config['iterations']):evaluate(step)
                if official and (step==1 or step%config['preview_every']==0 or step==config['iterations']):
                    save_preview(step,step==1 or step%snapshot_every==0 or step==config['iterations'])
                if not official:

                    for parameter in model.parameters():torch.nn.utils.clip_grad_norm_([parameter],10)
                    for o in optimizers.values():o.step()
                if strategy:
                    strategy.step_post_backward(params,optimizers,strategy_state,step,info)
                    model.synchronize_parameters(params)
                if official and strategy and step<config['densify_until_iter'] and (step%config['opacity_reset_interval']==0 or (config.get('white_background',False) and step==config['densify_from_iter'])):
                    # gsplat 1.5.3's reset conditional has a bitwise/comparison precedence bug.
                    # Reset explicitly to reproduce GraphDeco's min(opacity, 0.01) schedule.
                    from gsplat.strategy.ops import reset_opa
                    reset_opa(params,optimizers,strategy_state,value=.01)
                    model.synchronize_parameters(params)
                if official and step<config['iterations']:
                    for o in optimizers.values():o.step()
                with torch.no_grad():
                    if not official:model.log_scale.clamp_(-9,2);model.opacity.clamp_(-10,10)
                    psnr=float(-10*torch.log10((pred-target).square().mean().clamp_min(1e-10)))
                value=float(loss);ema_weight=.4 if official else .05;loss_ema=value if loss_ema is None else (1-ema_weight)*loss_ema+ema_weight*value
                if device.type=='cuda':torch.cuda.synchronize()
                row=dict(step=step,view=idx,loss=value,loss_ema=loss_ema,l1=float(l1),dssim=float(structural),alpha_loss=float(alpha_loss),psnr=psnr,
                         elapsed=time.monotonic()-start,step_ms=(time.monotonic()-tick)*1000,gaussians=len(model.xyz),lr_position=optimizers['means'].param_groups[0]['lr'])
                mw.writerow(row);last_step=step
                if step==1 or step%config['preview_every']==0 or step==config['iterations']:
                    if not official:save_preview(step,step==1 or step%snapshot_every==0 or step==config['iterations'])
                    update(**row,gpu_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2,1) if device.type=='cuda' else 0)
                elif step%10==0:update(**row)
                if not official and (step%evaluation_every==0 or step==config['iterations']):evaluate(step)
                if step==1 or step%50==0:print(f"step={step} loss={value:.6f} ema={loss_ema:.6f} psnr={psnr:.2f} gaussians={len(model.xyz)}",flush=True)
            if last_evaluated!=last_step:evaluate(last_step)
        model.export_ply(run_dir/'gaussians.ply');checkpoint(run_dir/'checkpoint.pt',last_step);save_preview(last_step,True)
        from PIL import ImageDraw
        comparison=Image.new('RGB',(frames[0]['w']*3,frames[0]['h']+24),(22,26,29))
        draw=ImageDraw.Draw(comparison)
        for column,(name,label) in enumerate([('initial.png','INITIAL'),('render.png','TRAINED'),('target.png','REFERENCE')]):
            with Image.open(run_dir/name) as image:comparison.paste(image,(column*frames[0]['w'],24))
            draw.text((column*frames[0]['w']+5,5),label,fill=(203,244,132))
        comparison.save(run_dir/'comparison.png')
        if len(replay_frames)>1:
            from .replay import export_gif
            export_gif(run_dir);update(animation=True)
        update(**row)
        update(status='stopped' if (run_dir/'stop.request').exists() else 'completed',step=last_step,
               finished_at=time.time(),elapsed=time.monotonic()-start)
    except InterruptedError as exc:update(status='stopped',error=str(exc),finished_at=time.time())
    except Exception as exc:
        traceback.print_exc();update(status='failed',error=str(exc),finished_at=time.time())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');args=parser.parse_args()
    if not args.run_id.replace('-','').isalnum():raise ValueError('Invalid run id')
    train(RUNS/args.run_id)
