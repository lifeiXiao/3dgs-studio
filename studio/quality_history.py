"""Backfill SSIM/LPIPS from real model snapshots without changing training state."""
import argparse
import csv
import io
import os
from pathlib import Path
import time
import traceback
from .storage import ROOT,RUNS,read_json,write_json

QUALITY_FIELDS=('train_ssim','train_lpips','val_ssim','val_lpips')


def quality_status(run_dir):
    state=read_json(run_dir/'quality_status.json',{})
    if state.get('status')=='running':
        try:command=Path(f"/proc/{state.get('pid')}/cmdline").read_bytes()
        except OSError:command=b''
        if b'studio.quality_history' not in command:
            state.update(status='failed',error='指标评测进程已退出，请重新补算')
    return state


def evaluation_rows(run_dir):
    path=run_dir/'evaluations.csv';rows={}
    if path.exists():
        for row in csv.DictReader(io.StringIO(path.read_text())):
            try:parsed={k:float(v) for k,v in row.items() if v not in ('',None)}
            except (ValueError,TypeError):continue
            if 'step' in parsed:rows[int(parsed['step'])]=parsed
    for row in read_json(run_dir/'quality_metrics.json',{}).get('rows',[]):
        step=int(row['step']);merged=rows.setdefault(step,{'step':step})
        for key in QUALITY_FIELDS:
            if key in row:merged.setdefault(key,row[key])
    return [rows[step] for step in sorted(rows)]


def backfill(run_dir):
    state=dict(status='running',pid=os.getpid(),done=0,total=0,started_at=time.time())
    def update(**values):
        state.update(values);write_json(run_dir/'quality_status.json',state)
    update()
    try:
        import numpy as np
        import torch
        from .dataset import read_scene,load_images
        from .evaluation import evaluate_model,evaluation_frames,load_checkpoint
        from .quality import QUALITY_SPEC,lpips_model
        from .runtime import configure_cuda_build
        torch.set_num_threads(4)
        config=read_json(run_dir/'config.json');status=read_json(run_dir/'status.json',{})
        official=config.get('profile')=='official';background=1. if config.get('white_background') else 0. if official else .025
        source=ROOT/config.get('prepared_dataset',config['dataset']);resolution=config['resolution']
        frames,_=read_scene(source)
        if official and not config.get('eval',False) and (source/'transforms_test.json').exists():
            frames.extend(read_scene(source,split='test')[0])
        indices=np.linspace(0,len(frames)-1,min(8,len(frames)),dtype=int)
        fixed=load_images([frames[i] for i in indices],resolution,background=background,official=official)
        validation=evaluation_frames(source,resolution,8,'val',background=background,official=official) if (source/'transforms_val.json').exists() else []
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':configure_cuda_build()
        lpips_model(str(device))
        paths={int(p.stem):p for p in sorted((run_dir/'replay/models').glob('[0-9]*.pt')) if p.stem.isdigit()}
        existing=evaluation_rows(run_dir);evaluation_steps={int(row['step']) for row in existing}
        if evaluation_steps:paths={step:path for step,path in paths.items() if step in evaluation_steps}
        if (run_dir/'checkpoint.pt').exists():paths.setdefault(int(status.get('step',0)),run_dir/'checkpoint.pt')
        elif not paths and (run_dir/'best_checkpoint.pt').exists():paths[int(status.get('best_step',0))]=run_dir/'best_checkpoint.pt'
        if not paths:raise ValueError('此任务尚未保存可供评测的完整模型')
        saved=read_json(run_dir/'quality_metrics.json',{})
        cached=saved.get('spec')==QUALITY_SPEC and saved.get('resolution')==resolution
        rows={int(row['step']):row for row in saved.get('rows',[])} if cached else {}
        update(total=len(paths),done=sum(step in rows for step in paths))
        for step,path in sorted(paths.items()):
            if step in rows:continue
            model,metadata=load_checkpoint(path,device,'gsplat' if device.type=='cuda' else 'torch',
                                            mmap=path.parent.name=='models' or status.get('checkpoint_atomic',False),model_only=True)
            assert int(metadata['step'])==step,'模型步数与历史快照不一致'
            del metadata;model.eval();model.requires_grad_(False)
            train=evaluate_model(model,fixed,device)
            val=evaluate_model(model,validation,device) if validation else None
            row=dict(step=step,train_ssim=train['ssim'],train_lpips=train['lpips'])
            if val:row.update(val_ssim=val['ssim'],val_lpips=val['lpips'])
            rows[step]=row
            write_json(run_dir/'quality_metrics.json',dict(spec=QUALITY_SPEC,resolution=resolution,
                       views=len(fixed),validation_views=len(validation),rows=[rows[k] for k in sorted(rows)]))
            update(done=sum(k in rows for k in paths),step=step)
            print(f"step={step} train_ssim={train['ssim']:.5f} train_lpips={train['lpips']:.5f}"+
                  (f" val_ssim={val['ssim']:.5f} val_lpips={val['lpips']:.5f}" if val else ''),flush=True)
            del model
        # Export the supplemented columns after the training writer has finished.
        if read_json(run_dir/'status.json',{}).get('status') in ('completed','stopped'):
            merged=evaluation_rows(run_dir);fields=list(dict.fromkeys(key for row in merged for key in row))
            temporary=run_dir/'evaluations.quality.tmp.csv'
            with temporary.open('w') as file:
                writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(merged)
            os.replace(temporary,run_dir/'evaluations.csv')
        update(status='completed',finished_at=time.time())
    except Exception as error:
        traceback.print_exc();update(status='failed',error=str(error),finished_at=time.time());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');args=parser.parse_args()
    if not args.run_id.replace('-','').isalnum():raise ValueError('Invalid run id')
    backfill(RUNS/args.run_id)
