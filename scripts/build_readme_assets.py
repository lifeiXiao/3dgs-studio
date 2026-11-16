"""Build README GIFs and a metric figure from a real, local Lego training run."""
import argparse
import io
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id');parser.add_argument('--url',default='http://127.0.0.1:7860')
    parser.add_argument('--output',type=Path,default=Path('docs/assets'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    def request(path,body=None):
        data=None if body is None else json.dumps(body).encode()
        for attempt in range(20):
            req=urllib.request.Request(args.url+path,data=data,headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=60) as response:return response.read()
            except urllib.error.HTTPError as error:
                if error.code!=429 or attempt==19:raise
                time.sleep(.1)
    get=lambda path:json.loads(request(path))
    status=get(f'/api/runs/{args.run_id}');config=status['config']
    dataset=json.loads(request('/api/datasets/inspect',{'path':config.get('prepared_dataset',config['dataset'])}))
    metadata=dataset['cameras'][0];camera=np.array(metadata['matrix'])
    scene=get(f'/api/runs/{args.run_id}/scene');metrics=get(f'/api/runs/{args.run_id}/evaluations')
    snapshots=get(f'/api/runs/{args.run_id}/replay');steps={frame['step'] for frame in snapshots if frame.get('model')}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    train_images=[]
    for row in metrics:
        step=int(row['step'])
        if step not in steps:continue
        body=dict(c2w=camera.tolist(),width=640,height=640,fx=640*metadata['fx']/metadata['width'],
                  fy=640*metadata['fy']/metadata['height'],cx=320,cy=320,snapshot_step=step)
        image=Image.open(io.BytesIO(request(f'/api/runs/{args.run_id}/render-view',body))).convert('RGB')
        frame=Image.new('RGB',(640,680),(15,20,23));frame.paste(image,(0,40))
        draw=ImageDraw.Draw(frame)
        draw.text((16,12),f"STEP {step:>5}    VAL SSIM {row.get('val_ssim',row.get('train_ssim',0)):.4f}    LPIPS {row.get('val_lpips',row.get('train_lpips',0)):.4f}",font=font,fill=(203,244,132))
        train_images.append(frame)
    if len(train_images)<2:raise ValueError('At least two full model snapshots are required')
    train_images[0].save(args.output/'training.gif',save_all=True,append_images=train_images[1:],loop=0,
                         duration=[500]+[130]*(len(train_images)-2)+[1800])
    orient=lambda p:np.array([p[0],p[2],-p[1]])
    unorient=lambda p:np.array([p[0],-p[2],p[1]])
    position=orient(camera[:3,3]);back=orient(camera[:3,2])
    center=np.median([orient(point) for point in scene['points']],axis=0)
    distance=np.linalg.norm(position-center);center=position-distance*back
    yaw0=math.atan2(back[0],back[2]);pitch0=math.asin(back[1]);images=[]
    final_step=max(steps)
    for index in range(72):
        phase=index/72*2*math.pi;yaw=yaw0+phase;pitch=pitch0+.12*math.sin(phase)
        cy,sy,cp,sp=math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch)
        right=unorient([cy,0,-sy]);up=unorient([-sp*sy,cp,-sp*cy]);back=unorient([cp*sy,sp,cp*cy])
        matrix=np.eye(4);matrix[:3,:3]=np.stack([right,up,back],axis=1);matrix[:3,3]=unorient(center)+distance*back
        body=dict(c2w=matrix.tolist(),width=640,height=480,fx=480*metadata['fy']/metadata['height'],
                  fy=480*metadata['fy']/metadata['height'],cx=320,cy=240,snapshot_step=final_step)
        image=Image.open(io.BytesIO(request(f'/api/runs/{args.run_id}/render-view',body))).convert('RGB');images.append(image)
    images[0].save(args.output/'orbit.gif',save_all=True,append_images=images[1:],duration=80,loop=0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.style.use('dark_background')
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained',facecolor='#101719')
    for ax,key,title in zip(axes.flat,['loss','psnr','ssim','lpips'],['RGB Loss (lower is better)','PSNR / dB (higher is better)','SSIM (higher is better)','LPIPS / VGG (lower is better)']):
        ax.set_facecolor('#151e22')
        for split,label,color in [('train','Fixed training','#cbf484'),('val','Validation','#b39bec')]:
            rows=[row for row in metrics if split+'_'+key in row]
            ax.plot([row['step'] for row in rows],[row[split+'_'+key] for row in rows],label=label,color=color,linewidth=1.8,linestyle='--' if split=='val' else '-')
        ax.set_title(title,fontsize=11);ax.set_xlabel('Training step',fontsize=9);ax.grid(alpha=.15)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value/1000:g}k'));ax.tick_params(labelsize=8)
    axes[0,0].legend(fontsize=8);fig.suptitle('NeRF Synthetic Lego · 800×800 · 31 real evaluation stages',fontsize=13)
    fig.savefig(args.output/'training-metrics.png',dpi=160);plt.close(fig)
    (args.output/'SOURCE.md').write_text(f"# Demonstration assets\n\nGenerated from local run `{args.run_id}` using all Gaussians and SH parameters.\nTraining GIF: {len(train_images)} actual model stages. Orbit GIF: 72 views at step {final_step}.\nMetrics: fixed 8 training and 8 validation cameras, native 800×800 training.\nDataset: NeRF Synthetic Lego, https://github.com/bmild/nerf .\nRebuild with `python scripts/build_readme_assets.py <run_id>` while Studio is running.\n")
    print(json.dumps(dict(training_frames=len(train_images),orbit_frames=len(images),output=str(args.output.resolve()))))


if __name__=='__main__':main()
