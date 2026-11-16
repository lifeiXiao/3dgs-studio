"""Exercise the local GPU API with multiple poses and exact historical models."""
import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path
import httpx
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--url',default='http://127.0.0.1:7860')
    parser.add_argument('--history',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('data/previews/playback-check'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    client=httpx.Client(base_url=args.url,timeout=60)
    scene=client.get(f'/api/runs/{args.run_id}/scene').raise_for_status().json()
    camera=np.asarray(scene['cameras'][0]['matrix'])
    # Same display-space orbit as the web viewer, for the Lego Z-up dataset.
    orient=lambda p:np.array([p[0],p[2],-p[1]])
    unorient=lambda p:np.array([p[0],-p[2],p[1]])
    points=np.array([orient(p) for p in scene['points']])
    position=orient(camera[:3,3]);back=orient(camera[:3,2])
    distance=np.linalg.norm(position-np.median(points,axis=0));center=position-distance*back
    yaw0=math.atan2(back[0],back[2]);pitch0=math.asin(back[1])
    steps=[None]*36
    if args.history:
        frames=client.get(f'/api/runs/{args.run_id}/replay').raise_for_status().json()
        steps=[frame['step'] for frame in frames if frame.get('model')]
        if len(steps)<2:raise RuntimeError('At least two full model snapshots are required')
    rows=[];images=[]
    for index,step in enumerate(steps):
        phase=index/max(1,len(steps))*2*math.pi;yaw=yaw0+phase;pitch=pitch0+.12*math.sin(phase)
        cy,sy,cp,sp=math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch)
        right=unorient([cy,0,-sy]);up=unorient([-sp*sy,cp,-sp*cy]);back=unorient([cp*sy,sp,cp*cy])
        matrix=np.eye(4);matrix[:3,:3]=np.stack([right,up,back],axis=1)
        matrix[:3,3]=unorient(center)+distance*back
        body=dict(c2w=matrix.tolist(),width=640,height=480,fx=667,fy=667,cx=320,cy=240,snapshot_step=step)
        started=time.perf_counter()
        for _ in range(20):
            response=client.post(f'/api/runs/{args.run_id}/render-view',json=body)
            if response.status_code!=429:break
            time.sleep(.1)
        response.raise_for_status()
        actual=int(response.headers['X-Model-Step']);count=int(response.headers['X-Gaussians'])
        if step is not None:assert actual==step,(step,actual)
        image=Image.open(io.BytesIO(response.content)).convert('RGB');images.append(image)
        row=dict(requested_step=step,actual_step=actual,gaussians=count,yaw_degrees=round(math.degrees(yaw),2),
                 sha256=hashlib.sha256(response.content).hexdigest(),seconds=round(time.perf_counter()-started,3))
        rows.append(row)
        if index%max(1,len(steps)//4)==0:image.save(args.output/f'view-{index:02d}.png')
    assert len({row['sha256'] for row in rows})==len(rows),'Views must change across the orbit'
    images[0].save(args.output/'animation.gif',save_all=True,append_images=images[1:],duration=160,loop=0)
    report=dict(run_id=args.run_id,mode='evolution+orbit' if args.history else 'orbit',frames=rows,
                verification='Real CUDA API renders; browser visual verification unavailable')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(run=args.run_id,frames=len(rows),min_gaussians=min(r['gaussians'] for r in rows),
                          max_gaussians=max(r['gaussians'] for r in rows),output=str(args.output.resolve()))))


if __name__=='__main__':main()
