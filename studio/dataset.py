import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

EXTS = {'.png','.jpg','.jpeg','.webp'}


def contained(root, relative):
    path=(root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('数据路径不能越出数据集目录')
    return path


def read_scene(root, split=None):
    root=Path(root).resolve()
    names=[f'transforms_{split}.json'] if split else ['transforms.json','transforms_train.json']
    source=next((root/n for n in names if (root/n).exists()),None)
    if split and source is None:
        raise ValueError(f'没有 {split} 数据划分')
    if source:
        raw=json.loads(source.read_text())
        frames=[]
        for frame in raw.get('frames',[]):
            p=contained(root,frame['file_path'])
            if not p.suffix:
                p=next((p.with_suffix(e) for e in sorted(EXTS) if p.with_suffix(e).exists()),p)
            with Image.open(p) as im: w,h=im.size
            c2w=np.asarray(frame['transform_matrix'],dtype=np.float32)
            if c2w.shape != (4,4) or not np.isfinite(c2w).all() or abs(np.linalg.det(c2w))<1e-8:
                raise ValueError('相机矩阵必须为可逆的有限 4×4 矩阵')
            opts={**raw,**frame}
            fx=opts.get('fl_x')
            if fx is None:
                angle=opts.get('camera_angle_x')
                if angle is None: raise ValueError('缺少 fl_x 或 camera_angle_x')
                fx=0.5*w/math.tan(float(angle)/2)
            fy=opts.get('fl_y',fx)
            if min(float(fx),float(fy))<=0: raise ValueError('焦距必须大于零')
            frames.append(dict(path=p,c2w=c2w,w=w,h=h,fx=float(fx),fy=float(fy),cx=float(opts.get('cx',w/2)),cy=float(opts.get('cy',h/2))))
        points=None
        pp=root/'points3D.txt'
    else:
        sparse=next((p for p in [root/'sparse/0',root/'sparse',root] if (p/'cameras.txt').exists()),None)
        if sparse is None: raise ValueError('需要 transforms.json 或 COLMAP 文本模型。仅有照片时请先估计相机位姿。')
        cameras={}
        for line in (sparse/'cameras.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'): continue
            a=line.split(); cid=int(a[0]); model=a[1]; w,h=map(int,a[2:4]); v=list(map(float,a[4:]))
            if model=='PINHOLE': fx,fy,cx,cy=v
            elif model=='SIMPLE_PINHOLE': fx,cx,cy=v; fy=fx
            else: raise ValueError(f'相机 {model} 需要先用 COLMAP image_undistorter 去畸变')
            cameras[cid]=(w,h,fx,fy,cx,cy)
        frames=[]
        lines=[l for l in (sparse/'images.txt').read_text().splitlines() if not l.startswith('#')]
        for i in range(0,len(lines),2):
            if not lines[i].strip(): continue
            a=lines[i].split(maxsplit=9); qw,qx,qy,qz=map(float,a[1:5]); tx,ty,tz=map(float,a[5:8])
            q=np.array([qw,qx,qy,qz]); q=q/np.linalg.norm(q); qw,qx,qy,qz=q
            r=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                        [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                        [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
            view=np.eye(4); view[:3,:3]=r; view[:3,3]=[tx,ty,tz]
            c2w=np.linalg.inv(view); c2w[:3,1:3]*=-1
            w,h,fx,fy,cx,cy=cameras[int(a[8])]
            frames.append(dict(path=contained(root,'images/'+a[9]),c2w=c2w.astype(np.float32),w=w,h=h,fx=fx,fy=fy,cx=cx,cy=cy))
        pp=sparse/'points3D.txt'; points=None
    if not frames: raise ValueError('数据集没有有效相机帧')
    if pp.exists():
        rows=[l.split() for l in pp.read_text().splitlines() if l.strip() and not l.startswith('#')]
        if rows:
            points=(np.array([[float(x) for x in r[1:4]] for r in rows],np.float32),np.array([[float(x)/255 for x in r[4:7]] for r in rows],np.float32))
    return frames,points


def load_images(frames,resolution,background=.025,official=False):
    result=[]
    for frame in frames:
        ow,oh=frame['w'],frame['h']
        if resolution in (1,2,4,8):w,h=max(1,round(ow/resolution)),max(1,round(oh/resolution))
        elif resolution==-1:
            scale=max(1,ow/1600);w,h=int(ow/scale),int(oh/scale)
        else:
            scale=resolution/(ow if official else max(ow,oh));w,h=max(8,round(ow*scale)),max(8,round(oh*scale))
        with Image.open(frame['path']) as im:
            rgba=im.convert('RGBA')
            if official:
                original=np.asarray(rgba,dtype=np.float32)/255
                rgb=original[:,:,:3]*original[:,:,3:]+background*(1-original[:,:,3:])
                # Match the official alpha-composite → 8-bit RGB → PIL resize pipeline.
                image=Image.fromarray((rgb*255).astype(np.uint8)).resize((w,h),Image.Resampling.BICUBIC)
                rgb=np.asarray(image,dtype=np.float32)/255
                alpha=np.asarray(rgba.getchannel('A').resize((w,h),Image.Resampling.BICUBIC),dtype=np.float32)/255
            else:
                a=np.asarray(rgba.resize((w,h),Image.Resampling.LANCZOS),dtype=np.float32)/255
                rgb=a[:,:,:3]*a[:,:,3:]+background*(1-a[:,:,3:]);alpha=a[:,:,3]
        result.append({**frame,'w':w,'h':h,'fx':frame['fx']*w/ow,'fy':frame['fy']*h/oh,
                       'cx':frame['cx']*w/ow,'cy':frame['cy']*h/oh,'image':rgb,'alpha':alpha})
    return result
