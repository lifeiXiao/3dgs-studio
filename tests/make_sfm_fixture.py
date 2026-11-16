"""Generate genuinely multi-view textured geometry; does not embed camera poses."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def make_fixture(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(8); textures=[]
    for i in range(12):
        noise=rng.integers(20,235,(512,512,3),dtype=np.uint8)
        im=Image.fromarray(noise).filter(ImageFilter.GaussianBlur(1.2)); draw=ImageDraw.Draw(im)
        for j in range(280):
            x,y=rng.integers(0,480,2); size=int(rng.integers(4,24)); color=tuple(map(int,rng.integers(10,245,3)))
            draw.rectangle((x,y,x+size,y+size),fill=color)
        textures.append(np.asarray(im))
    yy,xx=np.mgrid[:480,:640]; camera_rays=np.stack([(xx-320)/560,-(yy-240)/560,-np.ones_like(xx)],-1)
    # Textured rear wall and three cubes give repeatable SIFT and nonplanar geometry.
    planes=[(np.array([0.,0.,-1.5]),np.array([1.,0,0]),np.array([0.,1,0]),3.,2.5)]
    for center,half in [(np.array([-.65,-.1,.15]),.4),(np.array([.5,-.35,-.35]),.42),(np.array([.1,.65,-.7]),.3)]:
        for axis,sign in [(0,-1),(0,1),(1,1),(2,1)]:
            normal=np.eye(3)[axis]*sign; other=[a for a in range(3) if a!=axis]
            planes.append((center+normal*half,np.eye(3)[other[0]],np.eye(3)[other[1]],half,half))
    for idx in range(8):
        pos=np.array([-.9+idx*.25,.3+.12*np.sin(idx),3.2+.1*np.cos(idx)])
        back=pos/np.linalg.norm(pos); right=np.cross([0,1,0],back); right/=np.linalg.norm(right); up=np.cross(back,right)
        rotation=np.stack([right,up,back],1); rays=camera_rays@rotation.T
        depth=np.full((480,640),np.inf); pixels=np.full((480,640,3),35,dtype=np.uint8)
        for j,(origin,u,v,uw,vw) in enumerate(planes):
            normal=np.cross(u,v); denom=rays@normal
            denom=np.where(np.abs(denom)<1e-8,1e-8,denom)
            dist=((origin-pos)@normal)/denom
            hit=pos+rays*dist[:,:,None]-origin
            a=hit@u; b=hit@v
            mask=(dist>0)&(dist<depth)&(np.abs(a)<=uw)&(np.abs(b)<=vw)
            tx=np.clip(((a/uw+1)*255.5).astype(int),0,511); ty=np.clip(((b/vw+1)*255.5).astype(int),0,511)
            tex=textures[j%len(textures)]
            pixels[mask]=tex[ty[mask],tx[mask]]; depth[mask]=dist[mask]
        Image.fromarray(pixels).save(folder/f'{idx:03}.png')
    return folder

if __name__=='__main__':
    import sys
    make_fixture(sys.argv[1])
