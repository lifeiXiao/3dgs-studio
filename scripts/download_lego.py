#!/usr/bin/env python3
"""Download only NeRF Synthetic Lego from a public mirror; preserve original poses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE='https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/resolve/main/lego/'
DEST=ROOT/'data/datasets/lego'


def fetch(relative):
    dest=DEST/relative; dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size>0: return dest
    for attempt in range(5):
        try:
            req=urllib.request.Request(BASE+relative,headers={'User-Agent':'3dgs-studio/0.1'})
            with urllib.request.urlopen(req,timeout=90) as response: data=response.read()
            temp=dest.with_suffix(dest.suffix+'.download'); temp.write_bytes(data)
            if dest.suffix=='.png':
                with Image.open(temp) as im: im.verify()
            elif dest.suffix=='.json': json.loads(data)
            temp.replace(dest); return dest
        except Exception:
            if attempt==4: raise
            time.sleep(attempt+1)


def download(splits):
    files=[]
    for split in splits:
        config=fetch(f'transforms_{split}.json')
        frames=json.loads(config.read_text())['frames']
        for f in frames:
            relative=f['file_path'].removeprefix('./')+'.png'
            p=(DEST/relative).resolve()
            if not p.is_relative_to(DEST.resolve()): raise ValueError('Unsafe dataset path')
            files.append(relative)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,path in enumerate(pool.map(fetch,files),1):
            if i%20==0 or i==len(files): print(f'Lego: {i}/{len(files)} images downloaded',flush=True)
    (DEST/'SOURCE.md').write_text('NeRF Synthetic / Lego\n\nOriginal: https://www.matthewtancik.com/nerf\nMirror: https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/tree/main/lego\nLego model: Heinzelnisse, CC-BY-NC, https://www.blendswap.com/blend/11490\nOriginal camera matrices and image files are retained unchanged.\nDownloaded splits: '+', '.join(splits)+'\n')
    print(DEST,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val','test']);args=parser.parse_args()
    download(args.splits)
