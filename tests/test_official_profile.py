import importlib.util
from argparse import ArgumentParser
import numpy as np
import pytest
import torch
from PIL import Image
from studio.official import OFFICIAL_DEFAULTS,SOURCE,one_minus_ssim,make_strategy
from studio.server import TrainConfig,app
from studio.model import Gaussians
from studio.dataset import load_images
from fastapi.testclient import TestClient


def test_official_defaults_match_pinned_graphdeco_arguments():
    spec=importlib.util.spec_from_file_location('official_arguments_test',SOURCE/'arguments/__init__.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parser=ArgumentParser();module.ModelParams(parser);module.OptimizationParams(parser)
    original=vars(parser.parse_args([]));ours=TrainConfig(dataset='data/datasets/lego').model_dump()
    for key,value in OFFICIAL_DEFAULTS.items():
        if key in original:assert value==original[key],key
        if key in ours:assert ours[key]==value,key
    assert ours['max_gaussians']==0 and ours['batch_size']==1


def test_default_api_and_model_width_are_official():
    result=TestClient(app).get('/api/defaults').json()
    assert result['defaults']['iterations']==30000
    assert result['defaults']['resolution']==-1
    assert result['model']['sh_coefficients']==16
    model=Gaussians(np.zeros((2,3)),np.full((2,3),.5),sh_degree=3,color_activation='sh')
    assert model.sh_rest.shape==(2,15,3)
    assert sum(p.numel() for p in model.parameters())==2*59
    assert model.color.shape==(2,3)


def test_native_resolution_is_not_capped_at_512(tmp_path):
    path=tmp_path/'large.png';Image.new('RGBA',(1800,900),(120,80,30,255)).save(path)
    frame=dict(path=path,w=1800,h=900,fx=1500,fy=1500,cx=900,cy=450,c2w=np.eye(4,dtype=np.float32))
    native=load_images([frame],1,background=0,official=True)[0]
    automatic=load_images([frame],-1,background=0,official=True)[0]
    half=load_images([frame],2,background=0,official=True)[0]
    assert (native['w'],native['h'],native['fx'])==(1800,900,1500)
    assert (automatic['w'],automatic['h'])==(1600,800)
    assert (half['w'],half['h'])==(900,450)
    assert native['image'].shape==(900,1800,3)


def test_official_ssim_has_finite_gradients_and_zero_identical_loss():
    torch.manual_seed(1);x=torch.rand(1,20,24,3,requires_grad=True)
    assert abs(float(one_minus_ssim(x,x)))<1e-6
    loss=one_minus_ssim(x,x.detach()*.7);loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_official_density_is_not_absgrad_or_artificially_capped():
    strategy,state=make_strategy(OFFICIAL_DEFAULTS,1.)
    assert not strategy.absgrad and strategy.grow_grad2d==.0002
    assert strategy.refine_start_iter==500 and strategy.refine_stop_iter==15000
    assert strategy.reset_every==3000 and strategy.refine_every==100
    assert not hasattr(strategy,'max_gaussians')


def test_initial_web_form_renders_official_values_and_native_option():
    from html.parser import HTMLParser
    from studio.storage import ROOT
    class FormParser(HTMLParser):
        def __init__(self):super().__init__();self.inputs={};self.selected={};self.options={};self.current=None
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='input' and 'id' in a:self.inputs[a['id']]=a.get('value')
            if tag=='select':self.current=a.get('id')
            if tag=='option' and self.current:
                self.options.setdefault(self.current,[]).append(a.get('value'))
                if 'selected' in a:self.selected[self.current]=a.get('value')
        def handle_endtag(self,tag):
            if tag=='select':self.current=None
    p=FormParser();p.feed((ROOT/'static/index.html').read_text())
    assert p.inputs['iterations']=='30000' and p.inputs['gaussians']=='100000'
    assert p.inputs['lr']=='0.0025' and p.inputs['batchSize']=='1'
    assert p.selected['resolution']=='-1' and '1' in p.options['resolution']
    assert p.selected['shDegree']=='3' and p.inputs['maxGaussians']=='0'
