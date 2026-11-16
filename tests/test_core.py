import json
import numpy as np
import pytest
import torch
from PIL import Image
from studio.dataset import read_scene, load_images
from studio.model import Gaussians, dssim
from studio.worker import device_for


def test_renderer_optimizes_and_all_parameters_receive_finite_gradients():
    torch.set_num_threads(2); torch.manual_seed(3)
    xyz=np.array([[-.2,.1,-2.],[.3,-.2,-2.5]],np.float32)
    target_model=Gaussians(xyz,np.array([[.9,.2,.1],[.1,.9,.2]]),.2)
    model=Gaussians(xyz+.07,np.full((2,3),.5),.15)
    camera=torch.eye(4)
    with torch.no_grad(): target=target_model.render(camera,30,30,12,12,24,24)
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    losses=[]
    for step in range(25):
        optimizer.zero_grad(); pred=model.render(camera,30,30,12,12,24,24)
        loss=(pred-target).square().mean()+.05*dssim(pred,target)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step(); losses.append(loss.item())
    assert losses[-1]<losses[0]*.5
    assert torch.allclose(model.covariance(),model.covariance().transpose(1,2))
    assert torch.linalg.eigvalsh(model.covariance()).min()>0


def test_near_splat_occludes_far_splat():
    model=Gaussians([[0,0,-2],[0,0,-3]],[[1,.001,.001],[.001,.001,1]],.3)
    with torch.no_grad(): model.opacity.fill_(7)
    image=model.render(torch.eye(4),20,20,8,8,16,16)
    assert image[8,8,0]>.9 and image[8,8,2]<.05


def test_transforms_loader_and_path_guard(tmp_path):
    Image.new('RGB',(40,20),'red').save(tmp_path/'frame.png')
    raw=dict(fl_x=30,frames=[dict(file_path='frame',transform_matrix=np.eye(4).tolist())])
    (tmp_path/'transforms.json').write_text(json.dumps(raw))
    frames,points=read_scene(tmp_path); assert points is None
    small=load_images(frames,20)[0]
    assert (small['w'],small['h'],small['fx'])==(20,10,15)
    raw['frames'][0]['file_path']='../secret.png'
    (tmp_path/'transforms.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='越出'): read_scene(tmp_path)


def test_colmap_opencv_to_opengl_conversion(tmp_path):
    (tmp_path/'images').mkdir(); (tmp_path/'sparse').mkdir()
    Image.new('RGB',(32,24)).save(tmp_path/'images/a.png')
    (tmp_path/'sparse/cameras.txt').write_text('1 PINHOLE 32 24 20 21 16 12\n')
    (tmp_path/'sparse/images.txt').write_text('# Images\n1 1 0 0 0 0 0 2 1 a.png\n\n')
    (tmp_path/'sparse/points3D.txt').write_text('1 0 0 0 255 0 0 0\n')
    frames,points=read_scene(tmp_path)
    assert frames[0]['c2w'][2,3]==-2
    assert frames[0]['c2w'][1,1]==-1
    assert points[1][0,0]==1


def test_cpu_fallback_when_cuda_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert str(device_for('auto'))=='cpu'
    with pytest.raises(ValueError): device_for('cuda')
