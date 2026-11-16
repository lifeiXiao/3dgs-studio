import csv
import json
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from studio.model import Gaussians
from studio.dataset import read_scene
from studio.evaluation import evaluate_model


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware not available')
def test_cuda_projection_matches_reference_pixel_centers():
    from studio.runtime import configure_cuda_build
    configure_cuda_build()
    model=Gaussians([[.03,.02,-2]],[[.8,.3,.1]],.15).cuda()
    camera=torch.eye(4,device='cuda')
    with torch.no_grad():
        reference=model.render(camera,30,30,12,12,24,24)
        model.backend='gsplat'
        actual=model.render(camera,30,30,12,12,24,24)
    # gsplat truncates alpha < 1/255, whereas the reference keeps Gaussian tails.
    assert (reference-actual).abs().mean()<.00015


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware not available')
def test_density_cap_remaps_optimizer_moments_and_parameters():
    from studio.density import BoundedStrategy
    model=Gaussians([[0,0,-2],[.2,0,-2]],[[.8,.3,.1],[.2,.7,.4]],.1,sh_degree=1).cuda()
    params=model.parameter_map()
    optimizers={k:torch.optim.Adam([p],lr=.001) for k,p in params.items()}
    # Populate Adam's moments before resizing.
    sum(p.square().sum() for p in params.values()).backward()
    for o in optimizers.values():o.step();o.zero_grad(set_to_none=True)
    strategy=BoundedStrategy(max_gaussians=3)
    state=strategy.initialize_state(scene_scale=1)
    state['grad2d']=torch.tensor([1.,2.],device='cuda');state['count']=torch.ones(2,device='cuda')
    strategy._grow_gs(params,optimizers,state,100)
    model.synchronize_parameters(params)
    assert len(model.xyz)==3
    for k,p in params.items():
        assert optimizers[k].param_groups[0]['params'][0] is p
        assert optimizers[k].state[p]['exp_avg'].shape==p.shape
    # The next gradient update must still optimize the resized tensors.
    sum(p.square().sum() for p in model.parameters()).backward()
    for o in optimizers.values():o.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_sh_color_changes_with_camera_direction():
    model=Gaussians([[0,0,-2]],[[.5,.5,.5]],sh_degree=1)
    model.active_sh_degree=1
    with torch.no_grad(): model.sh_rest[:,0,0]=.4
    a=torch.eye(4);b=torch.eye(4);a[1,3]=1;b[1,3]=-1
    assert model.colors_for(a)[0,0]>model.colors_for(b)[0,0]


def test_validation_loader_never_falls_back_to_train(tmp_path):
    Image.new('RGB',(16,16),'red').save(tmp_path/'train.png')
    raw=dict(fl_x=16,frames=[dict(file_path='train.png',transform_matrix=np.eye(4).tolist())])
    (tmp_path/'transforms_train.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='val'): read_scene(tmp_path,split='val')


def test_fixed_evaluation_is_repeatable_and_does_not_update_model():
    model=Gaussians([[0,0,-2]],[[.8,.3,.1]],.2)
    camera=torch.eye(4)
    with torch.no_grad(): target=model.render(camera,20,20,8,8,16,16).numpy()
    frame=dict(path=__import__('pathlib').Path('fixed.png'),c2w=np.eye(4,dtype=np.float32),fx=20,fy=20,cx=8,cy=8,w=16,h=16,image=target,alpha=np.ones((16,16),np.float32))
    before={k:v.clone() for k,v in model.state_dict().items()}
    a=evaluate_model(model,[frame],torch.device('cpu'),quality=False)
    b=evaluate_model(model,[frame],torch.device('cpu'),quality=False)
    assert a==b and a['loss']<1e-6
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_fixed_metrics_keep_rows_without_validation(tmp_path,monkeypatch):
    from studio import server
    run=tmp_path/'eval-test';run.mkdir();(run/'config.json').write_text('{}')
    (run/'evaluations.csv').write_text('step,train_loss,val_loss,val_psnr\n0,0.4,,\n100,0.1,,\n')
    monkeypatch.setattr(server,'RUNS',tmp_path)
    response=TestClient(server.app).get('/api/runs/eval-test/evaluations')
    assert response.status_code==200
    assert response.json()==[{'step':0.,'train_loss':.4},{'step':100.,'train_loss':.1}]
