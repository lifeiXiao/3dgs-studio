import io
import json
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from studio.viewer import SceneViewer,validate_camera
from studio.model import Gaussians


def test_orbit_camera_must_be_right_handed_and_finite():
    assert np.allclose(validate_camera(np.eye(4)),np.eye(4))
    left=np.eye(4);left[0,0]=-1
    with pytest.raises(ValueError,match='右手'):validate_camera(left)
    invalid=np.eye(4);invalid[0,3]=np.nan
    with pytest.raises(ValueError,match='有限'):validate_camera(invalid)
    with pytest.raises(ValueError):validate_camera(np.zeros((3,4)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware required for real splatting')
def test_novel_view_uses_full_checkpoint_and_camera_pose(tmp_path,monkeypatch):
    from studio import server
    root=tmp_path/'viewer-test';root.mkdir()
    model=Gaussians([[0,0,-2]],[[.9,.2,.1]],scale=.2,sh_degree=3,color_activation='sh')
    with torch.no_grad():model.opacity.fill_(7)
    config={'profile':'official','dataset':'test'}
    torch.save({'model':model.state_dict(),'config':config,'step':17,'active_sh_degree':3,'color_activation':'sh'},root/'checkpoint.pt')
    (root/'config.json').write_text(json.dumps(config));(root/'status.json').write_text(json.dumps({'checkpoint_atomic':True}))
    monkeypatch.setattr(server,'RUNS',tmp_path);monkeypatch.setattr(server,'scene_viewer',SceneViewer())
    client=TestClient(server.app)
    body=dict(c2w=np.eye(4).tolist(),transparent=True,width=64,height=64,fx=40,fy=40,cx=32,cy=32)
    a=client.post('/api/runs/viewer-test/render-view',json=body)
    assert a.status_code==200 and a.headers['content-type']=='image/png'
    assert a.headers['X-Model-Step']=='17' and a.headers['X-Gaussians']=='1'
    image=Image.open(io.BytesIO(a.content));assert image.mode=='RGBA' and image.size==(64,64)
    assert image.getpixel((32,32))[0]>200 and image.getpixel((32,32))[3]>240
    assert image.getpixel((0,0))[3]==0
    body['c2w'][0][3]=.5
    b=client.post('/api/runs/viewer-test/render-view',json=body)
    assert b.status_code==200 and b.content!=a.content
    # History loads the exact saved model and never substitutes the final model.
    from studio.replay import save_model_snapshot
    body['c2w']=np.eye(4).tolist()
    for step in (0,1,2,3):
        with torch.no_grad():model.color.fill_(step/3)
        save_model_snapshot(root/'replay/models'/f'{step:07d}.pt',model,config,step)
        body['snapshot_step']=step
        historical=client.post('/api/runs/viewer-test/render-view',json=body)
        assert historical.status_code==200 and historical.headers['X-Model-Step']==str(step)
        assert historical.content!=a.content
    assert len(server.scene_viewer.models)<=3
    body['snapshot_step']=7
    assert client.post('/api/runs/viewer-test/render-view',json=body).status_code==404
    body['snapshot_step']=None
    assert client.post('/api/runs/viewer-test/render-view',json=body).headers['X-Model-Step']=='17'
    body['c2w'][0][0]=-1
    assert client.post('/api/runs/viewer-test/render-view',json=body).status_code==400
