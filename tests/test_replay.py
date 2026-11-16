from PIL import Image
import pytest
from studio.replay import export_gif
from studio.storage import write_json


def test_training_replay_exports_real_frames_in_step_order(tmp_path):
    replay=tmp_path/'replay'; replay.mkdir()
    for step,color in [(0,'red'),(25,'blue'),(50,'green')]:
        Image.new('RGB',(32,32),color).save(replay/f'{step:07d}.png')
    write_json(replay/'index.json',[{'step':step} for step in [0,25,50]])
    assert export_gif(tmp_path)==3
    with Image.open(tmp_path/'training.gif') as gif:
        assert gif.n_frames==3 and gif.info['loop']==0
        assert gif.convert('RGB').getpixel((0,0))==(255,0,0)
        gif.seek(1)
        assert gif.convert('RGB').getpixel((0,0))==(0,0,255)
        gif.seek(2)
        assert gif.convert('RGB').getpixel((0,0))==(0,128,0)


def test_replay_requires_multiple_recorded_frames(tmp_path):
    with pytest.raises(ValueError,match='至少'): export_gif(tmp_path)


def test_full_snapshot_preserves_all_gaussians_and_sh(tmp_path):
    import torch
    from studio.model import Gaussians
    from studio.replay import save_model_snapshot
    from studio.evaluation import load_checkpoint
    model=Gaussians(torch.randn(8193,3),torch.rand(8193,3),sh_degree=3,color_activation='sh')
    model.active_sh_degree=2
    with torch.no_grad():model.sh_rest.normal_()
    path=tmp_path/'models/0001000.pt'
    save_model_snapshot(path,model,{'profile':'official'},1000)
    restored,metadata=load_checkpoint(path,torch.device('cpu'),mmap=True,model_only=True)
    assert len(restored.xyz)==8193 and restored.active_sh_degree==2
    assert metadata['step']==1000 and metadata['snapshot_kind']=='full_gaussian_model'
    assert 'optimizers' not in metadata and not path.with_suffix('.tmp.pt').exists()
    for key,value in model.state_dict().items():assert torch.equal(value,restored.state_dict()[key])
