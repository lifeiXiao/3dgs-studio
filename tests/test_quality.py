import json
import pytest
import torch
from studio.storage import ROOT
from studio.quality import image_quality,lpips_model
from studio.quality_history import evaluation_rows


@pytest.mark.skipif(not (ROOT/'.cache/torch/hub/checkpoints/vgg16-397923af.pth').exists(),reason='pretrained VGG weights are required')
def test_ssim_lpips_identical_and_distorted_images_use_frozen_network():
    torch.manual_seed(8);reference=torch.rand(32,32,3)
    identical=image_quality(reference,reference)
    distorted=image_quality(reference,reference*.4)
    assert identical['ssim']==pytest.approx(1,abs=1e-6)
    assert identical['lpips']==pytest.approx(0,abs=1e-8)
    assert distorted['ssim']<identical['ssim'] and distorted['lpips']>identical['lpips']
    network=lpips_model('cpu')
    assert not network.training and not any(parameter.requires_grad for parameter in network.parameters())


def test_quality_history_merges_steps_without_replacing_training_metrics(tmp_path):
    (tmp_path/'evaluations.csv').write_text('step,train_loss,train_ssim,val_ssim\n0,0.4,,\n1000,0.1,0.9,\n')
    (tmp_path/'quality_metrics.json').write_text(json.dumps({'rows':[
        {'step':1000,'train_ssim':.8,'train_lpips':.2,'val_ssim':.85,'val_lpips':.25},
        {'step':2000,'train_ssim':.95,'train_lpips':.1}]}))
    rows=evaluation_rows(tmp_path)
    assert [row['step'] for row in rows]==[0,1000,2000]
    assert rows[1]['train_ssim']==.9 and rows[1]['train_loss']==.1
    assert rows[1]['train_lpips']==.2 and rows[1]['val_ssim']==.85
    assert 'train_lpips' not in rows[0] and 'val_ssim' not in rows[2]
