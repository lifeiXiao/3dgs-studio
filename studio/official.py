"""GraphDeco-compatible model/optimizer defaults; Studio owns only orchestration/UI."""
from functools import lru_cache
import importlib.util
from .storage import ROOT

SOURCE=ROOT/'third_party/gaussian-splatting'
OFFICIAL_DEFAULTS=dict(
    profile='official',iterations=30000,resolution=-1,sh_degree=3,gaussians=100000,
    max_gaussians=0,batch_size=1,seed=0,backend='auto',device='auto',densify=True,
    position_lr_init=.00016,position_lr_final=.0000016,position_lr_delay_mult=.01,
    position_lr_max_steps=30000,feature_lr=.0025,opacity_lr=.025,scaling_lr=.005,
    rotation_lr=.001,percent_dense=.01,lambda_dssim=.2,densification_interval=100,
    opacity_reset_interval=3000,densify_from_iter=500,densify_until_iter=15000,
    densify_grad_threshold=.0002,white_background=False,eval=False,
    random_background=False,optimizer_type='default',antialiasing=False,
    exposure_lr_init=.01,exposure_lr_final=.001,exposure_lr_delay_steps=0,
    exposure_lr_delay_mult=0.,depth_l1_weight_init=1.,depth_l1_weight_final=.01,
    data_device='cuda',train_test_exp=False,
)


def source_info():
    path=SOURCE/'STUDIO_SOURCE_COMMIT'
    if not path.exists():path=ROOT/'third_party/graphdeco-commit.txt'
    return dict(repository='https://github.com/graphdeco-inria/gaussian-splatting',
                commit=path.read_text().strip() if path.exists() else 'unavailable')


@lru_cache(maxsize=2)
def official_module(name):
    path=SOURCE/'utils'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(f'graphdeco_{name}',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def one_minus_ssim(pred,target):
    if pred.ndim==3:pred,target=pred[None],target[None]
    return 1-official_module('loss_utils').ssim(pred.permute(0,3,1,2),target.permute(0,3,1,2))


def make_strategy(config,scene_scale):
    from gsplat.strategy import DefaultStrategy
    kwargs=dict(absgrad=False,prune_opa=.005,
        grow_grad2d=config['densify_grad_threshold'],grow_scale3d=config['percent_dense'],
        prune_scale3d=.1,refine_start_iter=config['densify_from_iter'],
        refine_stop_iter=config['densify_until_iter'],refine_every=config['densification_interval'],
        reset_every=config['opacity_reset_interval'],pause_refine_after_reset=0,
        revised_opacity=False,refine_scale2d_stop_iter=0)
    if config.get('max_gaussians',0):
        from .density import BoundedStrategy
        strategy=BoundedStrategy(config['max_gaussians'],growth_fraction=1.,**kwargs)
    else:strategy=DefaultStrategy(**kwargs)
    return strategy,strategy.initialize_state(scene_scale=scene_scale)
