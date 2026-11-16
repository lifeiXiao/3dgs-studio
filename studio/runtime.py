"""Keep native compilation inside this project and choose a compatible host compiler."""
import os
from pathlib import Path
import shutil
import sys
import importlib.util
import importlib.metadata
import json
from .storage import ROOT


def configure_cuda_build():
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    cache=ROOT/'.cache/torch_extensions';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(cache))
    os.environ.setdefault('MAX_JOBS','2')
    if shutil.which('gcc-12') and shutil.which('g++-12'):
        # Ubuntu's CUDA 12.0 toolkit rejects GCC 13. No system compiler is modified.
        os.environ.setdefault('CC',shutil.which('gcc-12'))
        os.environ.setdefault('CXX',shutil.which('g++-12'))
    import torch
    if torch.cuda.is_available():
        major,minor=torch.cuda.get_device_capability()
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST',f'{major}.{minor}')

    # Reuse our locally built binary when the Python/Torch/gsplat/SM ABI matches.
    # This avoids recompiling solely because an intact project was moved to another disk.
    prebuilt=ROOT/'.cache/prebuilt'
    if torch.cuda.is_available() and (prebuilt/'manifest.json').exists() and (prebuilt/'gsplat_cuda.so').exists():
        expected=dict(torch=torch.__version__,gsplat=importlib.metadata.version('gsplat'),
                      arch=f'{major}.{minor}',python=f'{sys.version_info.major}.{sys.version_info.minor}')
        if json.loads((prebuilt/'manifest.json').read_text())==expected:
            spec=importlib.util.spec_from_file_location('gsplat_cuda',prebuilt/'gsplat_cuda.so')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            import gsplat
            gsplat.csrc=module;sys.modules['gsplat.csrc']=module
