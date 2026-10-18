"""V6-local dependency/settings bootstrap; never import CODE experiments."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / 'results' / 'runtime_settings'
SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(SETTINGS)
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')


def environment():
    import importlib.metadata as metadata
    import platform
    import torch
    versions = {}
    for name in ['torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python',
                 'pillow', 'matplotlib', 'pyyaml', 'psutil', 'polars', 'ultralytics-thop']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'packages': versions,
            'torch_version': torch.__version__, 'cuda_build': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'torch_threads': torch.get_num_threads()}


def require_gpu(index=0):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. Check GPU connection and CUDA-enabled PyTorch; no automatic CPU fallback.')
    if index >= torch.cuda.device_count():
        raise RuntimeError(f'GPU {index} does not exist.')
    torch.cuda.set_device(index)
    tensor = torch.randn(128, 128, device=f'cuda:{index}', requires_grad=True)
    result = (tensor @ tensor.T).square().mean()
    result.backward()
    torch.cuda.synchronize(index)
    if not torch.isfinite(tensor.grad).all().item():
        raise RuntimeError('GPU backward produced nonfinite values.')
    return {'device': index, 'name': torch.cuda.get_device_name(index),
            'memory_gib': torch.cuda.get_device_properties(index).total_memory / 1024**3,
            'forward_backward_passed': True}


if __name__ == '__main__':
    import json
    print(json.dumps(environment(), ensure_ascii=False, indent=2))
