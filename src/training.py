"""Optional fresh A/B/C training; retained imported checkpoints are immutable."""
from .runtime import ROOT
from .data import read_json, write_json, sha, read_inputs, load_images
from .image_ops import clahe_image
from .core import setup
from ultralytics.models.yolo.detect import DetectionTrainer
from pathlib import Path
from PIL import Image
import argparse
import csv
import os
import random
import shutil
import numpy as np
import cv2
import torch


class NativeContrastChoice:
    def __init__(self): self.bank, self.counts = {}, {'raw': 0, 'clahe': 0}
    def __call__(self, sample):
        if random.random() >= .5:
            self.counts['raw'] += 1; return sample
        name = Path(sample['im_file']).stem
        if name not in self.bank: self.bank[name] = np.asarray(Image.open(ROOT/'Data/clahe/train/images'/(name+'.png')).convert('RGB')).copy()
        native = self.bank[name]
        if native.shape[:2] != tuple(sample['ori_shape']): raise RuntimeError('Mixed native shape differs')
        h,w = sample['img'].shape[:2]
        sample['img'] = cv2.resize(native, (w,h), interpolation=cv2.INTER_LINEAR)
        self.counts['clahe'] += 1; return sample


class MixedTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode=='train':
            choice = NativeContrastChoice(); dataset.transforms.transforms.insert(0, choice); dataset.contrast_choice = choice
        return dataset


def main():
    from ultralytics import YOLO
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['A','B','C'], required=True)
    parser.add_argument('--run-name', required=True, help='New directory name; existing runs are never overwritten')
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ['.','..']: raise ValueError('One directory name required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config, gpu = setup(); spec = read_json(ROOT/'configs/experiment.json'); recipe = spec['methods'][args.model]
    run = ROOT/'results/training'/args.run_name
    if run.exists(): raise RuntimeError('Run exists; choose a new --run-name')
    # Train/val integrity only. No test images are decoded.
    records, _ = read_inputs()
    for part in ['train','val']:
        subset = [r for r in records if r['split']==part]; images = load_images(subset)
        for r, image in zip(subset, images):
            txt = ROOT/'Data/clahe'/part/'labels'/(r['id']+'.txt')
            view = np.asarray(Image.open(ROOT/'Data/clahe'/part/'images'/(r['id']+'.png')).convert('L'))
            if sha(txt) != r['label_sha256'] or not np.array_equal(view, clahe_image(image, spec['clahe'])): raise RuntimeError('CLAHE training view changed')
    pretrained = ROOT/'models/pretrained/yolo26n.pt'
    if sha(pretrained) != config['pretrained_sha256']: raise RuntimeError('Pretrained checkpoint differs')
    run.mkdir(parents=True)
    write_json(run/'protocol.json', {'recipe': recipe, 'train_args': config['train'], 'GPU': gpu, 'pretrained_sha256': sha(pretrained),
                                     'test_images_loaded': 0, 'retained_models_modified': False})
    best, seen = (-1.,-1.,0), set()
    def on_save(trainer):
        nonlocal best
        epoch = trainer.epoch+1
        if epoch in seen: return
        seen.add(epoch)
        metrics = {k: float(v) for k,v in trainer.metrics.items()}
        rank = (metrics['metrics/mAP50-95(B)'], metrics['metrics/mAP50(B)'], -epoch)
        if rank > best:
            best = rank; shutil.copyfile(trainer.last, run/'selected.pt')
            write_json(run/'selected_epoch.json', {'epoch': epoch, 'metrics': metrics, 'rank': list(rank)})
        choice = getattr(trainer.train_loader.dataset, 'contrast_choice', None)
        write_json(run/'status.json', {'epoch': epoch, 'best_epoch': -best[2], 'contrast_counts': choice.counts if choice else {'clahe_probability': recipe['training_clahe_probability']}})
    model = YOLO(str(pretrained)); model.add_callback('on_model_save', on_save)
    kwargs = dict(config['train'], data=str(ROOT/f"Data/{recipe['checkpoint_validation_view']}.yaml"), project=str(run), name='training', exist_ok=True)
    if args.model=='C': kwargs['trainer'] = MixedTrainer
    model.train(**kwargs)
    lock = read_json(run/'selected_epoch.json'); checkpoint = torch.load(run/'selected.pt', map_location='cpu', weights_only=False)
    with (run/'training/results.csv').open(encoding='utf-8-sig') as stream:
        rows = [{k.strip(): float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    selected = max(rows, key=lambda r: (r['metrics/mAP50-95(B)'], r['metrics/mAP50(B)'], -r['epoch']))
    if checkpoint['epoch']+1 != lock['epoch'] or int(selected['epoch']) != lock['epoch'] or abs(checkpoint['train_metrics']['metrics/mAP50-95(B)']-lock['metrics']['metrics/mAP50-95(B)']) > 1e-7: raise RuntimeError('Checkpoint selection mismatch')
    lock.update(checkpoint_sha256=sha(run/'selected.pt'), completed_epochs=len(rows), checkpoint_epoch_metric_verified=True)
    write_json(run/'selection_lock.json', lock)
    print('Fresh training complete:', run, '; retained A/B/C/F unchanged', flush=True)
