from .runtime import ROOT, require_gpu
from .data import read_json, sha
from .fusion import blend
from .evaluation import iou_matrix
from pathlib import Path
import json
import numpy as np
import torch
import cv2
from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

class Tee:
    def __init__(self, terminal, stream): self.terminal, self.stream = terminal, stream
    def write(self, value): self.terminal.write(value); self.stream.write(value); self.stream.flush()
    def flush(self): self.terminal.flush(); self.stream.flush()
    def isatty(self): return False

def setup(inference=False):
    import ultralytics
    config = read_json(ROOT / 'configs/baseline.json')
    if ultralytics.__version__ != config['ultralytics_version']: raise RuntimeError('Ultralytics version differs')
    settings.update({'datasets_dir': str(ROOT/'Data'), 'weights_dir': str(ROOT/'models/pretrained'), 'runs_dir': str(ROOT/'results'), 'sync': False})
    torch.set_num_threads(2); cv2.setNumThreads(2)
    if inference:
        # Inference can run on CPU when CUDA is not available; original evaluation remains GPU-only.
        config['train']['device'] = config['train']['device'] if torch.cuda.is_available() else 'cpu'
        return config, {'inference_device': config['train']['device']}
    return config, require_gpu(config['train']['device'])

def verify_assets():
    """Check frozen training metadata and selected checkpoints without V4/V5 directories."""
    protocol = read_json(ROOT / 'results/protocol.json')
    checkpoints = {}
    for name, expected in protocol['model_sha256'].items():
        actual = sha(ROOT / f'models/{name}/selected.pt')
        if actual != expected:
            raise RuntimeError('Selected model checkpoint differs: '+name)
        checkpoints[name] = actual
    return {'selected_checkpoints_verified': sorted(checkpoints)}


def fuse(original, partner, spec, alpha):
    results, audits = [], []
    if [p['id'] for p in original] != [p['id'] for p in partner]: raise RuntimeError('Partner IDs differ')
    for a, b in zip(original, partner):
        box, detail = blend(a['boxes'], b['boxes'], spec, alpha)
        results.append({'id': a['id'], 'boxes': box.tolist()})
        audits.extend(dict(id=a['id'], **row) for row in detail if row['A_confidence'] >= spec['minimum_A_score'])
    return results, audits

def official(checkpoint, view, config, folder, partner=None, spec=None, alpha=0.):
    partner_bank = {p['id']: p for p in partner} if partner is not None else {}
    class CoordinateValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            if alpha:
                for i, pred in enumerate(preds):
                    name = Path(batch['im_file'][i]).stem
                    native = ops.scale_boxes(batch['img'].shape[2:], pred['bboxes'].clone(), batch['ori_shape'][i], batch['ratio_pad'][i])
                    values = np.c_[native.cpu().numpy(), pred['conf'].cpu().numpy()]
                    corrected, _ = blend(values, partner_bank[name]['boxes'], spec, alpha)
                    changed = np.any(values[:, :4] != corrected[:, :4], axis=1)
                    if changed.any():
                        gain, pad = batch['ratio_pad'][i][0][0], batch['ratio_pad'][i][1]
                        returned = torch.as_tensor(corrected[changed, :4], dtype=pred['bboxes'].dtype, device=pred['bboxes'].device)
                        returned *= gain; returned[:, [0, 2]] += pad[0]; returned[:, [1, 3]] += pad[1]
                        pred['bboxes'][np.flatnonzero(changed)] = returned
            super().update_metrics(preds, batch)
    metric = YOLO(str(checkpoint)).val(validator=CoordinateValidator, data=str(ROOT / f'Data/{view}.yaml'), split='val',
        imgsz=config['train']['imgsz'], batch=config['train']['batch'], device=config['train']['device'], workers=0,
        rect=False, conf=config['evaluation']['confidence_floor'], iou=config['evaluation']['nms_iou'],
        max_det=config['evaluation']['max_det'], half=False, plots=False, verbose=False, project=str(folder), name='official', exist_ok=True)
    result = {k: float(v) for k, v in metric.results_dict.items()}
    result['AP75'] = float(np.asarray(metric.box.all_ap)[:, 5].mean())
    return result

def replay_difference(a, b):
    if [p['id'] for p in a] != [p['id'] for p in b]: raise RuntimeError('Replay IDs differ')
    error = 0.
    for p, q in zip(a, b):
        av, bv = np.asarray(p['boxes']).reshape(-1, 5), np.asarray(q['boxes']).reshape(-1, 5)
        if av.shape != bv.shape: raise RuntimeError('Replay prediction count differs')
        if av.size: error = max(error, float(np.abs(av-bv).max()))
    return error
