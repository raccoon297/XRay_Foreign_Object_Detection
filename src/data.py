from .runtime import ROOT
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image
EDA = ROOT / "results/EDA"
PREP = ROOT / "Data"
PROCESSED = ROOT / "Data"

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def read_inputs():
    manifest = read_json(EDA / 'manifest.json')
    split = read_json(EDA / 'split.json')
    summary = read_json(EDA / 'summary.json')
    records = manifest['records']
    if len(records) != 497 or sum(len(r['boxes_xyxy']) for r in records) != 1140:
        raise RuntimeError('Unexpected EDA dataset; regenerate an explicit protocol before training.')
    for part, count in [('train', 299), ('val', 99), ('test', 99)]:
        selected = [r for r in records if r['split'] == part]
        if len(selected) != count or set(split['ids'][part]) != {r['id'] for r in selected}:
            raise RuntimeError(f'Frozen split changed: {part}')
    for field in ['id', 'date', 'group', 'pixel_sha256', 'gray_sha256']:
        sets = [{r[field] for r in records if r['split'] == p} for p in ['train', 'val', 'test']]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError(f'Split leakage: {field}')
    return records, summary['condition_cutoffs']

def remove_markers(rgb, config):
    # No TXT, bounding boxes, split, or model predictions enter this function.
    gray = np.median(rgb, axis=2).astype(np.uint8)
    color = (np.ptp(rgb.astype(np.int16), axis=2) > config['channel_spread_threshold']).astype(np.uint8)
    kernel = np.ones((config['dilation_kernel'], config['dilation_kernel']), np.uint8)
    mask = cv2.dilate(color, kernel) * 255
    clean = cv2.inpaint(gray, mask, config['inpaint_radius'], cv2.INPAINT_TELEA) if mask.any() else gray.copy()
    if not np.array_equal(clean[mask == 0], gray[mask == 0]):
        raise RuntimeError('Unmasked grayscale pixels changed.')
    return clean, gray, color, mask

def load_images(records):
    bank = {r['id']: r for r in read_json(PREP / 'preprocessing_manifest.json')['files']}
    result = []
    for record in records:
        path = PROCESSED / record['split'] / 'images' / (record['id'] + '.png')
        label = PROCESSED / record['split'] / 'labels' / (record['id'] + '.txt')
        if sha(path) != bank[record['id']]['image_sha256'] or sha(label) != record['label_sha256']:
            raise RuntimeError('Frozen input changed: ' + record['id'])
        result.append(np.asarray(Image.open(path).convert('L')).copy())
    return result
