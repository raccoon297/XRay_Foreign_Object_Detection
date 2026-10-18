"""Prepare the 497 frozen X-ray images after downloading the KAMP source dataset.

Input may be a recursively nested directory of original BMPs and TXT labels.
The manifest selects only matching, hash-verified originals; dataset files are
intentionally excluded from the public GitHub project.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import read_json, remove_markers, sha, write_json
from src.image_ops import clahe_image


def index_source(source: Path):
    entries = {}
    for p in source.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.bmp', '.txt'):
            entries.setdefault(p.name.lower(), []).append(p)
    return entries


def matching_file(index, name, digest):
    choices = index.get(name.lower(), [])
    for item in choices:
        if sha(item) == digest:
            return item
    raise FileNotFoundError(f'Matching KAMP source not found: {name} ({len(choices)} candidate(s), sha256={digest[:16]}...)')


def main():
    ap = argparse.ArgumentParser(description='Rebuild 497 fixed-split X-ray samples from a separately downloaded KAMP dataset')
    ap.add_argument('--source', type=Path, required=True, help='Folder containing original BMP/TXT files (nested folders allowed)')
    args = ap.parse_args()
    if not args.source.is_dir():
        ap.error('--source must be an extracted dataset directory')
    manifest = read_json(ROOT / 'results/EDA/manifest.json')
    records = manifest['records']
    if len(records) != 497:
        raise RuntimeError('The original 497-record manifest is required')
    config = read_json(ROOT / 'configs/baseline.json')
    clahe_cfg = read_json(ROOT / 'configs/experiment.json')['clahe']
    index = index_source(args.source)
    paths = []
    # Resolve every source before writing anything so missing files fail early.
    for r in records:
        paths.append((r, matching_file(index,r['id']+'.bmp',r['image_sha256']),
                      matching_file(index,r['id']+'.txt',r['label_sha256'])))
    bank=[]
    for ix,(r,bmp,txt) in enumerate(paths,1):
        stem, split = r['id'],r['split']
        rawim = ROOT/'Data/raw/images'/f'{stem}.bmp'
        rawlb = ROOT/'Data/raw/labels'/f'{stem}.txt'
        processed = ROOT/'Data'/split/'images'/f'{stem}.png'
        processedlb = ROOT/'Data'/split/'labels'/f'{stem}.txt'
        enhanced = ROOT/'Data/clahe'/split/'images'/f'{stem}.png'
        enhancedlb = ROOT/'Data/clahe'/split/'labels'/f'{stem}.txt'
        for p in [rawim,rawlb,processed,processedlb,enhanced,enhancedlb]:
            p.parent.mkdir(parents=True, exist_ok=True)
        if bmp.resolve()!=rawim.resolve(): shutil.copyfile(bmp,rawim)
        if txt.resolve()!=rawlb.resolve(): shutil.copyfile(txt,rawlb)
        rgb=np.asarray(Image.open(rawim).convert('RGB')).copy()
        clean,gray,color,mask=remove_markers(rgb,config['preprocessing'])
        Image.fromarray(clean).save(processed)
        Image.fromarray(clahe_image(clean,clahe_cfg)).save(enhanced)
        shutil.copyfile(rawlb,processedlb)
        shutil.copyfile(rawlb,enhancedlb)
        bank.append({'id':stem,'split':split,'image_sha256':sha(processed),
                     'label_sha256':sha(processedlb),
                     'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
        if ix%100==0: print(f'{ix}/{len(paths)} prepared',flush=True)
    write_json(ROOT/'Data/preprocessing_manifest.json',{'policy':'KAMP source + frozen split; marker removal + CLAHE; recreated by prepare_data.py','files':bank})
    for view, root in [('raw',ROOT/'Data'),('clahe',ROOT/'Data/clahe')]:
        cfg=f'path: {json.dumps(str(root.resolve()),ensure_ascii=False)}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: foreign_object\n'
        (ROOT/'Data'/f'{view}.yaml').write_text(cfg,encoding='utf-8')
    print('Prepared 497 images (train 299, val 99, test 99) and two training views.',flush=True)

if __name__=='__main__':main()
