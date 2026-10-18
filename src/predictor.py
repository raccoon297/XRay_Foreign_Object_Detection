"""A/B/C or selected D/E, pixels and fixed recipes only."""
from .runtime import ROOT
from .data import read_json, sha, remove_markers
from .image_ops import infer, clahe_image
from .core import setup, fuse
from PIL import Image
import numpy as np

class Detector:
    def __init__(self,name='E',candidate=None):
        from ultralytics import YOLO
        self.config,self.gpu=setup(inference=True);self.name=name
        self.spec=read_json(ROOT/'configs/experiment.json');protocol=read_json(ROOT/'results/protocol.json')
        # Historical source hashes describe the original V6 experiment, not this minimal release.
        self.candidate=None
        if name in ['D','E']:
            selection=read_json(ROOT/'results/selection_lock.json')
            if selection['protocol_sha256']!=sha(ROOT/'results/protocol.json'): raise RuntimeError('Selection protocol differs')
            self.candidate=candidate or selection[name]['candidate']
            partner=self.spec['families'][name]['partner']
            choices={f'{name}_A{p:02d}_{partner}{100-p:02d}':p for p in self.spec['A_percent_grid']}
            if self.candidate not in choices: raise ValueError('Candidate outside declared family/grid')
            ap=choices[self.candidate]
            self.recipe={'family':name,'anchor':'A','partner':partner,'A_percent':ap,'partner_percent':100-ap,'partner_weight':(100-ap)/100}
            self.safe_selection=selection[name]['safe_selection'] if candidate is None else None
            needed=['A',partner]
        else:
            if candidate is not None: raise ValueError('Only D/E have weight candidates')
            self.recipe=self.spec['methods'][name];self.safe_selection=None;needed=[name]
        self.models={}
        for model in needed:
            path=ROOT/f'models/{model}/selected.pt'
            if sha(path)!=protocol['model_sha256'][model]: raise RuntimeError('Checkpoint changed')
            self.models[model]=YOLO(str(path))
        self.confidence=read_json(ROOT/'configs/A_threshold_lock.json')['confidence']

    def predict_clean(self,images,ids):
        if len(images)!=len(ids) or any(im.ndim!=2 or im.dtype!=np.uint8 for im in images): raise ValueError('Expected native uint8 gray images and IDs')
        records=[{'id':ident,'width':im.shape[1],'height':im.shape[0]} for ident,im in zip(ids,images)]
        predictions,timing={},{}
        for model,detector in self.models.items():
            view=images if model=='A' else [clahe_image(im,self.spec['clahe']) for im in images]
            predictions[model],timing[model]=infer(detector,records,view,self.config)
        output=fuse(predictions['A'],predictions[self.recipe['partner']],self.spec['fusion'],self.recipe['partner_weight'])[0] if 'partner' in self.recipe else predictions[self.name]
        return output,timing

    def predict_file(self,path,marker_removed=False):
        with Image.open(path) as image:
            pixels=np.asarray(image.convert('L')).copy() if marker_removed else remove_markers(np.asarray(image.convert('RGB')).copy(),self.config['preprocessing'])[0]
        predictions,timing=self.predict_clean([pixels],[path.stem])
        boxes=[b for b in predictions[0]['boxes'] if b[4]>=self.confidence]
        return {'model':self.name,'candidate':self.candidate,'recipe':self.recipe,'boxes_xyxy_confidence':boxes,'confidence':self.confidence,
                'status':'detected' if boxes else 'review_required','timing':timing},pixels
