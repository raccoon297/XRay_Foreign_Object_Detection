from src.runtime import ROOT
from src.data import write_json
from src.predictor import Detector
from pathlib import Path
from PIL import Image,ImageDraw
import argparse
import os

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--model',choices=['A','B','C','D','E'],default='E');parser.add_argument('--candidate')
    parser.add_argument('--marker-removed',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'results/user_predictions')
    args=parser.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');args.output.mkdir(parents=True,exist_ok=True)
    prefix=args.image.stem+'_'+(args.candidate or args.model);paths=[args.output/(prefix+'.json'),args.output/(prefix+'.png')]
    if any(p.exists() for p in paths): raise RuntimeError('Output exists; choose a different --output')
    result,pixels=Detector(args.model,args.candidate).predict_file(args.image,args.marker_removed)
    image=Image.fromarray(pixels).convert('RGB');draw=ImageDraw.Draw(image)
    for x1,y1,x2,y2,conf in result['boxes_xyxy_confidence']:
        draw.rectangle((x1,y1,x2,y2),outline='red',width=1);draw.text((x1,max(0,y1-12)),f'{conf:.3f}',fill='red')
    image.save(paths[1]);write_json(paths[0],result);print(paths[0],flush=True);print(paths[1],flush=True)

if __name__=='__main__': main()
