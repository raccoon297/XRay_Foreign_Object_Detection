"""A/B/C and nineteen A-partner weights per D/E family, val only."""
from .runtime import ROOT, environment
from .data import read_json, write_json, read_inputs, load_images, sha
from .evaluation import matches, counts, condition_recall
from .image_ops import clahe_image, infer
from .fusion import self_checks
from .core import setup, verify_assets, fuse, official, replay_difference, Tee
from pathlib import Path
import argparse
import contextlib
import csv
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

KEYS = ['metrics/mAP50(B)','AP75','metrics/mAP50-95(B)']

def candidate_name(family, a_percent, partner):
    return f'{family}_A{a_percent:02d}_{partner}{100-a_percent:02d}'

def operating(records, predictions, config, cutoffs, threshold):
    tp,fp,gt=matches(records,predictions,config['evaluation']['match_iou'])
    return {'confidence':threshold,'operating_point':counts(tp,fp,sum(len(r['boxes_xyxy']) for r in records),len(records),threshold),
            'conditions':condition_recall(records,gt,cutoffs,threshold)}

def safe(a,row):
    return row['operating_point']['TP']>=a['operating_point']['TP'] and row['operating_point']['FP']<=a['operating_point']['FP'] and all(row['conditions'][k]['TP']>=v['TP'] for k,v in a['conditions'].items())

def rank(row):
    return (row['official'][KEYS[2]],row['official'][KEYS[1]],row['official'][KEYS[0]],row['recipe']['A_percent'])

def validate_geometry(records,original,predictions):
    changed=0
    for r,a,b in zip(records,original,predictions):
        av,bv=np.asarray(a['boxes']).reshape(-1,5),np.asarray(b['boxes']).reshape(-1,5)
        if av.shape!=bv.shape or not np.array_equal(av[:,4],bv[:,4]): raise RuntimeError('A scores/count changed')
        if not np.isfinite(bv).all() or np.any(bv[:,2:4]<bv[:,:2]): raise RuntimeError('Invalid fused geometry')
        if np.any(bv[:,:2]<-1e-5) or np.any(bv[:,[0,2]]>r['width']+1e-5) or np.any(bv[:,[1,3]]>r['height']+1e-5): raise RuntimeError('Fused box outside native image')
        changed+=int(np.any(av[:,:4]!=bv[:,:4],axis=1).sum())
    return changed

def load_result(name):
    path=ROOT/'results'/name/'evaluation.json'
    if not path.exists(): return None
    row=read_json(path)
    if row['prediction_sha256']!=sha(path.parent/'predictions.json'): raise RuntimeError('Saved prediction changed: '+name)
    return row

def reference_check(name,row,predictions):
    folder=ROOT/'provenance/references'/name
    if not folder.exists(): return {'scope':'new combination, no historical reference'}
    reference=read_json(folder/'evaluation.json')
    error=replay_difference(predictions,read_json(folder/'predictions.json'))
    deltas={k:row['official'][k]-reference['official'][k] for k in KEYS}
    if error>1e-5 or max(abs(v) for v in deltas.values())>1e-7 or row['operating_point']!=reference['operating_point'] or row['conditions']!=reference['conditions']:
        raise RuntimeError('Historical recipe not reproduced: '+name)
    return {'native_max_difference':error,'official_AP_differences':deltas,'operating_and_conditions_equal':True}

def write_result(name,row,predictions):
    folder=ROOT/'results'/name; folder.mkdir(exist_ok=True)
    write_json(folder/'predictions.json',predictions)
    row['prediction_sha256']=sha(folder/'predictions.json')
    row['historical_replay']=reference_check(name,row,predictions)
    write_json(folder/'evaluation.json',row)
    return row

def finish(spec,baselines,candidates,protocol):
    selections={}; methods=dict(baselines)
    for family in ['D','E']:
        bank={n:r for n,r in candidates.items() if r['recipe']['family']==family}
        if len(bank)!=19: return
        unconstrained=max(bank,key=lambda n:rank(bank[n]))
        feasible=[n for n in bank if bank[n]['safe_vs_A']]
        selected=max(feasible,key=lambda n:rank(bank[n])) if feasible else unconstrained
        selections[family]={'candidate':selected,'safe_selection':bool(feasible),'safe_candidates':len(feasible),
                            'unconstrained_best':unconstrained,'criterion':spec['selection'],'recipe':bank[selected]['recipe']}
        methods[family]=bank[selected]
    superiority={k:{'E_at_least_all_others':all(methods['E']['official'][k]>=methods[n]['official'][k]-1e-10 for n in ['A','B','C','D']),
                    'E_strictly_above_all_others':all(methods['E']['official'][k]>methods[n]['official'][k]+1e-10 for n in ['A','B','C','D']),
                    'best_models':[n for n in methods if abs(methods[n]['official'][k]-max(r['official'][k] for r in methods.values()))<1e-10]} for k in KEYS}
    original=candidates['E_A75_C25']
    result={'model_order':spec['model_order'],'methods':methods,'selections':selections,'candidates':candidates,
            'superiority_check':superiority,'retained_original_F':{'candidate':'E_A75_C25','metrics':original['official'],'operating_point':original['operating_point']},
            'final_model':'E','test_evaluated':False,'scope':spec['scope']}
    write_json(ROOT/'results/comparison.json',result)
    write_json(ROOT/'results/selection_lock.json',{'D':selections['D'],'E':selections['E'],'final_model':'E','protocol_sha256':sha(ROOT/'results/protocol.json')})
    rows=[]
    for family in ['D','E']:
        for a_pct in spec['A_percent_grid']:
            name=candidate_name(family,a_pct,spec['families'][family]['partner']); row=candidates[name]; ap=row['official']; op=row['operating_point']
            rows.append({'candidate':name,'family':family,'A_percent':a_pct,'partner_percent':100-a_pct,'AP50':ap[KEYS[0]],'AP75':ap[KEYS[1]],'mAP50_95':ap[KEYS[2]],
                         'TP':op['TP'],'FN':op['FN'],'FP':op['FP'],'safe_vs_A':row['safe_vs_A'],'selected':selections[family]['candidate']==name})
    with (ROOT/'results/weight_sweep.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(ROOT/'results/checks.json',{'GPU':read_json(ROOT/'results/environment_check.json')['GPU'],'new_training_steps':0,'test_images_loaded':0,'test_images_inferred':0,
        'D_candidates':19,'E_candidates':19,'selected_checkpoints_verified_after_run':verify_assets(),
        'sources_and_configuration_unchanged':all(sha(ROOT/n)==expected for n,expected in protocol['sources'].items()),
        'historical_replays':{n:r['historical_replay'] for n,r in {**baselines,**candidates}.items() if 'native_max_difference' in r['historical_replay']},
        'scores_counts_bounds_preserved':True})
    report(result,rows)
    print('V6 COMPLETE',json.dumps({'selections':selections,'superiority_check':superiority},ensure_ascii=False),flush=True)

def report(result,rows):
    methods=result['methods']; selections=result['selections']
    lines=['# V6: 원본·대비 학습과 듀얼 박스 알고리즘 비교','',
        'A(원본) → B(CLAHE) → C(혼합 학습) → D(A+B) → E(A+C)의 논리 순서로 비교했다. D와 E 각각 A 비율5%,10%,…,95%의19개 조합을 평가했다. 기존에 본 val과 학습 가중치를 재사용한 후향적·탐색적 비교이며 최초 수행 연대기나 독립 확증 실험으로 표현하지 않는다.','',
        '| 모델 | 학습 / 추론 | 선택 비율 | AP50 | AP75 | mAP50–95 | TP / FN / FP |','|---|---|---|---:|---:|---:|---|']
    for n in result['model_order']:
        row=methods[n];ap=row['official'];op=row['operating_point'];ratio='—' if n in ['A','B','C'] else f"A:{row['recipe']['partner']}={row['recipe']['A_percent']}:{row['recipe']['partner_percent']}"
        desc={'A':'원본 / 원본','B':'CLAHE / CLAHE','C':'원본·CLAHE 혼합 / CLAHE','D':'A+B 좌표 결합','E':'A+C 좌표 결합'}[n]
        lines.append(f"| {n} | {desc} | {ratio} | {ap[KEYS[0]]*100:.2f}% | {ap[KEYS[1]]*100:.2f}% | {ap[KEYS[2]]*100:.2f}% | {op['TP']} / {op['FN']} / {op['FP']} |")
    lines+=['','## 비율 선택 기준과 결과','',
        '두 family에 같은 기준을 적용했다. 고정 A confidence에서 TP·FP와 train 정의 조건별 TP를 A보다 악화시키지 않는 후보를 우선하며, 그 안에서 공식 mAP50–95, AP75, AP50, 높은 A 비율 순으로 선택한다. 안전 조건을 만족하는 후보가 없으면 무제약 최고 후보를 표시하되 안전 보전 모델이라고 주장하지 않는다. 제약 없는 최고 점수도 함께 공개한다.','']
    for family,item in selections.items():
        lines.append(f"- {family}: 선택 {item['candidate']}, 안전 조건 통과 후보 {item['safe_candidates']}/19, 안전 조건 기반 선택={item['safe_selection']}. 무제약 mAP 최고={item['unconstrained_best']}.")
    lines+=['','## 모든 점수에서의 우위 검증','','| 지표 | 최상위 모델 | E가 다른 네 모델 이상 | E가 다른 네 모델보다 엄격히 높음 |','|---|---|---|---|']
    for k,label in zip(KEYS,['AP50','AP75','mAP50–95']):
        s=result['superiority_check'][k]; lines.append(f"| {label} | {', '.join(s['best_models'])} | {s['E_at_least_all_others']} | {s['E_strictly_above_all_others']} |")
    if all(v['E_strictly_above_all_others'] for v in result['superiority_check'].values()):
        lines+=['','선택된 다섯 모델의 세 AP 지표에서는 E가 모두 가장 높았다. 이는 반복 사용한 validation의 결과이며 독립 데이터에서의 우위를 뜻하지 않는다.']
    else:
        lines+=['','**E가 모든 점수에서 우위를 점했다는 결론은 이 결과로 성립하지 않는다.** 표에서 실제로 개선한 지표와 유지·악화한 지표를 구분해야 한다. 최종 제안 family는 E로 두되 관찰되지 않은 전면적 우위를 주장하지 않는다.']
    lines+=['','## 기존 최종 F의 보존','',
        'V5 F는 V6 E_A75_C25와 동일하다. 해당 후보의 가중치·정제 영상·좌표 결합·confidence·예측과 공식 AP를 기존 결과와 대조했다. 선택 비율이 달라도 이 후보와 원래 결과는 삭제하거나 덮어쓰지 않는다.','',
        '## 해석 범위','',
        '- 원본은 색상 표시만 제거한 영상이다. 공식 TXT를 정답으로 사용하며 CLAHE·모델 입력·박스 대응에는 정답 정보를 넣지 않는다.',
        '- B는 CLAHE val, C는 원본 val로 기존 checkpoint를 선택했다. 학습은 같은 초기화·seed·나머지 설정이지만 checkpoint 선택 입력 차이가 남아 있어 순수한 혼합 학습 효과를 분리하지 못한다.',
        '- 혼합 학습은 매 방문마다 원본/CLAHE를50% 확률로 선택한 것으로, 학습 영상을 두 배로 늘린 것이 아니다. B 단독 대비 C 단독 개선 여부와 듀얼 결합 개선 여부를 구분한다.',
        '- A confidence는0.271832734로 고정했다. 공식 AP는 stock validator의 전체 confidence 평가, 운영 TP/FN/FP는 별도 native array 예측의 score-order IoU≥.5 매칭이다. 두 추론 경로의 전처리 차이를 숨기지 않는다.',
        '- 19+19개 비율을 같은 val99장/209개 객체에서 선택했다. 단일 seed·다중 탐색 편향을 포함하며 test는 평가하지 않았다. 정상 제품 영상이 없어 FP는 양성 영상의 unmatched 박스다.',
        '- 5:95 등의 비율은 A:보조 박스 좌표의 비율이다. confidence 평균이나 교집합 박스가 아니며 대응이 없으면 A를 유지한다. A의 점수·개수·순서를 보존한다.',
        '- 모델 두 번의 추론 비용이 필요한다. 추가 학습은 없지만 단일 A보다 추론 비용이 증가한다.','',
        '전체38개 수치: `weight_sweep.csv`. 재현 설정과 원본 출처: `protocol.json`, `../provenance/migration.json`.','',
        '![비율별 비교](weight_sweep.png)','','![선택된 다섯 모델](comparison.png)','']
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,key,title in zip(axes,KEYS,['AP50','AP75','mAP50-95']):
        vals=[100*methods[n]['official'][key] for n in result['model_order']]
        ax.bar(result['model_order'],vals,color=['#0e7490' if n=='E' else '#64748b' for n in result['model_order']]); ax.set(title=title+' | selected, reused val',ylabel='%',ylim=(0,max(vals)*1.18))
        for i,v in enumerate(vals): ax.text(i,v+max(vals)*.02,f'{v:.2f}',ha='center')
    fig.savefig(ROOT/'results/comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for family,color in [('D','#d97706'),('E','#0e7490')]:
        bank=[r for r in rows if r['family']==family]
        for ax,key,title in zip(axes,['mAP50_95','AP75','FN'],['mAP50-95','AP75','FN at frozen A confidence']):
            values=[r[key]*100 if key!='FN' else r[key] for r in bank]
            ax.plot([r['A_percent'] for r in bank],values,'o-',label=family,color=color,markersize=3)
            ax.set(title=title,xlabel='A coordinate weight (%)',ylabel='%' if key!='FN' else 'Objects')
            ax.legend()
    axes[0].axhline(methods['A']['official'][KEYS[2]]*100,color='#64748b',linestyle='--')
    axes[1].axhline(methods['A']['official'][KEYS[1]]*100,color='#64748b',linestyle='--')
    axes[2].axhline(methods['A']['operating_point']['FN'],color='#64748b',linestyle='--')
    fig.savefig(ROOT/'results/weight_sweep.png',dpi=180);plt.close(fig)

def run(stage='all',check=False):
    config,gpu=setup();verify_assets();spec=read_json(ROOT/'configs/experiment.json');self_checks(spec['fusion'])
    write_json(ROOT/'results/environment_check.json',{'GPU':gpu,'environment':environment()})
    if check: print('V6 GPU CHECK PASSED',gpu,flush=True);return
    sources=['configs/experiment.json','configs/baseline.json','src/experiment.py','src/core.py','src/runtime.py','src/data.py','src/evaluation.py','src/image_ops.py','src/fusion.py']
    protocol={'experiment':spec,'config':config,'sources':{n:sha(ROOT/n) for n in sources},'model_sha256':{n:sha(ROOT/f'models/{n}/selected.pt') for n in ['A','B','C']},
              'new_training_steps':0,'test_inferences':0}
    lock=ROOT/'results/protocol.json'
    # The curated source has new portable paths and no V4/V5 migration manifest.
    write_json(lock,protocol)
    records,cutoffs=read_inputs();val=[r for r in records if r['split']=='val'];images=load_images(val);enhanced=[clahe_image(im,spec['clahe']) for im in images]
    for r,im in zip(val,enhanced):
        if not np.array_equal(im,np.asarray(Image.open(ROOT/'Data/clahe/val/images'/(r['id']+'.png')).convert('L'))): raise RuntimeError('Native CLAHE differs')
    threshold=read_json(ROOT/'configs/A_threshold_lock.json')['confidence'];native={};baselines={}
    pixel_records=[{'id':r['id'],'width':r['width'],'height':r['height']} for r in val]
    for n in ['A','B','C']:
        saved=load_result(n)
        if saved is not None:
            baselines[n]=saved;native[n]=read_json(ROOT/f'results/{n}/predictions.json');continue
        print('V6 BASELINE',n,flush=True)
        native[n],timing=infer(YOLO(str(ROOT/f'models/{n}/selected.pt')),pixel_records,images if n=='A' else enhanced,config)
        folder=ROOT/'results'/n;folder.mkdir(exist_ok=True)
        ap=official(ROOT/f'models/{n}/selected.pt',spec['methods'][n]['inference_view'],config,folder)
        baselines[n]=write_result(n,{'recipe':spec['methods'][n],'official':ap,**operating(val,native[n],config,cutoffs,threshold),'timing':timing},native[n])
    candidates={}
    for family in ['D','E']:
        partner=spec['families'][family]['partner']
        for a_pct in spec['A_percent_grid']:
            name=candidate_name(family,a_pct,partner);saved=load_result(name)
            if saved is not None: candidates[name]=saved;continue
            if stage not in ['all',family]: continue
            recipe={'family':family,'anchor':'A','partner':partner,'A_percent':a_pct,'partner_percent':100-a_pct,'partner_weight':(100-a_pct)/100}
            folder=ROOT/'results'/name;folder.mkdir(exist_ok=True)
            print('V6 SWEEP',name,flush=True)
            pred,audit=fuse(native['A'],native[partner],spec['fusion'],recipe['partner_weight'])
            changed=validate_geometry(val,native['A'],pred)
            ap=official(ROOT/'models/A/selected.pt','raw',config,folder,native[partner],spec['fusion'],recipe['partner_weight'])
            row={'recipe':recipe,'official':ap,**operating(val,pred,config,cutoffs,threshold),'changed_boxes':changed}
            row['safe_vs_A']=safe(baselines['A'],row)
            candidates[name]=write_result(name,row,pred);write_json(folder/'fusion_audit.json',audit)
            write_json(ROOT/'results/status.json',{'completed_candidates':len(candidates),'current_candidate':name,'total_candidates':38,'test_inferences':0})
            print('V6 RESULT',name,json.dumps({'AP50':ap[KEYS[0]],'AP75':ap[KEYS[1]],'mAP50_95':ap[KEYS[2]],'counts':row['operating_point'],'safe':row['safe_vs_A']}),flush=True)
    finish(spec,baselines,candidates,protocol)
    if len(candidates)<38: print('V6 PARTIAL',len(candidates),'of38; run remaining family to select',flush=True)

def main(default_stage='all'):
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','baseline','D','E'],default=default_stage);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();os.chdir(ROOT);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');(ROOT/'results').mkdir(exist_ok=True)
    with (ROOT/'results/run.log').open('a',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(Tee(sys.stdout,stream)),contextlib.redirect_stderr(Tee(sys.stderr,stream)):
        run(args.stage,args.check)
