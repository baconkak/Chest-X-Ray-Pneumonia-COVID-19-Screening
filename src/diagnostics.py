"""Validation-only error review and controlled backbone-unfreezing experiments."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import textwrap
import numpy as np
import pandas as pd
from PIL import Image
import torch
from .data import _pad_square, transforms
from .engine import TrainConfig, train_model, save_json, load_trained
from .explain import grad_cam
from .metrics import binary_metrics, choose_threshold, patient_bootstrap_ap
from .plots import history_plot


def restore_splits(source_run, metadata):
    """Load exact manifests, not another random split; verify against the source hash."""
    root = Path(source_run)
    audit = json.loads((root/'data_audit.json').read_text(encoding='utf-8'))
    digest, splits = hashlib.sha256(), {}
    if metadata['Image Index'].duplicated().any():
        raise ValueError('Duplicate current image IDs')
    current = metadata.set_index('Image Index')
    for name in ['train', 'val', 'test']:
        raw = (root/'splits'/f'{name}.csv').read_bytes()
        digest.update(name.encode()+raw)
        frame = pd.read_csv(root/'splits'/f'{name}.csv', dtype={'patient_id':str, 'Patient ID':str})
        if frame['Image Index'].duplicated().any() or not frame['Image Index'].isin(current.index).all():
            raise ValueError(f'Missing/duplicate images in {name}')
        mapped = current.loc[frame['Image Index']]
        if (not np.array_equal(frame.patient_id.astype(str), mapped.patient_id.astype(str)) or
                not np.array_equal(frame.target, mapped.target)):
            raise ValueError(f'Patient IDs or labels changed in {name}')
        frame['path'] = mapped.path.to_numpy()
        splits[name] = frame
    fingerprint = digest.hexdigest()
    if fingerprint != audit['split_hash']:
        raise ValueError('Saved split manifests differ from the recorded hash')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if (set(splits[a].patient_id)&set(splits[b].patient_id) or
                set(splits[a]['Image Index'])&set(splits[b]['Image Index'])):
            raise ValueError('Patient/image leakage in saved manifests')
    return splits, fingerprint


def validation_errors(source_run, val_frame, out):
    root, out = Path(source_run), Path(out)
    # Development continues only while the existing final holdout has not been opened.
    if any((root/'final_test'/n).exists() for n in ['STARTED.json','test_comparison.csv']):
        raise RuntimeError('Final test was opened; a new independent holdout is required for further tuning')
    selection = json.loads((root/'locked_selection.json').read_text(encoding='utf-8'))
    trial, name = selection['selected_trial'], selection['selected_model']
    model_dir = root/trial/name
    pred = pd.read_csv(model_dir/'val_predictions.csv', dtype={'patient_id':str})
    if (pred['Image Index'].duplicated().any() or
            list(pred['Image Index']) != list(val_frame['Image Index']) or
            not np.array_equal(pred.patient_id.astype(str), val_frame.patient_id.astype(str)) or
            not np.array_equal(pred.target, val_frame.target) or
            not np.isfinite(pred.score).all() or not pred.score.between(0,1).all()):
        raise ValueError('Validation predictions are not aligned with the frozen manifest')
    frame = val_frame.copy()
    frame['score'] = pred.score.to_numpy()
    threshold = selection['thresholds'][trial]
    frame['prediction'] = (frame.score >= threshold).astype(int)
    frame['error'] = np.select([(frame.target==1)&(frame.prediction==0),
                               (frame.target==0)&(frame.prediction==1),
                               (frame.target==1)&(frame.prediction==1)], ['FN','FP','TP'], default='TN')
    out.mkdir(parents=True, exist_ok=True)
    frame.drop(columns='path').to_csv(out/'validation_errors.csv', index=False)
    summary = binary_metrics(frame.target, frame.score, threshold)
    save_json(dict(source_trial=trial, split='validation', metrics=summary,
                   interpretation='Errors are disagreement with NIH labels, not expert adjudication'), out/'summary.json')
    slices=[]
    for column in ['View Position','Patient Gender']:
        if column in frame:
            for value, group in frame.groupby(column, dropna=False):
                slices.append(dict(field=column, group=str(value),
                                   **binary_metrics(group.target, group.score, threshold)))
    pd.DataFrame(slices).to_csv(out/'validation_slices.csv', index=False)
    label_rows=[]
    for label in sorted(set('|'.join(frame['Finding Labels']).split('|'))):
        mask = frame['Finding Labels'].map(lambda s: label in s.split('|'))
        for error in ['FN','FP','TP','TN']:
            label_rows.append(dict(label=label, error=error, images=int((mask&(frame.error==error)).sum())))
    pd.DataFrame(label_rows).to_csv(out/'co_labels.csv', index=False)
    return frame, model_dir/'best.pt', selection


def review_sample(frame, seed=42, per_group=6):
    """Patient-diverse examples; all FN cases exported separately, no relabeling."""
    selected=[]
    for error, ascending in [('FN',True),('FP',False),('TP',False),('TN',True)]:
        group=frame[frame.error==error]
        extremes=group.sort_values('score',ascending=ascending).drop_duplicates('patient_id').head(per_group)
        extremes=extremes.assign(review_group=error+'_extreme')
        selected.append(extremes)
        if error=='FP':
            remaining=group[~group.patient_id.isin(extremes.patient_id)].drop_duplicates('patient_id')
            selected.append(remaining.sample(min(per_group,len(remaining)),random_state=seed).assign(review_group='FP_random'))
    return pd.concat(selected,ignore_index=True)


def render_review(frame, checkpoint, out, image_size=224):
    """Original vs exact input vs positive-logit CAM; outputs remain private in Drive."""
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    sample=review_sample(frame)
    sample.drop(columns='path').to_csv(out/'review_manifest.csv',index=False)
    annotations=sample[['Image Index','patient_id','review_group','target','score']].copy()
    for column in ['technical_issue','cam_observation','label_question_for_expert','notes']:
        annotations[column]=''
    if not (out/'review_annotations.csv').exists():
        annotations.to_csv(out/'review_annotations.csv',index=False)
    # weights_only prevents arbitrary Python objects from being loaded from a checkpoint.
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    from .models import build_model
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(state['name'],False).to(device).eval()
    model.load_state_dict(state['model'])
    images=[]
    reproduced=[]
    for group, cases in sample.groupby('review_group',sort=False):
        fig,axes=plt.subplots(len(cases),3,figsize=(10,3.2*len(cases)),squeeze=False)
        for row,(_,case) in enumerate(cases.iterrows()):
            with Image.open(case.path) as image:
                original=image.convert('RGB').copy()
            x=transforms(size=image_size)(original).unsqueeze(0).to(device)
            with torch.no_grad():
                score=float(model(x).sigmoid().item())
            reproduced.append(dict(image=case['Image Index'],saved_score=float(case.score),
                                   reproduced_score=score,absolute_difference=abs(score-case.score)))
            if not np.isclose(score,case.score,rtol=1e-3,atol=1e-5):
                raise ValueError('Selected checkpoint/input does not reproduce saved review scores')
            cam=grad_cam(model,x)[0]
            rgb=np.clip(x[0].detach().cpu().permute(1,2,0).numpy()*[.229,.224,.225]+[.485,.456,.406],0,1)
            axes[row,0].imshow(original)
            axes[row,1].imshow(rgb)
            axes[row,2].imshow(rgb); axes[row,2].imshow(cam,cmap='jet',alpha=.35,vmin=0,vmax=1)
            for ax,title in zip(axes[row],['Original','Model input','Positive-logit Grad-CAM']):
                ax.set_title(title,fontsize=9); ax.axis('off')
            axes[row,0].set_ylabel(str(case['Image Index']))
            label='\n'.join(textwrap.wrap(str(case['Finding Labels']),38))
            axes[row,0].set_title(f'{case["Image Index"]}\n{label}\nscore={case.score:.5f} label={case.target}',fontsize=8)
        fig.suptitle(f'Validation {group} — descriptive review, not diagnosis',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.98])
        dest=out/f'{group}.png';fig.savefig(dest,dpi=120);plt.close(fig);images.append(dest)
    del model
    pd.DataFrame(reproduced).to_csv(out/'score_reproduction.csv',index=False)
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return images


SCHEDULES = {
    'control_layer4': ('finetune','finetune','finetune'),
    'progressive_layer34': ('finetune','layer34','layer34'),
    'progressive_all': ('finetune','layer34','all'),
}


def run_backbone_trials(splits, split_hash, checkpoint, baseline, out, config=None, resume=False):
    """Equal budgets and same initialization; only unfreezing schedule differs."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=config or TrainConfig(epochs=9,head_epochs=0,batch_size=64,image_size=224,
        lr=1e-5,finetune_lr=3e-6,positive_weight=10.,num_workers=4,patience=3,
        progress=True,pretrained=False,stage_epochs=3,warm_start=str(checkpoint))
    cfg=replace(cfg,warm_start=str(checkpoint))
    warm=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if warm['split_hash']!=split_hash or warm['name']!='resnet18':
        raise ValueError('Diagnostic design expects the selected ResNet18 from these same splits')
    del warm
    plan=dict(config=asdict(cfg),schedules=SCHEDULES,split_hash=split_hash,
              source_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              primary_metric='validation Pneumonia AP; precision at recall>=0.90 secondary',
              final_test='not accessed',note='All arms start from identical selected v3 weights; matched phase boundaries')
    # Normalize tuples so round-tripped JSON comparisons remain exact.
    plan=json.loads(json.dumps(plan))
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise ValueError('Diagnostic plan changed; use a new run directory')
    save_json(plan,out/'plan.json')
    rows=[dict(trial='source_v3_best',**binary_metrics(baseline.target,baseline.score,
                         choose_threshold(baseline.target,baseline.score,cfg.target_recall)))]
    paired=baseline[['Image Index','patient_id','target']].copy();paired['source_score']=baseline.score.to_numpy()
    for trial,schedule in SCHEDULES.items():
        trial_cfg=replace(cfg,unfreeze_schedule=schedule)
        model,metrics,history=train_model('resnet18',splits,trial_cfg,split_hash,out/trial,
            resume=resume and (out/trial/'resnet18/last.pt').exists())
        history_plot(history,out/trial/'resnet18/learning_curves.png')
        pred=pd.read_csv(out/trial/'resnet18/val_predictions.csv')
        if list(pred['Image Index'])!=list(paired['Image Index']):
            raise ValueError('Trial predictions reordered validation images')
        paired[trial]=pred.score.to_numpy()
        ci_frame=paired[['patient_id','target','source_score',trial]]
        ci=patient_bootstrap_ap(ci_frame,'source_score',trial,repeats=200,seed=42)
        rows.append(dict(trial=trial,**metrics,delta_ap=metrics['ap']-rows[0]['ap'],
                         paired_delta_low=ci['low'],paired_delta_high=ci['high']))
        pd.DataFrame(rows).to_csv(out/'validation_comparison.csv',index=False)
        paired.to_csv(out/'paired_validation_predictions.csv',index=False)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return pd.DataFrame(rows)
