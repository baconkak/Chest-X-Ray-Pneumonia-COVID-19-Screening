"""Patient manifests, error review and equal-budget unfreezing regression tests."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import torch
from test_pipeline import fixture
from src.data import read_metadata,make_splits,save_splits
from src.engine import TrainConfig,train_model,save_json,atomic_save
from src.models import build_model,configure_stage,discriminative_groups
from src.diagnostics import restore_splits,validation_errors,review_sample,render_review,run_backbone_trials


class DiagnosticTests(TestCase):
    def test_stage_contracts(self):
        model=build_model('resnet18',False)
        counts=[]
        for stage in ['finetune','layer34','all']:
            configure_stage(model,'resnet18',stage)
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            self.assertTrue(next(model.layer4.parameters()).requires_grad)
            self.assertEqual(next(model.layer3.parameters()).requires_grad,stage!='finetune')
            self.assertEqual(next(model.conv1.parameters()).requires_grad,stage=='all')
        self.assertTrue(counts[0]<counts[1]<counts[2])
        groups=discriminative_groups(model,1e-5,3e-6)
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertTrue(groups[0]['lr']>groups[1]['lr']>groups[2]['lr']>groups[-1]['lr'])

    def test_frozen_split_error_review_and_progressive_training(self):
        import matplotlib
        matplotlib.use('Agg')
        import src.engine as engine
        with TemporaryDirectory() as temp,patch('torch.cuda.is_available',return_value=False):
            root=Path(temp);fixture(root)
            metadata,_=read_metadata(root);splits=make_splits(metadata,root,mode='grouped')
            source=root/'source';source.mkdir()
            digest,_=save_splits(splits,source/'splits')
            save_json({'split_hash':digest},source/'data_audit.json')
            restored,hash2=restore_splits(source,metadata)
            self.assertEqual(digest,hash2)
            bad=metadata.copy();bad.loc[0,'target']=1-bad.loc[0,'target']
            with self.assertRaisesRegex(ValueError,'labels changed'):restore_splits(source,bad)
            cfg=TrainConfig(epochs=2,head_epochs=1,batch_size=64,image_size=32,pretrained=False,amp=False)
            model,metrics,_=train_model('resnet18',splits,cfg,digest,source/'baseline')
            checkpoint=source/'baseline/resnet18/best.pt'
            save_json(dict(selected_trial='baseline',selected_model='resnet18',
                           thresholds={'baseline':metrics['threshold']}),source/'locked_selection.json')
            errors,path,_=validation_errors(source,splits['val'],root/'review')
            self.assertEqual(path,checkpoint)
            for group,cases in review_sample(errors).groupby('review_group'):
                self.assertFalse(cases.patient_id.duplicated().any())
            images=render_review(errors,checkpoint,root/'review',image_size=32)
            self.assertTrue(images and all(p.is_file() for p in images))
            diagcfg=TrainConfig(epochs=3,head_epochs=0,batch_size=64,image_size=32,
                lr=1e-5,finetune_lr=3e-6,positive_weight=10.,stage_epochs=1,pretrained=False,amp=False)
            table=run_backbone_trials(splits,digest,checkpoint,errors,root/'trials',diagcfg)
            self.assertEqual(len(table),4)
            for trial in ['control_layer4','progressive_layer34','progressive_all']:
                f=pd.read_csv(root/'trials'/trial/'resnet18/history.csv')
                self.assertEqual(f.phase.tolist(),[0,1,2])
                self.assertEqual(f.train_weighted_loss.iloc[0],pd.read_csv(
                    root/'trials/control_layer4/resnet18/history.csv').train_weighted_loss.iloc[0])
            # Resuming at a phase boundary must preserve optimizer moments and RNG.
            testcfg=TrainConfig(**{**diagcfg.__dict__,'warm_start':str(checkpoint),
                                   'unfreeze_schedule':('finetune','layer34','all')})
            def interrupt(state,path):
                atomic_save(state,path)
                if Path(path).name=='last.pt' and state['epoch']==0:raise InterruptedError('disconnect')
            with patch.object(engine,'atomic_save',side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train_model('resnet18',splits,testcfg,digest,root/'resume')
            train_model('resnet18',splits,testcfg,digest,root/'resume',resume=True)
            uninterrupted=torch.load(root/'trials/progressive_all/resnet18/last.pt',weights_only=False)
            resumed=torch.load(root/'resume/resnet18/last.pt',weights_only=False)
            for key,value in uninterrupted['model'].items():
                self.assertTrue(torch.equal(value,resumed['model'][key]),key)
            self.assertFalse((root/'trials/final_test').exists())

    def test_notebook_syntax_and_no_download(self):
        nb=nbformat.read(Path(__file__).resolve().parents[1]/'Chest_Xray_Backbone_Diagnostics.ipynb',4)
        nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type=='code':
                ast.parse(c.source)
                self.assertEqual(c.outputs,[])
                self.assertNotIn('dataset_download(',c.source)
