"""Synthetic structural tests; these are not clinical performance experiments."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, skipUnless
from unittest.mock import patch
import ast
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import read_metadata, make_splits, save_splits, ChestDataset
from src.models import build_model
from src.metrics import choose_threshold, binary_metrics, patient_bootstrap_ap
from src.engine import TrainConfig, train_model, predict, atomic_save
from src.explain import grad_cam_resnet

torch.set_num_threads(2)


def fixture(root):
    rows = []
    image_root = root/'images_001'/'images'; image_root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for patient in range(60):
        for followup in range(2):
            name = f'{patient:08d}_{followup:03d}.png'
            label = 'Pneumonia|Effusion' if patient % 3 == 0 else ('No Finding' if patient % 2 else 'Effusion')
            Image.fromarray(rng.integers(0,256,(36,28),dtype=np.uint8)).save(image_root/name)
            rows.append({'Image Index':name, 'Finding Labels':label, 'Patient ID':patient, 'View Position':'PA'})
    pd.DataFrame(rows).to_csv(root/'Data_Entry_2017.csv', index=False)


class PipelineTests(TestCase):
    @skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
    def test_cuda_amp_training(self):
        from src.engine import run_epoch, seed_all
        from src.models import configure_stage
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        seed_all(42)
        loader=DataLoader(TensorDataset(torch.rand(16,3,64,64),torch.tensor([[0.],[1.]]*8)),batch_size=2)
        for name in ['mlp','small_cnn','resnet18']:
            model=build_model(name,pretrained=False).cuda()
            configure_stage(model,name,'finetune')
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            scaler=torch.amp.GradScaler('cuda')
            loss,*_=run_epoch(model,loader,nn.BCEWithLogitsLoss(),torch.device('cuda'),name,optimizer,scaler,True)
            self.assertTrue(np.isfinite(loss))
            del model, optimizer, scaler
            torch.cuda.empty_cache()

    def test_patient_splits_and_metadata(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            df, audit = read_metadata(root)
            self.assertEqual(df.target.sum(), 40)
            self.assertEqual(audit['missing_images'], 0)
            splits = make_splits(df, root, mode='grouped')
            self.assertEqual(sum(map(len,splits.values())),120)
            h1,_ = save_splits(splits,root/'out')
            h2,_ = save_splits(make_splits(df,root,mode='grouped'),root/'out2')
            self.assertEqual(h1,h2)
            self.assertFalse(set(splits['train'].patient_id)&set(splits['test'].patient_id))
            # Official partition is checked separately, with a real patient boundary.
            test_ids = set(df.patient_id.unique()[-12:])
            for name, frame in [('test_list.txt',df[df.patient_id.isin(test_ids)]),
                                ('train_val_list.txt',df[~df.patient_id.isin(test_ids)])]:
                (root/name).write_text('\n'.join(frame['Image Index']))
            official = make_splits(df,root,mode='official')
            self.assertEqual(len(official['test']),24)
            # Missing files must never silently remove cases in the full protocol.
            Path(df.iloc[0].path).unlink()
            with self.assertRaises(FileNotFoundError): read_metadata(root)
            partial, audit = read_metadata(root,allow_partial=True)
            self.assertEqual(audit['missing_images'],1)
            with self.assertRaises(ValueError): make_splits(partial,root,mode='official')

    def test_threshold_metrics_and_group_bootstrap(self):
        y=np.array([0,0,1,1]); s=np.array([.1,.2,.8,.9])
        threshold=choose_threshold(y,s,1.)
        self.assertAlmostEqual(threshold,.8)
        metrics=binary_metrics(y,s,threshold)
        self.assertEqual((metrics['tn'],metrics['fp'],metrics['fn'],metrics['tp']),(2,0,0,2))
        self.assertEqual(metrics['ap'],1.)
        degenerate=binary_metrics([0,0],[.1,.2],.5)
        self.assertIsNone(degenerate['recall']); self.assertIsNone(degenerate['ap'])
        frame=pd.DataFrame({'patient_id':[1,2,3,4], 'target':y,'a':s,'b':s})
        ci=patient_bootstrap_ap(frame,'a','b',repeats=40)
        self.assertEqual(ci['low'],0.); self.assertEqual(ci['high'],0.)

    def test_models_and_gradcam(self):
        for name in ['mlp','small_cnn','resnet18']:
            model=build_model(name,pretrained=False).eval()
            x=torch.rand(2,3,32,32)
            self.assertEqual(tuple(model(x).shape),(2,1))
            if name=='resnet18':
                cam=grad_cam_resnet(model,x)
                self.assertEqual(cam.shape,(2,32,32)); self.assertTrue(np.isfinite(cam).all())

    def test_training_resume_and_frozen_validation(self):
        from torch.utils.data import DataLoader
        import src.engine as engine
        with TemporaryDirectory() as temp, patch('torch.cuda.is_available', return_value=False):
            root=Path(temp); fixture(root)
            df,_=read_metadata(root); splits=make_splits(df,root,mode='grouped')
            fingerprint,_=save_splits(splits,root/'splits')
            cfg=TrainConfig(epochs=3, batch_size=32,image_size=32,pretrained=False,amp=False,patience=10)
            model, val, _=train_model('small_cnn',splits,cfg,fingerprint,root/'continuous')
            original = {k:v.clone() for k,v in model.state_dict().items()}
            loader=DataLoader(ChestDataset(splits['val'],size=32),batch_size=16)
            predict(model,loader,torch.device('cpu'))
            self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.state_dict().items()))
            def interrupt(state,path):
                atomic_save(state,path)
                if Path(path).name=='last.pt' and state['epoch']==0:
                    raise InterruptedError('simulated runtime disconnect after epoch checkpoint')
            with patch.object(engine,'atomic_save',side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train_model('small_cnn',splits,cfg,fingerprint,root/'resumed')
            resumed, val2, _=train_model('small_cnn',splits,cfg,fingerprint,root/'resumed',resume=True)
            self.assertEqual(val,val2)
            for key,v in model.state_dict().items():
                self.assertTrue(torch.equal(v,resumed.state_dict()[key]), key)
            for dirname in ['continuous','resumed']:
                state=torch.load(root/dirname/'small_cnn'/'last.pt',weights_only=False,map_location='cpu')
                self.assertEqual(state['epoch'],2)
            wrong=TrainConfig(epochs=4,pretrained=False)
            with self.assertRaises(ValueError):
                train_model('small_cnn',splits,wrong,fingerprint,root/'resumed',resume=True)

    def test_notebook_has_valid_python_and_no_fabricated_outputs(self):
        import nbformat
        notebook=nbformat.read(Path(__file__).resolve().parents[1]/'Chest_Xray_Pneumonia_Colab.ipynb',as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs,[])

    def test_notebook_workflow_with_synthetic_data(self):
        """Execute actual audit/train/select/test/inference notebook cells offline."""
        import nbformat
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        notebook=nbformat.read(Path(__file__).resolve().parents[1]/'Chest_Xray_Pneumonia_Colab.ipynb',4)
        with TemporaryDirectory() as temp, patch('torch.cuda.is_available',return_value=False), patch.object(plt,'show',side_effect=lambda:plt.close('all')):
            root=Path(temp); fixture(root)
            cfg=TrainConfig(epochs=3,head_epochs=1,batch_size=32,image_size=32,
                            pretrained=False,amp=False,patience=10)
            out=root/'results'; out.mkdir()
            context=dict(Path=Path,torch=torch,pd=pd,np=np,CFG=cfg,OUT=out,DATA_ROOT=root,
                         ALLOW_PARTIAL=False,SPLIT_MODE='grouped',MAX_PATIENTS=None,
                         PROFILE='test',CODE_SHA='synthetic-test',KAGGLE_HANDLE='synthetic-fixture',
                         RESUME=False,MODELS=['mlp','small_cnn','resnet18'],
                         display=lambda value:None,save_json=__import__('src.engine',fromlist=['save_json']).save_json)
            for i in [8,10,12,14,16,18]:
                source=notebook.cells[i].source
                if i==16:
                    source=source.replace('RUN_FINAL_TEST = False','RUN_FINAL_TEST = True')
                if i==18:
                    source=source.replace('IMAGE_PATH = ""',f'IMAGE_PATH = {str(next((root/"images_001").rglob("*.png")))!r}')
                exec(compile(source,f'notebook_cell_{i}','exec'),context)
            results=pd.read_csv(out/'final_test'/'test_comparison.csv')
            self.assertEqual(len(results),4)
            self.assertTrue((out/'locked_selection.json').exists())
            self.assertTrue((out/'final_test'/'resnet18_evaluation.png').exists())
            self.assertTrue((out/'final_test'/'paired_ap_delta.json').exists())
            self.assertEqual(context['cam'].shape,(32,32))


if __name__=='__main__':
    main(verbosity=2)
