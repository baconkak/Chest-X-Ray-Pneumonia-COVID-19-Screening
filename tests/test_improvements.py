"""Regression checks for larger models, cached inputs and multiworker resume."""
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
from PIL import Image
from test_pipeline import fixture
from src.data import read_metadata, make_splits, save_splits, ChestDataset, prepare_image_cache
from src.engine import TrainConfig, train_model, atomic_save, run_epoch, seed_all
from src.models import build_model, configure_stage, training_mode
from src.explain import grad_cam


class ImprovedTests(TestCase):
    def test_larger_models_stages_gradcam_and_cuda(self):
        reference = sum(p.numel() for p in build_model('resnet18', False).parameters())
        for name in ['resnet50', 'densenet121']:
            seed_all(42)
            model = build_model(name, False)
            if name == 'resnet50':
                self.assertGreater(sum(p.numel() for p in model.parameters()), reference)
            configure_stage(model, name, 'head')
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                             2049 if name == 'resnet50' else 1025)
            configure_stage(model, name, 'finetune')
            training_mode(model, name)
            self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))
            self.assertFalse(next(model.parameters()).requires_grad)
            x = torch.rand(2, 3, 64, 64)
            model.eval()
            self.assertEqual(model(x).shape, (2, 1))
            cam = grad_cam(model, x)
            self.assertEqual(cam.shape, (2, 64, 64))
            self.assertTrue(np.isfinite(cam).all())
            if torch.cuda.is_available():
                model.cuda()
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    x, torch.tensor([[0.], [1.]])), batch_size=2)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
                loss, *_ = run_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), torch.device('cuda'),
                    name, optimizer, torch.amp.GradScaler('cuda', init_scale=1024), True)
                self.assertTrue(np.isfinite(loss))
                del optimizer
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def test_cache_preserves_pixels_and_rebuilds_corruption(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            frame, _ = read_metadata(root)
            frame = frame.iloc[:5]
            cached = prepare_image_cache(frame, root/'cache', size=32, workers=2)
            old = ChestDataset(frame, size=32)
            new = ChestDataset(cached, size=32)
            for i in range(len(old)):
                self.assertTrue(torch.equal(old[i][0], new[i][0]))
            path = Path(cached.path.iloc[0]); before = path.stat().st_mtime_ns
            again = prepare_image_cache(frame, root/'cache', size=32, workers=2)
            self.assertEqual(path.stat().st_mtime_ns, before)
            path.write_bytes(b'interrupted cache')
            repaired = prepare_image_cache(frame, root/'cache', size=32, workers=2)
            self.assertTrue(torch.equal(old[0][0], ChestDataset(repaired, size=32)[0][0]))
            source = Path(frame.path.iloc[0])
            with Image.open(source) as im:
                Image.fromarray(np.zeros_like(np.asarray(im))).save(source)
            changed = prepare_image_cache(frame, root/'cache', size=32, workers=2)
            self.assertNotEqual(changed.path.iloc[0], again.path.iloc[0])
            self.assertEqual(frame['Image Index'].tolist(), changed['Image Index'].tolist())

    def test_multiworker_resume_and_explicit_weight(self):
        import src.engine as engine
        with TemporaryDirectory() as temp, patch('torch.cuda.is_available', return_value=False):
            root = Path(temp); fixture(root)
            frame, _ = read_metadata(root); splits = make_splits(frame, root, mode='grouped')
            digest, _ = save_splits(splits, root/'split')
            cfg = TrainConfig(epochs=2, head_epochs=1, batch_size=64, image_size=32,
                              pretrained=False, amp=False, num_workers=2, positive_weight=10., patience=5)
            continuous, v1, _ = train_model('small_cnn', splits, cfg, digest, root/'a')
            def interrupt(state, path):
                atomic_save(state, path)
                if Path(path).name == 'last.pt' and state['epoch'] == 0:
                    raise InterruptedError('test interruption')
            with patch.object(engine, 'atomic_save', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train_model('small_cnn', splits, cfg, digest, root/'b')
            resumed, v2, _ = train_model('small_cnn', splits, cfg, digest, root/'b', resume=True)
            self.assertEqual(v1, v2)
            for key, value in continuous.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            self.assertEqual(json.loads((root/'a/small_cnn/run.json').read_text())['pos_weight'], 10.)

    def test_improved_notebook_workflow(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.engine import save_json
        from src.experiments import run_experiments
        notebook = nbformat.read(Path(__file__).resolve().parents[1]/'Chest_Xray_Improved_Colab.ipynb', 4)
        nbformat.validate(notebook)
        for c in notebook.cells:
            if c.cell_type == 'code':
                ast.parse(c.source)
                self.assertEqual(c.outputs, [])
        with TemporaryDirectory() as temp, patch('torch.cuda.is_available', return_value=False), \
                patch.object(plt, 'show', side_effect=lambda: plt.close('all')):
            root = Path(temp); fixture(root)
            cfg = TrainConfig(epochs=2, head_epochs=1, image_size=32, batch_size=64,
                              pretrained=False, amp=False, positive_weight=10.)
            out = root/'out'; out.mkdir()
            experiments = [dict(id='resnet18_pw10', model='resnet18', positive_weight=10.),
                           dict(id='resnet50_pw1', model='resnet50', positive_weight=1.),
                           dict(id='densenet121_pw10', model='densenet121', positive_weight=10.)]
            context = dict(Path=Path, torch=torch, pd=pd, np=np, CFG=cfg, OUT=out, DATA_ROOT=root,
                ALLOW_PARTIAL=False, SPLIT_MODE='grouped', MAX_PATIENTS=None, SPLIT_SEED=42,
                PROFILE='quick', CODE_SHA='synthetic-test', KAGGLE_HANDLE='synthetic-fixture',
                RESUME=False, EXPERIMENTS=experiments, TRAIN_SEEDS=[42], display=lambda value: None,
                save_json=save_json)
            for i in [8, 10, 12, 14]:
                source = notebook.cells[i].source.replace('CACHE_IMAGES = True', 'CACHE_IMAGES = False')
                exec(compile(source, f'improved_cell_{i}', 'exec'), context)
            self.assertFalse((out/'final_test').exists())
            self.assertTrue((out/'validation_operating_points.csv').exists())
            self.assertTrue((out/'resnet50_pw1_seed42/resnet50/best.pt').exists())
            changed = experiments + [dict(id='new', model='small_cnn', positive_weight=1.)]
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                run_experiments(context['splits'], cfg, changed, [42], context['split_hash'], out, True)
            for i in [16, 18]:
                source = notebook.cells[i].source.replace('RUN_FINAL_TEST = False', 'RUN_FINAL_TEST = True')
                image_path = str(next((root/'images_001').rglob('*.png')))
                source = source.replace('IMAGE_PATH = ""', f'IMAGE_PATH = {image_path!r}')
                exec(compile(source, f'improved_cell_{i}', 'exec'), context)
            self.assertEqual(len(pd.read_csv(out/'final_test/test_comparison.csv')), 4)
            with self.assertRaisesRegex(RuntimeError, 'Test was opened'):
                run_experiments(context['splits'], cfg, experiments, [42], context['split_hash'], out, True)
