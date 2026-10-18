"""Strict metadata checks and patient-disjoint partitions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

LABELS = {'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
          'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
          'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'}


def read_metadata(root, allow_partial=False):
    root = Path(root)
    csvs = sorted(root.rglob('Data_Entry_2017*.csv'))
    if len(csvs) != 1:
        raise ValueError(f'Expected one Data_Entry_2017*.csv; found {len(csvs)} in {root}')
    df = pd.read_csv(csvs[0], dtype={'Patient ID': str})
    required = ['Image Index', 'Finding Labels', 'Patient ID']
    if not set(required) <= set(df.columns) or df[required].isna().any().any():
        raise ValueError('Missing required metadata columns or values')
    if df['Image Index'].duplicated().any():
        raise ValueError('Duplicate Image Index in metadata')
    tokens = df['Finding Labels'].map(lambda s: set(s.split('|')))
    unknown = set().union(*tokens) - LABELS
    if unknown:
        raise ValueError(f'Unexpected labels: {unknown}; this loader is for NIH ChestX-ray14')
    if any('No Finding' in s and len(s) > 1 for s in tokens):
        raise ValueError('Contradictory No Finding metadata')
    df['target'] = tokens.map(lambda s: int('Pneumonia' in s))
    df['patient_id'] = df['Patient ID'].str.strip()
    if (df['patient_id'] == '').any():
        raise ValueError('Empty patient ID')
    paths = {}
    for p in root.rglob('*'):
        if p.suffix.lower() == '.png':
            if p.name in paths:
                raise ValueError(f'Duplicate image basename: {p.name}')
            paths[p.name] = str(p.resolve())
    df['path'] = df['Image Index'].map(paths)
    missing = int(df['path'].isna().sum())
    if missing and not allow_partial:
        raise FileNotFoundError(f'{missing} metadata images missing. Extract all archives or explicitly enable partial grouped experiment.')
    audit = {'metadata_file': csvs[0].name,
             'metadata_sha256': hashlib.sha256(csvs[0].read_bytes()).hexdigest(),
             'metadata_rows': len(df), 'missing_images': missing,
             'available_images': int(df['path'].notna().sum()),
             'target': 'Pneumonia label present vs absent (not healthy vs diseased)'}
    df = df.dropna(subset=['path']).reset_index(drop=True)
    if df.empty:
        raise ValueError('No extracted PNG images found')
    return df, audit


def _patient_split(frame, fraction, seed):
    patients = frame.groupby('patient_id')['target'].max()
    counts = patients.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError('Too few positive/negative patients for a stratified patient split')
    a, b = train_test_split(patients.index, test_size=fraction, random_state=seed,
                            stratify=patients.values)
    return frame[frame.patient_id.isin(a)].copy(), frame[frame.patient_id.isin(b)].copy()


def make_splits(df, root, mode='official', seed=42, max_patients=None):
    if mode == 'official':
        if max_patients is not None:
            raise ValueError('Official protocol requires max_patients=None')
        root = Path(root)
        lists = {}
        for name in ['train_val_list.txt', 'test_list.txt']:
            found = list(root.rglob(name))
            if len(found) != 1:
                raise FileNotFoundError(f'Expected one {name}; use grouped mode only for a declared subset experiment')
            lists[name] = set(found[0].read_text().splitlines())
        if lists['train_val_list.txt'] & lists['test_list.txt']:
            raise ValueError('Official image lists overlap')
        available = set(df['Image Index'])
        if available != (lists['train_val_list.txt'] | lists['test_list.txt']):
            raise ValueError('Official protocol requires complete image lists and all images')
        dev = df[df['Image Index'].isin(lists['train_val_list.txt'])]
        test = df[df['Image Index'].isin(lists['test_list.txt'])].copy()
        train, val = _patient_split(dev, 0.20, seed)
    elif mode == 'grouped':
        if max_patients is not None:
            patients = np.array(sorted(df.patient_id.unique()))
            if max_patients < 10:
                raise ValueError('Use at least 10 patients (usually thousands are needed)')
            chosen = np.random.default_rng(seed).choice(patients, min(len(patients), max_patients), replace=False)
            df = df[df.patient_id.isin(chosen)].copy()
        dev, test = _patient_split(df, 0.15, seed)
        train, val = _patient_split(dev, 0.15 / 0.85, seed + 1)
    else:
        raise ValueError('mode must be official or grouped')
    splits = {k: v.sort_values('Image Index').reset_index(drop=True)
              for k, v in [('train', train), ('val', val), ('test', test)]}
    for name, frame in splits.items():
        if frame.empty or frame.target.nunique() != 2:
            raise ValueError(f'{name} needs both classes; increase available patients')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(splits[a].patient_id) & set(splits[b].patient_id):
            raise ValueError(f'Patient leakage: {a}/{b}')
        if set(splits[a]['Image Index']) & set(splits[b]['Image Index']):
            raise ValueError(f'Image leakage: {a}/{b}')
    return splits


def save_splits(splits, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    summary = {}
    for name, frame in splits.items():
        manifest = frame.drop(columns=['path'])
        content = manifest.to_csv(index=False).encode('utf-8')
        digest.update(name.encode() + content)
        (out / f'{name}.csv').write_bytes(content)
        summary[name] = dict(images=len(frame), patients=int(frame.patient_id.nunique()),
                             positive_images=int(frame.target.sum()), prevalence=float(frame.target.mean()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return digest.hexdigest(), summary


def transforms(train=False, size=224):
    # Preserve the full field of view with padding; no random crop or left/right flip.
    ops = [T.Lambda(lambda im: _pad_square(im)), T.Resize((size, size))]
    if train:
        ops += [T.RandomAffine(degrees=5, translate=(0.02, 0.02), fill=0)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def _pad_square(im):
    w, h = im.size
    side = max(w, h)
    return T.functional.pad(im, [(side-w)//2, (side-h)//2, (side-w+1)//2, (side-h+1)//2], fill=0)


class ChestDataset(Dataset):
    def __init__(self, frame, train=False, size=224):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms(train, size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as im:
            x = self.transform(im.convert('RGB'))
        return x, torch.tensor([row.target], dtype=torch.float32)


def make_loaders(splits, batch_size=32, seed=42, size=224):
    # workers=0 lets checkpointed Python/Torch RNG reproduce epoch-boundary data order.
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(ChestDataset(frame, name == 'train', size),
                   batch_size=batch_size, shuffle=name == 'train', num_workers=0,
                   pin_memory=torch.cuda.is_available(), generator=generator if name == 'train' else None)
               for name, frame in splits.items()}
    return loaders, generator
