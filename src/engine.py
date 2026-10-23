"""Two-stage training with validation selection and epoch-boundary resumption."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import platform
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from .models import build_model, configure_stage, training_mode, TRANSFER_MODELS, discriminative_groups
from .data import make_loaders
from .metrics import choose_threshold, binary_metrics


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 8
    head_epochs: int = 2
    batch_size: int = 32
    image_size: int = 224
    lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    min_delta: float = 1e-4
    weighted_loss: bool = True
    pretrained: bool = True
    target_recall: float = .90
    amp: bool = True
    positive_weight: float | None = None  # None: legacy N_negative / N_positive.
    num_workers: int = 0
    amp_init_scale: float = 1024.
    progress: bool = False
    warm_start: str | None = None
    unfreeze_schedule: tuple = ()
    stage_epochs: int = 3
    backbone_lr_decay: float = .3


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    import torchvision
    import sklearn
    return dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def atomic_save(state, path):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, scores = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        if logits.shape != y.shape or not torch.isfinite(logits).all():
            raise RuntimeError('Invalid model output')
        labels.extend(y.numpy().ravel())
        scores.extend(logits.sigmoid().cpu().numpy().ravel())
    if not labels:
        raise ValueError('Empty evaluation loader')
    return np.asarray(labels), np.asarray(scores)


def run_epoch(model, loader, criterion, device, name, optimizer=None, scaler=None, amp=False,
              progress=False):
    train = optimizer is not None
    training_mode(model, name) if train else model.eval()
    total = torch.zeros((), device=device)
    grad_norm = torch.zeros((), device=device)
    n, amp_skips, updates = 0, 0, 0
    labels, scores = [], []
    # Text progress avoids notebook-widget/multiprocessing cleanup interactions.
    from tqdm import tqdm
    batches = tqdm(loader, desc=f'{name} {"train" if train else "val"}', disable=not progress)
    with torch.set_grad_enabled(train):
        for x, y in batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                if logits.shape != y.shape:
                    raise ValueError(f'Logits {logits.shape} != target {y.shape}')
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stop and inspect inputs / learning rate')
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               max_norm=5., error_if_nonfinite=not amp)
                if not torch.isfinite(norm):
                    # GradScaler recorded overflow during unscale_: skip the update and
                    # reduce scale. This is expected AMP scale adaptation, not NaN repair.
                    if not amp:
                        raise RuntimeError('Non-finite gradient in full precision')
                    amp_skips += 1
                    scaler.step(optimizer); scaler.update()
                    warnings.warn('AMP gradient overflow: update skipped and loss scale reduced')
                    if amp_skips > 10:
                        raise RuntimeError('Repeated AMP overflow: inspect data/LR or set amp=False in a new run')
                else:
                    grad_norm += norm.detach(); updates += 1
                    scaler.step(optimizer); scaler.update()
            total += loss.detach().float() * len(y); n += len(y)
            # Training predictions are unused; avoid copying every training batch to CPU.
            if not train:
                labels.append(y.detach().flatten())
                scores.append(logits.detach().float().sigmoid().flatten())
    if train and not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise RuntimeError('Non-finite parameter after optimizer update')
    if n == 0:
        raise ValueError('Empty loader')
    if train and updates == 0:
        raise RuntimeError('No optimizer updates succeeded; disable AMP in a new run or inspect data')
    labels = torch.cat(labels).cpu().numpy() if labels else np.empty(0)
    scores = torch.cat(scores).cpu().numpy() if scores else np.empty(0)
    return float(total)/n, labels, scores, float(grad_norm)/max(1, updates), amp_skips


def train_model(name, splits, config, split_hash, out_dir, resume=False):
    cfg = asdict(config)
    if config.epochs < 1 or config.patience < 1:
        raise ValueError('epochs and patience must be positive')
    diagnostic = bool(config.unfreeze_schedule)
    if diagnostic:
        if (not name.startswith('resnet') or not config.warm_start or config.stage_epochs < 1
                or config.epochs != len(config.unfreeze_schedule)*config.stage_epochs
                or any(s not in {'finetune', 'layer34', 'all'} for s in config.unfreeze_schedule)
                or not 0 < config.backbone_lr_decay <= 1):
            raise ValueError('Diagnostic schedule requires a ResNet warm start, valid stages and equal phase budgets')
    if not diagnostic and name in TRANSFER_MODELS and not 1 <= config.head_epochs < config.epochs:
        raise ValueError('Transfer learning needs 1 <= head_epochs < epochs')
    if config.positive_weight is not None and (not config.weighted_loss or
            not np.isfinite(config.positive_weight) or config.positive_weight <= 0):
        raise ValueError('positive_weight must be finite, positive and used with weighted_loss=True')
    if not np.isfinite(config.amp_init_scale) or config.amp_init_scale <= 0:
        raise ValueError('amp_init_scale must be finite and positive')
    out = Path(out_dir) / name; out.mkdir(parents=True, exist_ok=True)
    warm_hash = hashlib.sha256(Path(config.warm_start).read_bytes()).hexdigest() if config.warm_start else None
    signature = hashlib.sha256(json.dumps(dict(config=cfg, split_hash=split_hash, model=name,
                                              warm_start_sha256=warm_hash),
                                           sort_keys=True).encode()).hexdigest()
    last_path, best_path = out/'last.pt', out/'best.pt'
    if not resume and last_path.exists():
        raise FileExistsError(f'{out} already has a run; use resume=True or a new RUN_NAME')
    if resume and not last_path.exists():
        raise FileNotFoundError(f'{last_path}: resume requires an existing run')
    seed_all(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = config.amp and device.type == 'cuda'
    loaders, generator = make_loaders(splits, config.batch_size, config.seed, config.image_size,
                                      config.num_workers)
    model = build_model(name, config.pretrained and not resume and not config.warm_start).to(device)
    if config.warm_start and not resume:
        warm = torch.load(config.warm_start, map_location='cpu', weights_only=True)
        if warm['name'] != name or warm['split_hash'] != split_hash:
            raise ValueError('Warm-start model or patient split differs from this experiment')
        model.load_state_dict(warm['model'])
    positives = int(splits['train'].target.sum())
    negatives = len(splits['train']) - positives
    if positives == 0 or negatives == 0:
        raise ValueError('Training data requires both classes')
    pos_weight = negatives/positives if config.weighted_loss else 1.
    if config.positive_weight is not None:
        pos_weight = config.positive_weight
    train_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    val_loss = nn.BCEWithLogitsLoss()  # Comparable unweighted validation data loss
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=config.amp_init_scale)
    optimizer = scheduler = None
    state, history, start, best, waiting, previous_stage = None, [], 0, -float('inf'), 0, None
    previous_phase = None
    if diagnostic and not resume:
        initial_y, initial_scores = predict(model, loaders['val'], device)
        best = binary_metrics(initial_y, initial_scores, .5)['ap']
        atomic_save(dict(model=model.state_dict(), config=cfg, name=name, epoch=-1,
                         validation_ap=best, split_hash=split_hash), best_path)
    if resume:
        # Only load checkpoints produced by this trusted project.
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('Checkpoint config or split mismatch; start a new RUN_NAME')
        if not best_path.exists():
            raise FileNotFoundError('best.pt missing; restore both checkpoint files')
        model.load_state_dict(state['model'])
        start, history, best, waiting = state['epoch']+1, state['history'], state['best_ap'], state['waiting']
        previous_stage = state['stage']
        previous_phase = state.get('phase')
    save_json(dict(config=cfg, environment=environment(), model=name, split_hash=split_hash,
                   pos_weight=pos_weight, signature=signature,
                   warm_start_sha256=warm_hash,
                   parameters=sum(p.numel() for p in model.parameters())), out/'run.json')
    stopped = bool(state and state.get('stopped'))
    for epoch in range(start, config.epochs):
        if stopped:
            break
        phase = epoch//config.stage_epochs if diagnostic else None
        stage = (config.unfreeze_schedule[phase] if diagnostic else
                 ('head' if name in TRANSFER_MODELS and epoch < config.head_epochs else 'finetune'))
        if diagnostic:
            configure_stage(model, name, stage)
            if optimizer is None:
                optimizer = torch.optim.AdamW(discriminative_groups(model, config.lr,
                    config.finetune_lr, config.backbone_lr_decay), weight_decay=config.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=.5)
                if state:
                    optimizer.load_state_dict(state['optimizer'])
                    scheduler.load_state_dict(state['scheduler'])
                    scaler.load_state_dict(state['scaler'])
            if phase != previous_phase:
                # Same declared boundaries for every arm; retain Adam moments across unfreezing.
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr']
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=.5)
                waiting = 0
        elif optimizer is None or stage != previous_stage:
            configure_stage(model, name, stage)
            lr = config.finetune_lr if name in TRANSFER_MODELS and stage == 'finetune' else config.lr
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=lr, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=.5)
            if state and stage == state['stage']:
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                scaler.load_state_dict(state['scaler'])
            elif previous_stage is not None:
                waiting = 0  # Give fine-tuning its declared patience budget.
        if state:
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if device.type == 'cuda' and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            generator.set_state(state['loader_rng']); state = None
        before = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        lr_used = optimizer.param_groups[0]['lr']
        tr_loss, _, _, grad, amp_skips = run_epoch(model, loaders['train'], train_loss, device, name,
                                       optimizer, scaler, amp, progress=config.progress)
        train_seconds = time.perf_counter()-before
        va_loss, y, scores, _, _ = run_epoch(model, loaders['val'], val_loss, device, name,
                                             progress=config.progress)
        metrics = binary_metrics(y, scores, .5)
        ap = metrics['ap']
        scheduler.step(ap)
        improved = ap > best + config.min_delta
        if improved:
            best, waiting = ap, 0
            atomic_save(dict(model=model.state_dict(), config=cfg, name=name,
                             epoch=epoch, validation_ap=ap, split_hash=split_hash), best_path)
        else:
            waiting += 1
        # Do not early-stop before the scheduled fine-tuning phase starts.
        can_stop = ((phase == len(config.unfreeze_schedule)-1) if diagnostic else
                    (name not in TRANSFER_MODELS or stage == 'finetune'))
        stopped = can_stop and waiting >= config.patience
        history.append(dict(epoch=epoch+1, stage=stage, train_weighted_loss=tr_loss,
                            val_bce=va_loss, val_ap=ap, val_auroc=metrics['auroc'],
                            grad_norm=grad, amp_skipped_steps=amp_skips, lr=lr_used, seconds=time.perf_counter()-before,
                            train_seconds=train_seconds, train_images_per_second=len(splits['train'])/train_seconds,
                            peak_gpu_allocated_mib=(torch.cuda.max_memory_allocated(device)/2**20
                                                    if device.type == 'cuda' else 0.),
                            phase=phase, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            lr_groups=json.dumps({g.get('group_name', 'all'):g['lr'] for g in optimizer.param_groups}),
                            best=improved))
        pd.DataFrame(history).to_csv(out/'history.csv', index=False)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                         epoch=epoch, stage=stage, best_ap=best, waiting=waiting,
                         phase=phase,
                         stopped=stopped, history=history, signature=signature,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None), last_path)
        print(f'{name} epoch={epoch+1} stage={stage} train={tr_loss:.4f} val_BCE={va_loss:.4f} '
              f'val_AP={ap:.4f} train={len(splits["train"])/train_seconds:.1f} img/s '
              f'peak_GPU={history[-1]["peak_gpu_allocated_mib"]:.0f} MiB', flush=True)
        previous_stage = stage
        previous_phase = phase
    best_state = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model']); model.to(device).eval()
    y, scores = predict(model, loaders['val'], device)
    threshold = choose_threshold(y, scores, config.target_recall)
    validation = binary_metrics(y, scores, threshold)
    save_json(dict(validation=validation, selected_epoch=best_state['epoch']+1,
                   threshold_rule=f'max precision subject to validation recall >= {config.target_recall}; ties largest threshold'),
              out/'validation.json')
    result = splits['val'][['Image Index', 'patient_id', 'target']].copy()
    result['score'] = scores; result.to_csv(out/'val_predictions.csv', index=False)
    return model, validation, pd.DataFrame(history)


def load_trained(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(state['name'], pretrained=False)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state
