"""Declared validation experiments; test data never selects a model or threshold."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .engine import train_model, save_json
from .metrics import binary_metrics, choose_threshold
from .plots import history_plot


def run_experiments(splits, config, experiments, seeds, split_hash, out, resume=False):
    out = Path(out)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Declare unique training seeds; the first seed is used for deployment')
    if not experiments or len({e['id'] for e in experiments}) != len(experiments):
        raise ValueError('Declare experiments with unique IDs')
    if (out/'final_test'/'STARTED.json').exists():
        raise RuntimeError('Test was opened for this run. Create a new development run with a new holdout.')
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(base_config=asdict(config), experiments=experiments, training_seeds=seeds,
                split_hash=split_hash, selection='mean validation AP across declared seeds; ties declared order',
                deployment_seed=seeds[0])
    plan_path = out/'experiment_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Experiment plan changed; use a new RUN_NAME')
    save_json(plan, plan_path)
    rows, operating = [], []
    for experiment in experiments:
        for seed in seeds:
            trial = f'{experiment["id"]}_seed{seed}'
            cfg = replace(config, seed=seed, positive_weight=experiment['positive_weight'])
            model_name = experiment['model']
            trial_root = out/trial
            model_dir = trial_root/model_name
            model, validation, history = train_model(model_name, splits, cfg, split_hash, trial_root,
                resume=resume and (model_dir/'last.pt').exists())
            history_plot(history, model_dir/'learning_curves.png')
            rows.append(dict(experiment=experiment['id'], trial=trial, seed=seed,
                             model=model_name, **validation))
            predictions = pd.read_csv(model_dir/'val_predictions.csv')
            y, scores = predictions.target.to_numpy(), predictions.score.to_numpy()
            for recall in [.5, .8, .9]:
                threshold = choose_threshold(y, scores, recall)
                operating.append(dict(trial=trial, rule=f'validation_recall>={recall}',
                                      **binary_metrics(y, scores, threshold)))
            operating.append(dict(trial=trial, rule='fixed_threshold_0.5',
                                  **binary_metrics(y, scores, .5)))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out/'validation_comparison.csv', index=False)
            pd.DataFrame(operating).to_csv(out/'validation_operating_points.csv', index=False)
    table = pd.DataFrame(rows)
    summary = table.groupby('experiment', sort=False).agg(
        mean_ap=('ap', 'mean'), std_ap=('ap', 'std'), seeds=('seed', 'count'),
        mean_auroc=('auroc', 'mean')).reset_index()
    summary.to_csv(out/'validation_seed_summary.csv', index=False)
    return table, summary


def lock_selection(table, summary, seeds, split_hash, out):
    """Choose a family by mean AP, never choose the most favorable random seed."""
    if not np.isfinite(summary.mean_ap).all():
        raise ValueError('All experiments require finite validation AP')
    winner = summary.sort_values('mean_ap', ascending=False, kind='stable').iloc[0].experiment
    selected = table[(table.experiment == winner) & (table.seed == seeds[0])].iloc[0]
    selection = dict(selected_experiment=winner, selected_trial=selected.trial,
                     selected_model=selected.model, deployment_seed=seeds[0], split_hash=split_hash,
                     primary_metric='mean validation AP across declared training seeds',
                     thresholds={row.trial: float(row.threshold) for row in table.itertuples()},
                     trials=table[['trial', 'model', 'seed', 'experiment']].to_dict('records'))
    save_json(selection, Path(out)/'locked_selection.json')
    return selection
