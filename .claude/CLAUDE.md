# Experiment layout

Each experiment is its own folder, and the same name is used across `train/`, `evals/`
and the downloaded HPC outputs. Keep outputs from different experiments separate.

| Experiment | Training code | Eval code | Local eval outputs |
| --- | --- | --- | --- |
| `basic_transformer` | `train/basic_transformer/` | `evals/basic_transformer/` | `evals/data/basic_transformer_runs/` |
| `geo_transformer` | `train/geo_transformer/` | `evals/geo_transformer/` | `evals/data/geo_transformer_runs/` |

Within an experiment, runs differ by loss function (e.g. infonce, discriminative,
crossentropy), which is encoded in the checkpoint/output filenames.

## Files pulled from the HPC

The HPC host is `ll3023@login.cx3.hpc.imperial.ac.uk`, project dir `~/HGCAL`
(capital letters). Job scripts write relative to the dir `qsub` was run from.

- Eval outputs (`eval_*.out`, `*_metrics.json`, `*_by_multiplicity.png`, `*_roc.png`) go in
  the matching experiment folder under `evals/data/`: files named `transformer_*` or
  `eval_transformer_*` / `eval_crossentropy_*` go to `basic_transformer_runs/`; files named
  `geo_transformer_*` or `eval_geo_*` go to `geo_transformer_runs/`.
- Model checkpoints and configs (`*_best.pt`, `*_last.pt`, `*_config.json`) go directly in
  `trained_models/`.
- Only pull what is new: `rsync -avz --ignore-existing` against the destination folder.
