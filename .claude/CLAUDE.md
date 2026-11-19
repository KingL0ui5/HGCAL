# Experiment layout

Each experiment is its own folder, and the same name is used across `train/`, `evals/`
and the downloaded HPC outputs. Keep outputs from different experiments separate.

| Experiment | Training code | Eval code | Local eval outputs |
| --- | --- | --- | --- |
| `basic_transformer` | `train/basic_transformer/` | `evals/basic_transformer/` | `evals/data/basic_transformer_runs/` |
| `geo_transformer` | `train/geo_transformer/` | `evals/geo_transformer/` | `evals/data/geo_transformer_runs/` |
| `spatial_transformer` | `train/spatial_transformer/` | `evals/spatial_transformer/` | `evals/data/spatial_transformer_runs/` |

- `basic_transformer`: `models/classifier_transformer.py`, trained as a 5-slot classifier
  (crossentropy). Its 128-dim encoder output is clustered at eval as a baseline.
- `spatial_transformer`: `models/spatial_transformer.py`, an encoder with a linear projection to
  a 3-dim latent space (`--out-dim`), trained directly with infonce or discriminative loss. No
  classifier head.
- `geo_transformer`: geometry-biased attention model with a 3-dim output.

Within an experiment, runs differ by loss function (e.g. infonce, discriminative,
crossentropy), which is encoded in the checkpoint/output filenames.

## Files pulled from the HPC

The HPC host is `ll3023@login.cx3.hpc.imperial.ac.uk`, project dir `~/HGCAL`
(capital letters). Job scripts write relative to the dir `qsub` was run from.

- Eval outputs (`eval_*.out`, `*_metrics.json`, `*_by_multiplicity.png`, `*_roc.png`) go in
  the matching experiment folder under `evals/data/`: files named `transformer_*` or
  `eval_transformer_*` / `eval_crossentropy_*` go to `basic_transformer_runs/`; files named
  `geo_transformer_*` or `eval_geo_*` go to `geo_transformer_runs/`; files named
  `spatial_transformer_*` or `eval_spatial_*` go to `spatial_transformer_runs/`.
- Model checkpoints and configs (`*_best.pt`, `*_last.pt`, `*_config.json`) go in the matching
  experiment folder: `trained_models/basic_transformer/` for `transformer_*`,
  `trained_models/spatial_transformer/` for `spatial_*` and `trained_models/geo_transformer/`
  for `geo_*`. Training writes them there by default.
- Only pull what is new: `rsync -avz --ignore-existing` against the destination folder.
