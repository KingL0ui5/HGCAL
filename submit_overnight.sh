#!/bin/bash
# Usage (on the HPC, from ~/HGCAL): ./submit_overnight.sh

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p evals/data/logs trained_models/spatial_transformer trained_models/geo_transformer

train_job="$(qsub train_job.pbs)"
eval_job="$(qsub -W depend=afterany:"$train_job" -v EVAL_SET="${EVAL_SET:-all}" eval_job.pbs)"

echo "training: $train_job"
echo "evals:    $eval_job (starts once every training run has finished)"
