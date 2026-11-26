#!/bin/bash
# Usage (on the HPC, from ~/HGCAL): [EPOCHS=100] [WALLTIME=14:00:00] ./submit_overnight.sh

set -euo pipefail

EPOCHS="${EPOCHS:-50}"
WALLTIME="${WALLTIME:-08:00:00}"

cd "$(dirname "$0")"
mkdir -p evals/data/logs trained_models/spatial_transformer trained_models/geo_transformer

train_job="$(qsub -v EPOCHS="$EPOCHS" -l walltime="$WALLTIME" train_job.pbs)"
eval_job="$(qsub -W depend=afterany:"$train_job" -v EVAL_SET="${EVAL_SET:-all}" eval_job.pbs)"

echo "training: $train_job ($EPOCHS epochs, walltime $WALLTIME)"
echo "evals:    $eval_job (starts once every training run has finished)"
