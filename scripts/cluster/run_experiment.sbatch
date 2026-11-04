#!/bin/bash
#SBATCH --job-name=presupp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# All rounds x seeds of one experiment. Resubmit after a timeout; finished runs are skipped.
#   sbatch -J fixed_unweighted scripts/cluster/run_experiment.sbatch configs/fixed_unweighted.yaml
set -eo pipefail
CONFIG="${1:?usage: sbatch scripts/cluster/run_experiment.sbatch configs/<name>.yaml}"
source scripts/cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m presupp.train "$CONFIG" --device cuda
