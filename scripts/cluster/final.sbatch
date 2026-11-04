#!/bin/bash
#SBATCH --job-name=presupp-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Lockbox evaluation for every config given, in one opening.
#   sbatch scripts/cluster/final.sbatch "reason" configs/a.yaml configs/b.yaml
set -eo pipefail
REASON="${1:?usage: sbatch scripts/cluster/final.sbatch \"reason\" configs/<name>.yaml ...}"
shift
source scripts/cluster/env.sh
python -m presupp.final "$@" --reason "$REASON" --device cuda
