#!/bin/bash
#SBATCH --job-name=presupp-release
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Train and save the demo model. ~6 min on a V100.
#   sbatch scripts/cluster/release.sbatch configs/v3_mil.yaml [--force]
set -eo pipefail
CONFIG="${1:?usage: sbatch scripts/cluster/release.sbatch configs/<name>.yaml [--force]}"
shift
source scripts/cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m presupp.release "$CONFIG" --seed 0 --device cuda "$@"
