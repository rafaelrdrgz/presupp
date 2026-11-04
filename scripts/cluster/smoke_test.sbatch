#!/bin/bash
#SBATCH --job-name=presupp-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# One round, one seed, one epoch per config (default fixed_unweighted).
#   sbatch scripts/cluster/smoke_test.sbatch [configs/v3_mil.yaml ...]
set -eo pipefail
source scripts/cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'"
configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then configs=(configs/fixed_unweighted.yaml); fi
rm -rf runs/_smoke
for config in "${configs[@]}"; do
    echo "== $config"
    python -m presupp.train "$config" --rounds 0 --seeds 0 --max-epochs 1 \
        --device cuda --runs-dir runs/_smoke
done
echo "Smoke test passed."
