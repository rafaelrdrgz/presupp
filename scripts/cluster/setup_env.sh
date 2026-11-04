#!/bin/bash
# One-time setup, from the repo root on a login node: conda env with torch 2.4.1 + CUDA 12.1,
# and roberta-base cached for offline jobs.
set -eo pipefail
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
if ! conda env list | grep -q '^presupp '; then
    conda create -y --name presupp -c conda-forge python=3.12
fi
source activate presupp
pip install --upgrade pip
# cuDNN etc. come from PyPI, hence the extra index
pip uninstall -y torch || true
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple
grep -v -E '^torch==' requirements-lock-cpu.txt > /tmp/presupp-requirements-$USER.txt
pip install -r /tmp/presupp-requirements-$USER.txt
pip install -e . --no-deps
python - <<'PY'
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("roberta-base")
AutoModel.from_pretrained("roberta-base")
import torch
print(torch.__version__, torch.version.cuda)
PY
mkdir -p logs
