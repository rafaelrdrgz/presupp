# Sourced by the job scripts. PRESUPP_ENV overrides the conda environment name.
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate "${PRESUPP_ENV:-presupp}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/src:${PYTHONPATH:-}"
# use the cached roberta-base without network calls
if [ -d "$HOME/.cache/huggingface/hub/models--roberta-base" ]; then export HF_HUB_OFFLINE=1; fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"
