#!/bin/bash
# One GPU job per experiment named (default: the four v2 transformer experiments).
# Skips anything queued or finished, so it can be rerun.
#   bash scripts/cluster/submit_all.sh [v3_joint v3_mil ...]
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
EXPECTED_RUNS=27   # 9 rounds x 3 seeds

queued() { squeue -u "$USER" -h -o "%j" | grep -qx "$1"; }

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
    names=(v1_recipe_passage_cv v1_recipe_grouped_cv fixed_unweighted fixed_weighted)
fi

for name in "${names[@]}"; do
    if [ ! -f "configs/$name.yaml" ]; then
        echo "$name: no configs/$name.yaml, skipping"
        continue
    fi
    finished=$(ls runs/"$name"/round*/run.json 2>/dev/null | wc -l)
    if [ "$finished" -ge "$EXPECTED_RUNS" ]; then
        echo "$name: already finished ($finished runs), skipping"
        continue
    fi
    if queued "$name"; then
        echo "$name: already queued or running, skipping"
        continue
    fi
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if sbatch -J "$name" scripts/cluster/run_experiment.sbatch "configs/$name.yaml"; then
            break
        fi
        # sbatch sometimes errors after accepting the job
        sleep 15
        if queued "$name"; then
            echo "$name: queued"
            break
        fi
        echo "$name: scheduler error, retrying (attempt $attempt of 10)"
    done
done
echo
squeue -u "$USER"
