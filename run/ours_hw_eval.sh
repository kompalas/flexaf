#!/bin/bash
set -eou pipefail

maindir=$(dirname "$(dirname "$(realpath "$0")")")

dataset=$1
if [[ -z "$dataset" ]]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

if [[ "$dataset" == "wesad_merged" ]]; then
# wesad_merged
python3 $maindir/src/evaluation/hw_eval_single_model.py \
    --expdir '/home/balaskas/flexaf/saved_logs/wesad_merged/diff_fs_fcnn_wesad_merged___2025.08.08-08.00.07.792' \
    --fold 4 \
    --sparsity 0.2 \
    --trial 1 \
    --weight-precision 8 \
    --input-precision 4


elif [[ "$dataset" == "stressinnurses" ]]; then
# stress-in-nurses
python3 $maindir/src/evaluation/hw_eval_single_model.py \
    --expdir '/home/balaskas/flexaf/saved_logs/stressinnurses/diff_fs_fcnn_stressinnurses___2025.08.08-11.59.24.548' \
    --fold 1 \
    --sparsity 0.05 \
    --trial 1 \
    --weight-precision 8 \
    --input-precision 4


elif [[ "$dataset" == "spd" ]]; then
# SPD
python3 $maindir/src/evaluation/hw_eval_single_model.py \
    --expdir '/home/balaskas/flexaf/saved_logs/spd/diff_fs_fcnn_spd___2025.08.06-20.55.49.900' \
    --fold 1 \
    --sparsity 0.2 \
    --trial 4 \
    --weight-precision 8 \
    --input-precision 4


elif [[ "$dataset" == "daphnet" ]]; then
# DaphNET
echo "DaphNET dataset is not supported for lottery ticket pruning yet."
exit 1


elif [[ "$dataset" == "harth" ]]; then
# harth
echo "Harth dataset is not supported for lottery ticket pruning yet."
exit 1

else
    echo "Unknown dataset: $dataset"
    exit 1
fi


curl -d "HW eval for gates experiment finished for ${dataset}" ntfy.sh/flexaf