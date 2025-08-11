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
expdir="/home/balaskas/flexaf/saved_logs/simple_eval/simple_eval_fcnn_wesad_merged___2025.08.07-21.30.03.189"
python3 $maindir/src/evaluation/noselection_hweval.py \
    --expdir $expdir \
    --dataset-type wesad_merged \
    --dataset-file data/wesad_merged.csv \
    --weight-precision 8 \
    --input-precision 4 \
    --neurons 100 \
    --batch-size 32 2>&1 | tee "$expdir/simple_hw_eval.log"


elif [[ "$dataset" == "stressinnurses" ]]; then
# stress-in-nurses
expdir="/home/balaskas/flexaf/saved_logs/simple_eval/simple_eval_fcnn_stressinnurses___2025.08.07-21.54.23.161"
python3 $maindir/src/evaluation/noselection_hweval.py \
    --expdir $expdir \
    --dataset-type stressinnurses \
    --dataset-file data/stress_in_nurses.csv \
    --weight-precision 32 \
    --input-precision 4 \
    --neurons 100 \
    --batch-size 32 2>&1 | tee "$expdir/simple_hw_eval.log"


elif [[ "$dataset" == "spd" ]]; then
# SPD
expdir="/home/balaskas/flexaf/saved_logs/simple_eval/simple_eval_fcnn_spd___2025.08.07-22.15.48.757"
python3 $maindir/src/evaluation/noselection_hweval.py \
    --expdir $expdir \
    --dataset-type spd \
    --dataset-file data/spd.csv \
    --weight-precision 8 \
    --input-precision 4 \
    --neurons 100 \
    --batch-size 32 2>&1 | tee "$expdir/simple_hw_eval.log"


elif [[ "$dataset" == "daphnet" ]]; then
# DaphNET
expdir="/home/balaskas/flexaf/saved_logs/simple_eval/simple_eval_fcnn_daphnet___2025.08.11-06.34.13.865"
python3 $maindir/src/evaluation/noselection_hweval.py \
    --expdir $expdir \
    --dataset-type daphnet \
    --dataset-file data/daphnet.csv \
    --weight-precision 8 \
    --input-precision 4 \
    --neurons 100 \
    --batch-size 32 2>&1 | tee "$expdir/simple_hw_eval.log"


elif [[ "$dataset" == "harth" ]]; then
# harth
expdir="/home/balaskas/flexaf/saved_logs/simple_eval/simple_eval_fcnn_harth___2025.08.07-22.21.00.646"
python3 $maindir/src/evaluation/noselection_hweval.py \
    --expdir $expdir \
    --dataset-type harth \
    --dataset-file data/harth.csv \
    --weight-precision 8 \
    --input-precision 4 \
    --neurons 100 \
    --batch-size 32 2>&1 | tee "$expdir/simple_hw_eval.log"


else
    echo "Unknown dataset: $dataset"
    exit 1
fi


curl -d "SoA hw eval experiment finished for ${dataset} analog features" ntfy.sh/flexaf