#!/bin/bash
set -x

maindir=$(dirname "$(dirname "$(realpath "$0")")")

dataset=$1
if [[ -z "$dataset" ]]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

if [[ "$dataset" == "wesad_merged" ]]; then
# wesad_merged - Analog Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/wesad_merged/sota_fcnn_wesad_merged_analogfeat___2025.08.08-19.54.25.800' \
    --selector-name DISR \
    --num-features 20 \
    --sparsity 0.5 \
    --weight-precision 32


elif [[ "$dataset" == "stressinnurses" ]]; then
# stress-in-nurses - Analog Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/stressinnurses/sota_fcnn_stressinnurses_analogfeat___2025.08.08-22.34.30.019' \
    --selector-name JMI \
    --num-features 20 \
    --sparsity 0.9 \
    --weight-precision 32


elif [[ "$dataset" == "spd" ]]; then
# SPD - Analog Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir "/home/balaskas/flexaf/saved_logs/spd/sota_fcnn_spd_analogfeat___2025.08.09-00.31.50.908" \
    --selector-name Fisher \
    --num-features 10 \
    --sparsity 0.5 \
    --weight-precision 32


elif [[ "$dataset" == "daphnet" ]]; then
# DaphNET - Analog Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/daphnet/sota_fcnn_daphnet_analogfeat___2025.08.10-10.19.47.641' \
    --selector-name Fisher \
    --num-features 10 \
    --sparsity 0.2 \
    --weight-precision 32


elif [[ "$dataset" == "harth" ]]; then
# harth - Analog Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/harth/sota_fcnn_harth_analogfeat___2025.08.08-21.07.19.545' \
    --selector-name DISR \
    --num-features 10 \
    --sparsity 0.2 \
    --weight-precision 8


else
    echo "Unknown dataset: $dataset"
    exit 1
fi


curl -d "SoA hw eval experiment finished for ${dataset} analog features" ntfy.sh/flexaf