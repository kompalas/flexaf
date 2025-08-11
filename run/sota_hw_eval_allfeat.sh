#!/bin/bash
set -x

maindir=$(dirname "$(dirname "$(realpath "$0")")")

dataset=$1
if [[ -z "$dataset" ]]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

if [[ "$dataset" == "wesad_merged" ]]; then

# wesad_merged - All Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/wesad_merged/sota_fcnn_wesad_merged_allfeat___2025.08.09-07.32.07.336' \
    --selector-name JMI \
    --num-features 20 \
    --sparsity 0.5 \
    --weight-precision 32 \
    --use-all-features

elif [[ "$dataset" == "stressinnurses" ]]; then

# stress-in-nurses - All Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/stressinnurses/sota_fcnn_stressinnurses_allfeat___2025.08.09-20.00.29.325' \
    --selector-name JMI \
    --num-features 20 \
    --sparsity 0.9 \
    --weight-precision 8 \
    --use-all-features

elif [[ "$dataset" == "spd" ]]; then

# SPD - All Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/spd/sota_fcnn_spd_allfeat___2025.08.10-02.55.53.940' \
    --selector-name Fisher \
    --num-features 15 \
    --sparsity 0.9 \
    --weight-precision 32 \
    --use-all-features

elif [[ "$dataset" == "daphnet" ]]; then

# DaphNET - All Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/daphnet/sota_fcnn_daphnet_allfeat___2025.08.10-10.19.04.484' \
    --selector-name DISR \
    --num-features 15 \
    --sparsity 0.9 \
    --weight-precision 32 \
    --use-all-features

elif [[ "$dataset" == "harth" ]]; then

# harth - All Features
python3 $maindir/src/evaluation/statistical_islped_hweval.py \
    --expdir '/home/balaskas/flexaf/saved_logs/harth/sota_fcnn_harth_allfeat___2025.08.09-15.06.02.647' \
    --selector-name JMI \
    --num-features 20 \
    --sparsity 0.5 \
    --weight-precision 32 \
    --use-all-features

else
    echo "Unknown dataset: $dataset"
    exit 1
fi

curl -d "SoA hw eval experiment finished for ${dataset} all features" ntfy.sh/flexaf