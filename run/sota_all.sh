#!/bin/bash
set -eou pipefail

maindir=$(dirname "$(dirname "$(realpath "$0")")")

datasets=("wesad_merged" "harth" "stressinnurses" "spd" "affectiveroad")

sed -i "/statistical_use_all_features:/c\  statistical_use_all_features: True" $maindir/run/args.yaml
sed -i "/statistical_num_features:/c\  statistical_num_features: [5, 10, 15, 20]" $maindir/run/args.yaml

for dataset in "${datasets[@]}"; do
    source $maindir/run/sota.sh fcnn $dataset
done