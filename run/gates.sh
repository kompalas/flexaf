#!/bin/bash
set -eou pipefail

maindir=$(dirname "$(dirname "$(realpath "$0")")")

classifier="${1:-}"
exp_name="diff_fs"
if [[ -n "$classifier" ]]; then
    exp_name="${exp_name}_$classifier"
    sed -i "/classifier_type:/c\  classifier_type: $classifier  \# decisiontree, mlp, svm, bnn, tnn, fcnn" $maindir/run/args.yaml
fi

dataset="${2:-}"
if [[ -n "$dataset" ]]; then

    if [[ "$dataset" == "wesad_merged" ]]; then
        dataset_file="data/wesad_merged.csv"
    elif [[ "$dataset" == "wesad_rb" ]]; then
        dataset_file="data/wesad_merged.csv"
    elif [[ "$dataset" == "wesad_e4" ]]; then
        dataset_file="data/wesad_merged.csv"
    elif [[ "$dataset" == "stressinnurses" ]]; then
        dataset_file="data/stress_in_nurses.csv"
    elif [[ "$dataset" == "spd" ]]; then
        dataset_file="data/spd.csv"
    elif [[ "$dataset" == "affectiveroad" ]]; then
        dataset_file="data/affective_road.csv"
    elif [[ "$dataset" == "drivedb" ]]; then
        dataset_file="data/DriveDB.csv"
    elif [[ "$dataset" == "har" ]]; then
        dataset_file="data/HAR.csv"
    elif [[ "$dataset" == "wisdm" ]]; then
        dataset_file="data/wisdm.csv"
    elif [[ "$dataset" == "harth" ]]; then
        dataset_file="data/harth.csv"
    elif [[ "$dataset" == "daphnet" ]]; then
        dataset_file="data/daphnet.csv"
    else
        echo "Unknown dataset: $dataset"
        exit 1
    fi

    exp_name="${exp_name}_${dataset}"
    sed -i "/dataset_type:/c\  dataset_type: $dataset  \# wesad_merged, wesad_rb, wesad_e4, stressinnurses, spd, affectiveroad, drivedb, har, wisdm, harth" $maindir/run/args.yaml
    sed -i "/dataset_file:/c\  dataset_file: $dataset_file" $maindir/run/args.yaml
fi

sed -i "/name:/c\  name: $exp_name" $maindir/run/args.yaml
sed -i "/execute_differentiable_feature_selection:/c\  execute_differentiable_feature_selection: True" $maindir/run/args.yaml
sed -i "/execute_statistical_feature_selection:/c\  execute_statistical_feature_selection: False" $maindir/run/args.yaml
sed -i "/execute_soa_statistical_feature_selection:/c\  execute_soa_statistical_feature_selection: False" $maindir/run/args.yaml
sed -i "/execute_heuristic_feature_selection:/c\  execute_heuristic_feature_selection: False" $maindir/run/args.yaml
sed -i "/execute_greedy_feature_selection:/c\  execute_greedy_feature_selection: False" $maindir/run/args.yaml

python3 $maindir/main.py \
    --yaml-cfg-file $maindir/run/args.yaml

logdir="$(find $maindir/logs/ -type d -name "*$exp_name*" | sort -rV | head -n 1)"
curl -d "Diff. gates experiment finished ($logdir)" ntfy.sh/flexaf
