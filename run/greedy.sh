#!/bin/bash
set -eou pipefail

maindir=$(dirname "$(dirname "$(realpath "$0")")")

classifier="${1:-}"
exp_name="greedy_fs"
if [[ -n "$classifier" ]]; then
    exp_name="greedy_fs_$classifier"
    sed -i "/classifier_type:/c\  classifier_type: $classifier  \# decisiontree, mlp, svm, bnn, tnn, fcnn" $maindir/run/args.yaml
fi

sed -i "/name:/c\  name: $exp_name" $maindir/run/args.yaml
sed -i "/execute_greedy_feature_selection:/c\  execute_greedy_feature_selection: True" $maindir/run/args.yaml
sed -i "/execute_statistical_feature_selection:/c\  execute_statistical_feature_selection: False" $maindir/run/args.yaml
sed -i "/execute_statistical_soa_feature_selection:/c\  execute_statistical_soa_feature_selection: False" $maindir/run/args.yaml
sed -i "/execute_heuristic_feature_selection:/c\  execute_heuristic_feature_selection: False" $maindir/run/args.yaml
sed -i "/execute_differentiable_feature_selection:/c\  execute_differentiable_feature_selection: False" $maindir/run/args.yaml

python3 $maindir/main.py \
    --yaml-cfg-file $maindir/run/args.yaml

logdir="$(find $maindir/logs/ -type d -name "*$exp_name*" | sort -rV | head -n 1)"
curl -d "Greedy experiment finished ($logdir)" ntfy.sh/flexaf
