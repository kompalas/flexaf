#!/bin/bash
set -eou pipefail

maindir=$(dirname "$(dirname "$(realpath "$0")")")

classifier="${1:-}"
exp_name="ga"
if [[ -n "$classifier" ]]; then
    sed -i "/name:/c\  name: ga_$classifier" $maindir/run/args.yaml
    sed -i "/classifier_type:/c\  classifier_type: $classifier  \# decisiontree, mlp, svm, bnn, tnn" $maindir/run/args.yaml
    exp_name="ga_$classifier"
fi

python3 $maindir/main.py \
    --yaml-cfg-file $maindir/run/args.yaml

logdir="$(find $maindir/logs/ -type d -name "*$exp_name*" | sort -rV | head -n 1)"
logfile="$(find $logdir -type f -name "*.log")"
results=$(grep -A 2 "Pareto front contains" $logfile | tail -n 3)
curl -d "GA finished ($logdir): $results" ntfy.sh/pestress
