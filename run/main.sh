#!/bin/bash
set -eou pipefail

maindir=$(dirname "$(dirname "$(realpath "$0")")")

python3 $maindir/main.py \
    --yaml-cfg-file $maindir/run/args.yaml
