#!/bin/bash
set -eou pipefail

maindir="$HOME/pestress"
resampling_rate="64"

# Download WESAD dataset
if [ ! -d "$maindir/data/wesad_raw/WESAD" ]; then
    mkdir -p $maindir/data
    mkdir -p $maindir/data/wesad_raw
    cd $maindir/data/wesad_raw
    wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
    unzip download
    rm -f download
fi

exit 0
# extract the raw data
if [ ! -f "$maindir/data/wesad_sr${resampling_rate}.csv" ]; then
    cd $maindir
    python3 src/misc/preprocess_wesad.py \
        --save-data-file $maindir/data/wesad_sr${resampling_rate}.csv \
        --resampling-rate $resampling_rate
fi