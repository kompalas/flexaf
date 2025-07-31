#!/bin/bash
source ./scripts/env.sh

export ENV_VERILOG_FILES=$(find ./hdl/*.v | awk 'ORS=" "')
echo "Verilog files found: $ENV_VERILOG_FILES"

mkdir -p ./gate
mkdir -p ./reports
mkdir -p ./logs

dc_shell-xg-t -f ./scripts/synthesis.tcl | tee ./logs/dcsyn.log

rm -rf transcript *.wlf *.mr *.syn *.log *.svf *.pvl *~

