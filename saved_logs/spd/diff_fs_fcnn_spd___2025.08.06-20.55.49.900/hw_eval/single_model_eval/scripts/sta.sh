#!/bin/bash
source ./scripts/env.sh

mkdir -p ./reports
mkdir -p ./logs

if ! grep -q Error "logs/dcsyn.log"; 
then  
    pt_shell  -f ./scripts/sta.tcl | tee ./logs/ptsta.log
    rm -rf transcript *.wlf *.mr *.syn *.log *.svf *.pvl *~
else
    echo "Error from synthesis!! Check the dcsyn.log"> ./logs/ptsta.log
fi;