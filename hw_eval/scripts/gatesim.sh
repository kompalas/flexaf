#!/bin/bash

source ./scripts/env.sh

work="./work_lib"
work_gate="./work_gate_lib"
tech="./tech_lib"
simv="gate_simv"

cat > ./synopsys_sim.setup << EOM
--Mapping default work directory
WORK > WORK_LIB
WORK_LIB : ${work_gate}
TECH_LIB : ${tech}

EOM


if [ ! -d ${tech} ]; then
	echo "Creating Library TECH_LIB"
	#find ${ENV_LIBRARY_VERILOG_PATH}/*.v > tech
	vlogfiles=$(find ${ENV_LIBRARY_VERILOG_PATH}/*.v | awk 'ORS=" "')
	vlogan -full64 -kdb -work TECH_LIB $vlogfiles
	
	exstatus="$?"
	if [ "$exstatus" -ne  0 ]; then
        	echo "ERROR in lib"
		rm -rf ${tech}
        	exit 1
	fi
fi

tbfiles=$(find ./sim/*.v | awk 'ORS=" "')
vlogan -full64 -work WORK_LIB -kdb $tbfiles
exstatus="$?"
if [ "$exstatus" -ne  0 ]; then
	echo "ERROR in testbench"
	exit 1
fi

vlogan -full64 -work WORK_LIB -kdb ./gate/${ENV_TOP_DESIGN}.sv
exstatus="$?"
if [ "$exstatus" -ne 0 ]; then
        echo "ERROR in synthesized verilog"
        exit 1
fi

# vcs -full64 -kdb -lca -debug_access ${ENV_TB_NAME} -sdf max:${ENV_TB_NAME}.${ENV_DUT_NAME}:./gate/${ENV_TOP_DESIGN}.sdf -o ${simv} -l ./logs/vcs_gate.log
vcs -full64 -kdb -lca -debug_access ${ENV_TB_NAME} -o ${simv} -l ./logs/vcs_gate.log
exstatus="$?"
err="$(grep -E -o '[0-9]+[ ]*error' ./logs/vcs.log | grep -E -o '[0-9]+')"
err=${err:=0}
if [ "$exstatus" -ne 0 ] || [ "$err" -ne 0 ]; then
        echo "ERROR in vcs"
        exit 1
fi

# ./${simv} -gui -ucli -do ./scripts/dump.tcl
./${simv}  -ucli -do ./scripts/dump.tcl
