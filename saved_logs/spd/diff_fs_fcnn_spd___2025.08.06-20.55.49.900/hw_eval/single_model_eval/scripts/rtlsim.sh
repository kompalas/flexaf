#!/bin/bash

source ./scripts/env.sh

mkdir -p ./logs

work="./work_lib"
simv="rtl_simv"

cat > ./synopsys_sim.setup << EOM
--Mapping default work directory
WORK > WORK_LIB
WORK_LIB : ${work}

EOM

vfiles=$(find ./hdl/*.v 2>/dev/null | awk 'ORS=" "')
if [ -n "$vfiles" ]; then
	vlogan -full64 -work WORK_LIB -kdb $vfiles
	exstatus="$?"
	if [ "$exstatus" -ne  0 ]; then
		echo "ERROR in testbench"
		exit 1
	fi
fi


vhdlfiles=$(find ./hdl/*.vhd 2>/dev/null | awk 'ORS=" "')
if [ -n "$vhdlfiles" ]; then
	vhdlan -full64 -work WORK_LIB -kdb $vhdlfiles
	exstatus="$?"
	if [ "$exstatus" -ne  0 ]; then
		echo "ERROR in testbench"
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


vcs -full64 -debug_access+r -kdb -lca ${ENV_TB_NAME} -o ${simv} -l ./logs/vcs_rtl.log
exstatus="$?"
err="$(grep -E -o '[0-9]+[ ]*error' ./logs/vcs.log | grep -E -o '[0-9]+')"
err=${err:=0}
if [ "$exstatus" -ne 0 ] || [ "$err" -ne 0 ]; then
        echo "ERROR in vcs"
        exit 1
fi


if [ "$1" == "-g" ]; then
	./${simv} -gui=verdi
else
	./${simv}	
fi
