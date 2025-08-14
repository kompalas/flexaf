#!/bin/bash
set -eou pipefail
set -x

# Directories
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
testdir=$(dirname "$script_dir")
cd $testdir

# Hyperparameters
top_design="top"
tunit="ns"
synclk="${1:-1000000.0}"  # 1,000,000 ns = 1 ms
libspath="/usr/local/eda/synLibs/FlexIC/G3SC_T5_v1"
voltage="${2:-3}"
if [[ "$voltage" == "1.5" ]]; then
    libfile="flexic3_T5_typ_1p50v.db"
elif [[ "$voltage" == "3" ]]; then
    libfile="flexic3_T5_typ_3p00v.db"
else
    echo "Unsupported voltage: $voltage. Supported voltages are 1.5 and 3.0."
    exit 1
fi

# Setting up environment
sed -i "/ENV_CLK_PERIOD=/ c\export ENV_CLK_PERIOD=\"$synclk\"" $testdir/scripts/env.sh
sed -i "/ENV_LIBRARY_PATH=/ c\export ENV_LIBRARY_PATH=\"$libspath/db\"" $testdir/scripts/env.sh
sed -i "/ENV_LIBRARY_DB=/ c\export ENV_LIBRARY_DB=\"$libspath/db/$libfile\"" $testdir/scripts/env.sh
sed -i "/ENV_LIBRARY_VERILOG_PATH=/ c\export ENV_LIBRARY_VERILOG_PATH=\"$libspath/verilog\"" $testdir/scripts/env.sh
# activate/deactivate the virtual clock if necessary
if grep -rq "input clk" $testdir/hdl/; then
    sed -i "/ENV_VIRTUAL_CLOCK=/c\export ENV_VIRTUAL_CLOCK=\"false\"" $testdir/scripts/env.sh
else
    sed -i "/ENV_VIRTUAL_CLOCK=/c\export ENV_VIRTUAL_CLOCK=\"true\"" $testdir/scripts/env.sh
fi

# run synthesis, STA, gate-level simulation and power analysis
rm -rf $testdir/work_gate_lib
rm -rf $testdir/work_lib
rm -rf $testdir/gate_simv.daidir
rm -f $testdir/gate_simv
rm -rf $testdir/rtl_simv.daidir
rm -f $testdir/rtl_simv
rm -rf $testdir/tech_lib
rm -rf $testdir/logs
make all
# make dcsyn

area_rpt="$testdir/reports/${top_design}_${synclk}${tunit}.area.rpt"  # area report from DC
delay_rpt="$testdir/reports/${top_design}_${synclk}${tunit}.timing.pt.rpt"  # delay report from PrimeTime
# delay_rpt="$testdir/reports/${top_design}_${synclk}${tunit}.timing.rpt"  # delay report from DC
power_rpt="$testdir/reports/${top_design}_${synclk}${tunit}.power.ptpx.rpt"  # power report from PrimeTime
# power_rpt="$testdir/reports/${top_design}_${synclk}${tunit}.power.rpt"  # power report from DC

resfile="$testdir/reports/results.csv"
echo "Synclk,Area,Delay,Power" > $resfile

# get area, delay and power from the reports
area="$(awk '/Total cell area/ {print $NF}' $area_rpt)"
if (( $(echo "$area == 0.0" | bc -l) )); then
    echo "Area is zero, likely due to netlist with only wiring. Skipping the rest of the evaluation."
    delay="0.0"
    power="0.0"
    
elif grep "No constrained paths." $delay_rpt; then
    echo "No constrained paths in the design. Skipping the rest of the evaluation."
    delay="0.0"
    power="0.0"

elif grep "Error: No activity is available in the VCD file for the given time interval for power calculation. (PWR-255)" $power_rpt; then
    echo "Error in the power report, due to lack of switching activity. The circuit is probably too simple."
    echo "Skipping the rest of the evaluation..."
    delay="$(grep "data arrival time" $delay_rpt | awk 'NR==1 {print $NF}')"
    power="0.0"

else
    delay="$(grep "data arrival time" $delay_rpt | awk 'NR==1 {print $NF}')"
    power="$(awk '/Total Power/ {print $4}' $power_rpt)"  # to get power from the PrimeTime power report
    # power="$(awk '/^Total  / {print $8}' $power_rpt)"  # to get power from the DC power report
fi

echo "$synclk,$area,$delay,$power" >> $resfile
