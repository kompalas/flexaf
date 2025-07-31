
#setup libs
set dcpath $::env(SNPS_SYN)

if {[string index $dcpath end] != "/"} {
	set DC_LIB_PATH ${dcpath}/libraries/syn/
} else {
	set DC_LIB_PATH ${dcpath}libraries/syn/
}

set LIB_DB_PATH $::env(ENV_LIBRARY_PATH)
if {[string index $LIB_DB_PATH end] != "/"} { set LIB_DB_PATH "${LIB_DB_PATH}/" }

set curDir [pwd]

set LIB_DB_NAME $::env(ENV_LIBRARY_DB)
set LIB_SDB_NAME " "

set search_path [list ${LIB_DB_PATH} ${DC_LIB_PATH} ${curDir}/hdl/ ]
set target_library [list ${LIB_DB_NAME}]
set synthetic_library {dw_foundation.sldb}
set link_library [list ${LIB_DB_NAME} ${synthetic_library}]
set symbol_library [list ${LIB_SDB_NAME}]

#read files
set verilog_files $::env(ENV_VERILOG_FILES)
set top_design $::env(ENV_TOP_DESIGN)

analyze -f Verilog $verilog_files
elaborate $top_design
current_design $top_design
link
set auto_wire_load_selection true
set_max_area 0

#sdc
#clock constraints
source ./scripts/sdc.tcl
#units ns | ps
source ./scripts/units.tcl

#compile option
compile_ultra

#reports
set clk_period $::env(ENV_CLK_PERIOD)
report_qor > ./reports/${top_design}_${clk_period}${tunit}.qor.rpt
report_area > ./reports/${top_design}_${clk_period}${tunit}.area.rpt
report_area -hierarchy > ./reports/${top_design}_${clk_period}${tunit}.hierarchy.area.rpt
report_timing > ./reports/${top_design}_${clk_period}${tunit}.timing.rpt 
report_power > ./reports/${top_design}_${clk_period}${tunit}.power.rpt 

report_script_runtime > ./reports/runtime.rpt 

#write files
change_names -h -rules verilog
write -h -f ddc -output ./gate/${top_design}.ddc
write -h -f verilog -output ./gate/${top_design}.sv
write_sdc ./gate/${top_design}.sdc
write_sdf ./gate/${top_design}.sdf

#exit
quit
