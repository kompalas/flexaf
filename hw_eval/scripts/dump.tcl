set dumpfile $::env(ENV_DUMPFILE)
set tb $::env(ENV_TB_NAME)
set dut $::env(ENV_DUT_NAME)

dump -file ${dumpfile} -type FSDB
dump -add ${tb}.${dut} -fid FSDB0

run
quit



