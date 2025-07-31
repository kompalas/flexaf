#units
set unit "xx"
set suppress_errors "IFS-001"
set isps [set_units  -time ps]
if {$isps == 1} {
        set tunit "ps"
}
set isns [set_units  -time ns]
if {$isns == 1} {
        set tunit "ns"
}
