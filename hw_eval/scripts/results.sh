
cd reports
area=$(cat *ns.area.rpt | grep 'Total cell area:' | awk '{printf "%.2f", ($NF)}')
# area=$(cat *ns.area.rpt | grep 'Total cell area:' | awk '{printf "%.5f", ($NF/10^8)}')
power=$(cat *ns.power.ptpx.rpt | grep 'Total Power' | awk '{printf "%.6f", ($4*10^3)}')
cd ../sim
acc=$(paste output.txt sim.Ytest | awk 'BEGIN{score=0}{if ($1==$2) score++;}END{print (score/NR)*10^2}')
cd ../reports
# Extract BITWIDTH value from hdl/top.v
bitwidth=$(grep -oP 'parameter\s+BITWIDTH\s*=\s*\K[0-9]+' ../hdl/top.v)

# Write output to reports.txt with BITWIDTH included next to accuracy
echo -e "Accuracy: $acc%\tBITWIDTH: $bitwidth\tArea: $area um^2\tPower: $power mW" > reports.txt
# echo -e "Accuracy: $acc\tArea: $area cm^2\tPower: $power mW">reports.txt
# echo -e "Accuracy: $acc\tArea: $area um^2\tPower: $power mW">reports.txt
