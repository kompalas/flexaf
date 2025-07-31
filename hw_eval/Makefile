env:
	@vim ./scripts/env.sh
dcsyn:
	./scripts/syn.sh
sta:
	./scripts/sta.sh
rtl_sim:
	./scripts/rtlsim.sh
rtl_sim_gui:
	./scripts/rtlsim.sh -g

gate_sim:
	./scripts/gatesim.sh
power:
	./scripts/power.sh
wave:
	./scripts/wave.sh
results:
	./scripts/results.sh
clean:
	./scripts/clean.sh
all:
	@make -s dcsyn
	@make -s sta
	@make -s gate_sim
	@make -s power

help:
	@echo -e "make dcsyn: Run Circuit Synthesis with Synopsys Design Compiler. Design files: ./hdl/*.v"
	@echo -e "make sta: Run Static Time Analysis (STA) with Synopsys PrimeTime."
	@echo -e "make rtl_sim: Run RTL simulation with VCS. Testbench files: ./sim/*.v"
	@echo -e "make rtl_sim_gui: Run RTL simulation with VCS. Uses Verdi Gui. Testbench files: ./sim/*.v"
	@echo -e "make gate_sim: Run gate-level simulation with VCS. Gate-level netlist is generated with make dcsyn."
	@echo -e "make power: Run power analysis with Synopsys PrimeTime PX. Requires the fsdb generated with make gate_sim."
	@echo -e "make env: Set environment"
	@echo -e "make results: Accuracy Power and Area reports"
	@echo -e "make help: Print this message"
