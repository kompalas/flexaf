`timescale 1ns/1ns
module top_tb;

    parameter BITWIDTH =  8;
	parameter CLK_PERIOD = 1000000;
    
    reg clk;
    reg reset;
    reg signed [BITWIDTH-1:0] signal_in;
    wire signed [BITWIDTH-1:0] signal_out;
    // File I/O variables
    integer file, scan_file;
    integer output_file;

    // Instantiate the min_calculator module (DUT)
    top
    // #(
    //     .WIDTH(BITWIDTH)
    // ) 
    DUT (
        .clk(clk),
        .rst_n(reset),
        .signal_in(signal_in),
        .signal_out(signal_out)
    );
    
    // Clock generation
    always begin
        #(CLK_PERIOD/2) clk = ~clk;  // Toggle clock every half period
    end

    // Testbench logic
    initial begin
        // Initialize signals
        $printtimescale(top_tb);
        clk = 0;
        reset = 0;
        
        // Open the input and output test files
        file = $fopen("./sim/inputs.txt", "r");
        if (file == 0) begin
            $display("Error: Could not open test input file.");
            $finish;
        end
        output_file = $fopen("./sim/output.txt", "w");
        if (output_file == 0) begin
            $display("Error: Could not open test output file.");
            $finish;
        end

        // Apply reset for a few clock cycles
        #CLK_PERIOD;
        #CLK_PERIOD;
        reset = 1;

        // Read values from the file and apply them as inputs to the module
        while (!$feof(file)) begin
            scan_file = $fscanf(file, "%b\n", signal_in);
            #CLK_PERIOD;
            // Display the input and the current output value in the console
            $display("Input: %d, Output: %d", signal_in, signal_out);
            $fwrite(output_file, "Input: %d, Output: %d\n", signal_in, signal_out);
        end
        
        // Close the input and output files and end the simulation
        $fclose(file);
        $fclose(output_file);
        $finish;
    end
endmodule