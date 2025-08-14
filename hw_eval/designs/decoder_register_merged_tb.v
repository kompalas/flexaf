`timescale 1ns/1ns
module feature_shift_reg_tb;

    parameter N          = 4;   // bits per feature
    parameter SAR_CYCLES = 5;   // SAR counter max
    parameter NUM_FEAT   = 2;  // number of features
    parameter CLK_PERIOD = 100000;

    reg clk;
    reg rst;
    reg en;
    reg  [N-1:0] quant_feat;
    wire [NUM_FEAT*N-1:0] shift_reg;
    wire [NUM_FEAT-1:0]   feat_decoded;

    // DUT
    feature_shift_reg 
    // #(
    //     .N(N),
    //     .SAR_CYCLES(SAR_CYCLES),
    //     .NUM_FEAT(NUM_FEAT)
    // ) 
    DUT (
        .clk(clk),
        .rst(rst),
        .en(en),
        .quant_feat(quant_feat),
        .shift_reg(shift_reg),
        .feat_decoded(feat_decoded)
    );

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // File I/O variables
    integer file, scan_file;
    integer output_file;
    integer i;

    // Helper task to print shift_reg as features
    task print_shift_reg;
        integer idx;
        begin
            $write("[");
            for (idx = 0; idx < NUM_FEAT; idx = idx + 1) begin
                $write("%b", shift_reg[(idx+1)*N-1 -: N]);
                if (idx < NUM_FEAT-1) $write(" ");
            end
            $write("]");
        end
    endtask

    initial begin
        $printtimescale(feature_shift_reg_tb);

        clk         = 0;
        rst         = 1;
        en          = 1; // always enabled in this test
        quant_feat  = 0;

        // Open input file
        file = $fopen("./sim/inputs.txt", "r");
        if (file == 0) begin
            $display("Error: Could not open test input file.");
            $finish;
        end

        // Open output file
        output_file = $fopen("./sim/output.txt", "w");
        if (output_file == 0) begin
            $display("Error: Could not open test output file.");
            $finish;
        end

        // Apply reset for a few cycles
        #(CLK_PERIOD);
        rst = 0;

        // Main test loop
        while (!$feof(file)) begin
            scan_file = $fscanf(file, "%b\n", quant_feat);
            #(CLK_PERIOD);

            // Console output
            $write("Input: %b | Decoded: %b | Shift Reg: ", quant_feat, feat_decoded);
            print_shift_reg();
            $write("\n");

            // File output
            $fwrite(output_file, "Input: %b | Decoded: %b | Shift Reg: [", quant_feat, feat_decoded);
            for (i = 0; i < NUM_FEAT; i = i + 1) begin
                $fwrite(output_file, "%b", shift_reg[(i+1)*N-1 -: N]);
                if (i < NUM_FEAT-1) $fwrite(output_file, " ");
            end
            $fwrite(output_file, "]\n");
        end

        $fclose(file);
        $fclose(output_file);
        $finish;
    end

endmodule
