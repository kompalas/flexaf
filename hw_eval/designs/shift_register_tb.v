`timescale 1ns/1ns
module shift_register_tb;

    parameter WORD_WIDTH = 4;
    parameter DEPTH      = 21;
    parameter CLK_PERIOD = 1000;

    reg clk;
    reg rst_n;
    reg  [WORD_WIDTH-1:0] din;
    wire [WORD_WIDTH-1:0] dout;
    wire [WORD_WIDTH*DEPTH-1:0] data_out_flat; // flattened internal storage

    // DUT
    shift_register #(
        .WORD_WIDTH(WORD_WIDTH),
        .DEPTH(DEPTH)
    ) DUT (
        .clk(clk),
        .rst_n(rst_n),
        .din(din),
        .dout(dout),
        .data_out_flat(data_out_flat)
    );

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // File I/O variables
    integer file, scan_file;
    integer output_file;
    integer i;

    // Helper task to print flattened bus as individual words
    task print_data;
        integer idx;
        begin
            $write("[");
            for (idx = 0; idx < DEPTH; idx = idx + 1) begin
                $write("%b", data_out_flat[(idx+1)*WORD_WIDTH-1 -: WORD_WIDTH]);
                if (idx < DEPTH-1) $write(" ");
            end
            $write("]");
        end
    endtask

    initial begin
        $printtimescale(shift_register_tb);

        clk   = 0;
        rst_n = 0;
        din   = 0;

        // Open files
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

        // Reset
        #(CLK_PERIOD);
        rst_n = 1;

        // Read from file until EOF
        while (!$feof(file)) begin
            scan_file = $fscanf(file, "%b\n", din); // read binary values
            #(CLK_PERIOD);

            // Console output
            $write("Input: %b | Output: %b | Data: ", din, dout);
            print_data();
            $write("\n");

            // File output
            $fwrite(output_file, "Input: %b | Output: %b | Data: ", din, dout);
            for (i = 0; i < DEPTH; i = i + 1) begin
                $fwrite(output_file, "%b", data_out_flat[(i+1)*WORD_WIDTH-1 -: WORD_WIDTH]);
                if (i < DEPTH-1) $fwrite(output_file, " ");
            end
            $fwrite(output_file, "\n");
        end

        $fclose(file);
        $fclose(output_file);
        $finish;
    end

endmodule
