`timescale 1ns/1ns
module counter_tb;

    parameter WIDTH = 2;
    parameter CLK_PERIOD = 1000;

    reg clk;
    reg rst_n;
    wire [WIDTH-1:0] count;

    // Instantiate DUT
    counter
    //  #(
    //     .WIDTH(WIDTH)
    // ) 
    DUT (
        .clk(clk),
        .rst_n(rst_n),
        .count(count)
    );

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test sequence
    initial begin
        $printtimescale(counter_tb);
        clk = 0;
        rst_n = 0;

        // Apply reset
        #(CLK_PERIOD);
        rst_n = 1;

        // Run for some cycles
        repeat (20) begin
            #(CLK_PERIOD);
            $display("Time=%0t | Count=%0d (%b)", $time, count, count);
        end

        $finish;
    end

endmodule
