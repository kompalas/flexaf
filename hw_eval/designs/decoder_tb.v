`timescale 1ns/1ns
module decoder_tb;

    parameter WIDTH = 5; // 5-bit select → 32 outputs
    localparam OUTS = (1 << WIDTH);

    reg  [WIDTH-1:0] sel;
    wire [OUTS-1:0]  out;

    decoder #(
        .WIDTH(WIDTH)
    ) DUT (
        .sel(sel),
        .out(out)
    );

    integer i;
    initial begin
        $printtimescale(decoder_tb);

        // Test all possible select values
        for (i = 0; i < OUTS; i = i + 1) begin
            sel = i[WIDTH-1:0];
            #10;
            $display("sel=%0d (%b) -> out=%b", sel, sel, out);
        end

        $finish;
    end

endmodule
