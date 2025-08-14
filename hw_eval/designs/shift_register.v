module shift_register #(
    parameter WORD_WIDTH = 4,
    parameter DEPTH      = 21
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [WORD_WIDTH-1:0] din,
    output wire [WORD_WIDTH-1:0] dout,
    output wire [WORD_WIDTH*DEPTH-1:0] data_out_flat
);

    reg [WORD_WIDTH-1:0] data [0:DEPTH-1];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1)
                data[i] <= {WORD_WIDTH{1'b0}};
        end else begin
            for (i = DEPTH-1; i > 0; i = i - 1)
                data[i] <= data[i-1];
            data[0] <= din;
        end
    end

    assign dout = data[DEPTH-1];

    // Flatten memory into single output bus
    genvar gi;
    generate
        for (gi = 0; gi < DEPTH; gi = gi + 1) begin : FLATTEN
            assign data_out_flat[ (gi+1)*WORD_WIDTH-1 : gi*WORD_WIDTH ] = data[gi];
        end
    endgenerate

endmodule
