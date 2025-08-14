module decoder #(
    parameter WIDTH = 5  // determines both input width and outputs count
)(
    input  wire [WIDTH-1:0] sel,          // binary select
    output reg  [(1<<WIDTH)-1:0] out      // one-hot output
);
    integer i;
    always @(*) begin
        out = {(1<<WIDTH){1'b0}};
        out[sel] = 1'b1;
    end
endmodule
