module d_flipflop (
    input  wire clk,
    input  wire rst,      // active-high reset
    input  wire D,
    output reg  Q
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            Q <= 1'b0;
        else
            Q <= D;
    end
endmodule
