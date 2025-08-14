module dff_chain (
    input  wire clk,
    input  wire D,
    output wire Q2
);
    reg q1_reg, q2_reg;
    wire Q1;
    always @(posedge clk) q1_reg <= D;
    always @(posedge clk) q2_reg <= q1_reg;
    assign Q1 = q1_reg;
    assign Q2 = q2_reg;
endmodule
