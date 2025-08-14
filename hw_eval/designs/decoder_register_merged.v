module feature_shift_reg #(
    parameter N = 4,              // Bits per feature
    parameter SAR_CYCLES = 5,     // sar_cnt max value
    parameter NUM_FEAT = 10       // Number of features
)(
    input  wire                   clk,
    input  wire                   rst,         // async posedge reset
    input  wire                   en,
    input  wire [N-1:0]           quant_feat,

    output reg  [NUM_FEAT*N-1:0]  shift_reg,   // full shift register
    output reg  [NUM_FEAT-1:0]    feat_decoded // one-hot decoded feat_cnt
);

    reg  [$clog2(SAR_CYCLES)-1:0]   sar_cnt;
    reg  [$clog2(NUM_FEAT)-1:0] feat_cnt;

    // sar_cnt and feat_cnt control logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sar_cnt  <= 0;
            feat_cnt <= 0;
            shift_reg <= {NUM_FEAT*N{1'b0}};
        end else
            if ((sar_cnt == SAR_CYCLES-1) && (feat_cnt == NUM_FEAT-1)) begin
                if (en) begin
                    // Restart both counters
                    sar_cnt  <= 0;
                    feat_cnt <= 0;
                end
                else begin
                    sar_cnt  <= sar_cnt;
                    feat_cnt <= feat_cnt;
                end
                // Shift register update
                shift_reg <= {shift_reg[(NUM_FEAT-2)*N-1:0], quant_feat};
            end else if (sar_cnt == SAR_CYCLES-1) begin
                sar_cnt  <= 0;
                feat_cnt <= feat_cnt + 1;
                // Shift register update
                shift_reg <= {shift_reg[(NUM_FEAT-2)*N-1:0], quant_feat};
            end else begin
                sar_cnt <= sar_cnt + 1;
            end
        end


    // One-hot decode feat_cnt
    always @(*) begin
        feat_decoded = {NUM_FEAT{1'b0}};
        feat_decoded[feat_cnt] = 1'b1;
    end

endmodule