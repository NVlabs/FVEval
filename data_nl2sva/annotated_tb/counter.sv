
module counter_tb (
clk, reset_, count, incr_vld, incr_value, decr_vld, decr_value, jump_vld, jump_value
);

    parameter width = 1;
    parameter min = 0;
    parameter [width:0] max = ((1<<width)-1);

input clk;
input reset_;
input [width-1:0] count;
input             incr_vld;
input [width-1:0] incr_value;
input             decr_vld;
input [width-1:0] decr_value;
input             jump_vld;
input [width-1:0] jump_value;


wire tb_reset;
assign tb_reset = (reset_ == 1'b0);


wire [width-1:0] eff_incr = {width{incr_vld}} & incr_value;
wire [width-1:0] eff_decr = {width{decr_vld}} & decr_value;
wire [width-1:0] net_incr = (eff_incr>eff_decr) ? (eff_incr-eff_decr) : {width{1'b0}};
wire [width-1:0] net_decr = (eff_decr>eff_incr) ? (eff_decr-eff_incr) : {width{1'b0}};


reg [width-1:0] eff_incr_d1;
reg [width-1:0] eff_decr_d1;
reg [width-1:0] net_incr_d1;
reg [width-1:0] net_decr_d1;

reg [width-1:0] count_d1;
reg             incr_vld_d1;
reg [width-1:0] incr_value_d1;
reg             decr_vld_d1;
reg [width-1:0] decr_value_d1;
reg             jump_vld_d1;
reg [width-1:0] jump_value_d1;

reg [width:0]     count_d1_next;
reg [width:0]     count_d1_next_p;
reg [width:0]     count_d1_next_m;

reg tb_reset_d1;
reg tb_reset_d2;
reg tb_reset_1_cycle_pulse_shadow;

always @(posedge clk) begin
    if (!reset_) begin 
        incr_vld_d1 <= 1'b0;
        decr_vld_d1 <= 1'b0;
        jump_vld_d1 <= 1'b0;
        eff_incr_d1 <= {width{1'b0}};
        eff_decr_d1 <= {width{1'b0}};
        net_incr_d1 <= {width{1'b0}};
        net_decr_d1 <= {width{1'b0}};
    end else begin
        eff_incr_d1 <= eff_incr;
        eff_decr_d1 <= eff_decr;
        net_incr_d1 <= net_incr;
        net_decr_d1 <= net_decr;
        incr_vld_d1 <= incr_vld;
        decr_vld_d1 <= decr_vld;
        jump_vld_d1 <= jump_vld;         
    end
end

always @(posedge clk) begin
        count_d1 <= count;
        incr_value_d1 <= incr_value;
        decr_value_d1 <= decr_value;
        jump_value_d1 <= jump_value;
        tb_reset_d1 <= tb_reset;          
        tb_reset_d2 <= tb_reset_d1;         
end
always @(posedge clk) begin
    count_d1_next <= count_d1 + net_incr_d1 - net_decr_d1;
    count_d1_next_m <= count_d1 - net_decr_d1;
    count_d1_next_p <= count_d1 + net_incr_d1;
    tb_reset_1_cycle_pulse_shadow <=!tb_reset_d2 && tb_reset_d1 && !tb_reset;
end

endmodule