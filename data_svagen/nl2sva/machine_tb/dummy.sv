module dummy (
clk, reset_, 
sig_A,
sig_B,
sig_C,
sig_D,
sig_E
);

input clk;
input reset_;//clock and reset
input sig_A;  
input sig_B;
input sig_C;
input sig_D;  
input sig_E;

logic sig_F;
logic sig_G;
logic sig_H;
logic sig_I;
logic sig_J;

wire tb_reset;
assign tb_reset = (reset_ == 1'b0);

assign sig_F = sig_A;
assign sig_G = sig_B;
assign sig_H = sig_C;
assign sig_I = sig_D;
assign sig_J = sig_E;


endmodule