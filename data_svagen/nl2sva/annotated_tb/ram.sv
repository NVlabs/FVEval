module ram_tb (clk, reset_, we, wa, wd, re, ra, rd, stall);

  
    parameter   addr_width = 1;
    parameter   data_width = 1;
    parameter   entries = 1;
    parameter   wr_latency=0;                   //checks max latency between a write to read. wr_latency=0 means disable this check


parameter tb_max_length_width = $clog2(wr_latency);

input    clk,reset_;
input    we,re;
input [addr_width-1:0] wa,ra;
input [data_width-1:0] wd,rd;
input    stall;

wire tb_reset;
assign tb_reset = (reset_ == 1'b0);


wire [addr_width-1:0] symbolic_constant_a;
asum_tb_ram__constant_addr_a : assume property (@(posedge clk) disable iff (tb_reset)
  ($stable(symbolic_constant_a))
);

reg addr_ar_seen;
reg addr_aw_seen;
reg addr_first_aw_seen;
reg [tb_max_length_width-1:0]  latency_cnt;       //max latency is wr_latency       

always @(posedge clk) begin
  if(!reset_) begin
    addr_aw_seen <= 1'd0;
  end else if (we && (wa == symbolic_constant_a)) begin
    addr_aw_seen <= 1'd1;    
  end else if (re && (ra == symbolic_constant_a)) begin
    addr_aw_seen <= 1'd0;    
  end
end
always @(posedge clk) begin
  if(!reset_) begin
    addr_ar_seen <= 1'd0;
  end else if (we && (re == symbolic_constant_a)) begin
    addr_ar_seen <= 1'd1;    
  end else if (re && (we == symbolic_constant_a)) begin
    addr_ar_seen <= 1'd0;    
  end
end

always @(posedge clk) begin
  if(!reset_) begin
    addr_first_aw_seen <= 1'd0;
  end else if (we && (wa == symbolic_constant_a)) begin
    addr_first_aw_seen <= 1'd1;
  end
end

always @(posedge clk) begin
    latency_cnt <= (we && wa == symbolic_constant_a) ? 16'd1 : (re && ra == symbolic_constant_a) ? 16'd0 : (latency_cnt>0) ? (latency_cnt+(!stall)) : latency_cnt;
end


endmodule