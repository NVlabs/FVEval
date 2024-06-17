module arbiter_sticky_lru_tb (
clk, reset_, req, gnt, gnt_id, busy, hold, cont_gnt);

    parameter       NUM_OF_CLIENTS = 6;

localparam tb_num_client_req_width = $clog2(NUM_OF_CLIENTS); 
localparam busy_latency_width = $clog2(NUM_OF_CLIENTS);
localparam cont_gnt_latency_width = $clog2(NUM_OF_CLIENTS);
localparam hold_latency_width = $clog2(NUM_OF_CLIENTS);

input clk;
input reset_;//clock and reset
input busy; //busy 
input hold;//hold
input [NUM_OF_CLIENTS-1 : 0] req;
input [NUM_OF_CLIENTS-1 : 0] gnt;//request and grant. grant is assumed to be one hot.
input [tb_num_client_req_width-1:0] gnt_id;
input cont_gnt; //same as cont_gnt from arbgen

wire tb_reset;
assign tb_reset = (reset_ == 1'b0);

wire [NUM_OF_CLIENTS-1 : 0] tb_req;
wire [NUM_OF_CLIENTS-1 : 0] tb_gnt;
wire [NUM_OF_CLIENTS-1 : 0] tb_req_for_starvation;
wire [NUM_OF_CLIENTS-1 : 0] tb_hold;

reg [NUM_OF_CLIENTS-1 : 0] sticky_req;
reg arbiter_in_sticky;

genvar a;
assign tb_req = req;
assign tb_gnt = gnt;
assign tb_req_for_starvation = req;
assign tb_hold = hold;

reg [NUM_OF_CLIENTS-1 : 0] last_gnt;
always @(posedge clk) begin
    if (!reset_) begin 
        last_gnt <= 0; 
   end else if (|tb_gnt && !cont_gnt) begin 
        last_gnt <= tb_gnt; 
    end
end

// sticky model    
always @(posedge clk) begin
    if (!reset_) begin 
        arbiter_in_sticky <= 1'b0;
        sticky_req <= 'd0;
    end else if (|(tb_gnt & tb_req & tb_hold)) begin 
        arbiter_in_sticky <= 1'b1;
        sticky_req <= tb_gnt & tb_req & tb_hold;
    end else if (|(tb_gnt & tb_req & ~tb_hold)) begin 
        arbiter_in_sticky <= 1'b0;
        sticky_req <= 'd0;
    end
end    

reg [tb_num_client_req_width:0] lru [NUM_OF_CLIENTS-1:0];
wire [tb_num_client_req_width:0] lru_gnt_num;
reg [tb_num_client_req_width:0] temp_lru_gnt_num_reg;

assign lru_gnt_num = temp_lru_gnt_num_reg;

always_comb begin
    temp_lru_gnt_num_reg = 'd0;
    for (int k=0; k<NUM_OF_CLIENTS; k++) begin
        if (tb_gnt[k]) begin
            temp_lru_gnt_num_reg = lru[k];
        end
    end
end
genvar i;
reg [NUM_OF_CLIENTS-1 : 0] req_seen_flag;
for (i = 0; i < NUM_OF_CLIENTS; i++) begin
    always @(posedge clk) begin
        if (!reset_) begin req_seen_flag[i] <= 0; end
        else if (tb_req_for_starvation[i] && tb_gnt[i]) begin req_seen_flag[i] <= 0; end
        else if (tb_req_for_starvation[i]) begin req_seen_flag[i] <= 1; end
        else if (tb_gnt[i]) begin req_seen_flag[i] <= 0; end
    end
end


endmodule