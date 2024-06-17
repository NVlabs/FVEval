module fifo_tb (
     clk,
     reset_,
     wr_vld,
     wr_data,
     wr_ready,
     rd_vld,
     rd_data,
     rd_ready
   );
  
    parameter   FIFO_DEPTH                              = 4;
    parameter   DATA_WIDTH                              = 1;

localparam FIFO_DEPTH_log2 = $clog2(FIFO_DEPTH); 
localparam DATA_WIDTH_log2 = $clog2(DATA_WIDTH); 

    input                   clk;
    input                   reset_;
    input                   wr_vld;
    input  [DATA_WIDTH-1:0] wr_data;
    input                   wr_ready;
    input                   rd_vld;
    input  [DATA_WIDTH-1:0] rd_data;
    input                   rd_ready;

wire wr_push;
wire rd_pop;

wire tb_reset;
assign tb_reset = (reset_ == 1'b0);

wire fifo_full;
assign wr_push = wr_vld && wr_ready;
assign rd_pop = rd_vld && rd_ready; 

wire colour_1bit;                          // pattern
wire [DATA_WIDTH_log2-1:0] rand_bit_sel;   // do this on one bit of fifo only
asum_tb_inorder__rand_stable: assume property (@(posedge clk) disable iff (!reset_)
$stable(colour_1bit) && $stable(rand_bit_sel) && (rand_bit_sel < DATA_WIDTH)
);
reg [FIFO_DEPTH_log2:0] rd_pending_ctr; 
reg [1:0] input_fsm; //0 - init, 1- 1st seen, 2 - 2nd seen, 3 - dead
reg [2:0] output_fsm; //0 - init, 1- 1st seen, 2 - nth seen, 3 - dead
wire fifo_empty;
always @(posedge clk) begin
    // ---- input fsm ----
    if (!reset_) begin
        input_fsm <= 2'd0;
    end else if ((input_fsm == 2'd0) && wr_push && (wr_data[rand_bit_sel] == colour_1bit)) begin //init state -> seen 1st state
        input_fsm <= 2'd1; 
    end else if ((input_fsm == 2'd1) && wr_push && (wr_data[rand_bit_sel] == colour_1bit)) begin //seen 1st state -> seen 2nd state
        input_fsm <= 2'd2; 
    end else if ((input_fsm == 2'd1) && wr_push && (wr_data[rand_bit_sel] != colour_1bit)) begin //seen 1st state -> dead state
        input_fsm <= 2'd3; 
    end else if ((input_fsm == 2'd2) && wr_push && (wr_data[rand_bit_sel] == colour_1bit)) begin //seen 2nd state -> dead state
        input_fsm <= 2'd3; 
    end else begin
        input_fsm <= input_fsm; 
    end
    // ---- output fsm ----
    if (!reset_) begin
        output_fsm <= 2'd0;
    end else if ((output_fsm == 2'd0) && rd_pop && (rd_data[rand_bit_sel] == colour_1bit)) begin //init state -> seen 1st state
        output_fsm <= 2'd1; 
    end else if ((output_fsm == 2'd1) && rd_pop && (rd_data[rand_bit_sel] == colour_1bit)) begin //seen 1st state -> seen 2nd state
        output_fsm <= 2'd2; 
    end else if ((output_fsm == 2'd1) && rd_pop && (rd_data[rand_bit_sel] != colour_1bit)) begin //seen 1st state -> dead state
        output_fsm <= 2'd3; 
    end else if ((output_fsm == 2'd2) && rd_pop && (rd_data[rand_bit_sel] == colour_1bit)) begin //seen 2nd state -> dead state
        output_fsm <= 2'd3; 
    end else begin
        output_fsm <= output_fsm; 
    end
    // ---- pending counter ----
    if (!reset_) begin
        rd_pending_ctr <= 'd0;
    end else begin
        rd_pending_ctr <= rd_pending_ctr + wr_push - rd_pop;
    end
end
assign fifo_full = rd_pending_ctr == (FIFO_DEPTH);
assign fifo_empty = rd_pending_ctr == 'd0;

endmodule