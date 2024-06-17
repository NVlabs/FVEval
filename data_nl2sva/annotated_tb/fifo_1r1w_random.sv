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

reg [FIFO_DEPTH_log2:0]  rd_pending_ctr;            // this is the true counter for overflow/underflow checks
reg [FIFO_DEPTH_log2:0]  rand_packet_tracker;       // pending counter
wire [FIFO_DEPTH_log2:0] rand_packet_tracker_next;  // pending counter next
reg [DATA_WIDTH-1:0]     free_selected_data;        // randomly picked data
wire                     rand_pulse;                // random pulse which picks the packet
reg                      rand_pulse_seen;           // to make sure we do this check only once
wire                     fifo_empty;

assign rand_packet_tracker_next = (rand_packet_tracker == 'd0) && rand_pulse_seen ? 'd0 : rand_packet_tracker + (wr_push && !rand_pulse_seen) - rd_pop;
always @(posedge clk) begin
    if (!reset_) begin
        rand_pulse_seen <= 1'b0;
    end else if (rand_pulse && wr_push && !rand_pulse_seen) begin
        rand_pulse_seen <= 1'b1;
        free_selected_data <= wr_data;
    end else begin
        rand_pulse_seen <= rand_pulse_seen;
        free_selected_data <= free_selected_data;
    end
    if (!reset_) begin
        rd_pending_ctr <= 'd0;
    end else begin
        rd_pending_ctr <= rd_pending_ctr + wr_push - rd_pop;
    end
    if (!reset_) begin
        rand_packet_tracker <= 'd0;
    end else begin
        rand_packet_tracker <= rand_packet_tracker_next;
    end
end

assign fifo_full = rd_pending_ctr == (FIFO_DEPTH);
assign fifo_empty = rd_pending_ctr == 'd0;


endmodule