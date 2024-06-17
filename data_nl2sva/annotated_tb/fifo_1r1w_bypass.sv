module fifo_with_bypass_tb (
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

reg [DATA_WIDTH-1:0]             fifo_array [FIFO_DEPTH-1:0]; //fifo array - shift register
reg [FIFO_DEPTH_log2-1:0]        fifo_rd_ptr;                 //fifo array - rd_ptr
wire                             actual_fifo_pop;             // actual pop == pop
reg                              fifo_empty;                  // fifo empty
wire [DATA_WIDTH-1:0]            fifo_out_data;               // dout

always @(posedge clk) begin
    if (!reset_) fifo_array[0] <= 'd0;
    else if (wr_push) begin
        fifo_array[0] <= wr_data;
    end else fifo_array[0] <= fifo_array[0];
end
for (genvar i = 1; i < FIFO_DEPTH; i++ ) begin : loop_id
    always @(posedge clk) begin
        if (!reset_) fifo_array[i] <= 'd0;
        else if (wr_push) begin
            fifo_array[i] <= fifo_array[i-1];
        end else fifo_array[i] <= fifo_array[i];
    end
end

always @(posedge clk) begin
    if (!reset_) begin
        fifo_rd_ptr <= 'd0;
    end else if (wr_push && fifo_empty)  begin
        fifo_rd_ptr <= 'd0;
    end else if (rd_pop && !fifo_empty && (fifo_rd_ptr == 'd0)) begin
        fifo_rd_ptr <= 'd0;
    end else begin
        fifo_rd_ptr <= fifo_rd_ptr + wr_push - rd_pop;
    end
    if (!reset_) begin
        fifo_empty <= 'd1;
    end else if (rd_pop && !fifo_empty && (fifo_rd_ptr == 'd0) && !wr_push) begin
        fifo_empty <= 'd1;
    end else if ((fifo_rd_ptr != 'd0) || wr_push && !rd_pop) begin
        fifo_empty <= 'd0;
    end
end
assign fifo_full = (fifo_rd_ptr == (FIFO_DEPTH - 1)) && !fifo_empty;
assign fifo_out_data = fifo_array[fifo_rd_ptr];


endmodule