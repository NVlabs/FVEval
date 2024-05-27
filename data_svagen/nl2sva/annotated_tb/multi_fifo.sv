module multi_fifo_tb (
clk,
reset_,
wr_vld,
wr_data,
wr_ready,
rd_vld,
rd_data,
rd_ready
);

    parameter   max_fifo_entries = 4;
    parameter   wr_port = 1;
    parameter   rd_port = 1;
    parameter   data_width = 1;

    localparam fifo_entry_cntr = $clog2(max_fifo_entries +1 );
    localparam wr_port_log2 = $clog2(wr_port +1 );
    localparam rand_bit_selector = $clog2(data_width);


input clk;
input reset_;
input [(wr_port)-1:0] wr_vld;
input [(data_width * wr_port)-1:0] wr_data;
input [(wr_port)-1:0] wr_ready;
input [(rd_port)-1:0] rd_vld;
input [(data_width * rd_port)-1:0] rd_data;
input [(rd_port)-1:0] rd_ready;

wire tb_reset;
assign tb_reset = (reset_ == 1'b0);

reg [(fifo_entry_cntr-1):0] tb_wr_ptr;
reg [(fifo_entry_cntr-1):0] tb_rd_ptr;

reg [(wr_port - 1):0] tb_wr_data_for_fv;
reg [(rd_port - 1):0] tb_rd_data_for_fv;

reg [data_width-1:0]tb_wr_data_for_sim[(wr_port - 1):0];
reg [data_width-1:0]tb_rd_data_for_sim[(rd_port - 1):0];



wire [rd_port-1:0] tb_rd_data_is_correct_in_sim_err;
wire tb_fifo_should_not_underflow_err;
wire tb_filled_cnt_less_or_equal_max_entries_err;
wire tb_fifo_should_not_overflow_err;
wire [wr_port-1:0] tb_wr_vld_is_contiguous_err;
wire [wr_port-1:0] tb_wr_ready_is_contiguous_err;
wire [rd_port-1:0] tb_rd_vld_is_contiguous_err;
wire [rd_port-1:0] tb_rd_ready_is_contiguous_err;


wire [rand_bit_selector-1:0] tb_random_1_bit_data_selector[wr_port-1:0]; 
generate
for (genvar i=0; i<wr_port; i++) begin : abstract_write_data_for_fv
    always_comb
    begin
        tb_wr_data_for_fv[i] = wr_data[(i*data_width) + tb_random_1_bit_data_selector[i]];
    end
end
endgenerate

generate
for (genvar i=0; i<rd_port; i++) begin : abstract_read_data_for_fv
    always_comb
    begin
        tb_rd_data_for_fv[i] = rd_data[(i * data_width) + tb_random_1_bit_data_selector[i]];
    end
end
endgenerate

generate
for (genvar i=0; i<wr_port; i++) begin : abstract_write_data_for_sim
    always_comb
    begin
        tb_wr_data_for_sim[i] = wr_data[((i+1)*data_width)-1 : (i*data_width)];
    end
end
endgenerate

generate
for (genvar i=0; i<rd_port; i++) begin : abstract_read_data_for_sim
    always_comb
    begin
        tb_rd_data_for_sim[i] = rd_data[((i+1)*data_width)-1 : (i*data_width)];
    end
end
endgenerate

wire [(fifo_entry_cntr-1):0] tb_pop_cnt  = $countones(rd_vld & rd_ready);
wire [(fifo_entry_cntr-1):0] tb_push_cnt = $countones(wr_vld & wr_ready);
wire [(fifo_entry_cntr-1):0] tb_fifo_filled_cnt = tb_wr_ptr - tb_rd_ptr;

wire fifo_will_overflow   = (tb_fifo_filled_cnt + tb_push_cnt - tb_pop_cnt) > max_fifo_entries;

always @(posedge clk)
begin
    if (!reset_) begin
        tb_wr_ptr <= 0;
        tb_rd_ptr <= 0;
    end else begin
        tb_wr_ptr <= tb_wr_ptr + tb_push_cnt; // wr_ptr always move when write
        if ((tb_fifo_filled_cnt + tb_push_cnt - tb_pop_cnt) > max_fifo_entries) begin
        tb_rd_ptr <= tb_wr_ptr + tb_push_cnt - max_fifo_entries;  // overflow will push rd_ptr
        end else begin
        tb_rd_ptr <= tb_rd_ptr + tb_pop_cnt;  // when not overflow, rd_ptr moves when fifo read
        end
    end
end

reg registered_data;
reg [(fifo_entry_cntr-1):0] registered_ptr;
reg [(wr_port)-1:0] registered_data_next;
reg [(wr_port)-1:0] registered_data_update;

reg [rand_bit_selector-1:0] tb_random_bit_selector_next[wr_port-1:0];
reg [rand_bit_selector-1:0] tb_random_1_bit_data_selector_registered;

always @(posedge clk)
begin
    if (!reset_) begin
        registered_ptr <= 'x;                 // registered_ptr is random during reset
    end else begin
        registered_ptr <= registered_ptr;     // else it retiains its value
    end
end

// when wr_ptr == registered_ptr, register data
generate
for (genvar i=0; i<wr_port; i++) begin:register_write_data
    always_comb
    begin
        registered_data_next[i] = 0;
        registered_data_update[i] = 0;
        tb_random_bit_selector_next[i] = 0;
        if (wr_vld[i] && ((tb_wr_ptr + i[(fifo_entry_cntr-1):0]) == registered_ptr)) begin
            registered_data_update[i] = 1;
            registered_data_next[i] = tb_wr_data_for_fv[i];
            tb_random_bit_selector_next[i] = tb_random_1_bit_data_selector[i];
        end
    end
end
endgenerate

// register selector
reg [wr_port_log2-1:0] wr_index;
reg [rand_bit_selector-1:0] tb_random_bit_next;
always_comb begin
    tb_random_bit_next = 0;
    for (wr_index=0; wr_index < wr_port; wr_index++) begin
        if (registered_data_update[wr_index]) begin
            tb_random_bit_next = tb_random_bit_selector_next[wr_index];
        end
    end
end

// register data
always @ (posedge clk)
begin
    if (|registered_data_update) begin
        registered_data <= |registered_data_next;
        tb_random_1_bit_data_selector_registered <= tb_random_bit_next;
    end
end

reg [data_width-1:0]fifo_data_tracker [(max_fifo_entries+wr_port-1):0];
reg [data_width-1:0]fifo_data_tracker_next [(max_fifo_entries+wr_port-1):0];
reg [(fifo_entry_cntr-1):0] tb_shift;

reg [(fifo_entry_cntr-1):0] n;
always @ (posedge clk)
begin
if (tb_shift == 0) 
    for (n=0; n < max_fifo_entries; n++) begin
        fifo_data_tracker[n] <= fifo_data_tracker_next[n];
    end
else if (tb_shift > (max_fifo_entries+wr_port-1)) begin
  for (n=0; n < max_fifo_entries;n++) fifo_data_tracker[n][data_width-1:0] <= 0;
end else begin
    for (n=0; n<max_fifo_entries; n++) begin 
        if ((tb_shift+n) <= (max_fifo_entries+wr_port-1)) begin
          fifo_data_tracker[n] <= fifo_data_tracker_next[n+tb_shift];
        end
    end  
end
end     

generate
for (genvar i=0; i<(max_fifo_entries+wr_port); i++) begin:assign_data_tracker_next
    always_comb
    begin
        if (i<tb_fifo_filled_cnt) begin
        fifo_data_tracker_next[i] = fifo_data_tracker[i];           // assign kept data
        end
        else if (i <tb_fifo_filled_cnt + wr_port) begin
        fifo_data_tracker_next[i] = tb_wr_data_for_sim[i-tb_fifo_filled_cnt];  // assign new written data
        end
        // else dont care
    end
end
endgenerate

always_comb
begin
    if ((tb_fifo_filled_cnt + tb_push_cnt - tb_pop_cnt) > max_fifo_entries) begin
        tb_shift = tb_push_cnt + tb_fifo_filled_cnt - max_fifo_entries;
    end else begin
        tb_shift = tb_pop_cnt;
    end
end

endmodule