module fsm_transition_tb (
clk, 
reset_, 
fsm_state,
fsm_sequence
);
    parameter fsm_width = 2; //actual width of the states in the RTL
    parameter num_of_states=2; //number of states provided in the fsm_sequence
    parameter num_of_times_initial_state_repeats=1; //Number of times the initial state of the "fsm_sequence" is repeated in the "fsm_sequence"  

input clk;
input reset_;
input [fsm_width-1:0]fsm_state;
input [fsm_width*num_of_states-1:0]fsm_sequence;

wire tb_reset;
assign tb_reset = (reset_ == 1'b0);

//Proper sequencing of the states
wire [fsm_width-1:0] tb_fsm_sequence[num_of_states-1:0]; 

//match the current "fsm_state" with the states provided in the "fsm_sequence"
wire [num_of_states-1:0]match_tracker[num_of_times_initial_state_repeats-1:0]; 
reg [num_of_states-1:0]match_tracker_d1[num_of_times_initial_state_repeats-1:0]; 

//match the current "fsm_state" with the states provided in the "fsm_sequence" when checking for individual states
wire [num_of_states-1:0]ind_state_match_tracker[num_of_times_initial_state_repeats-1:0]; 
reg [num_of_states-1:0]ind_state_match_tracker_d1[num_of_times_initial_state_repeats-1:0]; 

//Track all the states of the "fsm_sequence"
reg [num_of_states-1:0]state_tracker[num_of_times_initial_state_repeats-1:0]; 

reg [fsm_width-1:0] fsm_state_d1;
reg tb_reset_d1;
wire [fsm_width-1:0] tb_random_state;
wire [$clog2(num_of_times_initial_state_repeats):0]tb_sequence_seen;


//storing the states of the fsm_sequence in the correct order
for (genvar i=num_of_states-1; i >=0; i--) begin : storing_of_fsm_states
    assign  tb_fsm_sequence[num_of_states-1-i] = fsm_sequence[(fsm_width*(i+1))-1 : fsm_width*i];
end

//Delayed versions of fsm_state and tb_reset
always @(posedge clk) begin
    if (!reset_) begin
        fsm_state_d1 <= 'd0;
        tb_reset_d1 <= 1;
    end else begin
        fsm_state_d1 <= fsm_state;
        tb_reset_d1 <= tb_reset;  
    end
end

for (genvar n=0; n<num_of_times_initial_state_repeats; n++) begin : matching_of_states_as_per_initial_state_repeat
    if (n==0) begin : matching_of_states_for_certain_cases
        for (genvar i=0; i<num_of_states; i++) begin : matching_of_states_as_per_num_of_states
            if (i==0) begin : matching_of_states_for_first_state 
                assign ind_state_match_tracker[n][0] = (fsm_state == tb_fsm_sequence[0]);
            end else begin : matching_of_states_for_other_states 
                assign ind_state_match_tracker[n][i] = (fsm_state == tb_fsm_sequence[i]);
            end
        end 
    end else begin : matching_of_states_for_other_cases
        for (genvar i=0; i<num_of_states; i++) begin : matching_of_states_as_per_num_of_states
            if (i==0) begin : matching_of_states_for_first_state 
            assign ind_state_match_tracker[n][0] = ((fsm_state != fsm_state_d1) && !tb_reset_d1) 
                                            ? (!(|state_tracker[n]) && (|state_tracker[n-1]) && (fsm_state == tb_fsm_sequence[0])) 
                                            : ind_state_match_tracker[n][0] ;
            end else begin : matching_of_states_for_other_states
            assign ind_state_match_tracker[n][i] = ((fsm_state != fsm_state_d1) && !tb_reset_d1) 
                                            ? (!state_tracker[n][i] && state_tracker[n][i-1] && (fsm_state == tb_fsm_sequence[i])) 
                                            : ind_state_match_tracker[n][i] ;
            end
        end 
    end
end

reg [$clog2(num_of_times_initial_state_repeats):0] j;

always @(posedge clk) begin
    if (!reset_) begin
        for (j=0; j< num_of_times_initial_state_repeats; j++) begin
            state_tracker[j] <= 'd0;
            match_tracker_d1[j] <= 'd0;
            ind_state_match_tracker_d1[j] <= 'd0;
        end
    end else begin
        for (j=0; j< num_of_times_initial_state_repeats; j++) begin
            match_tracker_d1[j] <= match_tracker[j];
            ind_state_match_tracker_d1[j] <= ind_state_match_tracker[j];
            if (j==0) 
            state_tracker[j] <= (((state_tracker[j]==(match_tracker[j]-1'b1)) || 
                                    (state_tracker[j] == ((match_tracker[j]-1'b1) | match_tracker[j]))) && 
                                    (|match_tracker[j] != 'd0)) 
                                        ? state_tracker[j]|match_tracker[j] 
                                        : ((((|match_tracker[j]) == 0) && (fsm_state == tb_fsm_sequence[0])) 
                                            ? 'd1 
                                            : 'd0
                                        );
            else 
            state_tracker[j] <= (((state_tracker[j]==(match_tracker[j]-1'b1)) || 
                                    (state_tracker[j] == ((match_tracker[j]-1'b1) | match_tracker[j]))) && 
                                    (|match_tracker[j] != 'd0)) 
                                    ? state_tracker[j]|match_tracker[j] 
                                    : 'd0;
        end
    end
end


endmodule