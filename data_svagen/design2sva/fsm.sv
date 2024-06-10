
`define WIDTH 32
module fsm(
    clk,
    reset_,
    in_A,
    in_B,
    fsm_out
);
    parameter WIDTH = `WIDTH;
    parameter FSM_WIDTH = 3;

    parameter S0 = 3'b000;
    parameter S1 = 3'b001;
    parameter S2 = 3'b010;
    parameter S3 = 3'b011;
    parameter S4 = 3'b100;
    parameter S5 = 3'b101;
    parameter S6 = 3'b110;
    parameter S7 = 3'b111;

    input clk;
    input reset_;
    input [WIDTH-1:0] in_A;
    input [WIDTH-1:0] in_B;
    output reg [FSM_WIDTH-1:0] fsm_out;
    reg [FSM_WIDTH-1:0] state, next_state;
    always_ff @(posedge clk or negedge reset_) begin
        if (!reset_) begin
            state <= S0;
        end else begin
            state <= next_state;
        end
    end
    always_comb begin
        case(state)
            S0: begin
                if (~^((in_A || in_B))) begin
                    next_state = S1;
                end
                else if (((in_A ^ in_B) != 'd1)) begin
                    next_state = S2;
                end
                else if (((in_A != 'd1) != 'd0)) begin
                    next_state = S6;
                end
                else begin
                    next_state = S7;
                end
            end
            S1: begin
                if ((in_A && (in_B || in_A))) begin
                    next_state = S2;
                end
                else if ((in_B ^ &(in_A))) begin
                    next_state = S7;
                end
                else begin
                    next_state = S5;
                end
            end
            S2: begin
                next_state = S0;
            end
            S3: begin
                if (((in_B || in_A) && in_B)) begin
                    next_state = S0;
                end
                else if (((in_A || in_B) == 'd0)) begin
                    next_state = S5;
                end
                else begin
                    next_state = S7;
                end
            end
            S4: begin
                if ((in_A && (in_B ^ in_A))) begin
                    next_state = S3;
                end
                else if (((in_B <= 'd0) && (in_A != 'd1))) begin
                    next_state = S6;
                end
                else if (((in_B <= 'd0) ^ in_A)) begin
                    next_state = S7;
                end
                else begin
                    next_state = S2;
                end
            end
            S5: begin
                next_state = S4;
            end
            S6: begin
                if (((in_B == in_A) != 'd1)) begin
                    next_state = S1;
                end
                else if ((in_A && in_B)) begin
                    next_state = S5;
                end
                else begin
                    next_state = S2;
                end
            end
            S7: begin
                if ((in_A != 'd0)) begin
                    next_state = S2;
                end
                else begin
                    next_state = S0;
                end
            end
        endcase
    end
endmodule