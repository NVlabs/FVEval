import argparse
from dataclasses import asdict
import pathlib
import random

import pandas as pd
import numpy as np
import networkx as nx

from fv_eval.data import InputData

def decimal_to_binary(decimal: int, num_bits:int):
    return f"{num_bits}'b{decimal:0{num_bits}b}"

def generate_module_header(is_design: bool, num_inputs: int, num_nodes: int, width: int = 32, fsm_sequence_len: int = 0, fsm_sequence_repeats: int = 0):
    num_bits = int(np.ceil(np.log2(num_nodes)))
    input_ports = ",\n".join([f"    in_{chr(65+i)}" for i in range(num_inputs)])
    input_port_def = ";\n".join([f"    input [WIDTH-1:0] in_{chr(65+i)}" for i in range(num_inputs)])
    fsm_state_def = ";\n".join([f"    parameter S{i} = {decimal_to_binary(i, num_bits)}" for i in range(num_nodes)])
    additional_local_params = "" if is_design else f"""
    localparam num_of_states={fsm_sequence_len};
    localparam num_of_times_initial_state_repeats={fsm_sequence_repeats};
"""
    module_name = "fsm" if is_design else "fsm_tb"
    port_direction = "output" if is_design else "input"
    return f"""
`define WIDTH {width}
module {module_name}(
    clk,
    reset_,
{input_ports},
    fsm_out
);
    parameter WIDTH = `WIDTH;
    parameter FSM_WIDTH = {num_bits};
{additional_local_params}
{fsm_state_def};

    input clk;
    input reset_;
{input_port_def};
    {port_direction} reg [FSM_WIDTH-1:0] fsm_out;
"""
    
   
def generate_fsm_module(num_inputs: int, num_nodes: int, num_edges: int, max_recursive_dapth: int = 2, width: int = 32):
    G = generate_random_fsm(num_inputs=num_inputs, num_nodes=num_nodes, num_edges=num_edges, max_recursive_dapth=max_recursive_dapth)
    # Generate the module prefix
    
    module_header = generate_module_header(is_design=True, num_nodes=num_nodes, num_inputs=num_inputs, width=width)
    # Generate the module body
    module_body = "    reg [FSM_WIDTH-1:0] state, next_state;\n"
    module_body += "    always_ff @(posedge clk or negedge reset_) begin\n"
    module_body += "        if (!reset_) begin\n"
    module_body += f"            state <= S0;\n"
    module_body += "        end else begin\n"
    module_body += "            state <= next_state;\n"
    module_body += "        end\n"
    module_body += "    end\n"
    module_body += "    always_comb begin\n"
    module_body += "        case(state)\n"
    for src in range(num_nodes):
        module_body += f"            S{src}: begin\n"
        out_vertices = []
        for i, j in G.edges():
            if src == i:
                out_vertices.append(j)
        if len(out_vertices) == 1:
            dst = out_vertices[0]
            module_body += f"                next_state = S{out_vertices[0]};\n"
        elif len(out_vertices) == 2:
            module_body += f"                if ({G[src][out_vertices[0]]['operation']}) begin\n"
            module_body += f"                    next_state = S{out_vertices[0]};\n"
            module_body += "                end\n"
            module_body += f"                else begin\n"
            module_body += f"                    next_state = S{out_vertices[1]};\n"
            module_body += "                end\n"
        else:
            for i, dst in enumerate(out_vertices):
                if i==0:
                    module_body += f"                if ({G[src][dst]['operation']}) begin\n"
                    module_body += f"                    next_state = S{dst};\n"
                    module_body += "                end\n"
                elif i == len(out_vertices)-1:
                    module_body += f"                else begin\n"
                    module_body += f"                    next_state = S{dst};\n"
                    module_body += "                end\n"
                else:
                    module_body += f"                else if ({G[src][dst]['operation']}) begin\n"
                    module_body += f"                    next_state = S{dst};\n"
                    module_body += "                end\n"
        module_body += f"            end\n"    
    module_body += "        endcase\n"
    module_body += "    end\n"
    # Generate the module suffix
    module_suffix = "endmodule"
    return module_header + module_body + module_suffix, G

def generate_fsm_tb_module(G: nx.DiGraph, num_inputs: int, num_nodes: int, width: int = 32, max_fsm_sequence_len: int = 0):
    fsm_sequence = generate_fsm_sequence(G, max_fsm_sequence_len)
    fsm_sequence_len = len(fsm_sequence)
    # count number of S0 in sequence
    fsm_sequence_repeats = fsm_sequence.count("S0")

    # Generate the module prefix
    module_header = generate_module_header(is_design=False, 
                                           num_nodes=num_nodes, 
                                           num_inputs=num_inputs, 
                                           width=width,
                                           fsm_sequence_len=fsm_sequence_len,
                                           fsm_sequence_repeats=fsm_sequence_repeats
                                           )
    # Generate the module body
    module_body = f"""
    wire tb_reset;
    assign tb_reset = (reset_ == 1'b0);
    wire [FSM_WIDTH-1:0] tb_fsm_sequence[num_of_states-1:0]; 
    assign tb_fsm_sequence = {{S0, {", ".join(fsm_sequence[1:])}}};

    wire [num_of_states-1:0]match_tracker[num_of_times_initial_state_repeats-1:0]; 
    reg [num_of_states-1:0]match_tracker_d1[num_of_times_initial_state_repeats-1:0]; 

    reg [num_of_states-1:0]state_tracker[num_of_times_initial_state_repeats-1:0]; 

    reg [FSM_WIDTH-1:0] fsm_out_d1;
    reg tb_reset_d1;
    wire [FSM_WIDTH-1:0] tb_random_state;
    wire [$clog2(num_of_times_initial_state_repeats):0]tb_sequence_seen;


    //Delayed versions of fsm_out and tb_reset
    always @(posedge clk) begin
        if (!reset_) begin
            fsm_out_d1 <= 'd0;
            tb_reset_d1 <= 1;
        end else begin
            fsm_out_d1 <= fsm_out;
            tb_reset_d1 <= tb_reset;  
        end
    end

    for (genvar n=0; n<num_of_times_initial_state_repeats; n++) begin : matching_of_states_as_per_initial_state_repeat
        if (n==0) begin : matching_of_states_for_certain_cases
            for (genvar i=0; i<num_of_states; i++) begin : matching_of_states_as_per_num_of_states
                if (i==0) begin : matching_of_states_for_first_state 
                    assign match_tracker[n][0] = (fsm_out == tb_fsm_sequence[0]);
                end else begin : matching_of_states_for_other_states 
                    assign match_tracker[n][i] = (fsm_out == tb_fsm_sequence[i]);
                end
            end 
        end else begin : matching_of_states_for_other_cases
            for (genvar i=0; i<num_of_states; i++) begin : matching_of_states_as_per_num_of_states
                if (i==0) begin : matching_of_states_for_first_state 
                assign match_tracker[n][0] = ((fsm_out != fsm_out_d1) && !tb_reset_d1) 
                                                ? (!(|state_tracker[n]) && (|state_tracker[n-1]) && (fsm_out == tb_fsm_sequence[0])) 
                                                : match_tracker[n][0] ;
                end else begin : matching_of_states_for_other_states
                assign match_tracker[n][i] = ((fsm_out != fsm_out_d1) && !tb_reset_d1) 
                                                ? (!state_tracker[n][i] && state_tracker[n][i-1] && (fsm_out == tb_fsm_sequence[i])) 
                                                : match_tracker[n][i] ;
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
            end
        end else begin
            for (j=0; j< num_of_times_initial_state_repeats; j++) begin
                match_tracker_d1[j] <= match_tracker[j];
                if (j==0) 
                state_tracker[j] <= (((state_tracker[j]==(match_tracker[j]-1'b1)) || 
                                        (state_tracker[j] == ((match_tracker[j]-1'b1) | match_tracker[j]))) && 
                                        (|match_tracker[j] != 'd0)) 
                                            ? state_tracker[j]|match_tracker[j] 
                                            : ((((|match_tracker[j]) == 0) && (fsm_out == tb_fsm_sequence[0])) 
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

    for (genvar n=0; n<num_of_times_initial_state_repeats; n++) begin : fsm_sequence_seen
        assign tb_sequence_seen[n] = state_tracker[n][num_of_states-1];
    end

    reg check_state_legal_precondition;
    always @(posedge clk) begin
        if (!reset_) begin
            check_state_legal_precondition <= 1'b0;
        end else begin
            check_state_legal_precondition <= fsm_out == tb_fsm_sequence[0];
        end
    end

    target: assert property (@(posedge clk) disable iff (tb_reset)
        (|tb_sequence_seen) !== 1'b1     
    );
"""

    # Generate the module suffix
    module_suffix = "\nendmodule"
    module_suffix += f"""
bind fsm fsm_tb #(
    .WIDTH(WIDTH)
) fsm_tb_inst (.*);
    """
    return module_header + module_body + module_suffix

def generate_random_digraph(num_nodes: int, num_edges: int):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    
    # create edges such that each node has at least one outgoing edge
    # total number of edges is num_nodes + num_edges
    # ensure that graph is connected and there are no self edges
    for i in range(num_nodes):
        node_list = list(range(num_nodes))
        node_list.remove(i)
        j = random.choice(node_list)
        G.add_edge(j, i)

    # add random edges
    for i in range(num_edges - num_nodes):
        node_list = list(range(num_nodes))
        src = random.choice(node_list)
        node_list.remove(src)
        dst = random.choice(node_list)
        G.add_edge(src, dst)


    # ensure that there is an outedge from state 0
    if len(list(G.successors(0))) == 0:
        node_list = list(range(num_nodes))
        node_list.remove(0)
        dst = random.choice(node_list)
        G.add_edge(0, dst)
    return G


def generate_fsm_sequence(G: nx.DiGraph, fsm_sequence_len: int):
    # find all deadend nodes
    deadend_nodes = []
    for i in range(len(G.nodes())):
        if len(list(G.successors(i))) == 0:
            deadend_nodes.append(i)

    # generate a random path through the graph
    # start at node 0
    # pick a random edge to traverse
    # repeat until we have fsm_sequence_len states
    
    sequence = ["S0"]
    current_state = 0
    for i in range(fsm_sequence_len - 2):
        out_vertices = list(G.successors(current_state))
        if len(out_vertices) == 0 :
            import pdb; pdb.set_trace()
        next_state = random.choice(out_vertices)
        while next_state in deadend_nodes:
            # sample without replacement
            out_vertices.remove(next_state)
            if len(out_vertices) > 0:
                next_state = random.choice(out_vertices)
            else:
                next_state = random.choice(range(len(G.nodes())))
                sequence.append("S" + str(next_state))
                return sequence
        sequence.append("S" + str(next_state))
        current_state = next_state
    
    # append a final state that shouldn't reachable from the previous state
    out_vertices = list(G.successors(current_state))
    non_out_vertices = list(set(range(len(G.nodes()))) - set(out_vertices))
    next_state = random.choice(non_out_vertices)
    sequence.append("S" + str(next_state))
    return sequence

def generate_random_fsm(num_inputs:int, num_nodes: int, num_edges: int, max_recursive_dapth: int = 2):
    G = generate_random_digraph(num_nodes, num_edges)
    # annotate edges with random operations
    for src, dst in G.edges():
        expression_string = generate_transition_condition(num_inputs=num_inputs, depth=0, max_recursive_dapth=max_recursive_dapth)
            # check that if there are more than two signals in expr, they must all not be the same

        G[src][dst]["operation"] = expression_string
        print(f"Edge {src} -> {dst}: {G[src][dst]['operation']}")
    return G




def generate_binary_operator():
    # List of operators we can use in assertions
    logical_operators = ["&&", "||", "^"]
    relational_operators = ["<=", ">=", "<", ">"]
    equivalence_operator = ["==", "!="]

    # weighted random choice
    knob = random.random()
    if knob < 0.5:
        return f"{random.choice(logical_operators)}"
    elif knob < 0.9:
        return f"{random.choice(equivalence_operator)}"
    else:
        return f"{random.choice(relational_operators)}"

def generate_unary_operator():
    # List of operators we can use in assertions
    operators = ["!", "~", "&", "~&", "|", "~|", "^", "~^"]
    return f"{random.choice(operators)}"

def generate_transition_condition(num_inputs: int, depth: int = 0, max_recursive_dapth: int =2):
    expression = generate_expression(num_inputs=num_inputs, depth=depth, max_recursive_dapth=max_recursive_dapth)
    # map "signals" to symbolic names
    # count number of sigals in expression
    num_signals = expression.count("signal")
    prev_symbol = ""
    for i in range(num_signals):
        symbol = generate_random_signal(num_inputs=num_inputs)
        while symbol == prev_symbol:
            symbol = generate_random_signal(num_inputs=num_inputs)
        expression = expression.replace("signal", symbol, 1)
        prev_symbol = symbol
    return expression


def generate_random_signal(num_inputs: int = 10):
    # Generate a signal name randomly from sig_A
    # limit to max_int signals
    max_val = min(num_inputs, 26)
    return f"in_{chr(random.randint(65, 65 + max_val - 1))}"

def generate_expression(num_inputs: int, depth: int = 0, max_recursive_dapth: int =2):
    # Base case: Return a simple signal
    is_leaf = random.random() < (1.0 / max_recursive_dapth) * depth
    if is_leaf:
        return "signal"
    if random.random() < 0.2:
        # Generate a unary operator
        unary_operators = ["!", "~", "&", "~&", "|", "~|", "^", "~^"]
        return f"{random.choice(unary_operators)}({generate_expression(num_inputs=num_inputs, depth=depth + 1)})"
    # Generate a binary operator
    logical_operators = ["&&", "||", "^"]
    relational_operators = ["<=", ">=", "<", ">"]
    equivalence_operator = ["==", "!="]
    knob = random.random()
    if knob < 0.5:
        return f"({generate_expression(num_inputs=num_inputs, depth=depth + 1, max_recursive_dapth=max_recursive_dapth)} {random.choice(logical_operators)} {generate_expression(num_inputs=num_inputs, depth=depth + 1, max_recursive_dapth=max_recursive_dapth)})"
    elif knob < 0.9:
        options = [generate_expression(num_inputs=num_inputs, depth=depth + 1, max_recursive_dapth=max_recursive_dapth), "'d1", "'d0"]
        return f"({generate_expression(num_inputs=num_inputs, depth=depth + 1, max_recursive_dapth=max_recursive_dapth)} {random.choice(equivalence_operator)} {random.choice(options)})"
    else:
        options = [generate_expression(num_inputs=num_inputs, depth=depth + 1, max_recursive_dapth=max_recursive_dapth), "'d1", "'d0"]
        return f"({generate_expression(num_inputs=num_inputs, depth=depth + 1, max_recursive_dapth=max_recursive_dapth)} {random.choice(relational_operators)} {random.choice(options)})"


"""
Top-level function to generate each random pipeline design/testbench RTL
Both SV code is saved as .sv files
args:
"""
def generate_testcase(
    num_inputs: int = 4,
    num_nodes: int = 10,
    num_edges: int = 20,
    width: int = 32,
    op_recursive_depth: int = 2,
):
    
    fsm_rtl, G = generate_fsm_module(num_inputs=num_inputs, num_nodes=num_nodes, num_edges=num_edges, max_recursive_dapth=op_recursive_depth, width=width)
    fsm_tb_rtl = generate_fsm_tb_module(G, num_inputs=num_inputs, num_nodes=num_nodes, width=width, max_fsm_sequence_len=num_edges//2)
    return fsm_rtl, fsm_tb_rtl


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Generate Aribitrary FSM designs for the FVEVal-HelperGen Benchmark")
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to save directory",
        default=ROOT / "data",
    )
    parser.add_argument(
        "--num_test_cases",
        "-n",
        type=int,
        help="number of random pipeline designs to create per category",
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=0,
    )

    args = parser.parse_args()
    if not isinstance(args.save_dir, str):
        save_dir = args.save_dir.as_posix()
    else:
        save_dir = args.save_dir
    random.seed(args.seed)


    dataset = []
    experiment_id = "fsm"
    for i in range(args.num_test_cases):
        for ni in [8, 16]:
            for nn, ne in [(16,64), (32, 128)]:
                    for wd in [32]:
                        for opd in [5]:
                            tag = f"{ni}_{nn}_{ne}_{wd}_{opd}"
                            fsm_rtl, fsm_tb_rtl = generate_testcase(num_inputs=ni, num_nodes=nn, num_edges=ne, width=wd, op_recursive_depth=opd)
                            dataset.append(
                                InputData(
                                    design_name=experiment_id,
                                    task_id=tag,
                                    prompt=fsm_rtl,
                                    ref_solution="",
                                    testbench=fsm_tb_rtl,
                                )
                            )
    df = pd.DataFrame([asdict(d) for d in dataset])
    df.to_csv(args.save_dir / f"helpergen_{experiment_id}.csv", index=False)
    print(f"generated {len(df)} cases")

    with open("fsm.sv", "w") as f:
        f.write(dataset[-1].prompt)
    with open("fsm.sva", "w") as f:
        f.write(dataset[-1].testbench)