# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from dataclasses import asdict
import pathlib
import random

import pandas as pd
import numpy as np
import networkx as nx

from fv_eval.data import InputData


"""
Helper functions to generate SystemVerilog design of a random FSM module
"""


def decimal_to_binary(decimal: int, num_bits: int):
    return f"{num_bits}'b{decimal:0{num_bits}b}"


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


def generate_random_fsm(
    num_inputs: int, num_nodes: int, num_edges: int, max_recursive_dapth: int = 2
):
    G = generate_random_digraph(num_nodes, num_edges)
    # annotate edges with random operations
    for src, dst in G.edges():
        expression_string = generate_transition_condition(
            num_inputs=num_inputs, depth=0, max_recursive_dapth=max_recursive_dapth
        )
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


def generate_transition_condition(
    num_inputs: int, depth: int = 0, max_recursive_dapth: int = 2
):
    expression = generate_expression(
        num_inputs=num_inputs, depth=depth, max_recursive_dapth=max_recursive_dapth
    )
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


def generate_expression(num_inputs: int, depth: int = 0, max_recursive_dapth: int = 2):
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
        return f"({generate_expression(num_inputs=num_inputs, depth=depth + 1)} {random.choice(logical_operators)} {generate_expression(num_inputs=num_inputs, depth=depth + 1)})"
    elif knob < 0.9:
        options = [
            generate_expression(num_inputs=num_inputs, depth=depth + 1),
            "'d1",
            "'d0",
        ]
        return f"({generate_expression(num_inputs=num_inputs, depth=depth + 1)} {random.choice(equivalence_operator)} {random.choice(options)})"
    else:
        options = [
            generate_expression(num_inputs=num_inputs, depth=depth + 1),
            "'d1",
            "'d0",
        ]
        return f"({generate_expression(num_inputs=num_inputs, depth=depth + 1)} {random.choice(relational_operators)} {random.choice(options)})"


def generate_module_header(
    is_design: bool, num_inputs: int, num_nodes: int, width: int = 32
):
    num_bits = int(np.ceil(np.log2(num_nodes)))
    input_ports = ",\n".join([f"    in_{chr(65+i)}" for i in range(num_inputs)])
    input_port_def = ";\n".join(
        [f"    input [WIDTH-1:0] in_{chr(65+i)}" for i in range(num_inputs)]
    )
    fsm_state_def = ";\n".join(
        [
            f"    parameter S{i} = {decimal_to_binary(i, num_bits)}"
            for i in range(num_nodes)
        ]
    )
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

{fsm_state_def};

    input clk;
    input reset_;
{input_port_def};
    {port_direction} reg [FSM_WIDTH-1:0] fsm_out;
"""


def generate_fsm_module(
    num_inputs: int,
    num_nodes: int,
    num_edges: int,
    max_recursive_dapth: int = 2,
    width: int = 32,
):
    G = generate_random_fsm(
        num_inputs=num_inputs,
        num_nodes=num_nodes,
        num_edges=num_edges,
        max_recursive_dapth=max_recursive_dapth,
    )
    # Generate the module prefix

    module_header = generate_module_header(
        is_design=True, num_nodes=num_nodes, num_inputs=num_inputs, width=width
    )
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
            module_body += (
                f"                if ({G[src][out_vertices[0]]['operation']}) begin\n"
            )
            module_body += f"                    next_state = S{out_vertices[0]};\n"
            module_body += "                end\n"
            module_body += f"                else begin\n"
            module_body += f"                    next_state = S{out_vertices[1]};\n"
            module_body += "                end\n"
        else:
            for i, dst in enumerate(out_vertices):
                if i == 0:
                    module_body += (
                        f"                if ({G[src][dst]['operation']}) begin\n"
                    )
                    module_body += f"                    next_state = S{dst};\n"
                    module_body += "                end\n"
                elif i == len(out_vertices) - 1:
                    module_body += f"                else begin\n"
                    module_body += f"                    next_state = S{dst};\n"
                    module_body += "                end\n"
                else:
                    module_body += (
                        f"                else if ({G[src][dst]['operation']}) begin\n"
                    )
                    module_body += f"                    next_state = S{dst};\n"
                    module_body += "                end\n"
        module_body += f"            end\n"
    module_body += "        endcase\n"
    module_body += "    end\n"
    # Generate the module suffix
    module_suffix = "endmodule"
    return module_header + module_body + module_suffix, G


def generate_fsm_tb_module(
    G: nx.DiGraph, num_inputs: int, num_nodes: int, width: int = 32
):
    # Generate the module prefix
    module_header = generate_module_header(
        is_design=False, num_nodes=num_nodes, num_inputs=num_inputs, width=width
    )
    # Generate the module body
    module_body = "    wire tb_reset;\n"
    module_body += "    assign tb_reset = (reset_ == 1'b0);\n"
    # Generate the module suffix
    module_suffix = "\nendmodule"
    module_suffix += f"""
bind fsm fsm_tb #(
    .WIDTH(WIDTH)
) fsm_tb_inst (.*);
    """
    return module_header + module_body + module_suffix


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

    fsm_rtl, G = generate_fsm_module(
        num_inputs=num_inputs,
        num_nodes=num_nodes,
        num_edges=num_edges,
        max_recursive_dapth=op_recursive_depth,
        width=width,
    )
    fsm_tb_rtl = generate_fsm_tb_module(
        G, num_inputs=num_inputs, num_nodes=num_nodes, width=width
    )
    return fsm_rtl, fsm_tb_rtl


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate Aribitrary FSM designs for the FVEVal-Design2SVA Benchmark"
    )
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
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=0,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
    )

    args = parser.parse_args()
    if not isinstance(args.save_dir, str):
        save_dir = args.save_dir.as_posix()
    else:
        save_dir = args.save_dir
    random.seed(args.seed)

    dataset = []
    experiment_id = "fsm"
    for ni_log in [2, 4]:
        for nn_log in [2, 3, 4]:
            for ne_multiplier in [1, 2, 3, 4]:
                for wd in [32]:
                    for opd in [2, 3, 4, 5]:
                        for i in range(args.num_test_cases):
                            ni = 2**ni_log
                            nn = 2**nn_log
                            ne = nn * ne_multiplier
                            tag = f"ni_{ni}_nn_{nn}_ne_{ne}_wd_{wd}_opd_{opd}_{i}"
                            fsm_rtl, fsm_tb_rtl = generate_testcase(
                                num_inputs=ni,
                                num_nodes=nn,
                                num_edges=ne,
                                width=wd,
                                op_recursive_depth=opd,
                            )
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
    df.to_csv(args.save_dir / f"design2sva_{experiment_id}.csv", index=False)
    print(f"generated {len(df)} cases")

    if args.debug:
        with open("fsm.sv", "w") as f:
            f.write(dataset[-1].prompt)
        with open("fsm.sva", "w") as f:
            f.write(dataset[-1].testbench)
