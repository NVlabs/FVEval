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
from itertools import accumulate
import pathlib
import random

import pandas as pd

from fv_eval.data import InputData

PIPELINE_PREFIX = """
module pipeline (
    clk,
    reset_,
    in_vld,
    in_data,
    out_vld,
    out_data
);
    parameter WIDTH=`WIDTH;
    parameter DEPTH=`DEPTH;
    
    input clk;
    input reset_;
    input in_vld;
    input [WIDTH-1:0] in_data;
    output out_vld;
    output [WIDTH-1:0] out_data;

    wire [DEPTH:0] ready;
    wire [DEPTH:0][WIDTH-1:0] data;
    assign ready[0] = in_vld;
    assign data[0] = in_data;
    assign out_vld = ready[DEPTH];
    assign out_data = data[DEPTH];
"""

MULTI_PIPELINE_PREFIX = """
module pipeline (
    clk,
    reset_,
    in_vld,
    in_data_0,
    in_data_1,
    out_vld,
    out_data
);
    parameter WIDTH=`WIDTH;
    parameter DEPTH=`DEPTH;
    
    input clk;
    input reset_;
    input in_vld;
    input [WIDTH-1:0] in_data_0;
    input [WIDTH-1:0] in_data_1;
    output out_vld;
    output [WIDTH-1:0] out_data;

    wire [DEPTH:0] ready_0;
    wire [DEPTH:0] ready_1;
    wire [DEPTH:0][WIDTH-1:0] data_0;
    wire [DEPTH:0][WIDTH-1:0] data_1;

    assign ready_0[0] = in_vld;
    assign ready_1[0] = in_vld;
    assign data_0[0] = in_data_0;
    assign data_1[0] = in_data_1;
    assign out_vld = ready_0[DEPTH] & ready_1[DEPTH];
"""

PIPELINE_TB_PREFIX = """
module pipeline_tb (
    clk,
    reset_,
    in_vld,
    in_data,
    out_vld,
    out_data
);
    parameter WIDTH=`WIDTH;
    parameter DEPTH=`DEPTH;

    input clk;
    input reset_;
    input in_vld;
    input [WIDTH-1:0] in_data;
    input out_vld;
    input [WIDTH-1:0] out_data;

    assign tb_reset = (reset_ == 1'b0);
"""

MULTI_PIPELINE_TB_PREFIX = """
module pipeline_tb (
    clk,
    reset_,
    in_vld,
    in_data,
    out_vld,
    out_data
);
    parameter WIDTH=`WIDTH;
    parameter DEPTH=`DEPTH;

    input clk;
    input reset_;
    input in_vld;
    input [WIDTH-1:0] in_data;
    input out_vld;
    input [WIDTH-1:0] out_data;

    assign tb_reset = (reset_ == 1'b0);
"""

PIPELINE_TB_SUFFIX = """
endmodule


bind pipeline pipeline_tb #(
        .WIDTH(`WIDTH),
        .DEPTH(`DEPTH)
    ) pipeline_tb_inst (.*);
"""


def random_arithemtic_pairs(
    depth: int = 0, operation: str = "x", max_depth: int = 2
) -> list[str]:
    # Possible basic operations
    operators = ["+", "-", "<<<", ">>>"]
    dual_operators = ["-", "+",  ">>>", "<<<"]
    # Create a random compound operation from basic components
    index = random.randint(0, len(operators))  # Choose a random operator
    op = operators[index]
    dual_op = dual_operators[index]
    const = random.randint(1, 10)  # Choose a random constant or width
    return [f"({operation} {op} {const})", f"({operation} {dual_op} {const})"]



def random_single_input_arithmetic(
    depth: int = 0, operation: str = "x", max_depth: int = 2
):
    # Possible basic operations
    operators = ["+", "-", "&", "|", "^", "<<<", ">>>"]

    # Create a random compound operation from basic components
    op = random.choice(operators)  # Choose a random operator
    const = random.randint(1, 10)  # Choose a random constant or width
    if random.random() < (1.0 / max_depth) * depth:
        return f"({operation} {op} {const})"
    else:
        # Form a new part of the compound operation
        if random.random() < 0.5:
            return f"({random_single_input_arithmetic(depth+1, operation, max_depth=max_depth)} {op} {const})"
        return f"({random_single_input_arithmetic(depth+1, operation, max_depth=max_depth)} {op} {random_single_input_arithmetic(depth+1, operation, max_depth=max_depth)})"


def random_two_input_arithmetic(max_depth: int = 2):
    # Possible basic operations
    operators = ["+", "-", "&", "|", "^"]
    op = random.choice(operators)  # Choose a random operator
    left_operand = random_single_input_arithmetic(
        depth=0, operation="x", max_depth=max_depth
    )
    right_operand = random_single_input_arithmetic(
        depth=0, operation="y", max_depth=max_depth
    )
    return f"({left_operand} {op} {right_operand})"


def gen_arith_comb_module(module_idx: int, operation_str: str):
    # change operation_str
    operation_str = operation_str.replace("x", "in_data")
    # SystemVerilog module template with a random operation
    sv_module = f"""
module exec_unit_{module_idx} (
    clk,
    reset_,
    in_data,
    in_vld,
    out_data,
    out_vld
);
    parameter WIDTH = `WIDTH;

    input clk;
    input reset_;
    input [WIDTH-1:0] in_data;
    input in_vld;
    output reg [WIDTH-1:0] out_data;
    output reg out_vld;

    always_ff @(posedge clk or negedge reset_) begin
        if (!reset_) begin
            out_vld <= 'd0;
            out_data <= 'd0;
        end else begin
            out_vld <= in_vld;
            out_data <= {operation_str};
        end
    end
endmodule"""
    return sv_module


def gen_arith_pipeline_module(module_idx: int, operation_str: str, depth: int = 2):
    operation_str = operation_str.replace("x", "data[i]")
    # SystemVerilog module template with a random operation
    sv_module = f"""
module exec_unit_{module_idx} (
    clk,
    reset_,
    in_data,
    in_vld,
    out_data,
    out_vld
);  
    parameter WIDTH = `WIDTH;
    localparam DEPTH = {depth};
    input clk;
    input reset_;
    input [WIDTH-1:0] in_data;
    input in_vld;
    output [WIDTH-1:0] out_data;
    output out_vld;

    
    logic [DEPTH:0] ready;
    logic [DEPTH:0][WIDTH-1:0] data;
    assign ready[0] = in_vld;
    assign data[0] = in_data;
    assign out_vld = ready[DEPTH];
    assign out_data = data[DEPTH];

    generate
        for (genvar i=0; i < DEPTH; i=i+1) begin : gen
            always @(posedge clk) begin
                if (!reset_) begin
                    ready[i+1] <= 'd0;
                    data[i+1] <= 'd0;
                end else begin
                    ready[i+1] <= ready[i];
                    data[i+1] <= {operation_str};
                end
            end
        end
    endgenerate
endmodule
"""
    return sv_module


def gen_module_instatiation(
    module_idx: int, stage_idx: int, depth: int, pipeline_idx: int = -1
):
    if pipeline_idx >= 0:
        return f"""
    exec_unit_{module_idx} #(.WIDTH(WIDTH)) unit_{module_idx} (
        .clk(clk),
        .reset_(reset_),
        .in_data(data_{pipeline_idx}[{stage_idx}]),
        .in_vld(ready_{pipeline_idx}[{stage_idx}]),
        .out_data(data_{pipeline_idx}[{stage_idx + depth}]), 
        .out_vld(ready_{pipeline_idx}[{stage_idx + depth}])
    );"""
    else:
        return f"""
    exec_unit_{module_idx} #(.WIDTH(WIDTH)) unit_{module_idx} (
        .clk(clk),
        .reset_(reset_),
        .in_data(data[{stage_idx}]),
        .in_vld(ready[{stage_idx}]),
        .out_data(data[{stage_idx + depth}]), 
        .out_vld(ready[{stage_idx + depth}])
    );"""


def gen_pipeline(
    num_stages: int,
    depth: int,
    op_recursive_depth: int,
    ensure_consistency: bool = False
) -> str:
    """
    Wrapper function to generate a single pipeline with random arithemtic logic
    randomly inserted across it.
    Output is a tuple of two strings: (1) submodules specifying the random arithmetics
    and (2) submodule instantiation to be appended to the main module
    """
    if ensure_consistency: 
        operations_list = []
        dual_operations_list = []
        for _ in range(num_stages):
            ops = random_arithemtic_pairs(max_depth=op_recursive_depth)
            operations_list.append(ops[0])
            dual_operations_list.append(ops[1])
    else:
        operations_list = [
            random_single_input_arithmetic(max_depth=op_recursive_depth)
            for _ in range(num_stages)
        ]

    pipeline_module_rtl_text = []
    sv_modules_rtl_text = []

    # randomly assign latencies of each submodule, such that they add up to at most depth cycles
    sv_modules_depths = [
        1 if random.random() < 0.5 else random.randint(1, depth // 2)
        for _ in range(num_stages)
    ]
    while sum(sv_modules_depths) > depth:
        sv_modules_depths[random.randint(0, num_stages - 1)] = 1
    remaining_depth = depth - sum(sv_modules_depths)
    module_stage_indices = [0] + list(accumulate(sv_modules_depths))

    sv_modules_indices = list(range(num_stages))
    random.shuffle(sv_modules_indices)

    # package into a list of dictionaries
    operations = [
        {
            "op_str": operations_list[i],
            "depth": sv_modules_depths[i],
            "stage_idx": module_stage_indices[i],
            "idx_shuffled": sv_modules_indices[i],
            "idx_orig": i,
        }
        for i in range(num_stages)
    ]

    # for downstream generation, duplicate entries with depth > 1 in operations_list
    verbose_operations_list = []

    for op_dict in operations:
        if op_dict["depth"] == 1:
            sv_modules_rtl_text.append(
                gen_arith_comb_module(
                    module_idx=op_dict["idx_orig"], operation_str=op_dict["op_str"]
                )
            )
            verbose_operations_list.append(op_dict["op_str"])
        else:
            sv_modules_rtl_text.append(
                gen_arith_pipeline_module(
                    module_idx=op_dict["idx_orig"],
                    operation_str=op_dict["op_str"],
                    depth=op_dict["depth"],
                )
            )
            verbose_operations_list.extend([op_dict["op_str"]] * op_dict["depth"])
        pipeline_module_rtl_text.append(
            gen_module_instatiation(
                module_idx=op_dict["idx_orig"],
                stage_idx=op_dict["stage_idx"],
                depth=op_dict["depth"],
            )
        )
    if remaining_depth > 0:
        sv_modules_rtl_text.append(
            gen_arith_pipeline_module(
                module_idx=num_stages, operation_str="x", depth=remaining_depth
            )
        )
        pipeline_module_rtl_text.append(
            gen_module_instatiation(
                module_idx=num_stages,
                stage_idx=sum(sv_modules_depths),
                depth=remaining_depth,
            )
        )
    random.shuffle(sv_modules_rtl_text)
    # random.shuffle(pipeline_module_rtl_text)
    return (
        "\n".join(sv_modules_rtl_text),
        "\n".join(pipeline_module_rtl_text),
        verbose_operations_list,
    )


"""
Higher-level methods for pipeline design SV and testbench SV code generation
(1) gen_pipeline_design: generates a single pipeline design RTL code
    return: string of the SV RTL
(2) gen_multi_pipeline_design: generates a dual-pipeline design RTL code
    TODO: support variable # of pipelines
    return: string of the SV RTL
(3) gen_pipeline_tb_design: generates the TB for the single pipeline design RTL code
    Assumes a list of arithemtic operations provided as input
    return: string of the SV testbench RTL
(4) gen_multi_pipeline_tb_design: generates the TB for the dual pipeline design RTL code
    Assumes a list of list of arithemtic operations provided as input (one sub-list per pipeline)
    return: string of the SV testbench RTL
"""


def gen_pipeline_design(
    num_stages: int, depth: int, width: int = 32, op_recursive_depth: int = 2,
    ensure_consistency: bool=False
):
    ptr = 0
    operations_list = []
    # pipeline_module_rtl_text = PIPELINE_PREFIX
    full_rtl_text = f"`define WIDTH {width}\n`define DEPTH {depth}\n"
    sv_modules_rtl_text, pipeline_module_rtl_text, operations_list = gen_pipeline(
        num_stages=num_stages, depth=depth, op_recursive_depth=op_recursive_depth,
        ensure_consistency=ensure_consistency
    )
    pipeline_module_rtl_text = PIPELINE_PREFIX + pipeline_module_rtl_text
    full_rtl_text += (
        sv_modules_rtl_text + "\n\n" + pipeline_module_rtl_text + "\nendmodule"
    )
    return full_rtl_text, operations_list


def gen_multi_pipeline_design(
    num_pipelines: int,
    num_stages: int,
    depth: int,
    width: int = 32,
    op_recursive_depth: int = 2,
):

    multi_pipeline_operations_list = []
    pipeline_module_rtl_text = MULTI_PIPELINE_PREFIX
    full_rtl_text = f"`define WIDTH {width}\n`define DEPTH {depth}\n"
    sv_modules_rtl_text = ""
    i = 0
    for pipeline_idx in range(num_pipelines):
        ptr = 0
        operations_list = []
        for _ in range(num_stages):
            operation_str = random_single_input_arithmetic(max_depth=op_recursive_depth)

            if random.random() < 0.5 or ptr > depth - 2:
                module_pipeline_depth = 1
                sv_modules_rtl_text += "\n" + gen_arith_comb_module(
                    module_idx=i, operation_str=operation_str
                )
                operations_list.append(operation_str)
            else:
                module_pipeline_depth = random.randint(2, min(depth - ptr, 5))
                sv_modules_rtl_text += "\n" + gen_arith_pipeline_module(
                    module_idx=i,
                    operation_str=operation_str,
                    depth=module_pipeline_depth,
                )
                operations_list.extend([operation_str] * module_pipeline_depth)
            pipeline_module_rtl_text += "\n" + gen_module_instatiation(
                module_idx=i,
                stage_idx=ptr,
                depth=module_pipeline_depth,
                pipeline_idx=pipeline_idx,
            )
            ptr += module_pipeline_depth
            i += 1
        if ptr < depth:
            sv_modules_rtl_text += "\n" + gen_arith_pipeline_module(
                module_idx=i + 1, operation_str="x", depth=depth - ptr
            )
            pipeline_module_rtl_text += "\n" + gen_module_instatiation(
                module_idx=num_stages,
                stage_idx=ptr,
                depth=depth - ptr,
                pipeline_idx=pipeline_idx,
            )
            i += 1
        multi_pipeline_operations_list.append(operations_list)

    # combinational logic to combine the multiple pipeline outputs
    end_combination_rtl_text = random_two_input_arithmetic()
    multi_pipeline_operations_list.append([end_combination_rtl_text])
    end_combination_rtl_text = end_combination_rtl_text.replace("x", "data_0[DEPTH]")
    end_combination_rtl_text = end_combination_rtl_text.replace("y", "data_1[DEPTH]")
    end_combination_rtl_text = f"""
    assign out_data = {end_combination_rtl_text};
    """
    full_rtl_text += (
        sv_modules_rtl_text
        + "\n\n"
        + pipeline_module_rtl_text
        + end_combination_rtl_text
        + "\nendmodule"
    )
    return full_rtl_text, multi_pipeline_operations_list


def gen_pipeline_tb_design(operations_list: list[str], depth: int, width: int = 32):
    assert len(operations_list) > 1
    tb_rtl_text = f"`define WIDTH {width}\n`define DEPTH {depth}\n" + PIPELINE_TB_PREFIX
    for i, operation in enumerate(operations_list):
        # the last list entry in operations_list is the final arithmetic operation between pipeline results
        if i == 0:
            tb_rtl_text += f"    wire [WIDTH-1:0] data_{i};\n"
            tb_rtl_text += (
                f"    assign data_{i} = "
                + operation.replace("x", "in_data")
                + ";\n"
            )
        else:
            tb_rtl_text += f"    wire [WIDTH-1:0] data_{i};\n"
            tb_rtl_text += (
                f"    assign data_{i} = "
                + operation.replace("x", f"data_{i-1}")
                + ";\n"
            )
    tb_rtl_text += f"""
    target: assert property (@(posedge clk) disable iff(tb_reset)
        in_vld |-> ##DEPTH out_vld && (out_data == $past(data_{len(operations_list)-1}, DEPTH))
    );
    """
    tb_rtl_text += PIPELINE_TB_SUFFIX
    return tb_rtl_text


def gen_multi_pipeline_tb_design(
    operations_list: list[list[str]], depth: int, width: int = 32
):
    assert len(operations_list) > 1
    tb_rtl_text = f"`define WIDTH {width}\n`define DEPTH {depth}\n" + PIPELINE_TB_PREFIX
    for pipeline_idx, op_list in enumerate(operations_list[:-1]):
        for i, operation in enumerate(op_list):
            if i == 0:
                tb_rtl_text += f"    wire [WIDTH-1:0] data_{pipeline_idx}_{i};\n"
                tb_rtl_text += (
                    f"    assign data_{pipeline_idx}_{i} = "
                    + operation.replace("x", f"in_data_{pipeline_idx}")
                    + ";\n"
                )
            else:
                tb_rtl_text += f"    wire [WIDTH-1:0] data_{pipeline_idx}_{i};\n"
                tb_rtl_text += (
                    f"    assign data_{pipeline_idx}_{i} = "
                    + operation.replace("x", f"data_{pipeline_idx}_{i-1}")
                    + ";\n"
                )

    # currently hardcoded to two pipelines
    # TODO: extend to variable # of pipelines
    comb_operation = operations_list[-1][0]
    comb_operation = comb_operation.replace("x", f"data_0_{len(operations_list[0])-1}")
    comb_operation = comb_operation.replace("y", f"data_1_{len(operations_list[1])-1}")
    tb_rtl_text += f"    wire [WIDTH-1:0] comb_data;\n"
    tb_rtl_text += f"    assign comb_data = " + comb_operation + ";\n"
    tb_rtl_text += f"""
    asrt: assert property (@(posedge clk) disable iff(tb_reset)
        in_vld |-> ##DEPTH out_vld && (out_data == $past(comb_data, DEPTH))
    );
    """
    tb_rtl_text += PIPELINE_TB_SUFFIX
    return tb_rtl_text


"""
Top-level function to generate each random pipeline design/testbench RTL
Both SV code is saved as .sv files
args:
"""


def generate_testcase(
    num_pipelines: int,
    num_stages: int,
    depth: int,
    width: int = 32,
    op_recursive_depth: int = 2,
):
    if num_pipelines == 1:
        pipeline_rtl, operations_list = gen_pipeline_design(
            num_stages=num_stages,
            depth=depth,
            op_recursive_depth=op_recursive_depth,
            width=width,
        )
        pipeline_tb_rtl = gen_pipeline_tb_design(
            operations_list, depth=depth, width=width
        )
        return pipeline_rtl, pipeline_tb_rtl
    else:
        pipeline_rtl, operations_list = gen_multi_pipeline_design(
            num_pipelines=num_pipelines,
            num_stages=num_stages,
            depth=depth,
            op_recursive_depth=op_recursive_depth,
            width=width,
        )
        pipeline_tb_rtl = gen_multi_pipeline_tb_design(
            operations_list, depth=depth, width=width
        )
        return pipeline_rtl, pipeline_tb_rtl


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Generate Aribitrary Pipeline designs for the FVEVal-Design2SVA Benchmark")
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

    # (1) single pipeline designs
    experiment_id = "pipeline_single"
    for i in range(args.num_test_cases):
        # single pipeline & no guarantee of e2e data equivalence
        for ns in [2, 5, 10]:
            for width in [128, 256]:
                for depth in [50, 100]:
                    tag=f"ns_{ns}-w_{width}-d_{depth}-{i}"
                    pipeline_rtl, pipeline_tb_rtl = generate_testcase(
                        num_pipelines=1,
                        num_stages=ns,
                        depth=depth,
                        width=width,
                        op_recursive_depth=random.randint(1, 4),
                    )
                    dataset.append(
                        InputData(
                            design_name=experiment_id,
                            task_id=tag,
                            prompt=pipeline_rtl,
                            ref_solution="",
                            testbench=pipeline_tb_rtl,
                        )
                    )
    df = pd.DataFrame([asdict(d) for d in dataset])
    df.to_csv(args.save_dir / f"helpergen_{experiment_id}.csv", index=False)
    print(f"generated {len(df)} cases")

