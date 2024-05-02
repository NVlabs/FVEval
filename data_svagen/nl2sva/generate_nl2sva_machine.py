import argparse
from dataclasses import asdict
import os
import pathlib
import random

import pandas as pd
from tqdm import tqdm

from fv_eval.data import InputData
from adlrchat.langchain import LLMGatewayChat
from langchain.schema import HumanMessage, SystemMessage


def generate_random_signal(max_int: int = 10):
    # Generate a signal name randomly from sig_A
    # limit to max_int signals
    max_val = min(max_int, 26)
    return f"sig_{chr(random.randint(65, 65 + max_val - 1))}"

def generate_binary_operator():
    # List of operators we can use in assertions
    logical_operators = ['&&', '||', "^"]
    relational_operators = ['<=', '>=', "<", ">"]
    equivalence_operator = ['===', '!==', "==", "!="]
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

def generate_special_operator():
    # List of operators we can use in assertions
    operators = ["$stable", "$rose", "$fell", "$changed", "$past"]
    random_choice = random.choice(operators)
    return random_choice, random_choice == "$past"

def generate_temporal_operator():
    # Temporal operators are a bit different and are used specifically
    operators = ["|->", "|=>"]
    return f"{random.choice(operators)}"

def generate_temporal_bound():
    # Temporal operators are a bit different and are used specifically
    knob = random.random()
    start = random.randint(1, 5)
    end = start + random.randint(1, 5)
    # if knob < 0.01:
    #     return f"##[{start}:$]"
    if knob < 0.3:
        return f"##[{start}:{end}]"
    else:
        return f"##{start}"

def generate_s_temporal_operator():
    # Temporal operators are a bit different and are used specifically
    operators = ["s_eventually", "s_until", "s_always", "strong"]
    random_choice = random.choice(operators)
    return random_choice, random_choice == "strong"



def choose_from_unary(depth: int):
    random_val = random.random()
    if random_val < 0.8:
        # Generate simple singal
        return f"{generate_random_signal()}"
    else:
        # Generate a unary operator
        return f"{generate_unary_operator()}{generate_random_signal()}"

def choose_from_binary(depth: int):
    # Generate a binary operator
    if random.random() < 0.3:
        return f"({generate_expression(depth + 1)} {generate_binary_operator()} 1'b1)"
    else:
        return f"({generate_expression(depth + 1)} {generate_binary_operator()} {generate_expression(depth + 1)})"
        

def choose_from_temporal(depth: int):
    if random.random() < 0.8:
        # Generate a temporal operator
        expr = generate_temporal_operator()
        temporal_bound = generate_temporal_bound()
        expr = f"{expr} {temporal_bound}"
        return f"{generate_expression(depth + 1)} {expr} {generate_expression(depth + 1)}"
    elif random.random() < 0.1:
        # Generate a s_temporal operator
        expr, extra_args = generate_s_temporal_operator()
        if extra_args:
            expr = f"{expr}(##[1:$] {generate_expression(depth + 1)})"
        else:
            expr = f"{expr}({generate_expression(depth + 1)})"
        if random.random() < 0.8:
            return f"{generate_expression(depth + 1)} |-> " + expr
        else:
            return expr
    else:
        # Generate a special operator
        expr, extra_arg = generate_special_operator()
        if extra_arg:
            constant = random.randint(1, 10)
            return f"{expr}({generate_expression(depth + 1)}, {constant})"
        else:
            return f"{expr}({generate_expression(depth + 1)})"
    
def generate_expression(depth: int=0):
    # Base case: Return a simple signal
    is_leaf = random.random() < 0.25 * depth 
    if is_leaf:
        return choose_from_unary(depth)
    elif depth == 0:
        if random.random() < 0.5:
            return choose_from_temporal(depth)
        else:
            return choose_from_binary(depth)
    else:
        return choose_from_binary(depth)
        
def generate_assertion():
    # Generate a full assertion statement
    expression = generate_expression()
    return f"assert property(@(posedge clk)\n\t{expression}\n);"

if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Run LLM Inference for the FVEval-SVAGen Benchmark")
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to save directory",
        default=ROOT / "data",
    )
    parser.add_argument(
        "--dummy_testbench_path",
        type=str,
        help="path to dummy testbench",
        default=ROOT / "machine_tb" / "dummy.sv",
    )
    parser.add_argument(
        "--num_assertions",
        "-n",
        type=int,
        help="number of random SVA assertions to generate",
        default=100,
    )
    parser.add_argument(
        "--num_nldesc_per_assertion",
        "-m",
        type=int,
        help="number of NL descriptions to generate per assertion",
        default=5,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="LLM response sampling temperature for NL description generation",
        default=1.0,
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
        help="debug ",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    
    model_name = "gpt-3.5-turbo"
    system_prompt = f"You are tasked with generating natural language descriptions for SystemVerilog assertions"

    icl_prompt ="""
Here are examples:
Question: in a single sentence, explain the following SystemVerilog assertion in English under the context of the provided testbench.
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
Answer: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.

Question: in a single sentence, explain the following SystemVerilog assertion in English under the context of the provided testbench.
assert property(@(posedge clk)
    (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F)
);
Answer: If sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true
"""

    chat = LLMGatewayChat(
        streaming=True,
        temperature=args.temperature,
        model=model_name,
        max_tokens=150,
        stop=["\n", "."]
    )
    dataset = []
    testbech_text = ""
    with open(args.dummy_testbench_path, "r") as f:
        testbech_text = f.read()
    for i in tqdm(range(args.num_assertions)):
        # randomly generate assertion 
        assertion_text = generate_assertion()
        for j in range(args.num_nldesc_per_assertion):
            user_prompt = icl_prompt
            user_prompt += "\n\n Now here is your question to answer."
            user_prompt += f"\nQuestion: in a single sentence, explain the following SystemVerilog assertion in English.\n{assertion_text}\n"
            user_prompt += "\nDo NOT use phrases such as 'result of the expression ...'"
            user_prompt += "\nAnswer:"
            lm_response = chat(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            dataset.append(
                InputData(
                    design_name="nl2sva_machine",
                    task_id=f"{i}_{j}",
                    prompt=lm_response.content,
                    ref_solution=assertion_text,
                    testbench=testbech_text
                )
            )

        if args.debug:
            break
            
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    df = pd.DataFrame([asdict(d) for d in dataset])
    if args.debug:
        df.to_csv(args.save_dir / "nl2sva_machine_debug.csv", index=False)
    else:
        df.to_csv(args.save_dir / "nl2sva_machine.csv", index=False)
    