import argparse
from dataclasses import asdict
import os
import re
import pathlib
import random

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from fv_eval.data import InputData


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
        return "signal"
    else:
        # Generate a unary operator
        return f"{generate_unary_operator()}signal"

def choose_from_binary(depth: int, max_depth=3):
    # Generate a binary operator
    if random.random() < 0.3:
        return f"({generate_expression(depth + 1, max_depth=max_depth)} {generate_binary_operator()} 1'b1)"
    else:
        return f"({generate_expression(depth + 1, max_depth=max_depth)} {generate_binary_operator()} {generate_expression(depth + 1, max_depth=max_depth)})"
        

def choose_from_temporal(depth: int, max_depth=3):
    if random.random() < 0.7:
        # Generate a temporal operator
        expr = generate_temporal_operator()
        temporal_bound = generate_temporal_bound()
        expr = f"{expr} {temporal_bound}"
        return f"{generate_expression(depth + 1)} {expr} {generate_expression(depth + 1, max_depth=max_depth)}"
    elif random.random() < 0.5:
        # Generate a s_temporal operator
        expr, extra_args = generate_s_temporal_operator()
        if extra_args:
            expr = f"{expr}(##[1:$] {generate_expression(depth + 1, max_depth=max_depth)})"
        else:
            expr = f"{expr}({generate_expression(depth + 1, max_depth=max_depth)})"
        if random.random() < 0.8:
            return f"{generate_expression(depth + 1, max_depth=max_depth)} |-> " + expr
        else:
            return expr
    else:
        # Generate a special operator
        expr, extra_arg = generate_special_operator()
        if extra_arg:
            constant = random.randint(1, 10)
            return f"{expr}({generate_expression(depth + 1, max_depth=max_depth)}, {constant})"
        else:
            return f"{expr}({generate_expression(depth + 1, max_depth=max_depth)})"
    
def generate_expression(depth: int=0, max_depth=3):
    # Base case: Return a simple signal
    is_leaf = random.random() < (1.0 / max_depth) * depth 
    if is_leaf:
        return choose_from_unary(depth)
    elif depth == 0:
        if random.random() < 0.5:
            return choose_from_temporal(depth, max_depth=max_depth)
        else:
            return choose_from_binary(depth, max_depth=max_depth)
    else:
        return choose_from_binary(depth, max_depth=max_depth)
        
def generate_assertion(max_int: int = 10):
    # Generate a full assertion statement
    expression = generate_expression()

    # map "signals" to symbolic names
    # count number of sigals in expression
    num_signals = expression.count("signal")
    prev_symbol = ""
    for _ in range(num_signals):
        symbol = generate_random_signal(max_int=max_int)
        while symbol == prev_symbol:
            symbol = generate_random_signal(max_int=max_int)
        expression = expression.replace("signal", symbol, 1)
        prev_symbol = symbol
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
    client = OpenAI()
    model_name = "gpt-4o"
    max_tokens = 150
    stop = ["\n", "."]
    dataset = []
    testbech_text = ""
    with open(args.dummy_testbench_path, "r") as f:
        testbech_text = f.read()

    i = 0
    while True:
        # randomly generate assertion 
        assertion_text = generate_assertion()
        for j in range(args.num_nldesc_per_assertion):
            counter = 0
            for _ in range(5):
                user_prompt = icl_prompt
                user_prompt += "\n\n Now here is your question to answer."
                user_prompt += f"\nQuestion: in a single sentence, explain the following SystemVerilog assertion in English.\n{assertion_text}\n"
                user_prompt += "\nDo NOT use phrases such as 'result of the expression ...'"
                user_prompt += "\nAnswer:"
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=150,
                    temperature=args.temperature,
                    stop=stop,
                )
                lm_generated_annotation = completion.choices[0].message.content

                # gpt-4-turbo as a judge
                judge_system_prompt = "You are tasked with judging the quality of the following natural language description for a SystemVerilog assertion."
                judge_user_prompt = f"Judge whether the following natural language description is correct for the SystemVerilog\n\nAssertion: {assertion_text}\nDescription: {lm_generated_annotation}"
                judge_user_prompt += "\n\nPlease provide a score of 0 or 1, where 0 indicates the description is inaccurate or insufficient, and 1 indicates the description is accurate, clear, and sufficiently descriptive."
                judge_user_prompt += "\n\nScore:"
                judge_completion = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "system", "content": judge_system_prompt},
                        {"role": "user", "content": judge_user_prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.0,
                )
                judge_score = judge_completion.choices[0].message.content
                # regex match to get the score
                judge_score_numerical = re.search(r"\d+", judge_score)
                if not judge_score_numerical:
                    print(judge_score)
                    continue
                else:
                    judge_score_numerical = int(judge_score_numerical.group(0))
                if judge_score_numerical == 1:
                    break
            if judge_score_numerical == 0:
                continue
            dataset.append(
                InputData(
                    design_name="nl2sva_machine",
                    task_id=f"{i}_{j}",
                    prompt=completion.choices[0].message.content,
                    ref_solution=assertion_text,
                    testbench=testbech_text
                )
            )
            i += 1
            if i >= args.num_assertions:
                break

        if args.debug and i > 100:
            break
            
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    df = pd.DataFrame(dataset)
    df = pd.DataFrame([asdict(d) for d in dataset])
    if args.debug:
        df.to_csv(args.save_dir / "nl2sva_machine_debug.csv", index=False)
        print(f"Debug mode: Saved to {args.save_dir.as_posix() + f'/nl2sva_human_debug.csv'} | {len(df)}")
    else:
        df.to_csv(args.save_dir / "nl2sva_machine.csv", index=False)
        print(f"Saved to {args.save_dir.as_posix() + '/nl2sva_machine.csv'} | {len(df)}")
    