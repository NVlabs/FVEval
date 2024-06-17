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
import glob
import json
import os
import pathlib

from typing import Iterable, Dict

import pandas as pd

from fv_eval.data import read_sv_from_dir, InputData


def read_reference_solutions(solutions_dir: str, dut_name_list: Iterable[str]) -> Dict:
    benchmark_reference_outputs = {}
    for dut_name in dut_name_list:
        svafile_glob = glob.glob(solutions_dir + f"/{dut_name}/*.sva")
        if not svafile_glob:
            print(f"No reference solution found for:{dut_name}")
            continue
        benchmark_reference_outputs[dut_name] = {}
        for svafile in svafile_glob:
            with open(svafile, "r") as f:
                svatext = f.read()
            task_id = os.path.basename(svafile).split(".sva")[0]
            benchmark_reference_outputs[dut_name].update({task_id: svatext.strip()})
    return benchmark_reference_outputs


def read_svagen_raw_problems(problems_dir: str, dut_name_list: Iterable[str]) -> Dict:
    benchmark_inputs = {}
    for dut_name in dut_name_list:
        jsonfile_glob = glob.glob(problems_dir + f"/{dut_name}/*.jsonl")
        if not jsonfile_glob:
            print(f"No input prompt found for:{dut_name}")
            continue
        jsonlfile = jsonfile_glob[0]
        benchmark_inputs[dut_name] = []
        with open(jsonlfile, "r") as f:
            for line in list(f):
                json_object = json.loads(line)
                benchmark_inputs[dut_name].append(json_object)
    return benchmark_inputs


def preprocess_svagen_data(
    problems_dir: str,
    solutions_dir: str,
    testbench_dir: str,
    save_dir: str,
    debug: bool = False,
):
    """
    Preprocesses context (DUT, Testbench), input prompts, reference solutions, and in-context examples
    necessary for the svagen benchmark suite.
    Packages all necessary information into a single .jsonl file that run scripts
    can later read into administer evaluation of LMs

    Params:
        problems_dir: (str) source directory of svagen promp prompts
        solutions_dir: (str) source directory of svagen reference solutions
        testbench_dir: (str) source directory of svagen context testbenches
        save_dir: (str) save path
    """
    testbenches_dict = read_sv_from_dir(data_dir=testbench_dir)
    dut_names = list(testbenches_dict.keys())
    benchmark_inputs = read_svagen_raw_problems(
        problems_dir=problems_dir,
        dut_name_list=dut_names,
    )
    benchmark_reference_outputs = read_reference_solutions(
        solutions_dir=solutions_dir,
        dut_name_list=dut_names,
    )
    if debug:
        dut_names = [dut_names[0]]
    # Package into single dictionary
    full_dataset = []
    for dut_name in dut_names:
        for problem_dict in benchmark_inputs[dut_name]:
            task_id = problem_dict["task_id"]
            prompt = problem_dict["prompt"]
            testbench_context = testbenches_dict[dut_name]
            ref_solution = benchmark_reference_outputs[dut_name][task_id]
            full_dataset.append(
                InputData(
                    design_name=dut_name,
                    task_id=task_id,
                    prompt=prompt,
                    ref_solution=ref_solution,
                    testbench=testbench_context,
                )
            )
    if debug:
        pd.DataFrame([asdict(d) for d in full_dataset]).to_csv(
            save_dir + f"/nl2sva_human_debug.csv", sep=",", index=False
        )
        print(
            f"Debug mode: Saved to {save_dir + f'/nl2sva_human_debug.csv'} | {len(full_dataset)}"
        )
    else:
        pd.DataFrame([asdict(d) for d in full_dataset]).to_csv(
            save_dir + f"/nl2sva_human.csv", sep=",", index=False
        )
        print(f"Saved to {save_dir + f'/nl2sva_human.csv'} | {len(full_dataset)}")


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Run LLM Inference for the FVEval-SVAGen Benchmark"
    )
    parser.add_argument(
        "--svagen_nl2sva_input_dir",
        type=str,
        help="path to raw NL2SVA input dataset",
        default=ROOT / "annotated_instructions_with_signals",
    )
    parser.add_argument(
        "--svagen_nl2sva_tb_dir",
        type=str,
        help="path to raw NL2SVA testbench (TB) SystemVerilog files",
        default=ROOT / "annotated_tb",
    )
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to save directory, where processed datasets for SVAGen tasks will be saved",
        default=ROOT / "data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
    )

    args = parser.parse_args()
    nl2sva_prompt_dir = args.svagen_nl2sva_input_dir.as_posix()
    nl2sva_tb_dir = args.svagen_nl2sva_tb_dir.as_posix()
    save_dir = args.save_dir.as_posix()

    preprocess_svagen_data(
        testbench_dir=nl2sva_tb_dir,
        problems_dir=nl2sva_prompt_dir,
        solutions_dir=nl2sva_prompt_dir,
        save_dir=save_dir,
    )
