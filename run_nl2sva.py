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
from datetime import datetime
import pathlib


from fv_eval import utils, benchmark_launcher


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Run LLM Inference for the FVEval-SVAGen Benchmark"
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="path to input dataset",
    )
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to input dataset directory, potentially holding multiple .csv files",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="LLM decoder sampling temperature",
        default=0.0,
    )
    parser.add_argument(
        "--num_icl",
        "-k",
        type=int,
        help="number of in-context examples to use",
        default=3,
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        help="models to run with, ;-separated",
        default="gpt-4;gpt-4-turbo;gpt-3.5-turbo;llama-3-70b;mixtral-8x22b;llama-2-70b;mixtral-8x7b",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Evaluation mode: (1) 'human' where we evaluate NL to SVA generation against human-annotated assertions from real testbenches; (2) 'machine' where we ",
        default="human",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug ",
    )

    args = parser.parse_args()
    timestamp_str = datetime.now().strftime("%Y%m%d%H")
    temperature = 0.0

    if args.debug:
        print("Executing in debug mode")

    if args.mode == "human":
        if not args.dataset_path:
            dataset_path = ROOT / "data_nl2sva" / "data" / "nl2sva_human.csv"
            assert dataset_path.exists()
            dataset_path = dataset_path.as_posix()
        else:
            dataset_path = args.dataset_path

        if not args.save_dir:
            timestamp_str = datetime.now().strftime("%Y%m%d%H")
            save_dir = ROOT / f"results_nl2sva_human/{args.num_icl}/{timestamp_str}"
            save_dir = save_dir.as_posix()
        else:
            save_dir = args.save_dir
        utils.mkdir_p(save_dir)

        bmark_launcher = benchmark_launcher.NL2SVAHumanLauncher(
            save_dir=save_dir,
            dataset_path=dataset_path,
            task="nl2sva_human",
            model_name_list=args.models.split(";"),
            num_icl_examples=args.num_icl,
            debug=args.debug,
        )
        bmark_launcher.run_benchmark(
            temperature=temperature, max_tokens=200, num_cases=1
        )
    elif args.mode == "machine":
        if not args.dataset_path:
            dataset_path = ROOT / "data_nl2sva" / "data" / "nl2sva_machine.csv"
            assert dataset_path.exists()
            dataset_path = dataset_path.as_posix()
        else:
            dataset_path = args.dataset_path
        if not args.save_dir:
            timestamp_str = datetime.now().strftime("%Y%m%d%H")
            save_dir = ROOT / f"results_nl2sva_machine/{args.num_icl}/{timestamp_str}"
            save_dir = save_dir.as_posix()
        else:
            save_dir = args.save_dir
        utils.mkdir_p(save_dir)

        bmark_launcher = benchmark_launcher.NL2SVAMachineLauncher(
            save_dir=save_dir,
            dataset_path=dataset_path,
            task="nl2sva_machine",
            model_name_list=args.models.split(";"),
            num_icl_examples=args.num_icl,
            debug=args.debug,
        )
        bmark_launcher.run_benchmark(
            temperature=temperature, max_tokens=100, num_cases=1
        )
    else:
        print(f"Unsupported eval mode: {args.mode}")
        raise NotImplementedError
