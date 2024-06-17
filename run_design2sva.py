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

from fv_eval import data, utils, benchmark_launcher

if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Run LLM Inference for the FVEval-Design2SVA Benchmark"
    )
    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        help="path to input dataset directory, potentially holding multiple .csv files",
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
        "--num_assertions",
        "-k",
        type=int,
        help="Measure out of k assertions",
        default=1,
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        help="models to run with, ;-separated",
        default="gpt-4;gpt-4-turbo;gpt-3.5-turbo;llama-3-70b;mixtral-8x22b;llama-2-70b",
    )
    parser.add_argument(
        "--cot_strategy",
        "-c",
        type=str,
        help="chain of thought strategy: default, plan-act, plan-model-act",
        default="default",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
    )

    args = parser.parse_args()
    if not args.dataset_dir:
        dataset_dir = ROOT / "data_design2sva" / "data"
        assert dataset_dir.is_dir()
        dataset_dir = dataset_dir.as_posix()
    else:
        dataset_dir = args.dataset_dir

    timestamp_str = datetime.now().strftime("%Y%m%d%H")
    if not args.save_dir:
        save_dir = ROOT / f"results_design2sva/{args.cot_strategy}_{args.num_assertions}/{timestamp_str}"
        save_dir = save_dir.as_posix()
    else:
        save_dir = args.save_dir
    utils.mkdir_p(save_dir)

    temperature = args.temperature if args.num_assertions == 1 else 0.8
    dataset_paths = data.read_datasets_from_dir(dataset_dir)
    for dataset_path in dataset_paths:
        bmark_launcher = benchmark_launcher.Design2SVALauncher(
            save_dir=save_dir,
            dataset_path=dataset_path,
            task="design2sva",
            model_name_list=args.models.split(";"),
            debug=args.debug,
        )
        bmark_launcher.run_benchmark(
            temperature=temperature,
            max_tokens=2000,
            cot_strategy=args.cot_strategy,
            num_cases=args.num_assertions,
        )
