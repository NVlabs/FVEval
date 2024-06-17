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

from fv_eval import evaluation, utils

ROOT = pathlib.Path(__file__).parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run JG-based Evaluation of FVEval-SVAGen Results"
    )
    parser.add_argument(
        "--llm_output_dir",
        "-i",
        type=str,
        help="path to LLM results dir",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help="specific model name to evaluate for",
        default="",
    )
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to save JG eval results",
    )
    parser.add_argument(
        "--temp_dir",
        "-t",
        type=str,
        help="path to temp dir",
    )
    parser.add_argument(
        "--cleanup_temp",
        type=bool,
        help="Whether to clean up the temp dir afterwards",
        default=True,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="task you are evaluating for",
        default="nl2sva_human",
    )
    parser.add_argument(
        "--nparallel",
        "-n",
        type=int,
        help="parallel JG jobs",
        default=8,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug ",
    )

    args = parser.parse_args()
    if not args.llm_output_dir:
        utils.print_error(
            "Argument Error",
            "empty path to llm_output_dir. Provide correct args to --llm_output_dir",
        )
        raise ValueError("empty path to llm_output_dir")
    if not args.save_dir:
        save_dir = pathlib.Path(args.llm_output_dir) / "eval"
    else:
        save_dir = args.save_dir

    save_dir = save_dir.as_posix()
    if not args.temp_dir:
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tmp_dir = ROOT.as_posix() + f"/tmp_{args.task}/{args.model_name}_{datetime_str}"
    else:
        tmp_dir = args.temp_dir

    if "nl2sva" in args.task:
        if "human" in args.task:
            evaluator = evaluation.NL2SVAHumanEvaluator(
                llm_output_dir=args.llm_output_dir,
                model_name=args.model_name,
                temp_dir=tmp_dir,
                save_dir=save_dir,
                cleanup_temp_files=args.cleanup_temp,
                parallel_jobs=args.nparallel,
                debug=args.debug,
            )
            evaluator.run_evaluation()
        elif "machine" in args.task:
            evaluator = evaluation.NL2SVAMachineEvaluator(
                llm_output_dir=args.llm_output_dir,
                model_name=args.model_name,
                temp_dir=tmp_dir,
                save_dir=save_dir,
                cleanup_temp_files=args.cleanup_temp,
                parallel_jobs=args.nparallel,
                debug=args.debug,
            )
            evaluator.run_evaluation()
    elif "design2sva" in args.task:
        evaluator = evaluation.Design2SVAEvaluator(
            llm_output_dir=args.llm_output_dir,
            model_name=args.model_name,
            temp_dir=tmp_dir,
            save_dir=save_dir,
            cleanup_temp_files=args.cleanup_temp,
            parallel_jobs=args.nparallel,
            debug=args.debug,
        )
        evaluator.run_evaluation()
    # elif "helpergen" in args.task:
    #     evaluator = evaluation.HelperGenEvaluator(
    #         llm_output_dir=args.llm_output_dir,
    #         temp_dir=tmp_dir,
    #         save_dir=save_dir,
    #         cleanup_temp_files=args.cleanup_temp,
    #         parallel_jobs=args.nparallel,
    #         debug=args.debug,
    #     )
    #     evaluator.run_evaluation()
