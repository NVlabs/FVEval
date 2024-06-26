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

import os
import glob
import re
import shutil
import multiprocessing
from dataclasses import asdict

import pandas as pd
from tqdm import tqdm
import evaluate


from fv_eval import utils, fv_tool_execution
from fv_eval.data import LMResult, TextSimilarityEvaluationResult, JGEvaluationResult

"""
Base Evaluator Class for Executing FVEval Evaluation Flow
"""


class Evaluator(object):
    def __init__(
        self,
        task: str,
        llm_output_dir: str,
        model_name: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        self.task = task
        self.llm_output_dir = llm_output_dir
        self.temp_dir = temp_dir
        self.save_dir = save_dir
        self.parallel_jobs = parallel_jobs
        self.cleanup_temp_files = cleanup_temp_files
        self.debug = debug

        # load all results and store them in a list of FVEvalLMResponse
        if model_name:
            llm_results = glob.glob(f"{llm_output_dir}/*{model_name}_*.csv")
        else:
            # if empty model name, load all results in the directory
            llm_results = glob.glob(f"{llm_output_dir}/*.csv")
        assert len(llm_results) > 0, "No LLM results found"
        llm_results = [
            (f.split("/")[-1].split(".csv")[0], pd.read_csv(f)) for f in llm_results
        ]
        self.llm_results = [
            (exp_id, [LMResult(**r) for _, r in df.iterrows()])
            for exp_id, df in llm_results
        ]

        # setup temp and save directories
        if not os.path.isdir(temp_dir):
            utils.mkdir_p(temp_dir)
        if not os.path.isdir(save_dir):
            utils.mkdir_p(save_dir)

        # set tcl file path
        self.tcl_file_path = self.set_tcl_file_path()

        # load similarity metrics
        self.similarity_metrics = {
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
            "exact_match": evaluate.load("exact_match"),
        }

    def set_tcl_file_path(
        self,
    ):
        if self.task == "nl2sva_human":
            tcl_file_path = "tool_scripts/run_jg_nl2sva_human.tcl"
        elif self.task == "nl2sva_machine":
            tcl_file_path = "tool_scripts/run_jg_nl2sva_machine.tcl"
        elif self.task == "design2sva":
            tcl_file_path = "tool_scripts/run_jg_design2sva.tcl"
        elif self.task == "helpergen":
            tcl_file_path = "tool_scripts/run_jg_helpergen.tcl"
        else:
            utils.print_error(f"Task not supported", self.task)
            raise NotImplementedError
        if tcl_file_path is not None and not os.path.isfile(tcl_file_path):
            utils.print_error(f"TCL file not found", tcl_file_path)
            raise FileNotFoundError
        if self.debug:
            print(f"Set TCL file path to {tcl_file_path}")
        return tcl_file_path

    def write_design_sv(
        self,
        results_list: list[LMResult],
    ):
        # For each result, write the packaged testbench to a SystemVerilog file
        # in the subdir directory
        for r in results_list:
            with open(f"{self.temp_dir}/{r.experiment_id}_{r.task_id}.sv", "w") as f:
                f.write(r.design_rtl)

    def write_testbench_sv(
        self,
        results_list: list[LMResult],
    ):
        # For each result, write the packaged testbench to a SystemVerilog file
        # in the subdir directory
        for r in results_list:
            tb = r.output_tb
            with open(f"{self.temp_dir}/{r.experiment_id}_{r.task_id}.sva", "w") as f:
                f.write(tb)

    def setup_jg_evaluation(
        self,
        result_list: list[LMResult],
    ):
        # For each result, write the packaged testbench to a SystemVerilog file
        # in the temp directory
        for r in result_list:
            with open(f"{self.temp_dir}/{r.experiment_id}_{r.task_id}.sv", "w") as f:
                f.write(r.design_rtl)
            with open(f"{self.temp_dir}/{r.experiment_id}_{r.task_id}.sva", "w") as f:
                f.write(r.output_tb)

    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        raise NotImplementedError

    def calculate_similiarty_metric(
        self,
        lm_response: str,
        ref_solution: str,
    ):
        metric_values = {}
        for metric_name, metric_func in self.similarity_metrics.items():
            metric_output = metric_func.compute(
                predictions=[lm_response], references=[ref_solution]
            )
            if metric_name == "bleu":
                metric_values[metric_name] = metric_output["bleu"]
            elif metric_name == "rouge":
                metric_values[metric_name] = metric_output["rougeL"]
            elif metric_name == "exact_match":
                metric_values[metric_name] = metric_output["exact_match"]
            else:
                utils.print_error(f"Similarity metric not supported", metric_name)
                raise NotImplementedError
        return metric_values

    def evaluate_text_similiarty(
        self,
        result_list: list[LMResult],
    ):
        eval_results = []
        for lm_result in result_list:
            # calculate similarity metric
            metric_values = self.calculate_similiarty_metric(
                lm_result.response,
                lm_result.ref_solution,
            )
            eval_results.append(
                TextSimilarityEvaluationResult(
                    experiment_id=lm_result.experiment_id,
                    task_id=lm_result.task_id,
                    model_name=lm_result.model_name,
                    **metric_values,
                )
            )
        return eval_results

    def evaluate_jg(
        self,
        result_list: list[LMResult],
        with_rtl_design: bool = False,
    ):
        # setup temp and save directories
        if not os.path.isdir(self.temp_dir):
            utils.mkdir_p(self.temp_dir)
        # For each result, write the packaged testbench to a SystemVerilog file
        # in the temp directory
        if with_rtl_design:
            self.write_design_sv(result_list)
        self.write_testbench_sv(result_list)

        # set up execution parameters
        model_name = result_list[0].model_name
        experiment_id = result_list[0].experiment_id
        num_test_cases = len(result_list)
        num_batchs = num_test_cases // self.parallel_jobs + 1
        eval_results = []

        # iterate over all test cases
        for batch_id in tqdm(
            range(num_batchs),
            desc=f"Evaluating test cases {model_name} {experiment_id}",
        ):
            jasper_outputs = []

            # launch parallel jobs
            processes = []
            output_queue = multiprocessing.Queue()

            # iterate over all parallel workers
            for worker_id in range(self.parallel_jobs):
                index = batch_id * self.parallel_jobs + worker_id
                if index >= num_test_cases:
                    continue

                lm_result = result_list[index]
                assert lm_result.model_name == model_name
                assert lm_result.experiment_id == experiment_id

                if self.parallel_jobs == 1:
                    jasper_out_str = fv_tool_execution.launch_jg(
                        tcl_file_path=self.tcl_file_path,
                        sv_dir=self.temp_dir,
                        experiment_id=lm_result.experiment_id,
                        task_id=lm_result.task_id,
                    )
                    jasper_outputs.append(jasper_out_str)
                else:
                    p = multiprocessing.Process(
                        target=fv_tool_execution.launch_jg_with_queue,
                        kwargs={
                            "tcl_file_path": self.tcl_file_path,
                            "sv_dir": self.temp_dir,
                            "experiment_id": lm_result.experiment_id,
                            "task_id": lm_result.task_id,
                            "output_queue": output_queue,
                        },
                    )
                    processes.append(p)
                    p.start()
            if self.parallel_jobs > 1:
                for p in processes:
                    p.join()
                while not output_queue.empty():
                    jasper_outputs.append(output_queue.get())

            for jasper_out_str in jasper_outputs:
                # regex match *.sva in jasper_out_str
                task_id_match = re.findall(r"\bTASK_ID[^\n]*", jasper_out_str)
                if not task_id_match:
                    raise ValueError(f"Jasper output does not contain unique id (UID)")
                task_id = task_id_match[0].split("TASK_ID ")[-1]

                if self.debug:
                    print(jasper_out_str)
                result_dict = self.calculate_jg_metric(jasper_out_str)
                eval_results.append(
                    JGEvaluationResult(
                        experiment_id=experiment_id,
                        task_id=task_id,
                        model_name=model_name,
                        **result_dict,
                    )
                )
        if self.cleanup_temp_files:
            shutil.rmtree(self.temp_dir)
        return eval_results

    def run_evaluation(
        self,
    ):
        for exp_name, result_list in self.llm_results:
            text_similiarty_eval_results = self.evaluate_text_similiarty(result_list)
            text_similiarty_eval_results = [
                asdict(r) for r in text_similiarty_eval_results
            ]
            text_similiarty_eval_results = pd.DataFrame(text_similiarty_eval_results)
            text_similiarty_eval_results.to_csv(
                f"{self.save_dir}/{exp_name}_sim.csv", index=False
            )

            jg_eval_results = self.evaluate_jg(result_list)
            jg_eval_results = [asdict(r) for r in jg_eval_results]
            jg_eval_results = pd.DataFrame(jg_eval_results)
            jg_eval_results.to_csv(f"{self.save_dir}/{exp_name}_jg.csv", index=False)

            jg_eval_results["unique_task_id"] = jg_eval_results["task_id"].apply(
                lambda x: x.split("_trial")[0]
            )
            # for rows that share same unique_task_id, only take the max value
            jg_eval_results = jg_eval_results.groupby("unique_task_id").max()

            # take only the metric values from the evaluation results
            combined_results = pd.merge(
                text_similiarty_eval_results,
                jg_eval_results,
                on=["experiment_id", "task_id", "model_name"],
            )
            # drop the columns that are not metric values
            combined_results = combined_results.drop(
                columns=["experiment_id", "task_id", "model_name"]
            )
            # for each remaining column, calculate the mean value
            # add as a separate row
            mean_values = combined_results.mean(axis=0)
            mean_values = mean_values.to_frame().T
            mean_values["experiment_id"] = exp_name
            mean_values["task_id"] = "mean"
            mean_values["model_name"] = "mean"
            combined_results = pd.concat(
                [combined_results, mean_values], ignore_index=True
            )
            combined_results.to_csv(f"{self.save_dir}/{exp_name}.csv", index=False)
        return combined_results

    def save_evaluation_results(
        self,
        eval_results: list[JGEvaluationResult],
    ):
        eval_results = [asdict(r) for r in eval_results]
        eval_results = pd.DataFrame(eval_results)
        eval_results.to_csv(f"{self.save_dir}/jg_eval_results.csv", index=False)
        return eval_results


"""
Evaluator for NL2SVA-HUman
Implements a custom:
    - 'evaluate_jg' method that invokes proprety-equivalence checks
    - 'calculate_jg_metric' method that analyzes and reports syntax and funcational correctness specific to NL2SVA-Human
"""


class NL2SVAHumanEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        model_name: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="nl2sva_human",
            llm_output_dir=llm_output_dir,
            model_name=model_name,
            temp_dir=temp_dir,
            save_dir=save_dir,
            parallel_jobs=parallel_jobs,
            cleanup_temp_files=cleanup_temp_files,
            debug=debug,
        )

    def evaluate_jg(
        self,
        result_list: list[LMResult],
        with_rtl_design: bool = False,
    ):
        # setup temp and save directories
        if not os.path.isdir(self.temp_dir):
            utils.mkdir_p(self.temp_dir)
        # For each result, write the packaged testbench to a SystemVerilog file
        # in the temp directory
        if with_rtl_design:
            self.write_design_sv(result_list)
        self.write_testbench_sv(result_list)

        # set up execution parameters
        model_name = result_list[0].model_name
        experiment_id = result_list[0].experiment_id
        num_test_cases = len(result_list)
        num_batchs = num_test_cases // self.parallel_jobs + 1
        eval_results = []

        # iterate over all test cases
        for batch_id in tqdm(
            range(num_batchs),
            desc=f"Evaluating test cases {model_name} {experiment_id}",
        ):
            jasper_outputs = []

            # launch parallel jobs
            processes = []
            output_queue = multiprocessing.Queue()

            # iterate over all parallel workers
            for worker_id in range(self.parallel_jobs):
                index = batch_id * self.parallel_jobs + worker_id
                if index >= num_test_cases:
                    continue

                lm_result = result_list[index]
                assert lm_result.model_name == model_name
                assert lm_result.experiment_id == experiment_id

                lm_assertion_text = (
                    utils.parse_code_response(lm_result.response)
                    .strip()
                    .replace("\n", "")
                )
                lm_assertion_text = (
                    lm_assertion_text.split("tb_reset)")[-1]
                    .strip()
                    .split(");")[0]
                    .strip()
                )

                ref_assertion_text = lm_result.ref_solution.strip().replace("\n", "")
                ref_assertion_text = (
                    ref_assertion_text.split("tb_reset)")[-1]
                    .strip()
                    .split(");")[0]
                    .strip()
                )

                signal_list = re.findall(r"'([^'\s]+)'", lm_result.user_prompt)

                params = re.findall(
                    r"\b(parameter|localparam)\s+(int\s+|real\s+|bit\s+|\[[^]]+\]\s*)?(\w+)",
                    lm_result.output_tb,
                )
                params = [m[2] for m in params]
                signal_list.extend(params)
                signal_list_text = ",".join(signal_list)

                if self.parallel_jobs == 1:
                    jasper_out_str = fv_tool_execution.launch_jg_custom_equiv_check(
                        tcl_file_path=self.tcl_file_path,
                        lm_assertion_text=lm_assertion_text,
                        ref_assertion_text=ref_assertion_text,
                        signal_list_text=signal_list_text,
                        sv_dir=self.temp_dir,
                        experiment_id=lm_result.experiment_id,
                        task_id=lm_result.task_id,
                    )
                    jasper_outputs.append(jasper_out_str)
                else:
                    p = multiprocessing.Process(
                        target=fv_tool_execution.launch_jg_with_queue_custom_equiv_check,
                        kwargs={
                            "tcl_file_path": self.tcl_file_path,
                            "lm_assertion_text": lm_assertion_text,
                            "ref_assertion_text": ref_assertion_text,
                            "signal_list_text": signal_list_text,
                            "sv_dir": self.temp_dir,
                            "experiment_id": lm_result.experiment_id,
                            "task_id": lm_result.task_id,
                            "output_queue": output_queue,
                        },
                    )
                    processes.append(p)
                    p.start()
            if self.parallel_jobs > 1:
                for p in processes:
                    p.join()
                while not output_queue.empty():
                    jasper_outputs.append(output_queue.get())

            for jasper_out_str in jasper_outputs:
                # regex match *.sva in jasper_out_str
                task_id_match = re.findall(r"\bTASK_ID[^\n]*", jasper_out_str)
                if not task_id_match:
                    raise ValueError(f"Jasper output does not contain unique id (UID)")
                task_id = task_id_match[0].split("TASK_ID ")[-1]

                if self.debug:
                    print(jasper_out_str)
                result_dict = self.calculate_jg_metric(jasper_out_str)
                eval_results.append(
                    JGEvaluationResult(
                        experiment_id=experiment_id,
                        task_id=task_id,
                        model_name=model_name,
                        **result_dict,
                    )
                )
        if self.cleanup_temp_files:
            shutil.rmtree(self.temp_dir)
        return eval_results

    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {
                "syntax": 0.0,
                "functionality": 0.0,
                "func_relaxed": 0.0,
                # "bound_improve": 0.0,
            }

        # check for functionality error
        # match for "Full equivalence" in jaspert output string
        full_equiv_match = re.findall(r"Full equivalence", jasper_out_str)
        partial_equiv_match = re.findall(r"implies", jasper_out_str)
        if not full_equiv_match:
            if not partial_equiv_match:
                return {
                    "syntax": 1.0,
                    "functionality": 0.0,
                    "func_relaxed": 0.0,
                    # "bound_improve": 0.0,
                }
            else:
                return {
                    "syntax": 1.0,
                    "functionality": 0.0,
                    "func_relaxed": 1.0,
                }
        return {
            "syntax": 1.0,
            "functionality": 1.0,
            "func_relaxed": 1.0,
        }


"""
Evaluator for NL2SVA-Machine
Implements a custom:
    - 'evaluate_jg' method that invokes proprety-equivalence checks
    - 'calculate_jg_metric' method that analyzes and reports syntax and funcational correctness specific to NL2SVA-Machine
"""


class NL2SVAMachineEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        model_name: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="nl2sva_machine",
            llm_output_dir=llm_output_dir,
            model_name=model_name,
            temp_dir=temp_dir,
            save_dir=save_dir,
            parallel_jobs=parallel_jobs,
            cleanup_temp_files=cleanup_temp_files,
            debug=debug,
        )

    def evaluate_jg(
        self,
        result_list: list[LMResult],
        with_rtl_design: bool = False,
    ):
        # setup temp and save directories
        if not os.path.isdir(self.temp_dir):
            utils.mkdir_p(self.temp_dir)
        # For each result, write the packaged testbench to a SystemVerilog file
        # in the temp directory
        if with_rtl_design:
            self.write_design_sv(result_list)
        self.write_testbench_sv(result_list)

        # set up execution parameters
        model_name = result_list[0].model_name
        experiment_id = result_list[0].experiment_id
        num_test_cases = len(result_list)
        num_batchs = num_test_cases // self.parallel_jobs + 1
        eval_results = []

        # iterate over all test cases
        for batch_id in tqdm(
            range(num_batchs),
            desc=f"Evaluating test cases {model_name} {experiment_id}",
        ):
            jasper_outputs = []

            # launch parallel jobs
            processes = []
            output_queue = multiprocessing.Queue()

            # iterate over all parallel workers
            for worker_id in range(self.parallel_jobs):
                index = batch_id * self.parallel_jobs + worker_id
                if index >= num_test_cases:
                    continue
                lm_result = result_list[index]
                assert lm_result.model_name == model_name
                assert lm_result.experiment_id == experiment_id

                lm_assertion_text = (
                    utils.parse_code_response(lm_result.response)
                    .strip()
                    .replace("\n", "")
                )
                lm_assertion_text = (
                    lm_assertion_text.split("clk)")[-1].strip().split(");")[0].strip()
                )

                ref_assertion_text = lm_result.ref_solution.strip().replace("\n", "")
                ref_assertion_text = (
                    ref_assertion_text.split("clk)")[-1].strip().split(");")[0].strip()
                )

                signal_list = re.findall(r"\bsig_\w+", lm_result.ref_solution)
                signal_list = list(set(signal_list))
                signal_list_text = ",".join(signal_list)

                if self.parallel_jobs == 1:
                    jasper_out_str = fv_tool_execution.launch_jg_custom_equiv_check(
                        tcl_file_path=self.tcl_file_path,
                        lm_assertion_text=lm_assertion_text,
                        ref_assertion_text=ref_assertion_text,
                        signal_list_text=signal_list_text,
                        sv_dir=self.temp_dir,
                        experiment_id=lm_result.experiment_id,
                        task_id=lm_result.task_id,
                    )
                    jasper_outputs.append(jasper_out_str)
                else:
                    p = multiprocessing.Process(
                        target=fv_tool_execution.launch_jg_with_queue_custom_equiv_check,
                        kwargs={
                            "tcl_file_path": self.tcl_file_path,
                            "lm_assertion_text": lm_assertion_text,
                            "ref_assertion_text": ref_assertion_text,
                            "signal_list_text": signal_list_text,
                            "sv_dir": self.temp_dir,
                            "experiment_id": lm_result.experiment_id,
                            "task_id": lm_result.task_id,
                            "output_queue": output_queue,
                        },
                    )
                    processes.append(p)
                    p.start()
            if self.parallel_jobs > 1:
                for p in processes:
                    p.join()
                while not output_queue.empty():
                    jasper_outputs.append(output_queue.get())
            for jasper_out_str in jasper_outputs:
                # regex match *.sva in jasper_out_str
                task_id_match = re.findall(r"\bTASK_ID[^\n]*", jasper_out_str)
                if not task_id_match:
                    raise ValueError(f"Jasper output does not contain unique id (UID)")
                task_id = task_id_match[0].split("TASK_ID ")[-1]

                if self.debug:
                    print(jasper_out_str)
                result_dict = self.calculate_jg_metric(jasper_out_str)
                eval_results.append(
                    JGEvaluationResult(
                        experiment_id=experiment_id,
                        task_id=task_id,
                        model_name=model_name,
                        **result_dict,
                    )
                )
        if self.cleanup_temp_files:
            shutil.rmtree(self.temp_dir)
        return eval_results

    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {
                "syntax": 0.0,
                "functionality": 0.0,
                "func_relaxed": 0.0,
            }

        # check for functionality error
        # match for "Full equivalence" in jaspert output string
        full_equiv_match = re.findall(r"Full equivalence", jasper_out_str)
        partial_equiv_match = re.findall(r"implies", jasper_out_str)
        if not full_equiv_match:
            if not partial_equiv_match:
                return {
                    "syntax": 1.0,
                    "functionality": 0.0,
                    "func_relaxed": 0.0,
                    # "bound_improve": 0.0,
                }
            else:
                return {
                    "syntax": 1.0,
                    "functionality": 0.0,
                    "func_relaxed": 1.0,
                }
        return {
            "syntax": 1.0,
            "functionality": 1.0,
            "func_relaxed": 1.0,
        }


"""
Evaluator for Design2SVA
Implements a custom:
    - 'calculate_jg_metric' method that analyzes and reports syntax and funcational correctness specific to Design2SVA
"""


class Design2SVAEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        model_name: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="design2sva",
            llm_output_dir=llm_output_dir,
            model_name=model_name,
            temp_dir=temp_dir,
            save_dir=save_dir,
            parallel_jobs=parallel_jobs,
            cleanup_temp_files=cleanup_temp_files,
            debug=debug,
        )

    def run_evaluation(
        self,
    ):
        for exp_name, result_list in self.llm_results:
            jg_eval_results = self.evaluate_jg(result_list, with_rtl_design=True)
            jg_eval_results = [asdict(r) for r in jg_eval_results]
            jg_eval_results = pd.DataFrame(jg_eval_results)
            jg_eval_results.to_csv(f"{self.save_dir}/{exp_name}_jg.csv", index=False)

            jg_eval_results["unique_task_id"] = jg_eval_results["task_id"].apply(
                lambda x: x.split("_trial")[0]
            )
            final_results = jg_eval_results.copy()
            final_results.to_csv(f"{self.save_dir}/{exp_name}.csv", index=False)
        return final_results

    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {
                "syntax": 0.0,
                "functionality": 0.0,
                "func_relaxed": 0.0,
            }
        syntax_score = 1.0

        # check for number of assertions proven
        proof_result_match = re.findall(r"\bproofs:[^\n]*", jasper_out_str)
        if not proof_result_match:
            return {
                "syntax": syntax_score,
                "functionality": 0.0,
                "func_relaxed": 0.0,
            }
        proof_result_list = proof_result_match[-1].split(":")[-1].strip().split(" ")
        # count # of "proven"
        functionality_score = float(proof_result_list.count("proven")) / float(
            len(proof_result_list)
        )
        relaxed_funcality_score = (
            float(proof_result_list.count("proven"))
            + float(proof_result_list.count("undetermined"))
        ) / float(len(proof_result_list))
        return {
            "syntax": syntax_score,
            "functionality": functionality_score,
            "func_relaxed": relaxed_funcality_score,
        }
