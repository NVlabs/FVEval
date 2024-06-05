import os
import glob
import re
import shutil
import multiprocessing
from dataclasses import dataclass, asdict

from typing import Optional, Callable, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
import evaluate


from fv_eval import utils, fv_tool_execution
from fv_eval.data import LMResult, TextSimilarityEvaluationResult, JGEvaluationResult


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


class Design2SVAEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        model_name:str,
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
            # take only the metric values from the evaluation results
            final_results = jg_eval_results.copy()
            # # for rows that share same unique_task_id, only take the max value
            # # measure pass@k for each unique_task_id
            # count_num_trials = jg_eval_results.groupby("unique_task_id").count().iloc[0,0]
            # count_pass = jg_eval_results.groupby("unique_task_id").sum()
            
            # syntax_pass_at_1 = utils.pass_at_k(count_pass["syntax"].values, count_num_trials, 1)
            # syntax_pass_at_5 = utils.pass_at_k(count_pass["syntax"].values, count_num_trials, 5)
            # func_pass_at_1 = utils.pass_at_k(count_pass["functionality"].values, count_num_trials, 1)
            # func_pass_at_5 = utils.pass_at_k(count_pass["functionality"].values, count_num_trials, 5)
                
            # jg_eval_results = jg_eval_results.groupby("unique_task_id").mean()
            # jg_eval_results = jg_eval_results.drop(columns=["experiment_id", "task_id", "model_name"])

            # # for each remaining column, calculate the mean value
            # # add as a separate row
            # import pdb; pdb.set_trace()
            # pass_at_1 = jg_eval_results.copy()
            # pass_at_1["syntax"] = syntax_pass_at_1
            # pass_at_1["functionality"] = func_pass_at_1
            # pass_at_1 = pass_at_1.mean(axis=0)
            # pass_at_1 = pass_at_1.to_frame().T
            # pass_at_1["experiment_id"] = exp_name
            # pass_at_1["task_id"] = "Pass@1"
            # pass_at_1["model_name"] = "Pass@1"
            # pass_at_5 = jg_eval_results.copy()
            # pass_at_5["syntax"] = syntax_pass_at_5
            # pass_at_5["functionality"] = func_pass_at_5
            # pass_at_5 = pass_at_5.mean(axis=0)
            # pass_at_5 = pass_at_5.to_frame().T
            # pass_at_5["experiment_id"] = exp_name
            # pass_at_5["task_id"] = "Pass@5"
            # final_results = pd.concat([final_results, pass_at_1, pass_at_5], ignore_index=True)
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
        relaxed_funcality_score = (float(proof_result_list.count("proven")) + float(proof_result_list.count("undetermined")))/ float(
            len(proof_result_list)
        )
        return {
            "syntax": syntax_score,
            "functionality": functionality_score,
            "func_relaxed": relaxed_funcality_score,
        }
        # parse formal func_relaxed
        # cov_report_match = re.findall(r"\bformal_func_relaxed[^\n]*", jasper_out_str)
        # if not cov_report_match:
        #     return {"syntax": syntax_score, "functionality": functionality_score, "func_relaxed": 0.0, "bound_improve": 0.0}
        # testbench_name = f"{top_module_name}"
        # escaped_testbench_name = re.escape(testbench_name)
        # testbench_cov_match = re.findall(fr"\b{escaped_testbench_name}\b[^\n]*", cov_report_match[0])
        # if not testbench_cov_match:
        #     return {"syntax": syntax_score, "functionality": functionality_score, "func_relaxed": 0.0, "bound_improve": 0.0}
        # cov_value = re.search(r"func_relaxed_percentage {\s*(\d+\.\d+)%", testbench_cov_match[0]).group(1)
        # return {"syntax": syntax_score, "functionality": functionality_score, "func_relaxed": float(cov_value), "bound_improve": 0.0}


# class HelperGenEvaluator(Evaluator):
#     def __init__(
#         self,
#         llm_output_dir: str,
#         temp_dir: str,
#         save_dir: str,
#         parallel_jobs: int = 8,
#         cleanup_temp_files: bool = True,
#         debug: bool = False,
#     ):
#         super().__init__(
#             task="helpergen",
#             llm_output_dir=llm_output_dir,
#             temp_dir=temp_dir,
#             save_dir=save_dir,
#             parallel_jobs=parallel_jobs,
#             cleanup_temp_files=cleanup_temp_files,
#             debug=debug,
#         )

#     def run_evaluation(
#         self,
#     ):
#         for exp_name, result_list in self.llm_results:
#             jg_eval_results = self.evaluate_jg(result_list, with_rtl_design=True)
#             jg_eval_results = [asdict(r) for r in jg_eval_results]
#             jg_eval_results = pd.DataFrame(jg_eval_results)
#             jg_eval_results.to_csv(f"{self.save_dir}/{exp_name}_jg.csv", index=False)

#             # take only the metric values from the evaluation results
#             final_results = jg_eval_results.drop(
#                 columns=["experiment_id", "task_id", "model_name"]
#             )
#             # for each remaining column, calculate the mean value
#             # add as a separate row
#             mean_values = final_results.mean(axis=0)
#             mean_values = mean_values.to_frame().T
#             mean_values["experiment_id"] = exp_name
#             mean_values["task_id"] = "mean"
#             mean_values["model_name"] = "mean"
#             final_results = pd.concat([final_results, mean_values], ignore_index=True)
#             final_results.to_csv(f"{self.save_dir}/{exp_name}.csv", index=False)
#         return final_results

#     def calculate_jg_metric(
#         self,
#         jasper_out_str: str,
#     ):
#         # check for syntax error
#         top_module = re.findall(r"top: [^\n]*", jasper_out_str)
#         if not top_module:
#             return {
#                 "syntax": 0.0,
#                 "functionality": 0.0,
#                 "func_relaxed": 0.0,
#                 "bound_improve": 0.0,
#             }
#         top_module_name = top_module[-1].split(":")[-1].strip()

#         syntax_error_match = re.findall(r"syntax error", jasper_out_str)
#         if syntax_error_match:
#             return {
#                 "syntax": 0.0,
#                 "functionality": 0.0,
#                 "func_relaxed": 0.0,
#                 "bound_improve": 0.0,
#             }
#         syntax_score = 1.0

#         # check for number of assertions proven
#         proof_result_match = re.findall(r"\bproofs:[^\n]*", jasper_out_str)
#         if not proof_result_match:
#             return {
#                 "syntax": syntax_score,
#                 "functionality": 0.0,
#                 "func_relaxed": 0.0,
#                 "bound_improve": 0.0,
#             }
#         proof_result_list = proof_result_match[-1].split(":")[-1].strip().split(" ")
#         # count # of "proven"
#         functionality_score = float(proof_result_list.count("proven")) / float(
#             len(proof_result_list)
#         )

#         return {
#             "syntax": syntax_score,
#             "functionality": functionality_score,
#             "func_relaxed": 0.0,
#             "bound_improve": 0.0,
#         }
