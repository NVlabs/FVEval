import os
import glob
import re
import shutil
import multiprocessing
from dataclasses import dataclass, asdict

from typing import Optional, Callable, Dict

import pandas as pd
from tqdm import tqdm
import evaluate


from fv_eval import utils, fv_tool_execution
from fv_eval.data import LMResult, TextSimilarityEvaluationResult, JGEvaluationResult


class Evaluator(object):
    def __init__(
        self,
        task: str,
        llm_output_dir: str,
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
        llm_results = glob.glob(f"{llm_output_dir}/*.csv")
        assert len(llm_results) > 0, "No LLM results found"
        llm_results = [(f.split("/")[-1].split(".csv")[0], pd.read_csv(f)) for f in llm_results]
        self.llm_results = [(exp_id, [LMResult(**r) for _, r in df.iterrows()]) for exp_id, df in llm_results]

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
            "exact_match": evaluate.load("exact_match")
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
            # check for <CODE> and </CODE> tags in the response
            tb = r.output_tb
            # code_tags = re.findall(r"<CODE>(.*?)</CODE>", tb, re.DOTALL)
            # if len(code_tags) > 0:
            #     for code in code_tags:
            #         tb = tb.replace(f"<CODE>{code}</CODE>", code)
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
            metric_output = metric_func.compute(predictions=[lm_response], references=[ref_solution])
            if metric_name == "bleu":
                metric_values[metric_name] = metric_output["bleu"]
            elif metric_name == "rouge":
                metric_values[metric_name] = metric_output["rougeL"]
            elif metric_name == "exact_match":
                metric_values[metric_name] = metric_output["exact_match"]
            else:
                utils.print_error(f"Similarity metric not supported",metric_name)
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
                    **metric_values
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
        for batch_id in tqdm(range(num_batchs), desc=f"Evaluating test cases {model_name} {experiment_id}"):
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

                p = multiprocessing.Process(
                    target=fv_tool_execution.launch_jg_with_queue,
                    kwargs={
                        "tcl_file_path": self.tcl_file_path,
                        "sv_dir": self.temp_dir,
                        "experiment_id": lm_result.experiment_id,
                        "task_id": lm_result.task_id,
                        "output_queue": output_queue
                    }
                )
                processes.append(p)
                p.start()
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
                        **result_dict
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
            text_similiarty_eval_results = [asdict(r) for r in text_similiarty_eval_results]
            text_similiarty_eval_results = pd.DataFrame(text_similiarty_eval_results)
            text_similiarty_eval_results.to_csv(f"{self.save_dir}/{exp_name}_sim.csv", index=False)

            jg_eval_results = self.evaluate_jg(result_list)
            jg_eval_results = [asdict(r) for r in jg_eval_results]
            jg_eval_results = pd.DataFrame(jg_eval_results)
            jg_eval_results.to_csv(f"{self.save_dir}/{exp_name}_jg.csv", index=False)

            # take only the metric values from the evaluation results
            combined_results = pd.merge(text_similiarty_eval_results, jg_eval_results, on=["experiment_id", "task_id", "model_name"])
            # drop the columns that are not metric values
            combined_results = combined_results.drop(columns=["experiment_id", "task_id", "model_name"])
            # for each remaining column, calculate the mean value
            # add as a separate row
            mean_values = combined_results.mean(axis=0)
            mean_values = mean_values.to_frame().T
            mean_values["experiment_id"] = exp_name
            mean_values["task_id"] = "mean"
            mean_values["model_name"] = "mean"
            combined_results = pd.concat([combined_results, mean_values], ignore_index=True)
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
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="nl2sva_human",
            llm_output_dir=llm_output_dir,
            temp_dir=temp_dir,
            save_dir=save_dir,
            parallel_jobs=parallel_jobs,
            cleanup_temp_files=cleanup_temp_files,
            debug=debug,
        )

    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {"syntax": 0.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        
        # check for functionality error
        lm_coi_match = re.findall(r"\bLM_COI[^\n]*", jasper_out_str)
        ref_coi_match = re.findall(r"\bREF_COI[^\n]*", jasper_out_str) 
        if not lm_coi_match or not ref_coi_match:
            return {"syntax": 1.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        
        indexing_pattern = r"\[\s*\d+\]"
        lm_coi = lm_coi_match[-1].split(":")[-1].strip()
        lm_coi = re.sub(indexing_pattern, '', lm_coi)
        lm_coi = lm_coi.split(' ')
        ref_coi = ref_coi_match[-1].split(":")[-1].strip()
        ref_coi = re.sub(indexing_pattern, '', ref_coi)
        ref_coi = ref_coi.split(' ')
        return {"syntax": 0.0, "functionality": float(ref_coi == lm_coi), "coverage": 0.0, "bound_improve": 0.0}
    
class NL2SVAMachineEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="nl2sva_machine",
            llm_output_dir=llm_output_dir,
            temp_dir=temp_dir,
            save_dir=save_dir,
            parallel_jobs=parallel_jobs,
            cleanup_temp_files=cleanup_temp_files,
            debug=debug,
        )

    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {"syntax": 0.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        return {"syntax": 1.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}

class Design2SVAEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="design2sva",
            llm_output_dir=llm_output_dir,
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

            # take only the metric values from the evaluation results
            final_results = jg_eval_results.drop(columns=["experiment_id", "task_id", "model_name"])
            # for each remaining column, calculate the mean value
            # add as a separate row
            mean_values = final_results.mean(axis=0)
            mean_values = mean_values.to_frame().T
            mean_values["experiment_id"] = exp_name
            mean_values["task_id"] = "mean"
            mean_values["model_name"] = "mean"
            final_results = pd.concat([final_results, mean_values], ignore_index=True)
            final_results.to_csv(f"{self.save_dir}/{exp_name}.csv", index=False)
        return final_results
    
    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        top_module = re.findall(r"top: [^\n]*", jasper_out_str)
        if not top_module:
            return {"syntax": 0.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        top_module_name = top_module[-1].split(":")[-1].strip()

        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {"syntax": 0.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        syntax_score = 1.0
        
        # check for number of assertions proven
        proof_result_match = re.findall(r"\bproofs:[^\n]*", jasper_out_str)
        if not proof_result_match:
            return {"syntax": syntax_score, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        proof_result_list = proof_result_match[-1].split(":")[-1].strip().split(" ")
        # count # of "proven"
        functionality_score  = float(proof_result_list.count("proven")) / float(len(proof_result_list)) 

        # parse formal coverage
        cov_report_match = re.findall(r"\bformal_coverage[^\n]*", jasper_out_str)
        if not cov_report_match:
            return {"syntax": syntax_score, "functionality": functionality_score, "coverage": 0.0, "bound_improve": 0.0}
        testbench_name = f"{top_module_name}"
        escaped_testbench_name = re.escape(testbench_name)
        testbench_cov_match = re.findall(fr"\b{escaped_testbench_name}\b[^\n]*", cov_report_match[0])
        if not testbench_cov_match:
            return {"syntax": syntax_score, "functionality": functionality_score, "coverage": 0.0, "bound_improve": 0.0}
        cov_value = re.search(r"coverage_percentage {\s*(\d+\.\d+)%", testbench_cov_match[0]).group(1)
        return {"syntax": syntax_score, "functionality": functionality_score, "coverage": float(cov_value), "bound_improve": 0.0}



class HelperGenEvaluator(Evaluator):
    def __init__(
        self,
        llm_output_dir: str,
        temp_dir: str,
        save_dir: str,
        parallel_jobs: int = 8,
        cleanup_temp_files: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            task="helpergen",
            llm_output_dir=llm_output_dir,
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

            # take only the metric values from the evaluation results
            final_results = jg_eval_results.drop(columns=["experiment_id", "task_id", "model_name"])
            # for each remaining column, calculate the mean value
            # add as a separate row
            mean_values = final_results.mean(axis=0)
            mean_values = mean_values.to_frame().T
            mean_values["experiment_id"] = exp_name
            mean_values["task_id"] = "mean"
            mean_values["model_name"] = "mean"
            final_results = pd.concat([final_results, mean_values], ignore_index=True)
            final_results.to_csv(f"{self.save_dir}/{exp_name}.csv", index=False)
        return final_results
    
    def calculate_jg_metric(
        self,
        jasper_out_str: str,
    ):
        # check for syntax error
        top_module = re.findall(r"top: [^\n]*", jasper_out_str)
        if not top_module:
            return {"syntax": 0.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        top_module_name = top_module[-1].split(":")[-1].strip()

        syntax_error_match = re.findall(r"syntax error", jasper_out_str)
        if syntax_error_match:
            return {"syntax": 0.0, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        syntax_score = 1.0
        
        # check for number of assertions proven
        proof_result_match = re.findall(r"\bproofs:[^\n]*", jasper_out_str)
        if not proof_result_match:
            return {"syntax": syntax_score, "functionality": 0.0, "coverage": 0.0, "bound_improve": 0.0}
        proof_result_list = proof_result_match[-1].split(":")[-1].strip().split(" ")
        # count # of "proven"
        functionality_score  = float(proof_result_list.count("proven")) / float(len(proof_result_list)) 

        return {"syntax": syntax_score, "functionality": functionality_score,"coverage": 0.0, "bound_improve": 0.0}