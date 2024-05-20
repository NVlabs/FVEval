from dataclasses import dataclass, asdict
import os
import re
import time
import random

from openai import OpenAI
from together import Together
import google.generativeai as genai
from anthropic import Anthropic
import pandas as pd
from tqdm import tqdm

from fv_eval import utils, prompts_svagen_nl2sva, prompts_svagen_design2sva, prompts_avr_helpergen
from fv_eval.data import InputData, LMResult

class BenchmarkLauncher(object):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        debug: bool = False,
    ):
        self.save_dir = save_dir
        self.dataset_path = dataset_path
        df = pd.read_csv(dataset_path)
        self.dataset = [InputData(**row) for _, row in df.iterrows()]
        # convert dataset into list of InputData

        self.chat_client = OpenAI()
        self.task = task
        self.model_api_list = self._prepare_models(model_name_list)
        self.num_icl_examples = num_icl_examples
        self.debug = debug
        self.experiment_id = dataset_path.split(".csv")[0].split("/")[-1]

    def _build_iterator(self, model_name: str):
        if self.debug:
            # if debug, only take first 5 rows
            iterator = self.dataset[:2]
        else:
            iterator = tqdm(
                self.dataset,
                total=len(self.dataset),
                desc=f"Running for {model_name}",
            )
        return iterator

    def generate_system_prompt(self):
        raise NotImplementedError("generate_system_prompt not implemented")

    def generate_question_prompt(sefl, row: InputData):
        pass

    def generate_user_prompt_prefix(self, row: InputData):
        raise NotImplementedError("generate_user_prompt_prefix not implemented")

    def package_testbench(self, row: InputData, lm_response: str):
        raise NotImplementedError("package_testbench not implemented")
    
    def get_cot_strategy(self, cot_strategy: str) -> list[tuple[str, str]]:
        return []
    
    def parse_code_response(self, lm_response_str) -> str:
        code_tags = re.findall(r"```systemverilog(.*?)```", lm_response_str, re.DOTALL)
        if len(code_tags) > 0:
            for code in code_tags:
                lm_response_str = lm_response_str.replace(f"```systemverilog{code}```", code)
        code_tags = re.findall(r"```systemverilog(.*?)", lm_response_str, re.DOTALL)
        if len(code_tags) > 0:
            for code in code_tags:
                lm_response_str = lm_response_str.replace(f"```systemverilog{code}", code)
        return lm_response_str

    def _prepare_models(self, model_name_list: str):
        TOGETHER_MODEL_DICT = {
            "llama-3-70b": "meta-llama/Llama-3-70b-chat-hf",
            "code-llama-70b": "meta-llama/CodeLlama-70b-Instruct-hf",
            "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
            "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
        model_api_list = []
        for model_name in model_name_list:
            if "vllm" in model_name:
                api_provider = "vllm"
                api_key = "EMPTY"
                base_url = "http://localhost:8000/v1"
                full_model_name = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                ).models.list().data[0].id
            elif model_name in TOGETHER_MODEL_DICT:
                api_provider = "together"
                api_key = os.getenv("TOGETHER_API_KEY")
                base_url = "https://api.together.xyz/v1"
                full_model_name = TOGETHER_MODEL_DICT[model_name]
            elif "gemini" in model_name:
                api_provider = "google"
                api_key = os.getenv("GOOGLE_API_KEY")
                base_url = "https://gemini.googleapis.com/v1"
                full_model_name = model_name
            elif "claude" in model_name:
                api_provider = "anthropic"
                api_key = os.getenv("ANTHROPIC_API_KEY")
                base_url = "https://api.anthropic.com/v1"
                full_model_name = model_name
            else:
                api_provider = "openai"
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = "https://api.openai.com/v1"
                if "gpt-4" in model_name and "turbo" in model_name:
                    full_model_name = "gpt-4-0125-preview"
                elif "gpt-4" in model_name:
                    full_model_name = "gpt-4-0613"
                elif "gpt-3.5-turbo" in model_name:
                    full_model_name = "gpt-3.5-turbo-0125"
                    full_model_name = model_name
            model_api_list.append(
                {
                    "short_model_name": model_name,
                    "model_name": full_model_name,
                    "api_provider": api_provider,
                    "api_key": api_key,
                    "base_url": base_url,
                }
            )
        return model_api_list

    def setup_chat_client(
            self, 
            model_name:str,
            short_model_name: str,
            api_provider: str,
            api_key: str,
            base_url: str
        ):
        if api_provider == "vllm" or api_provider == "openai":
            self.chat_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self.api_provider = api_provider
        elif api_provider == "together":
            self.chat_client = Together(
                api_key=api_key,
                base_url=base_url,
            )
            self.api_provider = api_provider
        elif api_provider == "google":
            genai.configure(api_key=api_key)
            self.chat_client = genai.GenerativeModel(model_name)
            self.api_provider = api_provider
        elif api_provider == "anthropic":
            self.chat_client = Anthropic(
                api_key=api_key,
                base_url=base_url,
            )
            self.api_provider = api_provider
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")


    def run_lm(
            self, 
            model_name, 
            system_prompt, 
            user_prompt,
            temperature: float = 0.0,
            max_tokens: int = 100,
            max_retries = 20
        ):
        num_retries = 0
        delay = 1.0
        error = None
        api_provider = self.api_provider
        while num_retries <= max_retries:
            try:
                if api_provider == "vllm" or api_provider == "together" or api_provider == "openai":
                    completion = self.chat_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                
                    return completion.choices[0].message
                elif api_provider == "google":
                    completion = self.chat_client.generate_content(
                        prompt=user_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return completion.text
                elif api_provider == "anthropic":
                    completion = self.chat_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return completion.choices[0].message
                else:
                    utils.print_error("ERROR", f"Unknown API provider: {api_provider}")
                    break

            # Raise exceptions for any errors specified
            except Exception as e:
                # Sleep for the delay
                time.sleep(delay)
                # Increment the delay
                delay *= 2 * (1 + 1 * random.random())
                # Set the error to the last exception
                error = e
                # Increment retries
                num_retries += 1

                utils.print_error("Retrying  after error", f" {e} (retry {num_retries} of {max_retries})")

            if error is not None:
                raise error
        return None

    def run_lm_chain(
        self, 
        row: InputData, 
        model_name: str, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        max_retries: int = 20
    ):  
        lm_response = self.run_lm(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries
        )          
        lm_response = lm_response.content 

        if self.debug:
            utils.print_user_prompt(row.design_name + "/" + row.task_id, user_prompt)
            utils.print_lm_response(model_name, lm_response)
        return lm_response

    def run_experiment_single_model(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_question_chain: list[tuple[str, str]] = [],
    ):
        raise NotImplementedError("run_experiment_single_model not implemented")

    def run_benchmark(
        self,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_strategy: str = "default",
    ):  
        cot_question_chain = self.get_cot_strategy(cot_strategy)
        for model_dict in self.model_api_list:
            self.setup_chat_client(**model_dict)
            results = self.run_experiment_single_model(
                model_dict["model_name"],
                temperature=temperature,
                max_tokens=max_tokens,
                cot_question_chain=cot_question_chain,
            )
            self.save_results(model_dict["short_model_name"], results)
        return results

    def save_results(self, model_name: str, results: list[LMResult]):
        model_name = model_name.split('/')[-1].replace(" ", "_")
        results_df = pd.DataFrame([asdict(response) for response in results])
        results_df.to_csv(
            os.path.join(self.save_dir, f"{model_name}_{self.experiment_id}.csv"),
            index=False,
        )


class NL2SVALauncher(BenchmarkLauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, debug
        )

    def generate_system_prompt(self):
        return prompts_svagen_nl2sva.SVAGEN_HEADER

    def generate_question_prompt(sefl, row: InputData):
        return (
            prompts_svagen_nl2sva.SVAGEN_QUESTION_PREAMBLE
            + row.prompt
            + "\n"
            + prompts_svagen_nl2sva.SVAGEN_QUESTION_POSTAMBLE
        )

    def package_testbench(self, row: InputData, lm_response: str):
        question_prompt = self.generate_question_prompt(row)
        reference_assertion_text = row.ref_solution.replace("asrt", "reference")
        assertion_text = self.parse_code_response(lm_response)

        # retrieve question text
        commented_question_text = "\n//".join(question_prompt.split("\n"))
        testbench_text = row.testbench
        packaged_tb_text = (
            testbench_text.split("endmodule")[0]
            + "\n\n"
            + commented_question_text
            + "\n\n"
            + reference_assertion_text
            + "\n\n"
            + assertion_text
            + "\n\n"
            + "endmodule"
        )
        return packaged_tb_text

    def run_experiment_single_model(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_question_chain: list[tuple[str, str]] = [],
    ):
        results = []
        system_prompt = self.generate_system_prompt()
        for row in self._build_iterator(model_name):
            user_prompt = self.generate_user_prompt_prefix(row)
            if not cot_question_chain:
                user_prompt += "\n" + self.generate_question_prompt(row)
            else:
                raise NotImplementedError("COT question chain not implemented")
            lm_response = self.run_lm_chain(
                row=row, 
                model_name=model_name, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if self.debug:
                utils.print_lm_response("reference", row.ref_solution)
            packaged_tb_text = self.package_testbench(row, lm_response)
            response = LMResult(
                experiment_id=self.experiment_id,
                task_id=row.design_name + "_" + row.task_id,
                model_name=model_name,
                response=lm_response,
                ref_solution=row.ref_solution,
                user_prompt=user_prompt,
                output_tb=packaged_tb_text,
                design_rtl="\n",
                cot_response="cot_response\n",
            )
            results.append(response)
        return results


class NL2SVAHumanLauncher(NL2SVALauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, debug
        )

    def generate_user_prompt_prefix(self, row: InputData):
        if self.num_icl_examples == 0:
            user_prompt_prefix = ""
        elif self.num_icl_examples == 1:
            user_prompt_prefix = prompts_svagen_nl2sva.SVAGEN_HUMAN_ICL_EXAMPLE_1
        elif self.num_icl_examples == 3:
            user_prompt_prefix = prompts_svagen_nl2sva.SVAGEN_HUMAN_ICL_EXAMPLE_3
        else:
            utils.print_error(
                "ERROR",
                f"Unsupported number of in-context examples: {self.num_icl_examples}",
            )
        user_prompt_prefix += "\n\n" + prompts_svagen_nl2sva.SVAGEN_TB_PREAMBLE
        user_prompt_prefix += "\n" + row.testbench
        return user_prompt_prefix


class NL2SVAMachineLauncher(NL2SVALauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_machine",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, debug
        )

    def generate_user_prompt_prefix(self, row: InputData):
        if self.num_icl_examples == 0:
            user_prompt_prefix = ""
        elif self.num_icl_examples == 1:
            user_prompt_prefix = prompts_svagen_nl2sva.SVAGEN_MACHINE_ICL_EXAMPLE_1
        elif self.num_icl_examples == 2:
            user_prompt_prefix = prompts_svagen_nl2sva.SVAGEN_MACHINE_ICL_EXAMPLE_2
        elif self.num_icl_examples == 3:
            user_prompt_prefix = prompts_svagen_nl2sva.SVAGEN_MACHINE_ICL_EXAMPLE_3
        else:
            utils.print_error(
                "ERROR",
                f"unsupported number of in-context examples: {self.num_icl_examples}",
            )
        # user_prompt = prompts_svagen_nl2sva.SVAGEN_MACHINE_ICL_EXAMPLE
        return user_prompt_prefix


class Design2SVALauncher(BenchmarkLauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "design2sva",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, debug
        )

    def generate_system_prompt(self):
        return prompts_svagen_design2sva.SVAGEN_HEADER

    def generate_user_prompt_prefix(self, row: InputData):
        testbench_text = row.testbench
        testbench_text = testbench_text.split("assign tb_reset")[0]
        testbench_text += "assign tb_reset = (reset_ == 1'b0);\n"

        user_prompt_prefix = prompts_svagen_design2sva.SVAGEN_DUT_PREAMBLE
        user_prompt_prefix += row.prompt
        user_prompt_prefix += "\n\n" + prompts_svagen_design2sva.SVAGEN_TB_PREAMBLE
        user_prompt_prefix += "\n" + testbench_text
        return user_prompt_prefix

    def package_testbench(self, row: InputData, lm_response: str):
        testbench_lm = self.parse_code_response(lm_response)
        if not "endmodule" in testbench_lm:
            testbench_lm += "\nendmodule"
        bind_statement = row.testbench.split("endmodule")[-1]
        
        packaged_tb_text = (
            testbench_lm
            + bind_statement
        )
        return packaged_tb_text

    def get_cot_strategy(self, cot_strategy: str) -> list[tuple[str, str]]:
        if cot_strategy == "default":
            return [("question", prompts_svagen_design2sva.SVAGEN_QUESTION)]
        elif cot_strategy == "plan-act":
            return [
                ("plan", prompts_svagen_design2sva.SVAGEN_PLANNING_QUESTION),
                ("question", prompts_svagen_design2sva.SVAGEN_QUESTION),
            ]
        elif cot_strategy == "plan-model-act":
            return [
                ("plan", prompts_svagen_design2sva.SVAGEN_PLANNING_QUESTION),
                ("model", prompts_svagen_design2sva.SVAGEN_MODELING_QUESTION),
                ("question", prompts_svagen_design2sva.SVAGEN_QUESTION),
            ]
        else:
            utils.print_error("ERROR", f"Unsupported COT strategy: {cot_strategy}")
            raise NotImplementedError

    def run_experiment_single_model(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_question_chain: list[tuple[str, str]] = [
            ("question", prompts_svagen_design2sva.SVAGEN_QUESTION)
        ],
    ):
        results = []
        # generate system prompt
        system_prompt = self.generate_system_prompt()

        # iterate over dataset
        for row in self._build_iterator(model_name):
            if self.debug:
                print(len(self._build_iterator(model_name)))
            # generate user prompt
            user_prompt = self.generate_user_prompt_prefix(row)
            cot_responses = {}

            # iterate over COT question chain
            for q_type, q_str in cot_question_chain:
                # append question to user prompt
                user_prompt += "\n" + q_str
                # run LM chain
                lm_response = self.run_lm_chain(
                    row=row, 
                    model_name=model_name, 
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if q_type != cot_question_chain[-1][0]:
                    cot_responses[q_type] = lm_response
                    user_prompt += "\n" + lm_response
            
            # stringify cot_responses
            cot_response = "cot_response\n"
            for key, value in cot_responses.items():
                cot_response += f"{key}: {value}\n"

            # package testbench
            packaged_tb_text = self.package_testbench(row, lm_response)

            # construct response
            response = LMResult(
                experiment_id=self.experiment_id,
                task_id=row.design_name + "_" + row.task_id,
                model_name=model_name,
                response=lm_response,
                ref_solution=row.ref_solution,
                user_prompt=user_prompt,
                output_tb=packaged_tb_text,
                design_rtl=row.prompt,               
                cot_response=cot_responses,
            )
            results.append(response)
        return results
    

class HelperGenLauncher(BenchmarkLauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "helpergen",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 0,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, debug
        )

    def generate_system_prompt(self):
        return prompts_avr_helpergen.AGR_HELPERGEN_HEADER

    def generate_user_prompt_prefix(self, row: InputData):
        user_prompt_prefix = prompts_avr_helpergen.AGR_HELPERGEN_DUT_PREAMBLE
        user_prompt_prefix += row.prompt
        user_prompt_prefix += "\n\n" + prompts_avr_helpergen.AGR_HELPERGEN_TB_PREAMBLE
        user_prompt_prefix += row.testbench
        return user_prompt_prefix


    def package_testbench(self, row: InputData, lm_response: str):
        assertion_text = self.parse_code_response(lm_response)

        # retrieve question text
        testbench_text = row.testbench
        packaged_tb_text = (
            testbench_text.split("endmodule")[0]
            + "\n\n"
            + "\n\n"
            + assertion_text
            + "\n\n"
            + "endmodule"
        )
        return packaged_tb_text

    def get_cot_strategy(self, cot_strategy: str) -> list[tuple[str, str]]:
        if cot_strategy == "default":
            return [("question", prompts_avr_helpergen.AGR_HELPERGEN_QUESTION_COT_ANSWER)]
        elif cot_strategy == "plan-act":
            return [
                ("plan", prompts_avr_helpergen.AGR_HELPERGEN_QUESTION_COT_THOUGHT),
                ("question", prompts_avr_helpergen.AGR_HELPERGEN_QUESTION_COT_ANSWER),
            ]
        else:
            utils.print_error("ERROR", f"Unsupported COT strategy: {cot_strategy}")
            raise NotImplementedError

    def run_experiment_single_model(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_question_chain: list[tuple[str, str]] = [
            ("question", prompts_avr_helpergen.AGR_HELPERGEN_QUESTION_COT_ANSWER)
        ],
    ):
        results = []
        # generate system prompt
        system_prompt = self.generate_system_prompt()

        # iterate over dataset
        for row in self._build_iterator(model_name):
            if self.debug:
                print(len(self._build_iterator(model_name)))
            # generate user prompt
            user_prompt = self.generate_user_prompt_prefix(row)
            cot_responses = {}

            # iterate over COT question chain
            for q_type, q_str in cot_question_chain:
                # append question to user prompt
                user_prompt += "\n" + q_str
                # run LM chain
                lm_response = self.run_lm_chain(
                    row=row, 
                    model_name=model_name, 
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if q_type != cot_question_chain[-1][0]:
                    cot_responses[q_type] = lm_response
                    user_prompt += "\n" + lm_response
            
            # stringify cot_responses
            cot_response = "cot_response\n"
            for key, value in cot_responses.items():
                cot_response += f"{key}: {value}\n"

            # package testbench
            packaged_tb_text = self.package_testbench(row, lm_response)

            # construct response
            response = LMResult(
                experiment_id=self.experiment_id,
                task_id=row.design_name + "_" + row.task_id,
                model_name=model_name,
                response=lm_response,
                ref_solution=row.ref_solution,
                user_prompt=user_prompt,
                output_tb=packaged_tb_text,
                design_rtl=row.prompt,               
                cot_response=cot_responses,
            )
            results.append(response)
        return results