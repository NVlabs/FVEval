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

from dataclasses import asdict
import os
import time
import random

from openai import OpenAI
from together import Together
from anthropic import Anthropic
import google.generativeai
import pandas as pd
from tqdm import tqdm

from fv_eval import (
    prompts_design2sva,
    prompts_nl2sva_machine,
    prompts_nl2sva_human,
    utils,
)
from fv_eval.data import InputData, LMResult

"""
Base Class to Launch LLM Inference on Tasks in FVEval
"""


class BenchmarkLauncher(object):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        use_cot: bool = False,
        debug: bool = False,
    ):
        self.save_dir = save_dir
        self.dataset_path = dataset_path
        df = pd.read_csv(dataset_path)
        self.dataset = [InputData(**row) for _, row in df.iterrows()]
        self.use_cot = use_cot
        # convert dataset into list of InputData

        self.chat_client = None
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

    def _prepare_models(self, model_name_list: str):
        TOGETHER_MODEL_DICT = {
            "llama-3-8b": "meta-llama/Llama-3-8b-chat-hf",
            "llama-3-70b": "meta-llama/Llama-3-70b-chat-hf",
            "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "gemma-2-27b": "google/gemma-2-27b-it",
            "dbrx" : "databricks/dbrx-instruct",
            "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
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
                full_model_name = (
                    OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                    )
                    .models.list()
                    .data[0]
                    .id
                )
            elif model_name in TOGETHER_MODEL_DICT:
                api_provider = "together"
                api_key = os.getenv("TOGETHER_API_KEY")
                base_url = "https://api.together.xyz/v1"
                full_model_name = TOGETHER_MODEL_DICT[model_name]
            elif "claude" in model_name:
                api_provider = "anthropic"
                api_key = os.getenv("ANTHROPIC_API_KEY")
                base_url = "https://api.anthropic.com/v1"
                if "3.5" in model_name:
                    full_model_name = "claude-3-5-sonnet-20240620"
                elif "opus" in model_name:
                    full_model_name = "claude-3-opus-20240229"
                elif "sonnet" in model_name:
                    full_model_name = "claude-3-sonnet-20240229"
                elif "haiku" in model_name:
                    full_model_name = "claude-3-haiku-20240307"
                else:
                    raise ValueError(f"Unknown Anthropic model: {model_name}")
            elif "gemini" in model_name:
                api_provider = "google"
                api_key = os.getenv("GOOGLE_API_KEY")
                if "flash" in model_name:
                    full_model_name = "gemini-1.5-flash"
                else:
                    full_model_name = "gemini-1.5-pro"
                base_url = ""
            elif "gpt" in model_name:
                api_provider = "openai"
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = "https://api.openai.com/v1"
                if "gpt-4-turbo" in model_name:
                    full_model_name = "gpt-4-0125-preview"
                elif model_name == "gpt-4o":
                    full_model_name = "gpt-4o-2024-05-13"
                elif model_name == "gpt-4":
                    full_model_name = "gpt-4-0613"
                elif "gpt-3.5-turbo" in model_name:
                    full_model_name = "gpt-3.5-turbo-0125"
                    full_model_name = model_name
                else:
                    raise ValueError(f"Unknown OpenAI model: {model_name}")
            else:
                raise ValueError(f"Unknown model: {model_name}")
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
        model_name: str,
        short_model_name: str,
        api_provider: str,
        api_key: str,
        base_url: str,
    ):  
        self.api_provider = api_provider
        if api_provider == "vllm" or api_provider == "openai":
            self.chat_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        elif api_provider == "together":
            self.chat_client = Together(
                api_key=api_key,
                base_url=base_url,
            )
        elif api_provider == "anthropic":
            self.chat_client = Anthropic(
                api_key=api_key,
                base_url=base_url,
            )
        elif api_provider == "google":
            self.chat_client = google.generativeai.GenerativeModel(
                model_name=model_name
            )
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")

    def run_lm(
        self,
        model_name,
        system_prompt,
        user_prompt,
        temperature: float = 0.0,
        max_tokens: int = 100,
        max_retries: int = 40,
        num_cases: int = 1,
    ):
        num_retries = 0
        delay = 1.0
        error = None
        api_provider = self.api_provider
        if temperature == 0.0:
            top_p = 1.0
        else:
            top_p = 0.95
        
        while num_retries <= 20:
            try:
                if (
                    api_provider == "vllm"
                    or api_provider == "together"
                    or api_provider == "openai"
                ):
                    completion = self.chat_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_cases,
                    )
                    time.sleep(5)
                    return [choice.message.content for choice in completion.choices]
                elif api_provider == "anthropic":
                    completion = Anthropic().messages.create(
                        model=model_name,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    time.sleep(10)
                    return [textblock.text for textblock in completion.content]
                elif api_provider == "google":      
                        history=[
                            {"role": "user", "parts": system_prompt},
                            {"role": "model", "parts": "Understood."},
                        ]
                    )
                    
                    completion = chat.send_message(
                        user_prompt,
                        generation_config=google.generativeai.types.GenerationConfig(
                            candidate_count=1,
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                        ),
                    )
                    return [completion.text]

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

                utils.print_error(
                    "Retrying  after error",
                    f" {e} (retry {num_retries} of {max_retries})",
                )

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
        max_retries: int = 20,
        num_cases: int = 1,
    ):
        lm_response_list = self.run_lm(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            num_cases=num_cases,
        )

        if self.debug:
            utils.print_user_prompt(row.design_name + "/" + row.task_id, user_prompt)
            utils.print_lm_response(model_name, lm_response_list[0])
        return lm_response_list

    def run_experiment_single_model(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_question_chain: list[tuple[str, str]] = [],
        num_cases: int = 1,
    ):
        raise NotImplementedError("run_experiment_single_model not implemented")

    def run_benchmark(
        self,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_strategy: str = "default",
        num_cases: int = 1,
    ):
        cot_question_chain = self.get_cot_strategy(cot_strategy)
        for model_dict in self.model_api_list:
            self.setup_chat_client(**model_dict)
            results = self.run_experiment_single_model(
                model_dict["model_name"],
                temperature=temperature,
                max_tokens=max_tokens,
                cot_question_chain=cot_question_chain,
                num_cases=num_cases,
            )
            self.save_results(model_dict["short_model_name"], results)
        return results

    def save_results(self, model_name: str, results: list[LMResult]):
        model_name = model_name.split("/")[-1].replace(" ", "_")
        results_df = pd.DataFrame([asdict(response) for response in results])
        results_df.to_csv(
            os.path.join(self.save_dir, f"{model_name}_{self.experiment_id}.csv"),
            index=False,
        )


"""
LLM Inference Launcher Specific to NL2SVA Benchmark Tasks
- Launcher classes for NL2SVA-Human and NL2SVA-Machine inherit this class
"""


class NL2SVALauncher(BenchmarkLauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, use_cot, debug
        )

    def package_testbench(self, row: InputData, lm_response: str):
        question_prompt = self.generate_question_prompt(row)
        reference_assertion_text = row.ref_solution.replace("asrt", "reference")
        assertion_text = utils.parse_code_response(lm_response)

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
        num_cases: int = 1,
    ):
        results = []
        system_prompt = self.generate_system_prompt()
        for row in self._build_iterator(model_name):
            user_prompt = self.generate_user_prompt_prefix(row)
            if not cot_question_chain:
                user_prompt += "\n" + self.generate_question_prompt(row)
            else:
                raise NotImplementedError("COT question chain not implemented")
            lm_response_list = self.run_lm_chain(
                row=row,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                num_cases=num_cases,
            )
            for i, lm_response in enumerate(lm_response_list):
                if self.debug:
                    utils.print_lm_response("reference", row.ref_solution)
                packaged_tb_text = self.package_testbench(row, lm_response)
                response = LMResult(
                    experiment_id=self.experiment_id,
                    task_id=row.design_name + "_" + row.task_id + f"_trial_{i}",
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


"""
LLM Inference Launcher Specific to NL2SVA-Human
"""


class NL2SVAHumanLauncher(NL2SVALauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, use_cot, debug
        )

    def generate_system_prompt(self):
        return prompts_nl2sva_human.SVAGEN_HEADER

    def generate_question_prompt(self, row: InputData):
        question_prompt = prompts_nl2sva_human.SVAGEN_QUESTION_PREAMBLE
        question_prompt += row.prompt + "\n"
        return question_prompt + (prompts_nl2sva_human.SVAGEN_QUESTION_POSTAMBLE_COT if self.use_cot else prompts_nl2sva_human.SVAGEN_QUESTION_POSTAMBLE)

    def generate_user_prompt_prefix(self, row: InputData):
        if self.use_cot:
            user_prompt_prefix = prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_3_COT
        elif self.num_icl_examples == 0:
            user_prompt_prefix = ""
        elif self.num_icl_examples == 1:
            user_prompt_prefix = prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_1
        elif self.num_icl_examples == 3:
            user_prompt_prefix = prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_3
        else:
            utils.print_error(
                "ERROR",
                f"Unsupported number of in-context examples: {self.num_icl_examples}",
            )
        user_prompt_prefix += "\n\n" + prompts_nl2sva_human.SVAGEN_TB_PREAMBLE
        user_prompt_prefix += "\n" + row.testbench
        return user_prompt_prefix


"""
LLM Inference Launcher Specific to NL2SVA-Machine
"""


class NL2SVAMachineLauncher(NL2SVALauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_machine",
        model_name_list: list[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            save_dir, dataset_path, task, model_name_list, num_icl_examples, use_cot, debug
        )

    def generate_system_prompt(self):
        return prompts_nl2sva_machine.SVAGEN_HEADER

    def generate_question_prompt(self, row: InputData):
        question_prompt = prompts_nl2sva_machine.SVAGEN_QUESTION_PREAMBLE
        question_prompt += row.prompt + "\n"

        if self.use_cot:
            return question_prompt + prompts_nl2sva_machine.SVAGEN_QUESTION_POSTAMBLE_COT
        if self.num_icl_examples == 0:
            return (
                question_prompt
                + prompts_nl2sva_machine.SVAGEN_QUESTION_POSTAMBLE_ZERO_SHOT
            )
        return question_prompt + prompts_nl2sva_machine.SVAGEN_QUESTION_POSTAMBLE

    def generate_user_prompt_prefix(self, row: InputData):
        if self.use_cot:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_3_COT
        elif self.num_icl_examples == 0:
            user_prompt_prefix = ""
        elif self.num_icl_examples == 1:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_1
        elif self.num_icl_examples == 2:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_2
        elif self.num_icl_examples == 3:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_3
        else:
            utils.print_error(
                "ERROR",
                f"unsupported number of in-context examples: {self.num_icl_examples}",
            )
        # user_prompt = prompts_svagen_nl2sva.SVAGEN_MACHINE_ICL_EXAMPLE
        return user_prompt_prefix


"""
LLM Inference Launcher Specific to Design2SVA
"""


class Design2SVALauncher(BenchmarkLauncher):
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "design2sva",
        model_name_list: list[str] = ["gpt-4"],
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(save_dir, dataset_path, task, model_name_list, 0, use_cot, debug)

    def generate_system_prompt(self):
        return prompts_design2sva.SVAGEN_HEADER

    def generate_user_prompt_prefix(self, row: InputData):
        testbench_text = row.testbench
        testbench_text = testbench_text.split("assign tb_reset")[0]
        testbench_text += "assign tb_reset = (reset_ == 1'b0);\n"

        user_prompt_prefix = prompts_design2sva.SVAGEN_DUT_PREAMBLE
        user_prompt_prefix += row.prompt
        user_prompt_prefix += "\n\n" + prompts_design2sva.SVAGEN_TB_PREAMBLE
        user_prompt_prefix += "\n" + testbench_text
        return user_prompt_prefix

    def package_testbench(self, row: InputData, lm_response: str):
        testbench_text_prefix = row.testbench
        testbench_text_prefix = testbench_text_prefix.split("assign tb_reset")[0]
        testbench_text_prefix += "assign tb_reset = (reset_ == 1'b0);\n"
        testbench_text_postfix = "endmodule\n" + row.testbench.split("endmodule")[-1]
        lm_response = utils.parse_code_response(lm_response)
        packaged_tb_text = (
            testbench_text_prefix + "\n" + lm_response + "\n" + testbench_text_postfix
        )
        return packaged_tb_text

    def get_cot_strategy(self, cot_strategy: str) -> list[tuple[str, str]]:
        if cot_strategy == "default":
            return [
                (
                    "question",
                    prompts_design2sva.get_design2sva_direct_question_prompt(1),
                )
            ]
        elif cot_strategy == "plan-act":
            return [
                ("plan", prompts_design2sva.get_design2sva_planning_prompt(1)),
                ("question", prompts_design2sva.get_design2sva_question_prompt(1)),
            ]
        elif cot_strategy == "plan-model-act":
            return [
                ("plan", prompts_design2sva.get_design2sva_planning_prompt(1)),
                ("model", prompts_design2sva.SVAGEN_MODELING_QUESTION),
                ("question", prompts_design2sva.get_design2sva_question_prompt(1)),
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
            ("question", prompts_design2sva.get_design2sva_direct_question_prompt(1))
        ],
        num_cases: int = 1,
    ):
        results = []
        # generate system prompt
        system_prompt = self.generate_system_prompt()

        # iterate over dataset
        for row in self._build_iterator(model_name):
            # for trial_id in range(self.num_assertions):
            if self.debug:
                print(len(self._build_iterator(model_name)))
            # generate user prompt
            user_prompt = self.generate_user_prompt_prefix(row)
            cot_responses = {}

            for i in range(num_cases):
                # iterate over COT question chain
                for q_type, q_str in cot_question_chain:
                    # append question to user prompt
                    user_prompt += "\n" + q_str
                    # run LM chain
                    lm_response_list = self.run_lm_chain(
                        row=row,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        num_cases=1,
                    )
                    lm_response = lm_response_list[0]
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
                    task_id=row.design_name + "_" + row.task_id + f"_trial_{i}",
                    model_name=model_name,
                    response=lm_response,
                    ref_solution=row.ref_solution,
                    user_prompt=user_prompt,
                    output_tb=packaged_tb_text,
                    design_rtl=row.prompt,
                    cot_response=cot_response,
                )
                results.append(response)
        return results
