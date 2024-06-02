from dataclasses import dataclass, asdict
import glob
import os
import pathlib
from typing import Iterable, Dict

import pandas as pd


@dataclass
class TextSimilarityEvaluationResult:
    experiment_id: str
    task_id: str
    model_name: str
    bleu: float
    rouge: float
    exact_match: float


@dataclass
class JGEvaluationResult:
    experiment_id: str
    task_id: str
    model_name: str
    syntax: float
    functionality: float
    func_relaxed: float
    # bound_improve: int


@dataclass
class LMResult:
    experiment_id: str
    task_id: str
    model_name: str
    response: str
    ref_solution: str
    design_rtl: str
    output_tb: str
    user_prompt: str
    cot_response: str


@dataclass
class InputData:
    design_name: str
    task_id: str
    prompt: str
    ref_solution: str
    testbench: str


ROOT = pathlib.Path(__file__).parent.parent

"""
    Read SystemVerilog files from given data directory
    args: directory path to SV files (str)
    returns: Dict of (SV file descripter, raw SV text)
"""


def read_sv_from_dir(data_dir: str, is_testbench: bool = False) -> Dict[str, str]:
    dut_texts = {}
    file_suffix = ".sva" if is_testbench else ".sv"
    for dut_sv in glob.glob(data_dir + f"/*{file_suffix}"):
        svtext = ""
        with open(dut_sv, "r") as f:
            svtext = f.read()
        dut_name = os.path.basename(dut_sv).split(file_suffix)[0]
        dut_texts.update({dut_name: svtext.strip()})
    return dut_texts


"""
    Read dataset CSV files from given data directory
    args: directory path to SV files (str)
    returns: Dict of (SV file descripter, raw SV text)
"""


def read_datasets_from_dir(data_dir: str) -> Dict[str, str]:
    file_suffix = ".csv"
    return glob.glob(data_dir + f"/*{file_suffix}")


# def write_nl2sva_human_testbenches(
#     dataset_path: str,
#     llm_response_save_dir: str,
#     save_dir: str = f"{ROOT}/tmp"
# ):
#     df = pd.read_csv(dataset_path)
#     result_csvs = glob.glob(f"{llm_response_save_dir}/*.csv")

#     for csvfile in result_csvs:
#         llm_response_df = pd.read_csv(csvfile)
#         experiment_tag_name = csvfile.split("/")[-1].split(".csv")[0]
#         utils.mkdir_p(f"{save_dir}/{experiment_tag_name}")

#         for idx, row in llm_response_df.iterrows():
#             # retrieve testbench (sv) string
#             testbench_text = df.loc[(df['dut_name'] == row.dut_name) & (df['task_id'] == row.task_id)]["testbench_context"].values[0]

#             # ref assertion
#             reference_assertion_text = row.ref_solution
#             reference_assertion_text = reference_assertion_text.replace("asrt", "reference")
#             assertion_text = row.lm_code_response

#             # retrieve question text
#             commented_question_text = "\n//".join(row.question_prompt.split('\n'))

#             if not isinstance(assertion_text, str):
#                 assertion_text = "error"
#             packaged_tb_text = (
#                 testbench_text.split("endmodule")[0]
#                 + "\n\n"
#                 + commented_question_text
#                 + "\n\n"
#                 + reference_assertion_text
#                 + "\n\n"
#                 + assertion_text
#                 + "\n\n"
#                 + "endmodule"
#             )
#             with open(f"{save_dir}/{experiment_tag_name}/{idx}.sv", "w") as f:
#                 f.write(packaged_tb_text)


# def write_nl2sva_machine_testbenches(
#     dummy_testbench_rtl_path: str,
#     llm_response_save_dir: str,
#     save_dir: str = f"{ROOT}/tmp"
# ):
#     testbench_dict = read_sv_from_dir(data_dir=dummy_testbench_rtl_path,)
#     testbench_text = testbench_dict['dummy']
#     result_csvs = glob.glob(f"{llm_response_save_dir}/*.csv")

#     for csvfile in result_csvs:
#         llm_response_df = pd.read_csv(csvfile)
#         experiment_tag_name = csvfile.split("/")[-1].split(".csv")[0]
#         utils.mkdir_p(f"{save_dir}/{experiment_tag_name}")

#         for idx, row in llm_response_df.iterrows():
#             # ref assertion
#             reference_assertion_text = row.ref_solution
#             reference_assertion_text = reference_assertion_text.replace("assert", "reference: assert")
#             assertion_text = row.lm_code_response
#             if assertion_text and isinstance(assertion_text, str):
#                 try:
#                     assertion_text = assertion_text.replace("assert", "lm_assertion: assert")
#                 except:
#                     import pdb; pdb.set_trace()
#             # retrieve question text
#             commented_question_text = "\n//".join(row.question_prompt.split('\n'))

#             if not isinstance(assertion_text, str):
#                 assertion_text = "error"
#             packaged_tb_text = (
#                 testbench_text.split("endmodule")[0]
#                 + "\n\n"
#                 + commented_question_text
#                 + "\n\n"
#                 + reference_assertion_text
#                 + "\n\n"
#                 + assertion_text
#                 + "\n\n"
#                 + "endmodule"
#             )
#             with open(f"{save_dir}/{experiment_tag_name}/{idx}.sv", "w") as f:
#                 f.write(packaged_tb_text)

# def write_design2sva_testbenches(
#     llm_response_save_dir: str,
#     save_dir: str = f"{ROOT}/tmp"
# ):
#     result_csvs = glob.glob(f"{llm_response_save_dir}/*.csv")

#     # read out LM response results per each model (stored as separate .csvs)
#     for csvfile in result_csvs:
#         llm_response_df = pd.read_csv(csvfile)
#         experiment_tag_name = csvfile.split("/")[-1].split(".csv")[0].split("_")[0]
#         utils.mkdir_p(f"{save_dir}/{experiment_tag_name}")

#         for idx, row in llm_response_df.iterrows():
#             # ref assertion
#             with open(f"{save_dir}/{experiment_tag_name}/{idx}.sv", "w") as f:
#                 f.write(row.lm_code_response)
