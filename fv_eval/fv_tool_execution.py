# from collections import defaultdict, Counter
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import List, Union, Iterable, Dict, Tuple, Optional
# import itertools
import os
import shutil
import subprocess
import multiprocessing
import pathlib
from typing import Optional, Callable, Dict

# import pandas as pd
# from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent

# def launch_jg_single(
#     tcl_file_path: str,
#     tb_dir: str,
#     exp_name: str,
#     exp_id: str,
# )-> None:
#     tmp_jg_proj_dir = tb_dir + f"/jg/{exp_name}/{exp_id}"
#     if os.path.isdir(tmp_jg_proj_dir):
#      shutil.rmtree(tmp_jg_proj_dir)
#     jg_command = [
#         "jg", "-fpv", "-batch", "-tcl", tcl_file_path,
#         "-define", "EXP_NAME", exp_name,
#         "-define", "EXP_ID", exp_id,
#         "-define", "TB_DIR", tb_dir,
#         "-proj", tmp_jg_proj_dir,
#     ]
#     result = subprocess.run(jg_command, capture_output=True, text=True)
#     return result.stdout

def launch_jg(
    tcl_file_path: str,
    sv_dir: str,
    experiment_id: str,
    task_id: str,
    output_queue: multiprocessing.Queue = None,
) -> None:
    tmp_jg_proj_dir = sv_dir + f"/jg/{experiment_id}_{task_id}"
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)
    jg_command = [
        "jg",
        "-fpv",
        "-batch",
        "-tcl",
        tcl_file_path,
        "-define",
        "EXP_ID",
        experiment_id,
        "-define",
        "TASK_ID",
        task_id,
        "-define",
        "SV_DIR",
        sv_dir,
        "-proj",
        tmp_jg_proj_dir,
        "-allow_unsupported_OS",
    ]
    result = subprocess.run(jg_command, capture_output=True, text=True)
    return result.stdout.strip()


def launch_jg_with_queue(
    tcl_file_path: str,
    sv_dir: str,
    experiment_id: str,
    task_id: str,
    output_queue: multiprocessing.Queue = None,
) -> None:
    tmp_jg_proj_dir = sv_dir + f"/jg/{experiment_id}_{task_id}"
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)
    jg_command = [
        "jg",
        "-fpv",
        "-batch",
        "-tcl",
        tcl_file_path,
        "-define",
        "EXP_ID",
        experiment_id,
        "-define",
        "TASK_ID",
        task_id,
        "-define",
        "SV_DIR",
        sv_dir,
        "-proj",
        tmp_jg_proj_dir,
        "-allow_unsupported_OS",
    ]
    result = subprocess.run(jg_command, capture_output=True, text=True)
    output_queue.put(result.stdout.strip())

def launch_jg_custom_equiv_check(
    tcl_file_path: str,
    sv_dir: str,
    experiment_id: str,
    task_id: str,
    lm_assertion_text: str,
    ref_assertion_text: str,
    signal_list_text: str,
    output_queue: multiprocessing.Queue = None,
) -> None:
    tmp_jg_proj_dir = sv_dir + f"/jg/{experiment_id}_{task_id}"
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)
    jg_command = [
        "jg",
        "-fpv",
        "-batch",
        "-tcl",
        tcl_file_path,
        "-define",
        "LM_ASSERT_TEXT",
        lm_assertion_text,
        "-define",
        "REF_ASSERT_TEXT",
        ref_assertion_text,
        "-define",
        "SIGNAL_LIST",
        signal_list_text,
        "-define",
        "EXP_ID",
        experiment_id,
        "-define",
        "TASK_ID",
        task_id,
        "-define",
        "SV_DIR",
        sv_dir,
        "-proj",
        tmp_jg_proj_dir,
        "-allow_unsupported_OS",
    ]
    result = subprocess.run(jg_command, capture_output=True, text=True)
    return result.stdout.strip()


def launch_jg_with_queue_custom_equiv_check(
    tcl_file_path: str,
    sv_dir: str,
    experiment_id: str,
    task_id: str,
    lm_assertion_text: str,
    ref_assertion_text: str,
    signal_list_text: str,
    output_queue: multiprocessing.Queue = None,
) -> None:
    tmp_jg_proj_dir = sv_dir + f"/jg/{experiment_id}_{task_id}"
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)
    jg_command = [
        "jg",
        "-fpv",
        "-batch",
        "-tcl",
        tcl_file_path,
        "-define",
        "LM_ASSERT_TEXT",
        lm_assertion_text,
        "-define",
        "REF_ASSERT_TEXT",
        ref_assertion_text,
        "-define",
        "SIGNAL_LIST",
        signal_list_text,
        "-define",
        "EXP_ID",
        experiment_id,
        "-define",
        "TASK_ID",
        task_id,
        "-define",
        "SV_DIR",
        sv_dir,
        "-proj",
        tmp_jg_proj_dir,
        "-allow_unsupported_OS",
    ]
    result = subprocess.run(jg_command, capture_output=True, text=True)
    output_queue.put(result.stdout.strip())
