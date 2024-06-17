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
import shutil
import subprocess
import multiprocessing
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

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
