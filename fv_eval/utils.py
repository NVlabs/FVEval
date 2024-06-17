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

import errno
import gzip
import json
import os
import numpy as np
from typing import Iterable, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_error(header: str, body: str):
    print(f"{bcolors.FAIL}{header}:{bcolors.ENDC}\n{body}")


def print_lm_response(header: str, body: str):
    print(
        f"{bcolors.OKCYAN}{header}:{bcolors.ENDC}\n{bcolors.BOLD}{body}{bcolors.ENDC}"
    )


def print_user_prompt(header: str, body: str):
    body = body.split("Question:")[-1]
    body = "Question:" + body
    print(
        f"{bcolors.OKBLUE}{header}:{bcolors.ENDC}\n{bcolors.BOLD}{body}{bcolors.ENDC}"
    )


def mkdir_p(path):
    """mkdir -p in python
    Args:
        path: directory path
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    Skipping None in data
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if x:
                        gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if x:
                    fp.write((json.dumps(x) + "\n").encode("utf-8"))


def parse_code_response(lm_response_str) -> str:
    if "```systemverilog" in lm_response_str:
        lm_response_str = lm_response_str.split("```systemverilog")[-1]
    if "```" in lm_response_str:
        lm_response_str = lm_response_str.split("```")[0]
    return lm_response_str.strip()


def pass_at_k(x, n, k):
    pass_at_k_values = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        c = x[i]
        if n - c < k:
            pass_at_k_values[i] = 1.0
        else:
            range_values = np.arange(n - c + 1, n + 1)
            pass_at_k_values[i] = 1.0 - np.prod(1.0 - k / range_values)

    return pass_at_k_values