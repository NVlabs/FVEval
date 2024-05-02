import errno
import gzip
import json
import os
from typing import Iterable, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_error(header:str, body:str):
    print(f"{bcolors.FAIL}{header}:{bcolors.ENDC}\n{body}")

def print_lm_response(header:str, body:str):
    print(f"{bcolors.OKCYAN}{header}:{bcolors.ENDC}\n{bcolors.BOLD}{body}{bcolors.ENDC}")

def print_user_prompt(header:str, body:str):
    body = body.split("Question:")[-1]
    body = "Question:" + body
    print(f"{bcolors.OKBLUE}{header}:{bcolors.ENDC}\n{bcolors.BOLD}{body}{bcolors.ENDC}")

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