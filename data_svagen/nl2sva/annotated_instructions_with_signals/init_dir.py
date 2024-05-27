import argparse
import gzip
import json
import os
import pathlib

from typing import Iterable, Dict

def write_jsonl(filepath: pathlib.Path, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    Skipping None in data
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    if not filepath.exists():
        filepath.touch()
    if filepath.as_posix().endswith(".gz"):
        fp = filepath.open(mode)
        with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
            for x in data:
                if x:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        fp = filepath.open(mode)
        for x in data:
            if x:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def parse_args():
    parser = argparse.ArgumentParser(
        description="initialize FV eval directory for a suite of DUT"
    )

    parser.add_argument(
        "-n","--name",
        type=str,
        help="name of DUT suite"
    )
    parser.add_argument(
        "-i","--num_test_cases",
        type=int,
        default=10,
        help="Number of test cases (assertions)"
    )
    return parser.parse_args()

def create_input_dir(name:str, num_test_cases:int):
    dir_path = pathlib.Path(__file__).parent / name
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    # create jsonl
    sample_jsons = []
    for i in range(num_test_cases):
        json =  {"task_id": f"{name}_{i}", "prompt": "replace"}
        sample_jsons.append(json)
        sva_file = dir_path / f"{name}_{i}.sva"
        sva_file.touch()
    
    write_jsonl(dir_path / f"{name}.jsonl", data=sample_jsons, append=True)

    
if __name__ == "__main__":
    args = parse_args()
    create_input_dir(args.name, args.num_test_cases)