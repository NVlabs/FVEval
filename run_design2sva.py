import argparse
from datetime import datetime
import pathlib

from fv_eval import data, utils, benchmark_launcher

if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Run LLM Inference for the FVEval-Design2SVA Benchmark")
    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        help="path to input dataset directory, potentially holding multiple .csv files",
    )
    parser.add_argument(
        "--temperature", type=float, help="LLM decoder sampling temperature", default=0.0
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        help="models to run with, ;-separated",
        default="gpt-4-turbo;gpt-3.5-turbo;llama-3-70b;mixtral-8x22b",
    )
    parser.add_argument(
        "--cot_strategy",
        "-c",
        type=str,
        help="chain of thought strategy: default, plan-act, plan-model-act",
        default="plan-model-act",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
    )

    args = parser.parse_args()
    if not args.dataset_dir:
        dataset_dir = ROOT / "data_svagen" / "design2sva" / "data"
        assert dataset_dir.is_dir()
        dataset_dir = dataset_dir.as_posix()
    
    timestamp_str = datetime.now().strftime("%Y%m%d%H")
    save_dir = f"results_design2sva/{args.cot_strategy}/{timestamp_str}"
    utils.mkdir_p(save_dir)

    dataset_paths = data.read_datasets_from_dir(dataset_dir)
    for dataset_path in dataset_paths:
        bmark_launcher = benchmark_launcher.Design2SVALauncher(
            save_dir=save_dir,
            dataset_path=dataset_path,
            task="design2sva",
            model_name_list=args.models.split(";"),
            debug=args.debug
        )
        bmark_launcher.run_benchmark(
            temperature=args.temperature,
            max_tokens=2000,
            cot_strategy=args.cot_strategy
        )
        
