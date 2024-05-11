import argparse
from datetime import datetime
import pathlib


from fv_eval import data, utils, benchmark_launcher


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Run LLM Inference for the FVEval-SVAGen Benchmark")
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="path to input dataset",
    )
    parser.add_argument(
        "--temperature", type=float, help="LLM decoder sampling temperature", default=0.0
    )
    parser.add_argument(
        "--num_icl", type=int, help="number of in-context examples to use", default=3
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        help="models to run with, ;-separated",
        default="gpt-4;gpt-3.5-turbo;chipllama_70b_chat_delta_withDSFT;mixtral-chat;llama-2-70b-chat",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Evaluation mode: (1) 'human' where we evaluate NL to SVA generation against human-annotated assertions from real testbenches; (2) 'machine' where we ",
        default="human",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug ",
    )

    args = parser.parse_args()
    timestamp_str = datetime.now().strftime("%Y%m%d%H")
    # if not isinstance(args.dataset_path, str):
    #     dataset_path = args.dataset_path.as_posix()
    # else:
    #     dataset_path = args.dataset_path


    if args.debug:
        print("Executing in debug mode")

    if args.mode == "human":
        if not args.dataset_path:
            dataset_path = ROOT / "data_svagen" / "nl2sva" / "data" / "nl2sva_human.csv"
            assert dataset_path.exists()
            dataset_path = dataset_path.as_posix()
        else:
            dataset_path = args.dataset_path

        save_dir = f"results_nl2sva_human/{args.num_icl}/{timestamp_str}"
        utils.mkdir_p(save_dir)

        bmark_launcher = benchmark_launcher.NL2SVAHumanLauncher(
            save_dir=save_dir,
            dataset_path=dataset_path,
            task="nl2sva_human",
            model_name_list=args.models.split(";"),
            num_icl_examples=args.num_icl,
            debug=args.debug
        )
        bmark_launcher.run_benchmark(
            temperature=args.temperature,
            max_tokens=200
        )
    elif args.mode == "machine":
        if not args.dataset_path:
            dataset_path = ROOT / "data_svagen" / "nl2sva" / "data" / "nl2sva_machine.csv"
            assert dataset_path.exists()
            dataset_path = dataset_path.as_posix()
        else:
            dataset_path = args.dataset_path
        save_dir = f"results_nl2sva_machine/{args.num_icl}/{timestamp_str}"
        utils.mkdir_p(save_dir)

        bmark_launcher = benchmark_launcher.NL2SVAMachineLauncher(
            save_dir=save_dir,
            dataset_path=dataset_path,
            task="nl2sva_machine",
            model_name_list=args.models.split(";"),
            num_icl_examples=args.num_icl,
            debug=args.debug
        )
        bmark_launcher.run_benchmark(
            temperature=args.temperature,
            max_tokens=100
        )
    else:
        print(f"Unsupported eval mode: {args.mode}")
        raise NotImplementedError
