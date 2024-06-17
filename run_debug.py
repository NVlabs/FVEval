import argparse
import pathlib

from fv_eval import evaluation, utils

ROOT = pathlib.Path(__file__).parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run JG-based Evaluation of FVEval-SVAGen Results"
    )
    parser.add_argument(
        "--llm_output_dir",
        "-i",
        type=str,
        help="path to LLM results dir",
    )
    parser.add_argument(
        "--save_dir",
        "-o",
        type=str,
        help="path to save JG eval results",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help="specific model name to evaluate for",
        default="",
    )
    parser.add_argument(
        "--temp_dir",
        "-t",
        type=str,
        help="path to temp dir",
        default=ROOT / "tmp",
    )
    parser.add_argument(
        "--cleanup_temp",
        type=bool,
        help="Whether to clean up the temp dir afterwards",
        default=True,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="task you are evaluating for",
        default="nl2sva_human",
    )
    parser.add_argument(
        "--nparallel",
        "-n",
        type=int,
        help="parallel JG jobs",
        default=8,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug ",
    )

    args = parser.parse_args()
    if not args.llm_output_dir:
        utils.print_error(
            "Argument Error",
            "empty path to llm_output_dir. Provide correct args to --llm_output_dir",
        )
        raise ValueError("empty path to llm_output_dir")
    if not args.save_dir:
        save_dir = pathlib.Path(args.llm_output_dir) / "eval"
    else:
        save_dir = args.save_dir

    save_dir = save_dir.as_posix()
    tmp_dir = args.temp_dir / args.task
    tmp_dir = tmp_dir.as_posix()

    if "nl2sva" in args.task:
        if "human" in args.task:
            evaluator = evaluation.NL2SVAHumanEvaluator(
                llm_output_dir=args.llm_output_dir,
                model_name=args.model_name,
                temp_dir=tmp_dir,
                save_dir=save_dir,
                cleanup_temp_files=args.cleanup_temp,
                parallel_jobs=args.nparallel,
                debug=args.debug,
            )
        elif "machine" in args.task:
            evaluator = evaluation.NL2SVAMachineEvaluator(
                llm_output_dir=args.llm_output_dir,
                model_name=args.model_name,
                temp_dir=tmp_dir,
                save_dir=save_dir,
                cleanup_temp_files=args.cleanup_temp,
                parallel_jobs=args.nparallel,
                debug=args.debug,
            )
    elif "design2sva" in args.task:
        evaluator = evaluation.Design2SVAEvaluator(
            llm_output_dir=args.llm_output_dir,
            model_name=args.model_name,
            temp_dir=tmp_dir,
            save_dir=save_dir,
            cleanup_temp_files=args.cleanup_temp,
            parallel_jobs=args.nparallel,
            debug=args.debug,
        )
    elif "helpergen" in args.task:
        evaluator = evaluation.HelperGenEvaluator(
            llm_output_dir=args.llm_output_dir,
            temp_dir=tmp_dir,
            save_dir=save_dir,
            cleanup_temp_files=args.cleanup_temp,
            parallel_jobs=args.nparallel,
            debug=args.debug,
        )
    evaluator.write_design_sv(results_list=evaluator.llm_results[0][1])
    evaluator.write_testbench_sv(results_list=evaluator.llm_results[0][1])
