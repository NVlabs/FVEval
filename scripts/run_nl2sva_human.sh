# python run_nl2sva.py --mode "human" --num_icl 3
python run_evaluation.py --task "nl2sva-human" -i "results_nl2sva_human/0" -n 4
python run_evaluation.py --task "nl2sva-human" -i "results_nl2sva_human/1" -n 4
python run_evaluation.py --task "nl2sva-human" -i "results_nl2sva_human/3" -n 4