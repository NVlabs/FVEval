python run_nl2sva.py --mode "machine" --num_icl 3 -o "results_nl2sva_machine/3" -m "llama-3-70b;mixtral-8x22b;llama-2-70b;mixtral-8x7b"
python run_evaluation.py --task "nl2sva-human" -i "results_nl2sva_machine/3" -n 4