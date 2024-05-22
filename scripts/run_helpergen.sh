python run_helpergen.py -o "results_helpergen/plan-act" --cot_strategy "plan-act" -m "gpt-4-turbo;gpt-3.5-turbo;gpt-4o"
python run_evaluation.py --task "helpergen" -i "results_helpergen/plan-act" -n 4