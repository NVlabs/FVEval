STRAT=$1
k=$2
# MODELS=("gpt-4o" "mixtral-8x22b" "mixtral-8x7b" "gpt-4-turbo")
MODELS=("claude-opus" "codellama-34b")

for MODEL in "${MODELS[@]}"; do
    (
        # Run the first script and wait for it to finish
        python run_design2sva.py -o "results_design2sva/${STRAT}_${k}" --cot_strategy "${STRAT}" -m "${MODEL}" -k "${k}"
        # Run the second script after the first has completed
        python run_evaluation.py --task "design2sva" -i "results_design2sva/${STRAT}_${k}" -m "${MODEL}" -n 1
    ) &
done
wait
