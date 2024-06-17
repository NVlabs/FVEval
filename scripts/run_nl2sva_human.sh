NUM_ICL=$1
k=$2
# MODELS=("codellama-34b" "gpt-4o" "gpt-4" "gpt-3.5-turbo" "llama-3-70b" "mixtral-8x22b" "mixtral-8x7b" "llama-3-8b")
MODELS=("codellama-34b" "gpt-4o" "gpt-4" "gpt-3.5-turbo")
# MODELS=("llama-3-70b" "mixtral-8x22b" "mixtral-8x7b" "llama-3-8b")

# MODELS=("gpt-4o" "gpt-3.5-turbo")

# for MODEL in "${MODELS[@]}"; do
#     python run_nl2sva.py --mode "human" --num_icl ${NUM_ICL} -o "results_nl2sva_human/${NUM_ICL}_${k}" -m "${MODEL}" -k "${k}" &
# done
# wait

for MODEL in "${MODELS[@]}"; do
    python run_evaluation.py --task "nl2sva-human" -i "results_nl2sva_human/${NUM_ICL}_${k}" -n 1 -m "${MODEL}" &
done
wait