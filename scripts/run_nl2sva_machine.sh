NUM_ICL=$1
k=$2
# MODELS=("gpt-4o" "gpt-4" "gpt-3.5-turbo" "llama-3-70b" "mixtral-8x22b" "llama-3-8b" "mixtral-8x7b" "claude-opus" "codellama-34b") 
MODELS=("claude-opus")
# for MODEL in "${MODELS[@]}"; do
#     python run_nl2sva.py --mode "machine" --num_icl ${NUM_ICL} -o "results_nl2sva_machine/${NUM_ICL}_${k}" -m "${MODEL}" -k "${k}" &
# done
# wait

for MODEL in "${MODELS[@]}"; do
    python run_evaluation.py --task "nl2sva-machine" -i "results_nl2sva_machine/${NUM_ICL}_${k}" -n 1 -m "${MODEL}" &
done
wait
