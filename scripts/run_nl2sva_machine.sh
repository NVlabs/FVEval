#!/bin/bash
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NUM_ICL=$1
USE_COT=$2
MODELS=("gpt-4o" "gemini-1.5-pro" "gemini-1.5-flash" "llama-3.1-8b" "llama-3.1-70b" "mixtral-8x22b")

if [ "$USE_COT" = "true" ]; then
    for MODEL in "${MODELS[@]}"; do
        (
            python run_nl2sva.py --mode "machine" --num_icl ${NUM_ICL} -o "results_nl2sva_machine/${NUM_ICL}" -m "${MODEL}" --use_cot
            python run_evaluation.py --task "nl2sva-machine" -i "results_nl2sva_machine/${NUM_ICL}_cot" --nparallel 1 -m "${MODEL}" &
        )&
    done
    wait
    exit 0
else
    for MODEL in "${MODELS[@]}"; do
        (
            python run_nl2sva.py --mode "machine" --num_icl ${NUM_ICL} -o "results_nl2sva_machine/${NUM_ICL}" -m "${MODEL}"
	    python run_evaluation.py --task "nl2sva-machine" -i "results_nl2sva_machine/${NUM_ICL}" --nparallel 1 -m "${MODEL}" &	    
        )&
    done
    wait
    exit 0
fi
