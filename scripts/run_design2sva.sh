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

n=$1
MODELS=("gpt-4o" "gemini-1.5-pro" "gemini-1.5-flash" "llama-3.1-8b" "llama-3.1-70b" "mixtral-8x22b")


for MODEL in "${MODELS[@]}"; do
    (
        # Run the first script and wait for it to finish
        python run_design2sva.py -o "results_design2sva/default_${n}" --cot_strategy "default" -m "${MODEL}" --num_assertions "${n}"
        # Run the second script after the first has completed
         python run_evaluation.py --task "design2sva" -i "results_design2sva/default_${n}" -m "${MODEL}" --nparallel 1
    ) &
done
wait
