# FVEval: Evaluating LLMs on Hardware Formal Verification


## Installation
Public-facing versions of our repository will be based on the same OpenAI API to support a wider range of models (and endpoint servers), where only the base_url and API_keys need to be modified.

```{python}
# NVIDIA internal only
pip config set global.index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple
```

```
conda create -n fveval python=3.10
conda activate fveval
pip install -r requirements.txt
pip install -e .
```
##  Running Dataset Generation and Preprocessing

```{python}
# Note that all datasets are already generated and stored as .csv files in the 
# data_svagen/nl2sva/data
# data_svagen/design2sva/data 
# data_agr/helpergen/data
# directories. The following is for generating them new or again.

# (1) generate the NL2SVA-Human dataset
cd data_svagen/nl2sva && python generate_nl2sva_human.py && cd ../..

# (2) generate the NL2SVA-Machine dataset
cd data_svagen/nl2sva && python generate_nl2sva_machine.py && cd ../..

# (3) generate the Design2SVA dataset
cd data_svagen/design2sva && python generate_nl2sva_human.py && cd ../..

# (1) generate the NL2SVA-Human dataset
cd data_svagen/nl2sva && python generate_nl2sva_human.py && cd ../..
```



##  Running LLM generation on each task
```{python}
# Run the following commands in your conda environment created above
# You can supply a list of models to test with the --models flag, with model names ;-separated
# Run with --debug to print all input and outputs to and from the LM
# Change LLM decoding temperature with the --temperature flag
# You can also see the flag options available for each run script by passing the '-h' flag

# Running LM inference on the NL2SVA-machine (assertion generation from directed NL instructions) task:
python run_svagen_nl2sva.py --mode machine --models "gpt-4;mixtral-chat" 

# Running LM inference on the NL2SVA-Human (assertion generation from testbench and high-level instructions) task:
python run_svagen_nl2sva.py --mode human --models "gpt-4;mixtral-chat" 

# Running LM inference on the Design2SVA (SV testbench generation) task:
python run_svagen_design2sva.py --models "gpt-4;mixtral-chat" 
```



##  Repo Structure
Overview of the repository:
```
fv_eval/
├── fv_eval/
│   ├── benchmark_launcher.py (methods for consuming input bmark data and run LM inference)
│   ├── evaluation.py (methods for LM response evaluation)
│   ├── fv_tool_execution.py (methods for launching FV tools, i.e. JasperGold)
│   ├── data.py (definitions for input/output data)
│   ├── prompts_*.py  (default prompts for each subtask)
│   ├── utils.py (misc. util functions)
|
├── data_agr/ 
│   ├── helper_gen/
|       |── data/ 
│       |── generate_pipelines_helpergen.py 
│       |── generate_arbitration-clouds_helpergen.py 
│       |── generate_fsm_helpergen.py 
|
├── data_svagen/ 
│   ├── design2sva/
|   |   |── data/  
│   |   |── generate_pipelines_design2sva.py 
│   |   |── generate_arbitration-clouds_design2sva.py 
│   |   |── generate_fsm_design2sva.py 
│   ├── nl2sva/
│       |── annotated_instructions/ 
│       |── annotated_tb/ 
│       |── data/ 
│       |── machine_tb/ 
│       |── generate_nl2sva_human.py 
│       |── generate_nl2sva_machine.py 
|
├── tool_scripts/ 
│   ├── run_jg_design2sva.tcl
│   ├── run_jg_helpergen.tcl
│   ├── run_jg_nl2sva_human.tcl
│   ├── run_jg_nl2sva_machine.tcl
|
├── run_helpergen.py
├── run_evaluation.py
├── run_svagen_design2sva.py
├── run_svagen_nl2sva.py
├── run_evaluation.py
|
├── setup.py
└── README.md


```

## Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.
This work is made available under the Apache 2.0 License.